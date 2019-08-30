#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/refcount.h"

#include "tbb/concurrent_unordered_map.h"

#include "../utils/common.h"

namespace tensorflow
{
class HashTable : public ResourceBase {
public:
    HashTable(
            bool rehash,
            const std::vector<int64>& slot_hash_sizes,
            const std::vector<int>& occurrence_threshold) {
        rehash_ = rehash;
        slot_hash_sizes_ = slot_hash_sizes;
        occurrence_threshold_ = occurrence_threshold;
        multi_slot_ = slot_hash_sizes_.size() != 1;
        freq_filter_ = occurrence_threshold_.size() > 0;
        CHECK(!multi_slot_ || slot_hash_sizes_.size() == neo::MAX_SLOT);
        CHECK(!freq_filter_ || slot_hash_sizes_.size() == occurrence_threshold_.size());
        hash_count.resize(multi_slot_ ? neo::MAX_SLOT : 1);
    }

    void Hash(OpKernelContext *context) {
        auto instance_ids = context->input(1).flat<int64>();
        auto fids = context->input(2).flat<int64>();

        std::vector<int64_t> vec_insts, vec_fids;
        vec_insts.reserve(fids.dimension(0));
        vec_fids.reserve(fids.dimension(0));

        {
            mutex_lock l(mu_);
            uint64_t slot_id = 0;
            for (int64 i = 0; i < fids.dimension(0); ++i) {
                auto fid = static_cast<uint64_t>(fids(i));
                if (multi_slot_) slot_id = neo::get_slot(fid);
                if (freq_filter_) {
                    auto threshold = occurrence_threshold_[slot_id];
                    if (threshold < 0) continue;
                    if (threshold > 0) {
                        uint32_t count = ++filter_map[fid];
                        if (count <= threshold) continue;
                    }
                } else if (slot_hash_sizes_[slot_id] == 0) {
                    continue;
                }
                if (rehash_) {
                    auto it = hash_map.find(fid);
                    if (it == hash_map.end()) {
                        uint64_t val = (hash_count[slot_id] % slot_hash_sizes_[slot_id]) | (slot_id << neo::FEATURE_BIT);
                        hash_count[slot_id]++;
                        vec_fids.push_back(static_cast<int64_t>(val));
                        hash_map[fid] = val;
                    } else {
                        vec_fids.push_back(static_cast<int64_t>(it->second));
                    }
                } else {
                    vec_fids.push_back(fids(i));
                }
                vec_insts.push_back(instance_ids(i));
            }
        }

        Tensor *out_inst_ids_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, {vec_fids.size()}, &out_inst_ids_tensor));
        auto out_inst_ids = out_inst_ids_tensor->flat<int64>();
        Tensor *out_fids_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(1, {vec_fids.size()}, &out_fids_tensor));
        auto out_fids = out_fids_tensor->flat<int64>();

        std::memcpy(out_inst_ids.data(), vec_insts.data(), sizeof(int64)*vec_insts.size());
        std::memcpy(out_fids.data(), vec_fids.data(), sizeof(int64)*vec_fids.size());
    }

    void Export(OpKernelContext *context) {
        mutex_lock l(mu_);

        int64_t total_size = 1 + filter_map.size() * 2 + 1 + hash_map.size() * 2 + 1 + hash_count.size();
        Tensor *buff_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, {total_size}, &buff_tensor));
        auto buff = buff_tensor->flat<uint64>();

        size_t p = 0;
        buff(p++) = filter_map.size();
        for (const auto& kv : filter_map) {
            buff(p++) = kv.first;
            buff(p++) = kv.second;
        }
        
        buff(p++) = hash_map.size();
        for (const auto& kv : hash_map) {
            buff(p++) = kv.first;
            buff(p++) = kv.second;
        }

        buff(p++) = hash_count.size();
        for (const auto& i : hash_count) buff(p++) = i;
    }

    void Restore(OpKernelContext *context) {
        mutex_lock l(mu_);

        auto buff = context->input(1).flat<uint64>();
        size_t p = 0;

        uint64_t size = buff(p++);
        for (size_t i = 0; i < size; ++i) {
            filter_map[buff(p)] = buff(p+1);
            p += 2;
        }

        size = buff(p++);
        for (size_t i = 0; i < size; ++i) {
            hash_map[buff(p)] = buff(p+1);
            p += 2;
        }

        size = buff(p++);
        for (size_t i = 0; i < size; ++i) {
            hash_count[i] = buff(p++);
        }
    }

    std::string DebugString() override {
        return "A hash table";
    }

private:
    bool rehash_;
    bool multi_slot_;
    bool freq_filter_;
    std::vector<int64> slot_hash_sizes_;
    std::vector<int> occurrence_threshold_;

    mutex mu_;
    std::unordered_map<uint64_t, uint64_t> filter_map;
    std::unordered_map<uint64_t, uint64_t> hash_map;
    std::vector<uint64_t> hash_count;

    TF_DISALLOW_COPY_AND_ASSIGN(HashTable);
};

class HashTableCreateOp : public ResourceOpKernel<HashTable> {
public:
    explicit HashTableCreateOp(OpKernelConstruction *context) : ResourceOpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("rehash", &rehash_));
        OP_REQUIRES_OK(context, context->GetAttr("slot_hash_sizes", &slot_hash_sizes_));
        OP_REQUIRES_OK(context, context->GetAttr("occurrence_threshold", &occurrence_threshold_));
    }

private:
    Status CreateResource(HashTable** map) override EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        *map = new HashTable(rehash_, slot_hash_sizes_, occurrence_threshold_);
        if (*map == nullptr) {
            return errors::ResourceExhausted("Failed to allocate evaluator");
        }
        return Status::OK();
    }

    Status VerifyResource(HashTable* map) override EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        return Status::OK();
    }

private:
    bool rehash_;
    std::vector<int64> slot_hash_sizes_;
    std::vector<int> occurrence_threshold_;

    TF_DISALLOW_COPY_AND_ASSIGN(HashTableCreateOp);
};


class HashTableHashFidOp : public OpKernel {
public:
    explicit HashTableHashFidOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        HashTable* map;
        OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0), &map));
        core::ScopedUnref unref_eval(map);
        map->Hash(context);
    }
};


class HashTableExportOp : public OpKernel {
public:
    explicit HashTableExportOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext *context) override {
        HashTable* map;
        OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0), &map));
        core::ScopedUnref unref_eval(map);
        map->Export(context);
    }
};


class HashTableRestoreOp : public OpKernel {
public:
    explicit HashTableRestoreOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext *context) override {
        HashTable* map;
        OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0), &map));
        core::ScopedUnref unref_eval(map);
        map->Restore(context);
    }
};


REGISTER_KERNEL_BUILDER(Name("HashTableCreate").Device(DEVICE_CPU),
                        HashTableCreateOp);

REGISTER_KERNEL_BUILDER(Name("HashTableHashFid").Device(DEVICE_CPU),
                        HashTableHashFidOp);

REGISTER_KERNEL_BUILDER(Name("HashTableExport").Device(DEVICE_CPU),
                        HashTableExportOp);

REGISTER_KERNEL_BUILDER(Name("HashTableRestore").Device(DEVICE_CPU),
                        HashTableRestoreOp);

} // namespace tensorflow

