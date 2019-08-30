#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow
{

class EulerHashFidOp : public OpKernel {
 public:
  explicit EulerHashFidOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES(c, IsRefType(c->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));  
  }

  void Compute(OpKernelContext* c) override {
    if (use_exclusive_lock_) {
      // Hold mutex while we apply updates
      mutex_lock l(*c->input_ref_mutex(0));
      DoCompute(c);
    } else {
      DoCompute(c);
    }
  }

 private:
  bool use_exclusive_lock_;

  void DoCompute(OpKernelContext* c) {
    Tensor params = c->mutable_input(0, use_exclusive_lock_);
    auto params_flat = params.flat<int64>();
    int64 param_size = params.shape().dim_size(0) - 1;

    auto fids = c->input(1).flat<int64>();
    auto start = c->input(2).flat<int64>();
    auto end = c->input(3).flat<int64>();

    auto start_ = start(0);
    auto end_ = end(0);

    OP_REQUIRES(c, params.IsInitialized(),
              errors::FailedPrecondition("Null ref for params"));
    OP_REQUIRES(c, TensorShapeUtils::IsMatrix(params.shape()),
              errors::InvalidArgument("params must be 2-D, got shape: ",
                                      params.shape().DebugString()));
    OP_REQUIRES(c, params.shape().dim_size(1) == 2,
              errors::InvalidArgument("params must be 2-D with 2 columns"));

    OP_REQUIRES(c, start_ <= end_, errors::InvalidArgument("start > end"));
    OP_REQUIRES(c, param_size >= 2 * (end_ - start_ + 1),
                errors::FailedPrecondition("hash space too low"));
    
    Tensor *out_fids_tensor = nullptr;
    OP_REQUIRES_OK(c,
                   c->allocate_output(0, {fids.dimension(0)}, &out_fids_tensor));
    auto out_fids = out_fids_tensor->flat<int64>();

    for (int64 i = 0; i < fids.dimension(0); ++i) {
      auto fid = fids(i);
      int64 s_idx = fid % param_size;
      while (params_flat(s_idx * 2) != 0 && 
        params_flat(s_idx * 2) != fid) {
        s_idx += 1;
        s_idx %= param_size;
      }
      
      if (params_flat(s_idx * 2) == fid) {
        out_fids(i) = params_flat(s_idx * 2 + 1);
      }
      else {
        OP_REQUIRES(c, params_flat(param_size*2) < end_-start_+ 1,
                    errors::FailedPrecondition("Out of Id space"));
        params_flat(s_idx * 2) = fid;
        params_flat(s_idx * 2 + 1) = start_ + params_flat(param_size*2);
        params_flat(param_size*2)++;
        out_fids(i) = params_flat(s_idx * 2 + 1);
      }              
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("EulerHashFid").Device(DEVICE_CPU),
            EulerHashFidOp);
} // namespace tensorflow

