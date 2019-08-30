#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
using namespace tensorflow::shape_inference;

REGISTER_OP("EulerHashFid")
    .Attr("use_locking: bool = true")
    .Input("ref: Ref(int64)")
    .Input("fids: int64")
    .Input("start: int64")
    .Input("end: int64")
    .Output("output_fids: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &x));
      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("HashTableCreate")
    .SetIsStateful()
    .Output("handle: resource")
    .Attr("rehash: bool")
    .Attr("slot_hash_sizes: list(int)")
    .Attr("occurrence_threshold: list(int)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("HashTableHashFid")
    .Attr("Tidx: {int64}")
    .Input("handle: resource")
    .Input("instance_ids: Tidx")
    .Input("fids: Tidx")
    .Output("output_instance_ids: Tidx")
    .Output("output_fids: Tidx");

REGISTER_OP("HashTableExport")
    .Input("handle: resource")
    .Output("buff: uint64");

REGISTER_OP("HashTableRestore")
    .Input("handle: resource")
    .Input("buff: uint64");

} // namespace tensorflow
