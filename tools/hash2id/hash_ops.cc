#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

REGISTER_OP("EulerHashFid")
    .Attr("use_locking: bool = true")
    .Input("ref: Ref(int64)")
    .Input("fids: int64")
    .Input("start: int64")
    .Input("end: int64")
    .Output("output_fids: int64");

REGISTER_OP("LagrangeHashTableCreate")
    .SetIsStateful()
    .Output("handle: resource")
    .Attr("rehash: bool")
    .Attr("slot_hash_sizes: list(int)")
    .Attr("occurrence_threshold: list(int)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("LagrangeHashTableHashFid")
    .Attr("Tidx: {int64}")
    .Input("handle: resource")
    .Input("instance_ids: Tidx")
    .Input("fids: Tidx")
    .Output("output_instance_ids: Tidx")
    .Output("output_fids: Tidx");

REGISTER_OP("LagrangeHashTableExport")
    .Input("handle: resource")
    .Output("buff: uint64");

REGISTER_OP("LagrangeHashTableRestore")
    .Input("handle: resource")
    .Input("buff: uint64");

} // namespace tensorflow
