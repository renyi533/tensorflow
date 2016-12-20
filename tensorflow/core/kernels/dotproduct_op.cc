/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/dotproduct_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class DotProductOp : public OpKernel {
 public:
  explicit DotProductOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {

  }

 private:
};

template <typename Device>
class DotProductOp<Device, float> : public OpKernel {
 public:
  explicit DotProductOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a.shape()),
                errors::InvalidArgument("In[0] is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(b.shape()),
                errors::InvalidArgument("In[1] is not a vector"));

    OP_REQUIRES(ctx,
                a.dim_size(0) == b.dim_size(0),
                errors::InvalidArgument("vector size-incompatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));

    TensorShape out_shape({1});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      return;
    }

    auto a_ptr = a.flat<float>().data();
    auto b_ptr = b.flat<float>().data();
    auto c_ptr = out->flat<float>().data();  
    *c_ptr = cblas_sdot (a.dim_size(0), a_ptr, 1, b_ptr, 1); 
  }

 private:
};

template <typename Device>
class DotProductOp<Device, double> : public OpKernel {
 public:
  explicit DotProductOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a.shape()),
                errors::InvalidArgument("In[0] is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(b.shape()),
                errors::InvalidArgument("In[1] is not a vector"));

    OP_REQUIRES(ctx,
                a.dim_size(0) == b.dim_size(0),
                errors::InvalidArgument("vector size-incompatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));

    TensorShape out_shape({1});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      return;
    }

    auto a_ptr = a.flat<double>().data();
    auto b_ptr = b.flat<double>().data();
    auto c_ptr = out->flat<double>().data();  
    *c_ptr = cblas_ddot (a.dim_size(0), a_ptr, 1, b_ptr, 1); 
  }

 private:
};

#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("BlasDotProduct").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
      DotProductOp<CPUDevice, T>);            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("BlasDotProduct").Device(DEVICE_CPU).TypeConstraint<T>("T").Label("eigen"), \
      DotProductOp<CPUDevice, T>)


TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

}  // namespace tensorflow
