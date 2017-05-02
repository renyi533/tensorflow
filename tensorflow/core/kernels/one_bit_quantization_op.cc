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


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class OneBitQuantizationOp : public OpKernel {
 public:
  explicit OneBitQuantizationOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& gradients = ctx->input(0);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(gradients.shape()),
                errors::InvalidArgument("gradients is not a matrix"));

    auto gradients_flat = gradients.flat<T>();

    Tensor* row_cnt = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({1}), &row_cnt));

    auto row_count_flat = row_cnt->flat<int64>();

    row_count_flat(0) = gradients.dim_size(0);
    const int64 row_count = row_count_flat(0);
    const int64 col_count = gradients.dim_size(1);
    const int64 row_count_compress = row_count / 8 + (row_count%8 !=0? 1 : 0);

    TensorShape out_compress_shape(
        {row_count_compress, col_count});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_compress_shape, &out));
    auto out_flat = out->flat<uint8>();


    TensorShape gradient_mean_shape(
        {2, col_count});
    Tensor* gradient_mean = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, gradient_mean_shape, &gradient_mean));
    auto gradient_mean_flat = gradient_mean->flat<T>();

    
    for (int i=0; i<col_count; i++)
    {
      gradient_mean_flat(get_idx(0, i, col_count)) = 0.0;
      gradient_mean_flat(get_idx(1, i, col_count)) = 0.0;
      for (int j=0; j<row_count; j++)
      {
        int64 row_idx_comp = j/8;
        int64 offset = j%8;
        if (gradients_flat(get_idx(j, i, col_count)) >0)
        {
          gradient_mean_flat(get_idx(0, i, col_count)) = 
              gradient_mean_flat(get_idx(0, i, col_count))+gradients_flat(get_idx(j, i, col_count));

          out_flat(get_idx(row_idx_comp, i, col_count)) = 
              out_flat(get_idx(row_idx_comp, i, col_count)) | 1<<offset;  
        }
        else
        {
          gradient_mean_flat(get_idx(1, i, col_count)) = 
              gradient_mean_flat(get_idx(1, i, col_count))+gradients_flat(get_idx(j, i, col_count));

          out_flat(get_idx(row_idx_comp, i, col_count)) = 
              out_flat(get_idx(row_idx_comp, i, col_count)) & ~(1<<offset);         
        }
      }
      gradient_mean_flat(get_idx(0, i, col_count)) = gradient_mean_flat(get_idx(0, i, col_count))/row_count;
      gradient_mean_flat(get_idx(1, i, col_count)) = gradient_mean_flat(get_idx(1, i, col_count))/row_count;
    }
  }
  

 private:
  int64 get_idx(int64 row, int64 col, int64 col_count)
  {
    return row*col_count+col;
  }
};

template <typename Device, typename T>
class OneBitDequantizationOp : public OpKernel {
 public:
  explicit OneBitDequantizationOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& row_cnt = ctx->input(0);
    const Tensor& one_bit_data = ctx->input(1);
    const Tensor& gradients_mean = ctx->input(2);

    const int64 row_count = row_cnt.flat<int64>()(0);
    const int64 col_count = one_bit_data.dim_size(1);


    OP_REQUIRES(ctx, one_bit_data.dim_size(1)==gradients_mean.dim_size(1),
                errors::InvalidArgument("inconsistent column count for gradients_mean/one_bit_data"));

    Tensor* gradients = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({row_count, col_count}), &gradients));
    auto gradients_flat = gradients->flat<T>();

    auto gradient_mean_flat = gradients_mean.flat<T>();
    auto one_bit_data_flat = one_bit_data.flat<uint8>();

    for (int i=0; i<col_count; i++)
    {
      for (int j=0; j<row_count; j++)
      {
        int64 row_idx_comp = j/8;
        int64 offset = j%8;

        bool isPos = one_bit_data_flat(get_idx(row_idx_comp, i, col_count)) & 1<<offset;
        if (isPos)
        {
          gradients_flat(get_idx(j, i, col_count)) = gradient_mean_flat(get_idx(0, i, col_count));
        }
        else
        {
          gradients_flat(get_idx(j, i, col_count)) = gradient_mean_flat(get_idx(1, i, col_count));
        }
      }
    }
  }
  

 private:
  int64 get_idx(int64 row, int64 col, int64 col_count)
  {
    return row*col_count+col;
  }
};


#define REGISTER_CPU_QUANTIZATION(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("OneBitQuantization").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
      OneBitQuantizationOp<CPUDevice, T>);            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("OneBitQuantization").Device(DEVICE_CPU).TypeConstraint<T>("T").Label("eigen"), \
      OneBitQuantizationOp<CPUDevice, T>)

#define REGISTER_CPU_DEQUANTIZATION(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("OneBitDequantization").Device(DEVICE_CPU).TypeConstraint<T>("T"),                \
      OneBitDequantizationOp<CPUDevice, T>);            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("OneBitDequantization").Device(DEVICE_CPU).TypeConstraint<T>("T").Label("eigen"), \
      OneBitDequantizationOp<CPUDevice, T>)


TF_CALL_float(REGISTER_CPU_QUANTIZATION);
TF_CALL_double(REGISTER_CPU_QUANTIZATION);

TF_CALL_float(REGISTER_CPU_DEQUANTIZATION);
TF_CALL_double(REGISTER_CPU_DEQUANTIZATION);

}  // namespace tensorflow

