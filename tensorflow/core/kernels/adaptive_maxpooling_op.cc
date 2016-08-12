/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/adaptive_pooling_ops_common.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/adaptive_maxpooling_op_gpu.h"
#endif // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

const int kInvalidMaxPoolingIndex = -1;

template<typename Device, typename T>
static void SpatialAdaptiveMaxPoolWithArgMaxHelper(
    OpKernelContext* context, Tensor* output, Tensor* output_arg_max,
    Tensor* input_backprop, const Tensor& tensor_in, const Tensor& out_backprop,
    const AdaptivePoolParameters& params) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      EigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>>
      EigenIndexMatrixMap;

  ConstEigenMatrixMap in_mat(
      tensor_in.flat<T>().data(), params.depth,
      params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
  EigenMatrixMap out_mat(
      output->flat<T>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);
  EigenIndexMatrixMap out_arg_max_mat(
      output_arg_max->flat<int64>().data(), params.depth,
      params.out_width * params.out_height * params.tensor_in_batch);

  const DeviceBase::CpuWorkerThreads& worker_threads =
      *(context->device()->tensorflow_cpu_worker_threads());

  auto shard = [&params, &in_mat, &out_mat, &out_arg_max_mat, &input_backprop,
                &output_arg_max, &out_backprop](int64 start, int64 limit) {
    const int32 depth = params.depth;
    const int32 in_rows = params.tensor_in_rows;
    const int32 in_cols = params.tensor_in_cols;
    const int32 out_height = params.out_height;
    const int32 out_width = params.out_width;
    const std::vector<int64>& begin_rows = params.begin_rows;
    const std::vector<int64>& size_rows = params.size_rows;
    const std::vector<int64>& begin_cols = params.begin_cols;
    const std::vector<int64>& size_cols = params.size_cols;

    {
      // Initializes the output tensor with MIN<T>
      const int32 output_image_size = out_height * out_width * depth;
      EigenMatrixMap out_shard(out_mat.data() + start * output_image_size, 1,
                               (limit - start) * output_image_size);
      out_shard.setConstant(Eigen::NumTraits<T>::lowest());
      EigenIndexMatrixMap out_arg_max_shard(
          out_arg_max_mat.data() + start * output_image_size, 1,
          (limit - start) * output_image_size);
      out_arg_max_shard.setConstant(kInvalidMaxPoolingIndex);
    }

    for (int32 b = start; b < limit; ++b) {
      const int32 start_row = begin_rows[b];
      const int32 start_col = begin_cols[b];
      const int32 num_rows = size_rows[b];
      const int32 num_cols = size_cols[b];

      for (int h = 0; h < num_rows; ++h) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        const int h_start = int32(floor(float(h) / num_rows * out_height));
        const int h_end = int32(ceil(float(h + 1) / num_rows * out_height));

        for (int w = 0; w < num_cols; ++w) {
          const int w_start = int32(floor(float(w) / num_cols * out_width));
          const int w_end = int32(ceil(float(w + 1) / num_cols * out_width));

          // compute elementwise max
          const int in_index = (b * in_rows + h + start_row) * in_cols + w + start_col;
          for (int ph = h_start; ph < h_end; ++ph) {
            const int out_index_base = (b * out_height + ph) * out_width;
            for (int pw = w_start; pw < w_end; ++pw) {
              const int out_index = out_index_base + pw;
              for (int d = 0; d < depth; ++d) {
                const T& input_ref = in_mat.coeffRef(d, in_index);
                T& output_ref = out_mat.coeffRef(d, out_index);
                int64& out_arg_max_ref = out_arg_max_mat.coeffRef(d, out_index);
                if (output_ref < input_ref ||
                    out_arg_max_ref == kInvalidMaxPoolingIndex) {
                  output_ref = input_ref;
                  int input_offset = in_index * depth + d;
                  out_arg_max_ref = input_offset;
                }
              }
            }
          }
        }
      }
    }

    {
      auto input_backprop_flat = input_backprop->flat<T>();
      auto out_arg_max_flat = output_arg_max->flat<int64>();
      auto out_backprop_flat = out_backprop.flat<T>();

      // Initialize output to 0.
      const int in_size = in_rows * in_cols * depth;
      const int in_start = start * in_size;
      const int in_end = limit * in_size;
      EigenMatrixMap in_shard(input_backprop_flat.data() + in_start, 1,
                              in_end - in_start);
      in_shard.setConstant(T(0));

      // Backpropagate.
      const int out_size = out_height * out_width * depth;
      const int out_start = start * out_size;
      const int out_end = limit * out_size;
      for (int index = out_start; index < out_end; ++index) {
        int input_backprop_index = out_arg_max_flat(index);
        // Although this check is in the inner loop, it is worth its value
        // so we don't end up with memory corruptions. Our benchmark shows that
        // the performance impact is quite small
        CHECK(input_backprop_index >= in_start && input_backprop_index < in_end)
            << "Invalid input backprop index: " << input_backprop_index << ", "
            << in_start << ", " << in_end;
        input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
      }
    }
  };

  const int64 shard_cost = params.tensor_in_rows * params.tensor_in_cols *
                           params.depth;
  Shard(worker_threads.num_threads, worker_threads.workers,
        params.tensor_in_batch, shard_cost, shard);
}

REGISTER_KERNEL_BUILDER(Name("AdaptiveMaxPool")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<float>("T"),
                        SpatialAdaptiveMaxPoolingOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("AdaptiveMaxPool")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<Eigen::half>("T"),
                        SpatialAdaptiveMaxPoolingOp<CPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("AdaptiveMaxPool")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<double>("T"),
                        SpatialAdaptiveMaxPoolingOp<CPUDevice, double>);


template<typename Device, typename T>
class SpatialAdaptiveMaxPoolingGradOp;

template<typename T>
class SpatialAdaptiveMaxPoolingGradOp<CPUDevice, T> : public OpKernel {
 public:
  explicit SpatialAdaptiveMaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default SpatialAdaptiveMaxPoolingGradOp "
                                "only supports NHWC on device type ",
                                DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("output_shape", &shape_));
    OP_REQUIRES(context, shape_.size() == 4,
                errors::InvalidArgument("Output shape field must"
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, GetTensorDim(shape_, data_format_, 'N') == -1,
                errors::Unimplemented(
                    "Adaptive pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context, GetTensorDim(shape_, data_format_, 'C') == -1,
                errors::Unimplemented(
                    "Adaptive pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(3);
    const Tensor& out_backprop = context->input(4);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    TensorShape output_shape = tensor_in.shape();

    Tensor tensor_out_dup;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::v(),
                                          tensor_out.shape(), &tensor_out_dup));

    Tensor tensor_out_arg_max;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int64>::v(),
                                                   tensor_out.shape(),
                                                   &tensor_out_arg_max));

    AdaptivePoolParameters params{context, shape_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    SpatialAdaptiveMaxPoolWithArgMaxHelper<CPUDevice, T>(
        context, &tensor_out_dup, &tensor_out_arg_max, output, tensor_in,
        out_backprop, params);
  }

 private:
  std::vector<int32> shape_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("AdaptiveMaxPoolGrad")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<float>("T"),
                        SpatialAdaptiveMaxPoolingGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("AdaptiveMaxPoolGrad")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<Eigen::half>("T"),
                        SpatialAdaptiveMaxPoolingGradOp<CPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(Name("AdaptiveMaxPoolGrad")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<double>("T"),
                        SpatialAdaptiveMaxPoolingGradOp<CPUDevice, double>);

#if GOOGLE_CUDA

template<typename T>
class SpatialAdaptiveMaxPoolingOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit SpatialAdaptiveMaxPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default SpatialAdaptiveMaxPoolingOp "
                                "only supports NHWC on device type ",
                                DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("output_shape", &shape_));
    OP_REQUIRES(context, shape_.size() == 4,
                errors::InvalidArgument("Output shape field must"
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, GetTensorDim(shape_, data_format_, 'N') == -1,
                errors::Unimplemented(
                    "Adaptive pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context, GetTensorDim(shape_, data_format_, 'C') == -1,
                errors::Unimplemented(
                    "Adaptive pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);

    AdaptivePoolParameters params{context, shape_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, params.forward_output_shape(), &output));

    bool status = AdaptiveMaxPoolForwardWithOptionalArgmax(
        tensor_in.flat<T>().data(), params.tensor_in_batch, params.tensor_in_rows,
        params.tensor_in_cols, params.depth, params.out_height,
        params.out_width, output->flat<T>().data(),  nullptr,
        params.begin_rows.data(), params.size_rows.data(),
        params.begin_cols.data(), params.size_cols.data(),
        context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching AdaptiveMaxPoolForwardWithOptionalArgmax"));
    }
  }
 private:
  std::vector<int32> shape_;
  TensorFormat data_format_;
};

/*
REGISTER_KERNEL_BUILDER(Name("AdaptiveMaxPool")
                        .Device(DEVICE_GPU)
                        .HostMemory("begin")
                        .HostMemory("size")
                        .TypeConstraint<float>("T"),
                        SpatialAdaptiveMaxPoolingOp<GPUDevice, float>);
*/

template<typename T>
class SpatialAdaptiveMaxPoolingGradOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;

  explicit SpatialAdaptiveMaxPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default SpatialAdaptiveMaxPoolingGradOp "
                                "only supports NHWC on device type ",
                                DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("output_shape", &shape_));
    OP_REQUIRES(context, shape_.size() == 4,
                errors::InvalidArgument("Output shape field must"
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, GetTensorDim(shape_, data_format_, 'N') == -1,
                errors::Unimplemented(
                    "Adaptive pooling is not yet supported on the batch dimension."));
    OP_REQUIRES(context, GetTensorDim(shape_, data_format_, 'C') == -1,
                errors::Unimplemented(
                    "Adaptive pooling is not yet supported on the depth dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    const Tensor& tensor_out = context->input(3);
    const Tensor& out_backprop = context->input(4);

    // For maxpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));
    OP_REQUIRES(context, tensor_out.dims() == 4,
                errors::InvalidArgument("tensor_out must be 4-dimensional"));
    // For maxpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    AdaptivePoolParameters params{context, shape_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, tensor_in.shape(), &output));

    bool status = AdaptiveMaxPoolBackwardNoMask(
        tensor_in.flat<T>().data(), params.tensor_in_batch,
        params.tensor_in_rows, params.tensor_in_cols, params.depth,
        params.out_height, params.out_width,
        out_backprop.flat<T>().data(),
        output->flat<T>().data(),
        params.begin_rows.data(), params.size_rows.data(),
        params.begin_cols.data(), params.size_cols.data(),
        context->eigen_gpu_device());
    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching AdaptiveMaxPoolBackwardNoMask"));
    }
  }

 private:
  std::vector<int32> shape_;
  TensorFormat data_format_;
};

/*
REGISTER_KERNEL_BUILDER(Name("AdaptiveMaxPoolGrad")
                        .Device(DEVICE_GPU)
                        .HostMemory("begin")
                        .HostMemory("size")
                        .TypeConstraint<float>("T"),
                        SpatialAdaptiveMaxPoolingGradOp<GPUDevice, float>);
*/

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
