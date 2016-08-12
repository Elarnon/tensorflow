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

#ifndef TENSORFLOW_KERNELS_ADAPTIVE_POOLING_OPS_COMMON_H_
#define TENSORFLOW_KERNELS_ADAPTIVE_POOLING_OPS_COMMON_H_

#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// A helper class to manage sizes and shapes for adaptive pooling operations.
struct AdaptivePoolParameters {
  // Updates context->status if there is an invalid input.
  AdaptivePoolParameters(OpKernelContext* context,
                         const std::vector<int32>& output_shape,
                         TensorFormat data_format,
                         const TensorShape& tensor_in_shape);

  // Returns the shape of the output for "forward" pooling operations.
  TensorShape forward_output_shape();

  int depth;

  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  std::vector<int64> begin_rows;
  std::vector<int64> size_rows;
  std::vector<int64> begin_cols;
  std::vector<int64> size_cols;

  int64 out_height;
  int64 out_width;

  TensorFormat data_format;
};

template<typename Device, typename T>
class SpatialAdaptiveMaxPoolingOp;

// An implementation of AdaptiveMaxPooling (forward).
template <typename T>
class SpatialAdaptiveMaxPoolingOp<CPUDevice, T> : public OpKernel {
 public:
  typedef CPUDevice Device;
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

    AdaptivePoolParameters params{context, shape_, FORMAT_NHWC, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, params.forward_output_shape(), &output));

    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        ConstEigenMatrixMap;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
        EigenMatrixMap;

    ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), params.depth,
                               params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
    EigenMatrixMap out_mat(
        output->flat<T>().data(), params.depth,
        params.out_width * params.out_height * params.tensor_in_batch);

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());

    // The following code basically does the following:
    // 1. Flattens the input and output tensors into two dimensional arrays.
    //    tensor_in_as_matrix:
    //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
    //    output_as_matrix:
    //      depth by (out_width * out_height * tensor_in_batch)
    //
    // 2. Walks through the set of columns in the flattened
    // tensor_in_as_matrix,
    //    and updates the corresponding column(s) in output_as_matrix with the
    //    max value.
    auto shard = [&params, &in_mat, &out_mat](int64 start, int64 limit) {
      const int32 in_rows = params.tensor_in_rows;
      const int32 in_cols = params.tensor_in_cols;
      const int32 out_height = params.out_height;
      const int32 out_width = params.out_width;
      const std::vector<int64>& begin_rows = params.begin_rows;
      const std::vector<int64>& size_rows = params.size_rows;
      const std::vector<int64>& begin_cols = params.begin_cols;
      const std::vector<int64>& size_cols = params.size_cols;

      {
        // Initialize the output tensor with MIN<T>
        const int32 output_image_size = out_height * out_width * params.depth;
        EigenMatrixMap out_shard(out_mat.data() + start * output_image_size,
                                 1, (limit - start) * output_image_size);
        out_shard.setConstant(Eigen::NumTraits<T>::lowest());
      }

      for (int32 b = start; b < limit; ++b) {
        const int32 out_offset_batch = b * out_height;
        const int32 start_row = begin_rows[b];
        const int32 start_col = begin_cols[b];
        const int32 num_rows = size_rows[b];
        const int32 num_cols = size_cols[b];

        for (int32 h = 0; h < num_rows; ++h) {
          // (h_start, h_end) * (w_start, w_end) is the range that the input
          // vector projects to.
          const int32 h_start = int32(floor(float(h) / num_rows * out_height));
          const int32 h_end = int32(ceil(float(h + 1) / num_rows * out_height));

          for (int32 w = 0; w < num_cols; ++w) {
            const int32 w_start = int32(floor(float(w) / num_cols * out_width));
            const int32 w_end = int32(ceil(float(w + 1) / num_cols * out_width));

            // compute elementwise max
            const int32 in_offset = (b * in_rows + h + start_row) * in_cols + w + start_col;
            for (int32 ph = h_start; ph < h_end; ++ph) {
              const int32 out_offset_base =
                  (out_offset_batch + ph) * out_width;
              for (int32 pw = w_start; pw < w_end; ++pw) {
                const int32 out_offset = out_offset_base + pw;
                out_mat.col(out_offset) =
                    out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
              }
            }
          }
        }
      }
    };

    // TODO(?) Consider sharding across batch x rows x cols.
    // TODO(?) Consider a higher resolution shard cost model.
    const int64 shard_cost =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    Shard(worker_threads.num_threads, worker_threads.workers,
          params.tensor_in_batch, shard_cost, shard);
  }

 private:
  std::vector<int32> shape_;
  TensorFormat data_format_;
};

} // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ADAPTIVE_POOLING_OPS_COMMON_H_
