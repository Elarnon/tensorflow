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

#include "tensorflow/core/kernels/adaptive_pooling_ops_common.h"

namespace tensorflow {

namespace {

void FillInt64Vecs(const Tensor& tensor, std::vector<int64> *rows, std::vector<int64> *cols) {
  auto batch_size = tensor.dim_size(0);
  if (tensor.dtype() == DT_INT32) {
    auto vector = tensor.shaped<int32, 2>({batch_size, 2});
    for (int64 i = 0; i < batch_size; ++i) {
      rows->push_back(vector(i, 0));
      cols->push_back(vector(i, 1));
    }
  } else if (tensor.dtype() == DT_INT64) {
    auto vector = tensor.shaped<int64, 2>({batch_size, 2});
    for (int64 i = 0; i < batch_size; ++i) {
      rows->push_back(vector(i, 0));
      cols->push_back(vector(i, 1));
    }
  } else {
    LOG(FATAL) << "tensor must be either int32 or int64";
  }
}

} // namespace

AdaptivePoolParameters::AdaptivePoolParameters(OpKernelContext* context,
                                               const std::vector<int32>& output_shape,
                                               TensorFormat data_format,
                                               const TensorShape& tensor_in_shape) {
  const Tensor& begin_tensor = context->input(1);
  const Tensor& size_tensor = context->input(2);
  OP_REQUIRES(context, tensor_in_shape.dims() == 4,
              errors::InvalidArgument("tensor_in must be 4-dimensional"));

  OP_REQUIRES(context, GetTensorDim(output_shape, data_format, 'N') == -1,
              errors::Unimplemented(
                  "Adaptive pooling is not yet supported on the batch dimension."));
  OP_REQUIRES(context, GetTensorDim(output_shape, data_format, 'C') == -1,
              errors::Unimplemented(
                  "Adaptive pooling is not yet supported on the depth dimension."));

  this->data_format = data_format;
  depth = GetTensorDim(tensor_in_shape, data_format, 'C');
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');

  out_height = GetTensorDim(output_shape, data_format, 'H');
  out_width = GetTensorDim(output_shape, data_format, 'W');

  OP_REQUIRES(context, -1 <= out_height && -1 <= out_width,
              errors::InvalidArgument("Expected output_shape are -1 or nonnegative values ",
                                      "but got ", out_height, " ", out_width));

  if (out_height == -1)
    out_height = tensor_in_rows;
  if (out_width == -1)
    out_width = tensor_in_cols;

  OP_REQUIRES(context,
              (
                  begin_tensor.shape().dims() == 2 &&
                  begin_tensor.shape().dim_size(0) == tensor_in_batch &&
                  begin_tensor.shape().dim_size(1) == 2 &&
                  size_tensor.shape().dims() == 2 &&
                  size_tensor.shape().dim_size(0) == tensor_in_batch &&
                  size_tensor.shape().dim_size(1) == 2
               ),
              errors::InvalidArgument(
                  "Expected begin and size arguments to be 2-D tensors of size (",
                  GetTensorDim(tensor_in_shape, data_format, 'N'), ", 2) but got shapes",
                  begin_tensor.shape().DebugString(),
                  " and ", size_tensor.shape().DebugString(), " instead."));

  FillInt64Vecs(begin_tensor, &begin_rows, &begin_cols);
  FillInt64Vecs(size_tensor, &size_rows, &size_cols);

  for (int i = 0; i < tensor_in_batch; ++i) {
    // A size of -1 means "all elements from begin to dim_size"
    if (size_rows[i] == -1) {
      size_rows[i] = tensor_in_rows - begin_rows[i];
    }
    int64 b = begin_rows[i];
    int64 s = size_rows[i];
    OP_REQUIRES(
        context, 0 <= b && b <= tensor_in_rows,
        errors::InvalidArgument("Expected begin[", i, ", 0] in [0, ",
                                tensor_in_rows, "] but got ", b));
    OP_REQUIRES(
        context, 0 <= s && b + s <= tensor_in_rows,
        errors::InvalidArgument("Expected begin[", i, ", 0] in [0, ",
                                tensor_in_rows - b, "] but got ", s));
  }

  for (int i = 0; i < tensor_in_batch; ++i) {
    // A size of -1 means "all elements from begin to dim_size"
    if (size_cols[i] == -1) {
      size_cols[i] = tensor_in_cols - begin_cols[i];
    }
    int64 b = begin_cols[i];
    int64 s = size_cols[i];
    OP_REQUIRES(
        context, 0 <= b && b <= tensor_in_cols,
        errors::InvalidArgument("Expected begin[", i, ", 1] in [0, ",
                                tensor_in_cols, "] but got ", b));
    OP_REQUIRES(
        context, 0 <= s && b + s <= tensor_in_cols,
        errors::InvalidArgument("Expected begin[", i, ", 1] in [0, ",
                                tensor_in_cols - b, "] but got ", s));
  }
}

TensorShape AdaptivePoolParameters::forward_output_shape() {
  return ShapeFromFormat(data_format, tensor_in_batch, out_height, out_width,
                         depth);
}

} // namespace tensorflow
