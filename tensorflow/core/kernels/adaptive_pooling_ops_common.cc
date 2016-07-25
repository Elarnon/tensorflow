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

AdaptivePoolParameters::AdaptivePoolParameters(OpKernelContext* context,
                                               const std::vector<int32>& output_shape,
                                               TensorFormat data_format,
                                               const TensorShape& tensor_in_shape) {
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

  if (out_height < 0)
    out_height = tensor_in_rows;
  if (out_width < 0)
    out_width = tensor_in_cols;
}

TensorShape AdaptivePoolParameters::forward_output_shape() {
  return ShapeFromFormat(data_format, tensor_in_batch, out_height, out_width,
                         depth);
}

} // namespace tensorflow
