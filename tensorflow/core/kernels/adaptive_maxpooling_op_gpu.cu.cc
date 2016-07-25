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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/adaptive_maxpooling_op_gpu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace {
// This is Yangqing's custom kernel for the maxpooling operation. There are
// three functions: MaxPoolForwardNCHW and MaxPoolForwardNHWC are the two
// forward functions, dealing with the forward case. MaxPoolBackward is the
// backward function that deals with the backward case for both storage orders.
// The parameters to the kernels in the forward function is as follows:
//     nthreads: the number of threads, which is equal to the output size.
//     bottom_data: the bottom data of N*H*W*C (or N*C*H*W) items.
//     height, width, pooled_height, pooled_width: the input and output sizes.
//     kernel_h, kernel_w: the kernel sizes.
//     stride_h, stride_w: the strides.
//     pad_t, pad_l: the padding values on the top and left side.
//     top_data: the maxpool output.
//     mask: the output mask of the same size as top_data. It is stored in
//         int form, keeping track of the flattened index of the input item that
//         produces the max output. If a nullptr is passed in for mask, no mask
//         will be produced.
//
// To call the forward and backward functions, use e.g.:
// const int kThreadsPerBlock = 1024
// const int output_size = batch * channels * pooled_height * pooled_width;
// MaxPoolForwardNCHW<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
//                      kThreadsPerBlock, 0, cuda_stream>>>(...);
template <typename dtype>
__global__ void AdaptiveMaxPoolForwardNCHW(
    const int nthreads, const dtype* bottom_data, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, dtype* top_data, int64* mask) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = int(floor(float(ph) / height * pooled_height));
    int wstart = int(floor(float(pw) / width * pooled_width));
    int hend = int(ceil(float(ph + 1) / height * pooled_height));
    int wend = int(ceil(float(pw + 1) / width * pooled_width));
    dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const dtype* bottom_data_n = bottom_data + n * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = c * height * width + h * width + w;
        if (bottom_data_n[idx] > maxval) {
          maxidx = idx;
          maxval = bottom_data_n[idx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask != nullptr) {
      mask[index] = maxidx;
    }
  }
}

template <typename dtype>
__global__ void AdaptiveMaxPoolForwardNHWC(
    const int nthreads, const dtype* bottom_data, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, dtype* top_data, int64* mask) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = int(floor(float(n % pooled_width) / width * pooled_width));
    int wend = int(ceil(float((n % pooled_width) + 1) / width * pooled_width));
    n /= pooled_width;
    int hstart = int(floor(float(n % pooled_height) / height * pooled_height));
    int hend = int(ceil(float((n % pooled_height) + 1) / height * pooled_height));
    n /= pooled_height;
    dtype maxval = Eigen::NumTraits<dtype>::lowest();
    int maxidx = -1;
    const dtype* bottom_data_n = bottom_data + n * height * width * channels;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = (h * width + w) * channels + c;
        if (bottom_data_n[idx] > maxval) {
          maxidx = idx;
          maxval = bottom_data_n[idx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask != nullptr) {
      mask[index] = maxidx;
    }
  }
}

template <typename dtype>
__global__ void AdaptiveMaxPoolBackwardNoMaskNHWC(
    const int nthreads, const dtype* bottom_data, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const dtype* top_diff, dtype* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // First find out the index to the maximum, since we have no mask.
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = int(floor(float(n % pooled_width) / width * pooled_width));
    int wend = int(ceil(float((n % pooled_width) + 1) / width * pooled_width));
    n /= pooled_width;
    int hstart = int(floor(float(n % pooled_height) / height * pooled_height));
    int hend = int(ceil(float((n % pooled_height) + 1) / height * pooled_height));
    n /= pooled_height;
    dtype maxval = Eigen::NumTraits<dtype>::lowest();
    int maxidx = -1;
    const dtype* bottom_data_n = bottom_data + n * height * width * channels;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = (h * width + w) * channels + c;
        if (bottom_data_n[idx] > maxval) {
          maxidx = idx;
          maxval = bottom_data_n[idx];
        }
      }
    }

    // Atomically accumulate the bottom diff. The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      CudaAtomicAdd(bottom_diff + n * height * width * channels + maxidx,
                    top_diff[index]);
    }
  }
}

// The parameters to the kernels in the backward function is as follows:
//     nthreads: the number of threads, which is equal to the output size.
//     top_diff: the gradient of the output data, of size N*Hout*Wout*C (or
//        N*C*Hout*Wout). As we have stored the flattened index of the input
//        entries, the backward function is agnostic of the input storage order.
//     mask: the output mask of the same size as top_data. It is stored in
//         int form, keeping track of the flattened index of the input item that
//         produces the max output.
//     top_offset: the pre-computed per-image offset of the maxpool output. This
//         is equal to Hout*Wout*C. We choose to pre-compute this so we do not
//         need to compute it every time inside the kernel.
//     bottom_offset: the pre-computed per-image offset of the maxpool input.
//         This is equal to H*W*C.
//     bottom_diff: the gradient with respect to the input.
// This function relies on CudaAtomicAdd to avoid race conditions. Also, before
// the kernel is run, you will need to make sure that bottom_diff is filled with
// zero first.
template <typename dtype>
__global__ void AdaptiveMaxPoolBackward(
    const int nthreads, const dtype* top_diff,
    const int64* mask, const int top_offset,
    const int bottom_offset, dtype* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int image_id = (index / top_offset);
    CudaAtomicAdd(bottom_diff + image_id * bottom_offset + mask[index],
                  top_diff[index]);
  }
}

#undef CUDA_1D_KERNEL_LOOP
}  // namespace

bool AdaptiveMaxPoolForwardWithOptionalArgmax(
    const float* bottom_data, const int batch, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, float* top_data, int64* mask,
    const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int output_size = batch * channels * pooled_height * pooled_width;

  AdaptiveMaxPoolForwardNHWC<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, height, width, channels, pooled_height,
      pooled_width, top_data, mask);
  return d.ok();
}

bool AdaptiveMaxPoolBackwardNoMask(
    const float* bottom_data, const int batch, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* top_diff, float* bottom_diff,
    const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int bottom_size = batch * channels * height * width;
  const int top_size = batch * channels * pooled_height * pooled_width;

  SetZero<<<(bottom_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
            kThreadsPerBlock, 0, d.stream()>>>(bottom_size, bottom_diff);

  AdaptiveMaxPoolBackwardNoMaskNHWC<<<(top_size + kThreadsPerBlock - 1) /
                                  kThreadsPerBlock,
                              kThreadsPerBlock, 0, d.stream()>>>(
      top_size, bottom_data, height, width, channels, pooled_height,
      pooled_width, top_diff, bottom_diff);
  return d.ok();
}

bool AdaptiveMaxPoolBackwardWithArgmax(const int output_size, const int input_size,
                                       const float* top_diff, const int64* mask,
                                       const int top_offset, const int bottom_offset,
                                       float* bottom_diff, const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  SetZero<<<(input_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
            kThreadsPerBlock, 0, d.stream()>>>(input_size, bottom_diff);
  AdaptiveMaxPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                    kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, mask, top_offset, bottom_offset, bottom_diff);
  return d.ok();
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
