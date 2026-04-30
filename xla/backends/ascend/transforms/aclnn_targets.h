/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_TARGETS_H_
#define XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_TARGETS_H_

#include "absl/strings/string_view.h"

namespace xla {
namespace ascend {

// ACLNN target constants
constexpr absl::string_view kAclnnGemmCallTarget = "__aclnn$gemm";
constexpr absl::string_view kAclnnConvolutionCallTarget = "__aclnn$convolution";
constexpr absl::string_view kAclnnConvolutionBackwardCallTarget = "__aclnn$convolution_backward";

// Check if a custom call target is an ACLNN target
bool IsAclnnTarget(absl::string_view target);

// Check if a custom call target is an ACLNN GEMM target
bool IsAclnnGemmTarget(absl::string_view target);

// Check if a custom call target is an ACLNN Convolution target
bool IsAclnnConvolutionTarget(absl::string_view target);

// Check if a custom call target is an ACLNN ConvolutionBackward target
bool IsAclnnConvolutionBackwardTarget(absl::string_view target);

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_TARGETS_H_
