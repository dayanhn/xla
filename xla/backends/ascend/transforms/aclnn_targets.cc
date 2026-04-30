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

#include "xla/backends/ascend/transforms/aclnn_targets.h"

namespace xla {
namespace ascend {

bool IsAclnnTarget(absl::string_view target) {
  constexpr absl::string_view prefix = "__aclnn$";
  return target.size() >= prefix.size() &&
         target.substr(0, prefix.size()) == prefix;
}

bool IsAclnnGemmTarget(absl::string_view target) {
  return target == kAclnnGemmCallTarget;
}

bool IsAclnnConvolutionTarget(absl::string_view target) {
  return target == kAclnnConvolutionCallTarget;
}

bool IsAclnnConvolutionBackwardTarget(absl::string_view target) {
  return target == kAclnnConvolutionBackwardCallTarget;
}

}  // namespace ascend
}  // namespace xla
