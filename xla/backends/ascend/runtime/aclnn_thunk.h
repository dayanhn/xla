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

#ifndef XLA_BACKENDS_ASCEND_RUNTIME_ACLNN_THUNK_H_
#define XLA_BACKENDS_ASCEND_RUNTIME_ACLNN_THUNK_H_

#include <variant>

#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnn/acl_meta.h"
#include "absl/types/span.h"

namespace xla {
namespace ascend {

// A thunk that executes aclnn operations directly
class AclnnThunk : public gpu::Thunk {
 public:
  // Parameter type for aclnn operations
  using Param = std::variant<
      aclTensor*,           // Tensor parameter
      float,                // Float parameter
      int64_t,              // Integer parameter
      bool,                 // Boolean parameter
      int8_t,               // Int8 parameter (e.g., cubeMathType)
      std::vector<int64_t>, // Dimensions parameter
      std::vector<bool>     // Boolean mask parameter (e.g., outputMask)
  >;

  AclnnThunk(gpu::Thunk::ThunkInfo thunk_info, std::string op_name,
             std::vector<NullableShapedSlice> operands,
             std::vector<NullableShapedSlice> results,
             std::vector<Param> params);

  // Execute the aclnn operation
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::string op_name_;  // Name of the aclnn operation
  std::vector<NullableShapedSlice> operands_;  // Input operands
  std::vector<NullableShapedSlice> results_;  // Output results
  std::vector<Param> params_;  // Additional parameters for the operation
};

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_RUNTIME_ACLNN_THUNK_H_
