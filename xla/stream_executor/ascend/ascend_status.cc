/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/ascend/ascend_status.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace stream_executor {
namespace ascend {

absl::Status ToStatus(aclError error, const char* message) {
  if (error == ACL_ERROR_NONE) {
    return absl::OkStatus();
  }

  std::string error_message;
  if (message != nullptr) {
    error_message = message;
  } else {
    error_message = "ACL error";
  }

  switch (error) {
    case ACL_ERROR_INVALID_PARAM:
      return absl::InvalidArgumentError(absl::StrCat(error_message, ": invalid parameter"));
    case ACL_ERROR_RT_FAIL:
      return absl::InternalError(absl::StrCat(error_message, ": runtime failure"));
    case ACL_ERROR_DEVICE_NOT_FOUND:
      return absl::NotFoundError(absl::StrCat(error_message, ": device not found"));
    case ACL_ERROR_MEMORY_ALLOCATION:
      return absl::ResourceExhaustedError(absl::StrCat(error_message, ": memory allocation failed"));
    default:
      return absl::InternalError(absl::StrCat(error_message, ": error code ", error));
  }
}

}  // namespace ascend
}  // namespace stream_executor