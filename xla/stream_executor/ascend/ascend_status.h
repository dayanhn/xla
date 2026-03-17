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

#ifndef XLA_STREAM_EXECUTOR_ASCEND_ASCEND_STATUS_H_ 
#define XLA_STREAM_EXECUTOR_ASCEND_ASCEND_STATUS_H_

#include "absl/status/status.h"
#include "third_party/acl/inc/acl/acl_base.h"

namespace stream_executor {
namespace ascend {

// Converts an ACL error code to an absl::Status.
absl::Status ToStatus(aclError error, const char* message = nullptr);

}  // namespace ascend
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ASCEND_ASCEND_STATUS_H_