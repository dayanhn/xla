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

#include "xla/stream_executor/ascend/ascend_context.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "xla/stream_executor/ascend/ascend_status.h"
#include "xla/stream_executor/ascend/ascend_context_map.h"
#include "xla/stream_executor/ascend/scoped_activate_context.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor::ascend {

namespace {

// Returns the current context or dies if it fails.
aclrtContext CurrentContextOrDie() {
  aclrtContext current = nullptr;
  aclError error = aclrtGetCurrentContext(&current);
  CHECK(error == ACL_ERROR_NONE) << "Failed to query current context: " << error;
  return current;
}

}  // namespace

// Returns the singleton ContextMap.
ContextMap<aclrtContext, AscendContext>* AscendContext::GetContextMap() {
  static ContextMap<aclrtContext, AscendContext>* context_map = 
      new ContextMap<aclrtContext, AscendContext>([](void* ptr) {
        // TODO: Implement device ordinal retrieval from pointer
        // This is a placeholder implementation
        return 0;
      });
  return context_map;
}


AscendContext::~AscendContext() {
  // Set the context to current before destroying it
  aclError error = aclrtSetCurrentContext(context_);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "failed to set current context before destruction: " << error;
  }

  // Destroy the context
  error = aclrtDestroyContext(context_);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "failed to destroy Ascend context; leaking: " << error;
  }

  // Remove from context map
  GetContextMap()->Remove(context_);
}

absl::StatusOr<AscendContext*> AscendContext::Create(int device_ordinal) {
  // Set the device
  aclError error = aclrtSetDevice(device_ordinal);
  if (error != ACL_ERROR_NONE) {
    return absl::InternalError(absl::StrCat("aclrtSetDevice failed with ", error));
  }

  // Create the context
  aclrtContext context;
  error = aclrtCreateContext(&context, device_ordinal);
  if (error != ACL_ERROR_NONE) {
    return absl::InternalError(absl::StrCat("aclrtCreateContext failed with ", error));
  }

  // Add to context map
  AscendContext* ascend_context = GetContextMap()->Add(context, device_ordinal);
  CHECK(ascend_context != nullptr) << "success in this call must entail non-null result";
  VLOG(2) << "created or reused context " << context << " for device " << device_ordinal;
  return ascend_context;
}

void AscendContext::SetActive() {
  aclError error = aclrtSetCurrentContext(context_);
  CHECK(error == ACL_ERROR_NONE) << "Failed setting context: " << error;
}

bool AscendContext::IsActive() const {
  aclrtContext current = CurrentContextOrDie();
  return current == context_;
}

absl::Status AscendContext::Synchronize() {
  ScopedActivateContext activation(this);
  aclError error = aclrtSynchronizeDevice();
  if (error != ACL_ERROR_NONE) {
    return ToStatus(error, "Failed to synchronize Ascend context");
  }
  return absl::OkStatus();
}

}  // namespace stream_executor::ascend
