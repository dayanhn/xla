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

#ifndef XLA_STREAM_EXECUTOR_ASCEND_ASCEND_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_ASCEND_ASCEND_CONTEXT_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "xla/stream_executor/ascend/ascend_context_map.h"
#include "xla/stream_executor/ascend/context.h"

namespace stream_executor::ascend {

// This class implements AscendContext for Ascend devices that use ACL libraries.
class AscendContext : public Context {
 public:
  AscendContext(aclrtContext context, int device_ordinal)
    : device_ordinal_(device_ordinal), context_(context) {}
    
  ~AscendContext() override;

  // Creates a new Ascend context for the given device ordinal.
  static absl::StatusOr<AscendContext*> Create(int device_ordinal);

  // Returns the underlying ACL context handle.
  aclrtContext context() const { return context_; }

  // Returns the singleton ContextMap.
  static ContextMap<aclrtContext, AscendContext>* GetContextMap();

  // gpu::Context interface implementation.
  void SetActive() override;
  bool IsActive() const override;
  int device_ordinal() const override { return device_ordinal_; }
  absl::Status Synchronize() override;

 private:
  // The device ordinal associated with this context.
  int device_ordinal_;

  // The underlying ACL context handle.
  aclrtContext context_;
};

}  // namespace stream_executor::ascend

#endif  // XLA_STREAM_EXECUTOR_ASCEND_ASCEND_CONTEXT_H_
