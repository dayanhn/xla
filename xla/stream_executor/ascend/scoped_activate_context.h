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

#ifndef XLA_STREAM_EXECUTOR_ASCEND_SCOPED_ACTIVATE_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_ASCEND_SCOPED_ACTIVATE_CONTEXT_H_

#include "xla/stream_executor/ascend/context.h"

namespace stream_executor::ascend {

// RAII helper that activates an Ascend context for the duration of a scope.
class ScopedActivateContext {
 public:
  explicit ScopedActivateContext(Context* context);
  ~ScopedActivateContext();

 private:
  // The context to restore when the scope is exited.
  Context* to_restore_ = nullptr;
};

}  // namespace stream_executor::ascend

#endif  // XLA_STREAM_EXECUTOR_ASCEND_SCOPED_ACTIVATE_CONTEXT_H_
