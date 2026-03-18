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

#include "xla/stream_executor/ascend/scoped_activate_context.h"
#include "xla/stream_executor/ascend/context.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

namespace stream_executor::ascend {

namespace {

// Thread-local data to track the current context and depth.
thread_local struct ThreadLocalData {
  Context* context;
  int device_ordinal;
  int depth;
} tls_data = {};

}  // namespace

ScopedActivateContext::ScopedActivateContext(Context* context) {
  auto* tls = &tls_data;

  // If this is an outermost scope, we must not assume that the Ascend context
  // has been left in the same state we left it. Other code may have run on
  // this thread and altered the context.
  if (tls->depth == 0) {
    VLOG(3) << "ScopedActivateContext switching to "
            << context->device_ordinal();
    context->SetActive();
    tls->depth = 1;
    tls->device_ordinal = context->device_ordinal();
    tls->context = context;
    to_restore_ = nullptr;
    return;
  }

  tls->depth++;
  if (tls->device_ordinal == context->device_ordinal()) {
    DCHECK(context->IsActive());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching context from "
          << tls->device_ordinal << " to " << context->device_ordinal();

  to_restore_ = tls->context;
  // Set the context and update thread local.
  context->SetActive();
  tls->device_ordinal = context->device_ordinal();
  tls->context = context;
}

ScopedActivateContext::~ScopedActivateContext() {
  auto* tls = &tls_data;

  tls->depth--;
  DCHECK_GE(tls->depth, 0);
  if (to_restore_ == nullptr) {
    // Leave context, tls->device_ordinal, and tls->context set.
    return;
  }

  // Set context and update thread local.
  to_restore_->SetActive();
  tls->device_ordinal = to_restore_->device_ordinal();
  tls->context = to_restore_;
}

}  // namespace stream_executor::ascend
