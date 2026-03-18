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

#include "xla/stream_executor/ascend/ascend_event.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/ascend/ascend_status.h"

namespace stream_executor {
namespace ascend {

absl::StatusOr<std::unique_ptr<AscendEvent>> AscendEvent::Create(StreamExecutor* executor,
                                                              bool allow_timing) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  
  aclrtEvent event;
  aclError ret = aclrtCreateEvent(&event);
  if (ret != ACL_ERROR_NONE) {
    return ToStatus(ret, "Failed to create Ascend event");
  }

  return std::unique_ptr<AscendEvent>(new AscendEvent(executor, event));
}

Event::Status AscendEvent::PollForStatus() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  
  aclrtEventRecordedStatus status;
  aclError ret = aclrtQueryEventStatus(event_handle_, &status);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "Failed to query event status: " << ret;
    return Event::Status::kError;
  }

  switch (status) {
    case ACL_EVENT_RECORDED_STATUS_COMPLETE:
      return Event::Status::kComplete;
    case ACL_EVENT_RECORDED_STATUS_NOT_READY:
      return Event::Status::kPending;
    default:
      return Event::Status::kError;
  }
}

absl::Status AscendEvent::WaitForEventOnExternalStream(std::intptr_t stream) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  aclError ret = aclrtStreamWaitEvent(reinterpret_cast<aclrtStream>(stream), event_handle_);
  if (ret != ACL_ERROR_NONE) {
    return ToStatus(ret, "Failed to wait for event on external stream");
  }
  return absl::OkStatus();
}

absl::Status AscendEvent::Synchronize() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  aclError ret = aclrtSynchronizeEvent(event_handle_);
  if (ret != ACL_ERROR_NONE) {
    return ToStatus(ret, "Failed to synchronize event");
  }
  return absl::OkStatus();
}

absl::StatusOr<float> AscendEvent::GetElapsedTime() const {
  // ACL doesn't support event timing, return 0.0
  return 0.0f;
}

AscendEvent::~AscendEvent() {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  aclError ret = aclrtDestroyEvent(event_handle_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "Failed to destroy Ascend event: " << ret;
  }
}

AscendEvent::AscendEvent(StreamExecutor* executor, aclrtEvent event_handle)
    : executor_(executor), event_handle_(event_handle) {}

}  // namespace ascend
}  // namespace stream_executor