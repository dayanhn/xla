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

#ifndef XLA_STREAM_EXECUTOR_ASCEND_ASCEND_EVENT_H_ 
#define XLA_STREAM_EXECUTOR_ASCEND_ASCEND_EVENT_H_

#include "absl/status/statusor.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace ascend {

class AscendExecutor;

class AscendEvent : public Event {
 public:
  ~AscendEvent() override;

  // Creates a new event for the given executor.
  static absl::StatusOr<std::unique_ptr<AscendEvent>> Create(StreamExecutor* executor,
                                                            bool allow_timing);

  // Returns the underlying ACL event handle.
  aclrtEvent event_handle() const { return event_handle_; }

  // Event interface implementation.
  Status PollForStatus() override;
  absl::Status WaitForEventOnExternalStream(std::intptr_t stream) override;
  absl::Status Synchronize() override;
  absl::StatusOr<float> GetElapsedTime() const;

 private:
  AscendEvent(StreamExecutor* executor, aclrtEvent event_handle);

  // The StreamExecutor to which this object and ACL event are bound.
  StreamExecutor* executor_;

  // The underlying ACL event handle.
  aclrtEvent event_handle_;

  friend class AscendExecutor;
};

}  // namespace ascend
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ASCEND_ASCEND_EVENT_H_