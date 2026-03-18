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

#ifndef XLA_STREAM_EXECUTOR_ASCEND_ASCEND_STREAM_H_ 
#define XLA_STREAM_EXECUTOR_ASCEND_ASCEND_STREAM_H_

#include <memory>
#include <optional>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"

namespace stream_executor {
namespace ascend {

class AscendExecutor;
class AscendEvent;

class AscendStream : public StreamCommon {
 public:
  ~AscendStream() override;

  // Creates a new stream for the given executor.
  static absl::StatusOr<std::unique_ptr<AscendStream>> Create(
      StreamExecutor* executor,
      std::optional<std::variant<StreamPriority, int>> priority);

  // Returns the underlying ACL stream handle.
  aclrtStream stream_handle() const { return stream_handle_; }

  // Stream interface implementation.
  absl::Status WaitFor(Stream* other) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status WaitFor(Event* event) override;
  absl::Status BlockHostUntilDone() override;
  absl::Status Memset32(DeviceAddressBase* location, uint32_t pattern, uint64_t size) override;
  absl::Status MemZero(DeviceAddressBase* location, uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* ascend_dst, const DeviceAddressBase& ascend_src, uint64_t size) override;
  absl::Status Memcpy(DeviceAddressBase* ascend_dst, const void* host_src, uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceAddressBase& ascend_src, uint64_t size) override;
  absl::Status LaunchKernel(const ThreadDim& thread_dims, const BlockDim& block_dims, const std::optional<ClusterDim>& cluster_dims, void* function, absl::string_view name, void** args, int64_t shmem_bytes) override;
  absl::Status DoHostCallbackWithStatus(absl::AnyInvocable<absl::Status() &&> callback) override;
  void SetName(std::string name) override;

 private:
  AscendStream(StreamExecutor* executor, std::unique_ptr<Event> completed_event,
              std::optional<std::variant<StreamPriority, int>> priority,
              aclrtStream stream_handle);

  // Records the completed event for this stream.
  absl::Status RecordCompletedEvent();

  // The StreamExecutor to which this object and ACL stream are bound.
  StreamExecutor* executor_;

  // The underlying ACL stream handle.
  aclrtStream stream_handle_;

  // Event used to track stream completion for inter-stream synchronization.
  std::unique_ptr<Event> completed_event_;

  // The priority of the stream.
  std::optional<std::variant<StreamPriority, int>> priority_;

  // Counter for pending host callbacks.
  std::atomic<int> num_pending_host_callbacks_{0};
  absl::Mutex mutex_;
  bool no_pending_host_callbacks_ = true;

  friend class AscendExecutor;
};

}  // namespace ascend
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ASCEND_ASCEND_STREAM_H_