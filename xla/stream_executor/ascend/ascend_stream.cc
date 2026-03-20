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

#include "xla/stream_executor/ascend/ascend_stream.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/ascend/ascend_event.h"
#include "xla/stream_executor/ascend/ascend_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor {
namespace ascend {

namespace {

absl::Status WaitStreamOnEvent(StreamExecutor* executor, aclrtStream stream,
                               aclrtEvent event) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  
  aclError ret = aclrtStreamWaitEvent(stream, event);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "Failed to wait event " << event << " on stream " << stream;
    return ToStatus(ret, "Failed to wait event");
  }
  
  return absl::OkStatus();
}

absl::Status RecordAscendEvent(StreamExecutor* executor, aclrtEvent event,
                            aclrtStream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  return ToStatus(aclrtRecordEvent(event, stream),
                  "Error recording Ascend event");
}

absl::StatusOr<aclrtStream> CreateStream(StreamExecutor* executor, int priority) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  aclrtStream stream;
  
  // ACL stream creation with priority
  // Note: ACL stream priority is different from CUDA, adjust accordingly
  aclError ret = aclrtCreateStream(&stream);
  if (ret != ACL_ERROR_NONE) {
    return ToStatus(ret, "Failed to create Ascend stream");
  }

  VLOG(2) << "successfully created stream " << stream << " for executor "
          << executor << " on thread";
  return stream;
}

absl::Status AsynchronousMemcpyD2H(StreamExecutor* executor, void* host_dst,
                                   void* ascend_src, uint64_t size,
                                   aclrtStream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  TF_RETURN_IF_ERROR(
      ToStatus(aclrtMemcpyAsync(host_dst, size, ascend_src, size, ACL_MEMCPY_DEVICE_TO_HOST, stream)));

  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << ascend_src << " to " << host_dst
          << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyH2D(StreamExecutor* executor,
                                   void* ascend_dst, const void* host_src,
                                   uint64_t size, aclrtStream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  TF_RETURN_IF_ERROR(
      ToStatus(aclrtMemcpyAsync(ascend_dst, size, host_src, size, ACL_MEMCPY_HOST_TO_DEVICE, stream)));

  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " from " << host_src << " to " << ascend_dst
          << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status AsynchronousMemcpyD2D(StreamExecutor* executor,
                                   void* ascend_dst, void* ascend_src,
                                   uint64_t size, aclrtStream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  TF_RETURN_IF_ERROR(
      ToStatus(aclrtMemcpyAsync(ascend_dst, size, ascend_src, size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream)));

  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes"
          << " from " << ascend_src << " to " << ascend_dst
          << " on stream " << stream;
  return absl::OkStatus();
}

void InternalHostCallback(void* data) {
  auto* callback = reinterpret_cast<absl::AnyInvocable<void() &&>*>(data);
  std::move(*callback)();
  delete callback;
}

}  // namespace

absl::StatusOr<std::unique_ptr<AscendStream>> AscendStream::Create(
    StreamExecutor* executor,
    std::optional<std::variant<StreamPriority, int>> priority) {
  int stream_priority = [&]() {
    if (priority.has_value() && std::holds_alternative<int>(priority.value())) {
      return std::get<int>(priority.value());
    }
    std::unique_ptr<ActivateContext> activation = executor->Activate();
    return executor->GetGpuStreamPriority(
        std::get<StreamPriority>(priority.value_or(StreamPriority::Default)));
  }();
  TF_ASSIGN_OR_RETURN(auto stream_handle,
                      CreateStream(executor, stream_priority));

  TF_ASSIGN_OR_RETURN(auto completed_event,
                      executor->CreateEvent());

  return std::unique_ptr<AscendStream>(new AscendStream(
      executor, std::move(completed_event), priority, stream_handle));
}

absl::Status AscendStream::WaitFor(Stream* other) {
  AscendStream* other_stream = static_cast<AscendStream*>(other);

  TF_RETURN_IF_ERROR(other_stream->RecordCompletedEvent());
  return WaitStreamOnEvent(executor_, stream_handle_,
                           static_cast<AscendEvent*>(other_stream->completed_event_.get())->event_handle());
}

absl::Status AscendStream::RecordEvent(Event* event) {
  return RecordAscendEvent(executor_, static_cast<AscendEvent*>(event)->event_handle(),
                        stream_handle_);
}

absl::Status AscendStream::WaitFor(Event* event) {
  return WaitStreamOnEvent(executor_, stream_handle_,
                           static_cast<AscendEvent*>(event)->event_handle());
}

absl::Status AscendStream::RecordCompletedEvent() {
  return RecordEvent(completed_event_.get());
}

namespace {
void DestroyStream(StreamExecutor* executor, aclrtStream stream) {
  if (stream == nullptr) {
    return;
  }

  std::unique_ptr<ActivateContext> activation = executor->Activate();
  aclrtStreamStatus  stream_status = ACL_STREAM_STATUS_RESERVED;
  aclError res = aclrtStreamQuery(stream,&stream_status);
  if (res != ACL_ERROR_NONE || stream_status != ACL_STREAM_STATUS_COMPLETE) {
    LOG(ERROR) << "stream not idle on destroy: " << ToStatus(res);
  }

  auto status = ToStatus(aclrtDestroyStream(stream));
  if (!status.ok()) {
    LOG(ERROR) << "failed to destroy Ascend stream for executor " << executor
               << ": " << status;
  } else {
    VLOG(2) << "successfully destroyed stream " << stream << " for executor "
            << executor;
  }
}

absl::Status SynchronizeStream(StreamExecutor* executor, aclrtStream stream) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CHECK(stream != nullptr);
  return ToStatus(aclrtSynchronizeStream(stream),
                  "Could not synchronize Ascend stream");
}

}  // namespace

AscendStream::~AscendStream() {
  BlockHostUntilDone().IgnoreError();
  executor_->DeallocateStream(this);

  DestroyStream(executor_, stream_handle_);
}

absl::Status AscendStream::BlockHostUntilDone() {
  TF_RETURN_IF_ERROR(SynchronizeStream(executor_, stream_handle_));
  absl::MutexLock lock(mutex_);
  mutex_.Await(absl::Condition(&no_pending_host_callbacks_));
  return absl::OkStatus();
}

absl::Status AscendStream::Memset32(DeviceAddressBase* location, uint32_t pattern,
                                  uint64_t size) {
  if (absl::bit_cast<uintptr_t>(location->opaque()) % alignof(uint32_t) != 0) {
    return absl::InvalidArgumentError("location must be 4 byte aligned.");
  }
  if (size % sizeof(uint32_t) != 0) {
    return absl::InvalidArgumentError("size must be a multiple of 4 bytes.");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  
  // ACL doesn't have a direct memset32 async API, we'll use memset instead
  // Note: This is a simplification, actual implementation may need to be adjusted
  void* ptr = const_cast<void*>(location->opaque());
  for (uint64_t i = 0; i < size / sizeof(uint32_t); ++i) {
    reinterpret_cast<uint32_t*>(ptr)[i] = pattern;
  }
  return absl::OkStatus();
}

absl::Status AscendStream::MemZero(DeviceAddressBase* location, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  void* ptr = const_cast<void*>(location->opaque());
  return ToStatus(aclrtMemsetAsync(ptr, size, 0, size,stream_handle_),
                  "Failed to enqueue async memset operation");
}

absl::Status AscendStream::Memcpy(DeviceAddressBase* ascend_dst,
                                 const DeviceAddressBase& ascend_src,
                                 uint64_t size) {
  return AsynchronousMemcpyD2D(
      executor_, const_cast<void*>(ascend_dst->opaque()),
      const_cast<void*>(ascend_src.opaque()), size, stream_handle_);
}

absl::Status AscendStream::Memcpy(DeviceAddressBase* ascend_dst,
                                 const void* host_src, uint64_t size) {
  return AsynchronousMemcpyH2D(executor_,
                               const_cast<void*>(ascend_dst->opaque()),
                               host_src, size, stream_handle_);
}

absl::Status AscendStream::Memcpy(void* host_dst,
                                 const DeviceAddressBase& ascend_src,
                                 uint64_t size) {
  return AsynchronousMemcpyD2H(executor_, host_dst,
                               const_cast<void*>(ascend_src.opaque()),
                               size, stream_handle_);
}

absl::Status AscendStream::DoHostCallbackWithStatus(
    absl::AnyInvocable<absl::Status() &&> callback) {
  auto callback_ptr = new absl::AnyInvocable<void() &&>(
      [cb = std::move(callback), this]() mutable {
        absl::Status s = (std::move(cb))();
        if (!s.ok()) {
          LOG(ERROR) << "Host callback failed: " << s;
        }
        int num_pending_host_callbacks = num_pending_host_callbacks_.fetch_sub(
                                             1, std::memory_order_acq_rel) - 1;
        if (num_pending_host_callbacks == 0) {
          absl::MutexLock lock(mutex_);
          no_pending_host_callbacks_ = num_pending_host_callbacks_ <= 0;
        }
      });
  
  // ACL doesn't have a direct host callback API, so we just execute the callback directly
  // Note: This is a simplification, actual implementation may need to use task queues
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  std::move(*callback_ptr)();
  delete callback_ptr;
  
  int num_pending_host_callbacks = num_pending_host_callbacks_.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (num_pending_host_callbacks == 1) {
    absl::MutexLock lock(mutex_);
    no_pending_host_callbacks_ = num_pending_host_callbacks_ <= 0;
  }
  return absl::OkStatus();
}

absl::Status AscendStream::LaunchKernel(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, void* function,
    absl::string_view name, void** args, int64_t shmem_bytes) {
  // TODO: Implement kernel launching for Ascend
  // This requires specific ACL API calls for kernel execution
  return absl::UnimplementedError("LaunchKernel not implemented");
}

void AscendStream::SetName(std::string name) {
  // ACL doesn't support stream naming, so we just store the name
  StreamCommon::SetName(std::move(name));
}

AscendStream::AscendStream(StreamExecutor* executor, std::unique_ptr<Event> completed_event,
                          std::optional<std::variant<StreamPriority, int>> priority,
                          aclrtStream stream_handle)
    : StreamCommon(executor),
      executor_(executor),
      completed_event_(std::move(completed_event)),
      priority_(priority),
      stream_handle_(stream_handle) {}

}  // namespace ascend
}  // namespace stream_executor