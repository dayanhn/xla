/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_NPU_COLLECTIVES_HCCL_COMMUNICATOR_H_
#define XLA_BACKENDS_NPU_COLLECTIVES_HCCL_COMMUNICATOR_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/hccl/inc/hccl/hccl.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"

namespace xla::npu {

class HcclCollectives;

// XLA collectives communicator wrapping an HCCL communicator.
class HcclCommunicator : public gpu::GpuCommunicator {
 public:
  // Creates a HCCL communicator.
  //
  // make_comm should construct and return a new HcclComm. For example, it
  // could call HcclCommInitRank. make_comm should not return a HcclComm that
  // was created by a different thread.
  //
  // If is_async is true, all collective methods (e.g., AllReduce) are performed
  // asynchronously on a separate thread. Otherwise, they are performed
  // synchronously on the calling thread.
  static absl::StatusOr<std::unique_ptr<HcclCommunicator>> Create(
      se::StreamExecutor* stream_executor,
      absl::AnyInvocable<absl::StatusOr<HcclComm>()> make_comm,
      std::shared_ptr<xla::gpu::CancellationToken> cancel, bool is_async = false,
      tsl::Env& env = *tsl::Env::Default());

  ~HcclCommunicator() override;

  // HcclCommunicator is not copyable or movable.
  HcclCommunicator(const HcclCommunicator&) = delete;
  HcclCommunicator(HcclCommunicator&&) = delete;
  HcclCommunicator& operator=(const HcclCommunicator&) = delete;
  HcclCommunicator& operator=(HcclCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::Status HealthCheck() const final;
  absl::StatusOr<size_t> NumRanks() const final;

  xla::gpu::PlatformCommunicatorHandle platform_comm() const final {
    return xla::gpu::PlatformCommunicatorHandle{comm_};
  }

  bool SupportsDeviceComm() const final { return false; }

  absl::StatusOr<std::unique_ptr<gpu::GpuDeviceCommunicator>> CreateDeviceComm(
      const gpu::GpuDeviceCommunicator::Requirements& requirements) final {
    return Unimplemented("HCCL device communicator is not implemented");
  }

  absl::StatusOr<std::unique_ptr<SymmetricMemory>> CreateSymmetricMemory(
      se::DeviceAddressBase addr) final {
    return Unimplemented("HCCL symmetric memory is not implemented");
  }

  // Since each XLA buffer is a slice into a larger BFCAllocator chunk, first
  // get the base address of buffer. We will use the base address to keep track
  // of which chunks we have registered.
  absl::Status RegisterBufferOnce(se::DeviceAddressBase buffer_range,
                                  int device_ordinal,
                                  bool use_symmetric_buffer) final {
    return Unimplemented("HCCL buffer registration is not implemented");
  }

  Future<> GroupExecute(
      absl::AnyInvocable<absl::Status(gpu::GpuCommunicator*)> f) final;

  Future<> AllReduce(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, ReductionKind reduction_kind,
                     const Executor& executor) final;

  Future<> Broadcast(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, RankId root, const Executor& executor) final;

  Future<> ReduceScatter(se::DeviceAddressBase send_buffer,
                         se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final;

  Future<> AllGather(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, const Executor& executor) final;

  Future<> AllToAll(absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
                    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
                    PrimitiveType dtype, size_t count,
                    const Executor& executor) final;

  Future<> CollectivePermute(se::DeviceAddressBase send_buffer,
                             se::DeviceAddressBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             std::optional<RankId> source_rank,
                             absl::Span<const RankId> target_ranks,
                             const Executor& executor) final;

  Future<> Send(se::DeviceAddressBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  Future<> Recv(se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  std::string ToString() const final;

  HcclComm comm() const { return comm_; }

  se::StreamExecutor* stream_executor() const { return stream_executor_; }

 private:
  // Polls the communicator until any pending non-blocking operations are "done"
  // or aborted.
  absl::Status PollUntilDone() const;

  // Executes f on executor_, or calls f directly if executor_ is null.
  Future<> Execute(absl::AnyInvocable<absl::Status() &&> f) const;

  // Executes f on executor_, or calls f directly if executor_ is null.
  template <typename T>
  Future<T> Execute(absl::AnyInvocable<absl::StatusOr<T>() &&> f) const;

  absl::Status ExecuteAwait(absl::AnyInvocable<absl::Status() &&> f) const {
    return Execute(std::move(f)).Await();
  }

  template <typename T>
  absl::StatusOr<T> ExecuteAwait(
      absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
    return Execute<T>(std::move(f)).Await();
  }

  HcclCommunicator(se::StreamExecutor* stream_executor, HcclComm comm,
                   std::unique_ptr<tsl::Executor> executor,
                   std::shared_ptr<xla::gpu::CancellationToken> cancel)
      : stream_executor_(stream_executor),
        comm_(comm),
        executor_(std::move(executor)),
        cancel_(std::move(cancel)) {
    VLOG(1) << absl::StreamFormat("[%d] Created HCCL communicator %s",
                                  stream_executor_->device_ordinal(),
                                  this->ToString());
  }

  absl::Status GroupStart();
  absl::Status GroupEnd();

  absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor);

  absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor);

  absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                   se::DeviceAddressBase recv_buffer,
                                   PrimitiveType dtype, size_t count,
                                   ReductionKind reduction_kind,
                                   const Executor& executor);

  absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               const Executor& executor);

  absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor);

  absl::Status LaunchCollectivePermute(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       std::optional<RankId> source_rank,
                                       absl::Span<const RankId> target_ranks,
                                       const Executor& executor);

  absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor);

  absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor);

  // The stream executor (underlying NPU device) on which this communicator is
  // instantiated. We need to know the stream executor to be able to active
  // context for all operations that create or destroy device comms.
  se::StreamExecutor* stream_executor_;

  // Underlying HCCL communicator.
  HcclComm comm_;

  // If not null, used to execute methods.
  //
  // HCCL communicators (instances of HcclComm) are not thread safe. Thus,
  // multiple threads cannot concurrently access the same HcclComm. In fact,
  // a HcclComm must be created by, live on, and be destroyed by a single thread.
  // A HcclComm cannot be accessed by any thread except the one that created it.
  // To accomplish this, we perform all comm_ operations on executor_, if it
  // is not null.
  std::unique_ptr<tsl::Executor> executor_;

  // Should all pending collectives cancel?
  std::shared_ptr<xla::gpu::CancellationToken> cancel_;

  // Has comm_ been aborted?
  bool aborted_ = false;

  // Nesting level of current HCCL group
  int group_nesting_level_ = 0;
};

}  // namespace xla::npu

#endif  // XLA_BACKENDS_NPU_COLLECTIVES_HCCL_COMMUNICATOR_H_
