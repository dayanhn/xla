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

#include "xla/backends/npu/collectives/hccl_communicator.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/hccl/inc/hccl/hccl.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"

namespace xla::npu {
namespace {

aclrtStream AsAclStream(se::Stream* stream) {
  return absl::bit_cast<aclrtStream>(stream->platform_specific_handle().stream);
}

se::Stream* ToStream(const Communicator::Executor& executor) {
  return tsl::down_cast<const gpu::GpuCollectives::Executor&>(executor).stream();
}

//==-----------------------------------------------------------------------===//
// Conversions between XLA and HCCL data types
//==-----------------------------------------------------------------------===//

static size_t ToHcclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

static absl::StatusOr<HcclDataType> ToHcclDataType(PrimitiveType dtype) {
  switch (dtype) {
    case S8:
      return HCCL_DATA_TYPE_INT8;
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      // Map FP8 variants to closest HCCL type if available
      return HCCL_DATA_TYPE_INT8;  // May need adjustment based on HCCL support
    case F8E5M2:
    case F8E4M3FN:
      return HCCL_DATA_TYPE_FP16;  // Approximate mapping
    case PRED:
    case U8:
      return HCCL_DATA_TYPE_UINT8;
    case S16:
      return HCCL_DATA_TYPE_INT16;
    case U16:
      return HCCL_DATA_TYPE_UINT16;
    case S32:
      return HCCL_DATA_TYPE_INT32;
    case U32:
      return HCCL_DATA_TYPE_UINT32;
    case S64:
      return HCCL_DATA_TYPE_INT64;
    case U64:
      return HCCL_DATA_TYPE_UINT64;
    case F16:
      return HCCL_DATA_TYPE_FP16;
    case BF16:
      return HCCL_DATA_TYPE_BFP16;
    case F32:
    case C64:
      return HCCL_DATA_TYPE_FP32;
    case F64:
    case C128:
      return HCCL_DATA_TYPE_FP64;
    default:
      return InvalidArgument("Unsupported data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(dtype));
  }
}

static HcclReduceOp ToHcclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return HCCL_REDUCE_SUM;
    case ReductionKind::PRODUCT:
      return HCCL_REDUCE_PROD;
    case ReductionKind::MIN:
      return HCCL_REDUCE_MIN;
    case ReductionKind::MAX:
      return HCCL_REDUCE_MAX;
  }
}

// Convert HcclResult to absl::Status
absl::Status HcclStatusToAbslStatus(HcclResult result, const std::string& message) {
  if (result == HCCL_SUCCESS) {
    return absl::OkStatus();
  }
  
  const char* error_str = HcclGetErrorString(result);
  return absl::InternalError(
      absl::StrCat(message, ": HCCL error: ", 
                   error_str ? error_str : "unknown error"));
}

// Polls the communicator until any pending non-blocking operations are "done"
// or aborted.
absl::Status PollUntilDone(HcclComm comm, const xla::gpu::CancellationToken& cancel) {
  auto poll = [](HcclComm comm,
                 const xla::gpu::CancellationToken& cancel) -> absl::Status {
    HcclResult state = HCCL_SUCCESS;
    while (state == HCCL_SUCCESS && !cancel.IsCancelled()) {
      HcclResult result = HcclGetCommAsyncError(comm, &state);
      if (result != HCCL_SUCCESS) {
        return HcclStatusToAbslStatus(result, "HcclGetCommAsyncError failed");
      }
    }
    if (cancel.IsCancelled()) {
      return Cancelled("HcclCommunicator cancelled");
    }
    // state now contains the final status of the operation
    if (state != HCCL_SUCCESS) {
      return HcclStatusToAbslStatus(state, "HCCL operation failed");
    }
    return absl::OkStatus();
  };

  if (!VLOG_IS_ON(1)) {
    return poll(comm, cancel);
  }

  absl::Time start = absl::Now();
  absl::Status s = poll(comm, cancel);
  absl::Time stop = absl::Now();
  VLOG(1) << "Polled HCCL communicator " << comm << " for " << (stop - start)
          << ": " << s;
  return s;
}

}  // namespace

//==-----------------------------------------------------------------------===//
// HCCL Communicator
//==-----------------------------------------------------------------------===//

absl::StatusOr<std::unique_ptr<HcclCommunicator>> HcclCommunicator::Create(
    se::StreamExecutor* stream_executor,
    absl::AnyInvocable<absl::StatusOr<HcclComm>()> make_comm,
    std::shared_ptr<xla::gpu::CancellationToken> cancel, bool is_async, tsl::Env& env) {
  auto f = [cancel, &make_comm]() -> absl::StatusOr<HcclComm> {
    TF_ASSIGN_OR_RETURN(HcclComm comm, make_comm());
    if (cancel) {
      TF_RETURN_IF_ERROR(xla::npu::PollUntilDone(comm, *cancel));
    } else {
      xla::gpu::CancellationToken never_cancelled;
      TF_RETURN_IF_ERROR(xla::npu::PollUntilDone(comm, never_cancelled));
    }
    return comm;
  };

  if (!is_async) {
    // If this HcclCommunicator is synchronous, construct HcclComm in the
    // calling thread.
    TF_ASSIGN_OR_RETURN(HcclComm comm, f());
    return absl::WrapUnique(new HcclCommunicator(stream_executor, comm, nullptr,
                                                 std::move(cancel)));
  }

  // If this HcclCommunicator is asynchronous, then all operations on the
  // underlying HcclComm, including its creation, must take place on the
  // single threaded executor.
  auto executor = std::make_unique<xla::gpu::SingleThreadedExecutor>(env);
  TF_ASSIGN_OR_RETURN(HcclComm comm,
                      MakeFutureOn<HcclComm>(*executor, f).Await());
  return absl::WrapUnique(new HcclCommunicator(
      stream_executor, comm, std::move(executor), std::move(cancel)));
}

HcclCommunicator::~HcclCommunicator() {
  auto f = [this]() -> absl::Status {
    if (comm_ == nullptr) {
      VLOG(1) << "Skipping destruction; null comm_ " << *this;
      return absl::OkStatus();
    }

    if (aborted_) {
      VLOG(1) << "Skipping destruction; already aborted " << *this;
      return absl::OkStatus();
    }

    // Note that we intentionally don't call PollUntilDone. Once comm_ has
    // been destroyed, we can no longer safely touch it.
    VLOG(1) << "Destroy " << *this;
    return HcclStatusToAbslStatus(HcclCommDestroy(comm_), 
                                   "Failed to destroy HCCL communicator");
  };

  if (absl::Status s = Execute(f).Await(); !s.ok()) {
    LOG(ERROR) << "HcclCommunicator::~HcclCommunicator: " << s;
  }
}

absl::Status HcclCommunicator::Abort() {
  // By setting the cancellation token all pending collectives scheduled on
  // executor_ will cancel. This will allow the aborting lambda below to run.
  cancel_->Cancel();

  return ExecuteAwait([this]() -> absl::Status {
    VLOG(1) << "Abort HCCL communicator: " << *this;
    if (aborted_) {
      return FailedPrecondition("HcclCommunicator already aborted");
    }
    aborted_ = true;
    // Note that we intentionally don't call PollUntilDone. Once comm_
    // has been aborted, we can no longer safely touch it.
    // HCCL doesn't have a direct abort API, so we just mark it as aborted
    return absl::OkStatus();
  });
}

absl::Status HcclCommunicator::HealthCheck() const {
  return ExecuteAwait([this]() -> absl::Status {
    VLOG(5) << "Health check for HCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return FailedPrecondition("HcclCommunicator aborted");
    }
#if 0
    // HCCL doesn't have a direct async error query like NCCL,
    // so we just check if the communicator is valid
    int rank;
    HcclResult result = HcclCommUserRank(comm_, &rank);
    if (result != HCCL_SUCCESS) {
      return HcclStatusToAbslStatus(result, "HCCL communicator health check failed");
    }
    return absl::OkStatus();
#endif
    return absl::UnimplementedError("HcclCommunicator health check not implemented");
  });
}

absl::StatusOr<size_t> HcclCommunicator::NumRanks() const {
#if 0
  return ExecuteAwait<size_t>([this]() -> absl::StatusOr<size_t> {
    VLOG(5) << "Get the number of ranks in HCCL communicator: " << *this;
    if (cancel_->IsCancelled()) {
      return FailedPrecondition("HcclCommunicator aborted");
    }

    int nranks;
    HcclResult result = HcclCommCount(comm_, &nranks);
    if (result != HCCL_SUCCESS) {
      return HcclStatusToAbslStatus(result, "Failed to get HCCL communicator size");
    }
    return static_cast<size_t>(nranks);
  });
#endif
  return absl::UnimplementedError("HcclCommunicator::NumRanks is not implemented");
}

Future<> HcclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status(gpu::GpuCommunicator*)> f) {
  return Execute([f = std::move(f), this]() mutable -> absl::Status {
    TF_RETURN_IF_ERROR(GroupStart());
    TF_RETURN_IF_ERROR(f(this));
    TF_RETURN_IF_ERROR(GroupEnd());
    return absl::OkStatus();
  });
}

Future<> HcclCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     const Communicator::Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor);
  });
}

Future<> HcclCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     RankId root, const Executor& executor) {
  return Execute(
      [send_buffer, recv_buffer, dtype, count, root, &executor, this]() {
        return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                               executor);
      });
}

Future<> HcclCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
                                         se::DeviceAddressBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         ReductionKind reduction_kind,
                                         const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() {
    return LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                               reduction_kind, executor);
  });
}

Future<> HcclCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                     se::DeviceAddressBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor, this]() {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> HcclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers, recv_buffers, dtype, count, &executor, this]() {
    return LaunchAllToAll(send_buffers, recv_buffers, dtype, count, executor);
  });
}

Future<> HcclCommunicator::CollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  std::vector<RankId> owned_target_ranks(target_ranks.begin(),
                                         target_ranks.end());
  return Execute([send_buffer, recv_buffer, dtype, count, source_rank,
                  owned_target_ranks = std::move(owned_target_ranks), &executor,
                  this]() {
    return LaunchCollectivePermute(send_buffer, recv_buffer, dtype, count,
                                   source_rank, owned_target_ranks, executor);
  });
}

Future<> HcclCommunicator::Send(se::DeviceAddressBase send_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([send_buffer, dtype, count, peer, &executor, this]() {
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

Future<> HcclCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                PrimitiveType dtype, size_t count, RankId peer,
                                const Executor& executor) {
  return Execute([recv_buffer, dtype, count, peer, &executor, this]() {
    return LaunchRecv(recv_buffer, dtype, count, peer, executor);
  });
}

absl::Status HcclCommunicator::GroupStart() {
  VLOG(5) << "Start HCCL group";
  // HCCL doesn't have explicit group start/end like NCCL
  // We track nesting level for consistency
  group_nesting_level_++;
  return absl::OkStatus();
}

absl::Status HcclCommunicator::GroupEnd() {
  VLOG(5) << "End HCCL group";
  group_nesting_level_--;
  if (group_nesting_level_ > 0) {
    return absl::OkStatus();
  }
  // Wait for the communicator to finish.
  return PollUntilDone();
}

absl::Status HcclCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("HcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch HCCL AllReduce operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%lld; reduction_kind=%v; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      static_cast<long long>(count), reduction_kind, comm_, stream);

  TF_ASSIGN_OR_RETURN(HcclDataType hccl_dtype, ToHcclDataType(dtype));
  HcclReduceOp hccl_op = ToHcclReduction(reduction_kind);

  HcclResult result = HcclAllReduce(
      const_cast<void*>(send_buffer.opaque()),
      const_cast<void*>(recv_buffer.opaque()),
      ToHcclCount(dtype, count), hccl_dtype, hccl_op, comm_,
      AsAclStream(stream));
  
  TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclAllReduce failed"));
  
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status HcclCommunicator::LaunchBroadcast(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("HcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch HCCL Broadcast operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%lld; root=%d; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      static_cast<long long>(count), root.value(), comm_, stream);

  TF_ASSIGN_OR_RETURN(HcclDataType hccl_dtype, ToHcclDataType(dtype));

  // HCCL Broadcast uses in-place buffer (same buffer for send/recv)
  void* buf = const_cast<void*>(send_buffer.opaque());
  
  HcclResult result = HcclBroadcast(buf, ToHcclCount(dtype, count), hccl_dtype,
                                    root.value(), comm_, AsAclStream(stream));
  
  TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclBroadcast failed"));
  
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status HcclCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("HcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch HCCL ReduceScatter operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%lld; reduction_kind=%v; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      static_cast<long long>(count), reduction_kind, comm_, stream);

  TF_ASSIGN_OR_RETURN(HcclDataType hccl_dtype, ToHcclDataType(dtype));
  HcclReduceOp hccl_op = ToHcclReduction(reduction_kind);

  HcclResult result = HcclReduceScatter(
      const_cast<void*>(send_buffer.opaque()),
      const_cast<void*>(recv_buffer.opaque()),
      ToHcclCount(dtype, count), hccl_dtype, hccl_op, comm_,
      AsAclStream(stream));
  
  TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclReduceScatter failed"));
  
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status HcclCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("HcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch HCCL AllGather operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%lld; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      static_cast<long long>(count), comm_, stream);

  TF_ASSIGN_OR_RETURN(HcclDataType hccl_dtype, ToHcclDataType(dtype));

  HcclResult result = HcclAllGather(
      const_cast<void*>(send_buffer.opaque()),
      const_cast<void*>(recv_buffer.opaque()),
      ToHcclCount(dtype, count), hccl_dtype, comm_, AsAclStream(stream));
  
  TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclAllGather failed"));
  
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

// If all buffers are contiguous returns a device address range that covers
// all of them, otherwise returns an empty optional.
static std::optional<se::DeviceAddressBase> IsContiguous(
    absl::Span<const se::DeviceAddressBase> buffers) {
  if (buffers.empty()) {
    return std::nullopt;
  }

  if (buffers.size() == 1) {
    return buffers[0];
  }

  size_t total_size = buffers[0].size();
  for (size_t i = 1; i < buffers.size(); ++i) {
    se::DeviceAddress<uint8_t> a(buffers[i - 1]);
    se::DeviceAddress<uint8_t> b(buffers[i]);
    total_size += b.size();

    if (a.base() + a.size() != b.base()) {
      return std::nullopt;
    }
  }

  return se::DeviceAddressBase(buffers[0].opaque(), total_size);
}

absl::Status HcclCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
#if 0
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("HcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  auto buffer_formatter = [](std::string* out, se::DeviceAddressBase buffer) {
    absl::StrAppendFormat(out, "%p", buffer.opaque());
  };

  auto send_contiguous = IsContiguous(send_buffers);
  auto recv_contiguous = IsContiguous(recv_buffers);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch HCCL AllToAll operation; send_buffers=[%s]; "
      "send_contiguous=%v; recv_buffers=[%s]; recv_contiguous=%v; dtype=%s; "
      "count=%lld; comm=%p; stream=%p",
      stream->parent()->device_ordinal(),
      absl::StrJoin(send_buffers, ", ", buffer_formatter),
      send_contiguous.has_value(),
      absl::StrJoin(recv_buffers, ", ", buffer_formatter),
      recv_contiguous.has_value(),
      primitive_util::LowercasePrimitiveTypeName(dtype),
      static_cast<long long>(count), comm_, stream);

  if (send_buffers.size() != recv_buffers.size()) {
    return InvalidArgument(
        "Number of send buffers must match number of recv buffers: %d != %d",
        send_buffers.size(), recv_buffers.size());
  }

  int32_t num_ranks;
  HcclResult result = HcclCommCount(comm_, &num_ranks);
  if (result != HCCL_SUCCESS) {
    return HcclStatusToAbslStatus(result, "Failed to get HCCL communicator size");
  }

  if (send_buffers.size() != num_ranks) {
    return InvalidArgument(
        "Number of send buffers must match number of ranks: %d != %d",
        send_buffers.size(), num_ranks);
  }

  TF_ASSIGN_OR_RETURN(HcclDataType hccl_dtype, ToHcclDataType(dtype));

  // If send and receive buffers are contiguous we can use all-to-all API from
  // HCCL directly without launching individual send/recv operations.
  if (send_contiguous && recv_contiguous) {
    HcclResult result = HcclAlltoAll(
        send_contiguous->opaque(), ToHcclCount(dtype, count), hccl_dtype,
        recv_contiguous->opaque(), ToHcclCount(dtype, count), hccl_dtype,
        comm_, AsAclStream(stream));
    TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclAlltoAll failed"));
    return absl::OkStatus();
  }

  // Fall back to using Send/Recv pairs for non-contiguous buffers
  TF_RETURN_IF_ERROR(GroupStart());
  for (size_t i = 0; i < send_buffers.size(); ++i) {
    se::DeviceAddressBase send_buffer = send_buffers[i];
    se::DeviceAddressBase recv_buffer = recv_buffers[i];
    
    HcclResult send_result = HcclSend(
        const_cast<void*>(send_buffer.opaque()), ToHcclCount(dtype, count),
        hccl_dtype, static_cast<uint32_t>(i), comm_, AsAclStream(stream));
    TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(send_result, "HcclSend failed"));
    
    HcclResult recv_result = HcclRecv(
        const_cast<void*>(recv_buffer.opaque()), ToHcclCount(dtype, count),
        hccl_dtype, static_cast<uint32_t>(i), comm_, AsAclStream(stream));
    TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(recv_result, "HcclRecv failed"));
  }
  TF_RETURN_IF_ERROR(GroupEnd());
  
  return absl::OkStatus();
#endif
  return absl::UnimplementedError("HcclCommunicator::LaunchAllToAll is not implemented");
}

absl::Status HcclCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("HcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch HCCL CollectivePermute operation; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%lld; source_rank=%v; target_ranks=%s; "
      "comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      static_cast<long long>(count),
      source_rank ? std::to_string(source_rank->value()) : "none",
      absl::StrJoin(target_ranks, ",",
                    [](std::string* out, RankId r) {
                      absl::StrAppend(out, std::to_string(r.value()));
                    }),
      comm_, stream);

  TF_ASSIGN_OR_RETURN(HcclDataType hccl_dtype, ToHcclDataType(dtype));

  // HCCL doesn't have a direct collective permute API.
  // We implement it using Send/Recv operations.
  if (source_rank.has_value()) {
    // Receive from source rank
    HcclResult result = HcclRecv(
        const_cast<void*>(recv_buffer.opaque()), ToHcclCount(dtype, count),
        hccl_dtype, source_rank->value(), comm_, AsAclStream(stream));
    TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclRecv failed"));
  } else {
    // No source rank, zero out the receive buffer
    std::memset(const_cast<void*>(recv_buffer.opaque()), 0,
                recv_buffer.size());
  }

  // Send to all target ranks
  for (const auto& target_rank : target_ranks) {
    HcclResult result = HcclSend(
        const_cast<void*>(send_buffer.opaque()), ToHcclCount(dtype, count),
        hccl_dtype, target_rank.value(), comm_, AsAclStream(stream));
    TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclSend failed"));
  }

  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status HcclCommunicator::LaunchSend(se::DeviceAddressBase send_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("HcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch HCCL Send operation; send_buffer=%p; dtype=%s; count=%lld; "
      "peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype),
      static_cast<long long>(count), peer.value(), comm_, stream);

  TF_ASSIGN_OR_RETURN(HcclDataType hccl_dtype, ToHcclDataType(dtype));

  HcclResult result = HcclSend(
      const_cast<void*>(send_buffer.opaque()), ToHcclCount(dtype, count),
      hccl_dtype, peer.value(), comm_, AsAclStream(stream));
  
  TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclSend failed"));
  
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status HcclCommunicator::LaunchRecv(se::DeviceAddressBase recv_buffer,
                                          PrimitiveType dtype, size_t count,
                                          RankId peer,
                                          const Executor& executor) {
  if (cancel_->IsCancelled()) {
    return FailedPrecondition("HcclCommunicator aborted");
  }
  se::Stream* stream = ToStream(executor);

  VLOG(3) << absl::StreamFormat(
      "[%d] Launch HCCL Recv operation; recv_buffer=%p; dtype=%s; count=%lld; "
      "peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype),
      static_cast<long long>(count), peer.value(), comm_, stream);

  TF_ASSIGN_OR_RETURN(HcclDataType hccl_dtype, ToHcclDataType(dtype));

  HcclResult result = HcclRecv(
      const_cast<void*>(recv_buffer.opaque()), ToHcclCount(dtype, count),
      hccl_dtype, peer.value(), comm_, AsAclStream(stream));
  
  TF_RETURN_IF_ERROR(HcclStatusToAbslStatus(result, "HcclRecv failed"));
  
  if (group_nesting_level_ == 0) {
    TF_RETURN_IF_ERROR(PollUntilDone());
  }
  return absl::OkStatus();
}

absl::Status HcclCommunicator::PollUntilDone() const {
  // HCCL operations are synchronous with respect to the host, but they
  // execute asynchronously on the device. We need to wait for the
  // operations to complete.
  //
  // For HCCL, we poll the communicator until all operations are done.
  if (cancel_) {
    return ::xla::npu::PollUntilDone(comm_, *cancel_);
  } else {
    xla::gpu::CancellationToken never_cancelled;
    return ::xla::npu::PollUntilDone(comm_, never_cancelled);
  }
}


template <typename T>
Future<T> HcclCommunicator::Execute(
    absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
  return executor_ ? MakeFutureOn<T>(*executor_, std::move(f))
                   : Future<T>(std::move(f)());
}


Future<> HcclCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return executor_ ? MakeFutureOn<void>(*executor_, std::move(f))
                   : Future<>(std::move(f)());
}

std::string HcclCommunicator::ToString() const {
  return absl::StrCat("HcclCommunicator(comm=", reinterpret_cast<uintptr_t>(comm_),
                      ", device_ordinal=", stream_executor_->device_ordinal(),
                      ")");
}

}  // namespace xla::npu
