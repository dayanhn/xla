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

#ifndef XLA_SERVICE_ASCEND_ASCEND_TRANSFER_MANAGER_H_
#define XLA_SERVICE_ASCEND_ASCEND_TRANSFER_MANAGER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/literal.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ascend {

// An implementation of the XLA GenericTransferManager that
// handles Ascend-specific data transfer.
class AscendTransferManager : public GenericTransferManager {
 public:
  AscendTransferManager(se::Platform::Id id, unsigned pointer_size);

  absl::Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                       const LiteralSlice& literal) override;
  absl::Status TransferLiteralFromOutfeed(
      se::StreamExecutor* executor, MutableBorrowingLiteral literal) override;
  absl::Status ReadDynamicShapes(se::Stream* stream,
                                 const ShapedBuffer* device_buffer,
                                 Shape* device_shape) override;

 private:
  // We use a fixed-size staging buffers and split transfer into multiple
  // operations if literal does not fit into it.
  static constexpr int64_t kStagingBufferSize = 128 * 1024 * 1024;

  // We use host memory allocation (pinned host memory) as a staging buffer for
  // transfering literals to and from device. We keep a separate staging
  // allocation per device so we don't need to do cross-device synchronization.
  // All transfers to and from a device are ordered via stream dependencies.
  struct StagingBuffer {
    StagingBuffer(std::unique_ptr<se::MemoryAllocation> allocation,
                  std::unique_ptr<se::Event> transfer_completed);

    absl::Mutex mutex;
    std::unique_ptr<se::MemoryAllocation> allocation ABSL_GUARDED_BY(mutex);
    std::unique_ptr<se::Event> transfer_completed ABSL_GUARDED_BY(mutex);
  };

  AscendTransferManager(const AscendTransferManager&) = delete;
  AscendTransferManager& operator=(const AscendTransferManager&) = delete;

  bool PackSubbyteTypes() const override { return true; }

  // Returns or creates the staging buffer for the given executor.
  absl::StatusOr<StagingBuffer*> GetOrCreateStagingBuffer(
      se::StreamExecutor* executor, bool& first_create);

  absl::Status TransferBufferFromDevice(se::Stream* stream,
                                        const se::DeviceAddressBase& source,
                                        int64_t size,
                                        void* destination) override;

  absl::Status TransferBufferToDevice(
      se::Stream* stream, int64_t size, const void* source,
      se::DeviceAddressBase* destination) override;

  // This class keeps a pool of pinned memory
  // (StreamExecutor::HostMemoryAllocate()) that serves ReadDynamicShapes().
  absl::Status EnsurePinnedBuffersAllocated(se::StreamExecutor* executor)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  static constexpr int64_t kPinnedChunkBytes = 128 * 1024;
  static constexpr int64_t kPinnedBufferBytes = 128;

  absl::Mutex mu_;

  // Chunk of pinned memory of size kPinnedChunkBytes.  The pointers in
  // pinned_buffers_ point into this chunk.  Lazily initialized.
  std::unique_ptr<se::MemoryAllocation> pinned_chunk_ ABSL_GUARDED_BY(mu_);

  // Host buffers for reading dynamic shapes.  Each buffer has size
  // kPinnedBufferBytes.  Lazily initialized.
  std::vector<void*> pinned_buffers_ ABSL_GUARDED_BY(mu_);

  // Staging buffers allocated for transfers to and from device.
  absl::Mutex mutex_;
  absl::node_hash_map<se::StreamExecutor*, StagingBuffer> staging_buffers_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace ascend
}  // namespace xla

#endif  // XLA_SERVICE_ASCEND_ASCEND_TRANSFER_MANAGER_H_