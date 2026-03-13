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

#ifndef XLA_STREAM_EXECUTOR_ASCEND_ASCEND_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_ASCEND_ASCEND_EXECUTOR_H_

#include <memory>
#include <optional>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_common.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/dnn.h"

namespace stream_executor::ascend {

// This class implements StreamExecutorCommon for Ascend devices that use ACL libraries.
class AscendExecutor : public StreamExecutorCommon {
 public:
  AscendExecutor(Platform* platform, int device_ordinal)
      : StreamExecutorCommon(platform), device_ordinal_(device_ordinal) {}

  ~AscendExecutor() override;
  absl::Status Init() override;
  int device_ordinal() const override { return device_ordinal_; }
  bool SynchronizeAllActivity() override;
  absl::StatusOr<DeviceAddressBase> GetMemoryRange(
      const DeviceAddressBase& location) const override;
  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      Stream* stream, bool use_delay_kernel) override;
  absl::StatusOr<DeviceAddressBase> GetSymbol(
      const std::string& symbol_name, ModuleHandle module_handle) override;
  absl::Status SynchronousMemZero(DeviceAddressBase* location,
                                  uint64_t size) override;
  absl::Status SynchronousMemcpy(DeviceAddressBase* gpu_dst,
                                 const void* host_src, uint64_t size) override;
  absl::Status SynchronousMemcpy(void* host_dst,
                                 const DeviceAddressBase& gpu_src,
                                 uint64_t size) override;
  void DeallocateStream(Stream* stream) override;
  absl::Status EnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(StreamExecutor* other) override;
  bool CanEnablePeerAccessTo(int other_device_ordinal) override;
  bool DeviceMemoryUsage(int64_t* free_out, int64_t* total_out) const override;
  absl::StatusOr<std::unique_ptr<Kernel>> LoadKernel(
      const KernelLoaderSpec& spec) override;
  void UnloadKernel(const Kernel* kernel) override;
  absl::StatusOr<ModuleHandle> LoadModule(
      const MultiModuleLoaderSpec& spec) override;
  bool UnloadModule(ModuleHandle module_handle) override;
  absl::StatusOr<std::shared_ptr<DeviceAddressBase>> CreateOrShareConstant(
      Stream* stream, absl::Span<const uint8_t> content) override;
  DeviceAddressBase Allocate(uint64_t size, int64_t memory_space) override;
  void Deallocate(DeviceAddressBase* mem) override;
  blas::BlasSupport* AsBlas() override;
  fft::FftSupport* AsFft() override;
  dnn::DnnSupport* AsDnn() override;
  absl::StatusOr<std::unique_ptr<Event>> CreateEvent() override;
  absl::StatusOr<std::unique_ptr<Stream>> CreateStream(
      std::optional<std::variant<StreamPriority, int>> priority) override;
  absl::StatusOr<std::unique_ptr<CommandBuffer>> CreateCommandBuffer(
      CommandBuffer::Mode mode) override;

  absl::StatusOr<std::unique_ptr<DeviceDescription>> CreateDeviceDescription()
      const override;
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> HostMemoryAllocate(
      uint64_t size) override;

  bool HostMemoryRegister(void* location, uint64_t size) override;
  bool HostMemoryUnregister(void* location) override;

  absl::StatusOr<MemorySpace> GetPointerMemorySpace(const void* ptr) override;

 private:
  // The device ordinal value that this executor was initialized with;
  int device_ordinal_;
};

}  // namespace stream_executor::ascend

#endif  // XLA_STREAM_EXECUTOR_ASCEND_ASCEND_EXECUTOR_H_