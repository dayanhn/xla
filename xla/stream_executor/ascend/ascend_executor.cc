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

#include "xla/stream_executor/ascend/ascend_executor.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/container/flat_hash_map.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/acl/acl_base.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/ascend/ascend_context.h"
#include "xla/stream_executor/ascend/ascend_event.h"
#include "xla/stream_executor/ascend/ascend_stream.h"
#include "xla/stream_executor/ascend/scoped_activate_context.h"
#include "xla/stream_executor/device_description.h"

namespace {

// Returns whether peer access is possible between two devices.
bool CanEnablePeerAccess(int from_device, int to_device) {
  int32_t result_s2d = -1;
  aclError error = aclrtDeviceCanAccessPeer(&result_s2d, from_device, to_device);
  if (error != ACL_ERROR_NONE || !result_s2d){
    return false;
  }
  return true;
}

// Enables peer access between two contexts.
absl::Status EnablePeerAccess(int from_device, int to_device) {
  aclError error = aclrtSetDevice(from_device);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclrtSetDevice failed with " << error << ",set device is " << from_device;
  }
  error = aclrtDeviceEnablePeerAccess(to_device, 0);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclrtDeviceEnablePeerAccess failed with " << error << ", from device " \
               << from_device << " to device " << to_device;
  }
  return absl::OkStatus();
}

}  // namespace

namespace stream_executor::ascend {

AscendExecutor::~AscendExecutor() {
  // Context is managed by AscendContext::GetContextMap()
}

absl::Status AscendExecutor::Init() {
  // Create Ascend context
  TF_ASSIGN_OR_RETURN(context_, AscendContext::Create(device_ordinal_));

  // Detect peer access capabilities
  uint32_t device_count = 0;
  aclError error = aclrtGetDeviceCount(&device_count);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclrtGetDeviceCount failed with " << error;
    device_count = 1; // Assume at least one device
  }

  for (int i = 0; i < device_count; ++i) {
    if (i == device_ordinal_) {
      peer_access_cache_[i] = true;
      continue;
    }

    peer_access_cache_[i] = CanEnablePeerAccess(device_ordinal_, i);
  }

  return absl::OkStatus();
}

std::unique_ptr<ActivateContext> AscendExecutor::Activate() {
  return std::make_unique<ScopedActivateContext>(context_);
}

bool AscendExecutor::SynchronizeAllActivity() {
  // TODO: Implement synchronization
  return true;
}

absl::StatusOr<DeviceAddressBase> AscendExecutor::GetMemoryRange(
    const DeviceAddressBase& location) const {
  // TODO: Implement memory range retrieval
  return absl::UnimplementedError("GetMemoryRange not implemented");
}

absl::StatusOr<std::unique_ptr<EventBasedTimer>> AscendExecutor::CreateEventBasedTimer(
    Stream* stream, bool use_delay_kernel) {
  // TODO: Implement event-based timer creation
  return absl::UnimplementedError("CreateEventBasedTimer not implemented");
}

absl::StatusOr<DeviceAddressBase> AscendExecutor::GetSymbol(
    const std::string& symbol_name, ModuleHandle module_handle) {
  // TODO: Implement symbol retrieval
  return absl::UnimplementedError("GetSymbol not implemented");
}

absl::Status AscendExecutor::SynchronousMemZero(DeviceAddressBase* location,
                                              uint64_t size) {
  // TODO: Implement memory zeroing
  return absl::UnimplementedError("SynchronousMemZero not implemented");
}

absl::Status AscendExecutor::SynchronousMemcpy(DeviceAddressBase* ascend_dst,
                                             const void* host_src, uint64_t size) {
  // TODO: Implement memory copy
  return absl::UnimplementedError("SynchronousMemcpy not implemented");
}

absl::Status AscendExecutor::SynchronousMemcpy(void* host_dst,
                                             const DeviceAddressBase& ascend_src,
                                             uint64_t size) {
  // TODO: Implement memory copy
  return absl::UnimplementedError("SynchronousMemcpy not implemented");
}

void AscendExecutor::DeallocateStream(Stream* stream) {
  AscendStream* ascend_stream = static_cast<AscendStream*>(stream);
  absl::MutexLock l(&alive_ascend_streams_mu_);
  alive_ascend_streams_.erase(ascend_stream->stream_handle());
}

absl::Status AscendExecutor::EnablePeerAccessTo(StreamExecutor* other) {
  AscendExecutor* other_ascend_executor = static_cast<AscendExecutor*>(other);
  int other_device_ordinal = other_ascend_executor->device_ordinal();

  if (other_device_ordinal == device_ordinal()) {
    return absl::OkStatus(); // A device can always access its own memory
  }

  if (!CanEnablePeerAccessTo(other_device_ordinal)) {
    return absl::InternalError(absl::StrCat("Peer access not supported from device ", 
                                           device_ordinal(), " to ", other_device_ordinal));
  }

  return EnablePeerAccess(device_ordinal(), other_device_ordinal);
}

bool AscendExecutor::CanEnablePeerAccessTo(StreamExecutor* other) {
  AscendExecutor* other_ascend_executor = static_cast<AscendExecutor*>(other);
  return CanEnablePeerAccessTo(other_ascend_executor->device_ordinal());
}

bool AscendExecutor::CanEnablePeerAccessTo(int other_device_ordinal) {
  auto it = peer_access_cache_.find(other_device_ordinal);
  if (it != peer_access_cache_.end()) {
    return it->second;
  }

  // If not in cache, check and add to cache
  bool can_access = CanEnablePeerAccess(device_ordinal_, other_device_ordinal);
  peer_access_cache_[other_device_ordinal] = can_access;
  return can_access;
}

bool AscendExecutor::DeviceMemoryUsage(int64_t* free_out, int64_t* total_out) const {
  // TODO: Implement memory usage query
  return false;
}

absl::StatusOr<std::unique_ptr<Kernel>> AscendExecutor::LoadKernel(
    const KernelLoaderSpec& spec) {
  // TODO: Implement kernel loading
  return absl::UnimplementedError("LoadKernel not implemented");
}

void AscendExecutor::UnloadKernel(const Kernel* kernel) {
  // TODO: Implement kernel unloading
}

absl::StatusOr<ModuleHandle> AscendExecutor::LoadModule(
    const MultiModuleLoaderSpec& spec) {
  // TODO: Implement module loading
  return absl::UnimplementedError("LoadModule not implemented");
}

bool AscendExecutor::UnloadModule(ModuleHandle module_handle) {
  // TODO: Implement module unloading
  return false;
}

absl::StatusOr<std::shared_ptr<DeviceAddressBase>> AscendExecutor::CreateOrShareConstant(
    Stream* stream, absl::Span<const uint8_t> content) {
  // TODO: Implement constant creation
  return absl::UnimplementedError("CreateOrShareConstant not implemented");
}

DeviceAddressBase AscendExecutor::Allocate(uint64_t size, int64_t memory_space) {
  // TODO: Implement memory allocation
  return DeviceAddressBase();
}

void AscendExecutor::Deallocate(DeviceAddressBase* mem) {
  // TODO: Implement memory deallocation
}

blas::BlasSupport* AscendExecutor::AsBlas() {
  // TODO: Implement BLAS support
  return nullptr;
}

fft::FftSupport* AscendExecutor::AsFft() {
  // TODO: Implement FFT support
  return nullptr;
}

dnn::DnnSupport* AscendExecutor::AsDnn() {
  // TODO: Implement DNN support
  return nullptr;
}

absl::StatusOr<std::unique_ptr<Event>> AscendExecutor::CreateEvent() {
  return AscendEvent::Create(this, /*allow_timing=*/false);
}

absl::StatusOr<std::unique_ptr<Stream>> AscendExecutor::CreateStream(
    std::optional<std::variant<StreamPriority, int>> priority) {
  TF_ASSIGN_OR_RETURN(auto stream, AscendStream::Create(this, priority));
  absl::MutexLock l(&alive_ascend_streams_mu_);
  alive_ascend_streams_[stream->stream_handle()] = stream.get();
  return std::move(stream);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>> AscendExecutor::CreateCommandBuffer(
    CommandBuffer::Mode mode) {
  // TODO: Implement command buffer creation
  return absl::UnimplementedError("CreateCommandBuffer not implemented");
}

absl::StatusOr<std::unique_ptr<DeviceDescription>> AscendExecutor::CreateDeviceDescription() const {
  DeviceDescription desc;

  // Set device address bits (Ascend devices support 64-bit addresses)
  desc.set_device_address_bits(64);

  std::string socName = aclrtGetSocName();
  desc.set_name(absl::StrCat(socName+"-", device_ordinal()));

  int64_t total_memory = 0;
  aclrtGetDeviceInfo(device_ordinal(), ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE, &total_memory);
  desc.set_device_memory_size(total_memory);

  // Set device vendor
  desc.set_device_vendor("Huawei");

  // Set other default values
  int64_t aicore_num = 0;
  aclrtGetDeviceInfo(device_ordinal(), ACL_DEV_ATTR_AICORE_CORE_NUM, &aicore_num);
  desc.set_core_count(aicore_num);

  int64_t l2_size = 0;
  aclrtGetDeviceInfo(device_ordinal(), ACL_DEV_ATTR_L2_CACHE_SIZE, &l2_size);
  desc.set_l2_cache_size(l2_size);


  return std::make_unique<DeviceDescription>(std::move(desc));
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> AscendExecutor::HostMemoryAllocate(
    uint64_t size) {
  // TODO: Implement host memory allocation
  return absl::UnimplementedError("HostMemoryAllocate not implemented");
}

bool AscendExecutor::HostMemoryRegister(void* location, uint64_t size) {
  // TODO: Implement host memory registration
  return false;
}

bool AscendExecutor::HostMemoryUnregister(void* location) {
  // TODO: Implement host memory unregistration
  return false;
}

absl::StatusOr<MemorySpace> AscendExecutor::GetPointerMemorySpace(const void* ptr) {
  // TODO: Implement memory space detection
  return absl::UnimplementedError("GetPointerMemorySpace not implemented");
}

}  // namespace stream_executor::ascend