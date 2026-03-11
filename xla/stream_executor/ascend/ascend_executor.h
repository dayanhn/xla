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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Forward declarations
class Platform;
class DeviceDescription;

namespace gpu {

// CollectiveAllocatorType from cuda_platform.h
enum class CollectiveAllocatorType {
  kNone,
  kNccl,
  kNvshmem,
};

class AscendExecutor : public StreamExecutor {
 public:
  AscendExecutor(Platform* platform, int device_ordinal,
                CollectiveAllocatorType collective_allocator_type);
  ~AscendExecutor() override;

  // StreamExecutor interface implementation
  absl::Status Init() override;
  int device_ordinal() const override;
  const Platform* GetPlatform() const override;
  const DeviceDescription& GetDeviceDescription() const override;

 private:
  // Platform this executor is associated with.
  Platform* platform_;

  // Device ordinal this executor is associated with.
  int device_ordinal_;

  // Collective allocator type.
  CollectiveAllocatorType collective_allocator_type_;

  // Device description for this executor.
  std::unique_ptr<DeviceDescription> device_description_;

  // Initialization status.
  absl::Status initialization_status_;
};

}  // namespace gpu

namespace ascend {

using AscendExecutor = gpu::AscendExecutor;

}  // namespace ascend
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ASCEND_ASCEND_EXECUTOR_H_
