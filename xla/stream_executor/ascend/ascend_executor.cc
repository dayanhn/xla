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

#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor {
namespace gpu {

AscendExecutor::AscendExecutor(Platform* platform, int device_ordinal,
                              CollectiveAllocatorType collective_allocator_type)
    : platform_(platform),
      device_ordinal_(device_ordinal),
      collective_allocator_type_(collective_allocator_type) {
}

AscendExecutor::~AscendExecutor() {
}

absl::Status AscendExecutor::Init() {
  LOG(INFO) << "Initializing Ascend executor for device " << device_ordinal_;
  
  device_description_ = std::make_unique<DeviceDescription>();
  device_description_->set_device_vendor("Huawei");
  device_description_->set_name("Ascend-910B");
  device_description_->set_core_count(300);
  device_description_->set_shared_memory_per_block_optin(64 * 1024);
  
  initialization_status_ = absl::OkStatus();
  return initialization_status_;
}

int AscendExecutor::device_ordinal() const {
  return device_ordinal_;
}

const Platform* AscendExecutor::GetPlatform() const {
  return platform_;
}

const DeviceDescription& AscendExecutor::GetDeviceDescription() const {
  CHECK(device_description_) << "Device description not initialized";
  return *device_description_;
}

}  // namespace gpu

}  // namespace stream_executor
