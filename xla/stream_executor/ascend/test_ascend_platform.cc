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

#include <iostream>
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/ascend/ascend_platform_id.h"

int main() {
  std::cout << "Testing Ascend platform registration..." << std::endl;
  
  // Try to get Ascend platform directly
  auto status_or_platform = stream_executor::PlatformManager::PlatformWithName("ASCEND");
  if (status_or_platform.ok()) {
    std::cout << "\n? Successfully retrieved Ascend platform by name!" << std::endl;
    auto platform = status_or_platform.value();
    std::cout << "  Platform name: " << platform->Name() << std::endl;
    std::cout << "  Device count: " << platform->VisibleDeviceCount() << std::endl;
    
    // Test device description
    if (platform->VisibleDeviceCount() > 0) {
      auto description = platform->DescriptionForDevice(0);
      if (description.ok()) {
        std::cout << "  Device 0: " << description.value()->name() << std::endl;
        std::cout << "  Vendor: " << description.value()->device_vendor() << std::endl;
        std::cout << "  Core count: " << description.value()->core_count() << std::endl;
      } else {
        std::cout << "  Failed to get device description: " << description.status() << std::endl;
      }
    }
  } else {
    std::cout << "\n? Failed to retrieve Ascend platform: " << status_or_platform.status() << std::endl;
  }
  
  // Also try by ID
  auto status_or_platform_by_id = stream_executor::PlatformManager::PlatformWithId(stream_executor::ascend::kAscendPlatformId);
  if (status_or_platform_by_id.ok()) {
    std::cout << "\n? Successfully retrieved Ascend platform by ID!" << std::endl;
    auto platform = status_or_platform_by_id.value();
    std::cout << "  Platform name: " << platform->Name() << std::endl;
  } else {
    std::cout << "\n? Failed to retrieve Ascend platform by ID: " << status_or_platform_by_id.status() << std::endl;
  }
  
  std::cout << "\nTest completed." << std::endl;
  return 0;
}
