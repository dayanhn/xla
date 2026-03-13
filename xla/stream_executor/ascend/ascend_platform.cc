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

#include "xla/stream_executor/ascend/ascend_platform.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "third_party/acl/inc/acl/acl.h"
#include "xla/stream_executor/ascend/ascend_executor.h"
#include "xla/stream_executor/ascend/ascend_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"

namespace stream_executor {
namespace ascend {
namespace {

static absl::Status InternalInit() {
  aclError error = aclInit(nullptr);
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "Failed call to aclInit: " << error;
    return absl::InternalError(absl::StrCat("aclInit failed with ", error));
  }
  LOG(INFO) << "Ascend platform initialized";
  return absl::OkStatus();
}

static absl::Status PlatformInitialize() {
  static absl::Status* initialization_status = [] {
    return new absl::Status(InternalInit());
  }();
  return *initialization_status;
}

}  // namespace

AscendPlatform::AscendPlatform() : name_("ASCEND") {}

AscendPlatform::~AscendPlatform() {
  aclFinalize();
}

Platform::Id AscendPlatform::id() const {
  return ascend::kAscendPlatformId;
}

int AscendPlatform::VisibleDeviceCount() const {
  static const int num_devices = [] {
    if (!PlatformInitialize().ok()) {
      return -1;
    }
    uint32_t device_count = 0;
    aclError error = aclrtGetDeviceCount(&device_count);
    if (error != ACL_ERROR_NONE) {
      LOG(ERROR) << "Could not retrieve Ascend device count: " << error;
      return 0;
    }
    LOG(INFO) << "Found " << device_count << " Ascend device(s)";
    return static_cast<int>(device_count);
  }();
  return num_devices;
}

const std::string& AscendPlatform::Name() const {
  return name_;
}

absl::StatusOr<std::unique_ptr<DeviceDescription>>
AscendPlatform::DescriptionForDevice(int ordinal) const {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  auto description = std::make_unique<DeviceDescription>();
  description->set_device_vendor("Huawei");
  std::string socName = aclrtGetSocName();
  //LOG(ERROR) << "Current Ascend chipset platform is: " << socName;
  description->set_name(absl::StrCat(socName+"-", ordinal));
  int64_t aicore_num = 0;
  aclError error = aclrtGetDeviceInfo(ordinal, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicore_num);
  //LOG(ERROR) << "Ascend device " << ordinal << " has " << aicore_num << " ai cores";
  //error = aclrtGetDeviceInfo(ordinal, ACL_DEV_ATTR_CUBE_CORE_NUM, &aicore_num);
  //LOG(ERROR) << "Ascend device " << ordinal << " has " << aicore_num << " cube cores";
  //error = aclrtGetDeviceInfo(ordinal, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aicore_num);
  //LOG(ERROR) << "Ascend device " << ordinal << " has " << aicore_num << " vector cores";
  if (error != ACL_ERROR_NONE) {
    LOG(ERROR) << "Could not retrieve Ascend cube core count: " << error;
    return absl::InternalError(absl::StrCat("aclrtGetDeviceInfo failed with ", error));
  }
  
  description->set_core_count(aicore_num);
  description->set_shared_memory_per_block_optin(64 * 1024);
  return std::move(description);
}

absl::StatusOr<StreamExecutor*> AscendPlatform::ExecutorForDevice(int ordinal) {
  TF_RETURN_IF_ERROR(PlatformInitialize());
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> AscendPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
AscendPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<AscendExecutor>(this, ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace ascend

static void InitializeAscendPlatform() {
  CHECK_OK(
      PlatformManager::RegisterPlatform(std::make_unique<ascend::AscendPlatform>()));
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    ascend_platform, stream_executor::InitializeAscendPlatform());
