#include "xla/service/ascend/ffi/ascend_ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_interop.h"
#include "absl/status/status.h"
#include "absl/log/log.h"

namespace xla::ffi {

void RegisterAscendFfiHandlers() {
  // Register GELU operator
  auto error = Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.gelu",
      "ASCEND",
      AscendGelu);
  
  auto status = TakeStatus(error);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to register ascend.gelu operator: " << status.ToString();
    throw std::runtime_error("Failed to register ascend.gelu operator: " + status.ToString());
  }

  // Register other operators here in the future
}

}  // namespace xla::ffi

namespace {
// 模块初始化时注册
bool InitModule() {
  try {
    xla::ffi::RegisterAscendFfiHandlers();
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to initialize Ascend FFI module: " << e.what();
    return false;
  }
}

bool module_initialized = InitModule();

}  // namespace