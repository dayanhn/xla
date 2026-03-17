#include "xla/service/ascend/ffi/ascend_ffi.h"
#include "xla/ffi/ffi.h"

namespace xla::ffi {

void RegisterAscendFfiHandlers() {
  // Register GELU operator
  Ffi::RegisterStaticHandler(
      GetXlaFfiApi(),
      "ascend.gelu",
      "ASCEND",
      AscendGelu,
      {Traits::kCmdBufferCompatible});

  // Register other operators here in the future
}

}  // namespace xla::ffi