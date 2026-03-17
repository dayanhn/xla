#ifndef XLA_SERVICE_ASCEND_FFI_ASCEND_FFI_H_
#define XLA_SERVICE_ASCEND_FFI_ASCEND_FFI_H_

#include "xla/ffi/api/api.h"

namespace xla::ffi {

// Register all Ascend FFI handlers
void RegisterAscendFfiHandlers();

// Declare external symbols for FFI handlers
extern "C" XLA_FFI_Error* AscendGelu(XLA_FFI_CallFrame* frame);

}  // namespace xla::ffi

#endif  // XLA_SERVICE_ASCEND_FFI_ASCEND_FFI_H_