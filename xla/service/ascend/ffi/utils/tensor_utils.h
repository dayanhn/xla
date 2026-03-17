#ifndef XLA_SERVICE_ASCEND_FFI_UTILS_TENSOR_UTILS_H_
#define XLA_SERVICE_ASCEND_FFI_UTILS_TENSOR_UTILS_H_

#include "xla/ffi/api/ffi.h"
#include "acl/acl.h"

namespace xla::ffi {
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* ConvertToAclTensor(const BufferBase& buffer);
  
  // Convert XLA PrimitiveType to Ascend aclDataType
  aclDataType ConvertToAclDataType(PrimitiveType type);
}

#endif  // XLA_SERVICE_ASCEND_FFI_UTILS_TENSOR_UTILS_H_