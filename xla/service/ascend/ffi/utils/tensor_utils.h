#ifndef XLA_SERVICE_ASCEND_FFI_UTILS_TENSOR_UTILS_H_
#define XLA_SERVICE_ASCEND_FFI_UTILS_TENSOR_UTILS_H_

#include <limits>

#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnn/acl_meta.h"
#include "xla/ffi/api/ffi.h"
#include "xla/xla_data.pb.h"

namespace xla::ffi {
  // Convert XLA Buffer to Ascend Tensor
  template <DataType dtype, size_t rank>
  aclTensor* ConvertToAclTensor(const Buffer<dtype, rank>& buffer);
  
  // Convert XLA PrimitiveType to Ascend aclDataType
  aclDataType ConvertToAclDataType(PrimitiveType type);
}

#endif  // XLA_SERVICE_ASCEND_FFI_UTILS_TENSOR_UTILS_H_
