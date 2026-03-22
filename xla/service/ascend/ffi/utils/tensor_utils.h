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
  
  // Convert XLA DataType to Ascend aclDataType
  aclDataType ConvertToAclDataType(DataType type);
  
  // Convert XLA PrimitiveType to Ascend aclDataType
  aclDataType ConvertToAclDataType(PrimitiveType type);
  
  // Template version to convert XLA DataType to Ascend aclDataType
  template <DataType dtype>
  aclDataType ConvertToAclDataType();
  
  // Template version to print first N elements of a tensor for debugging
  template <typename T>
  absl::Status PrintTensorFirstNElementsImpl(aclTensor* tensor, int num_elements, const std::string& tensor_name);
  
  // Print first N elements of a tensor for debugging (dispatcher)
  absl::Status PrintTensorFirstNElements(aclTensor* tensor, int num_elements, const std::string& tensor_name);
  
  // Test function to copy device memory to host and print values
  absl::Status TestDeviceMemoryCopy(void* device_addr, int64_t size_in_bytes);
  
  // Template version of TestDeviceMemoryCopy
  template <typename T>
  absl::Status TestDeviceMemoryCopyImpl(void* device_addr, int64_t num_elements);
}

#endif  // XLA_SERVICE_ASCEND_FFI_UTILS_TENSOR_UTILS_H_
