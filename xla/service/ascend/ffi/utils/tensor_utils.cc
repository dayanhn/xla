#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::ffi {
  template <DataType dtype, size_t rank>
  aclTensor* ConvertToAclTensor(const Buffer<dtype, rank>& buffer) {
  // Get buffer dimensions
  auto dims = buffer.dimensions();
  std::vector<int64_t> dimensions;
  for (auto dim : dims) {
    dimensions.push_back(dim);
  }

  // Determine Ascend data type
  aclDataType data_type = ConvertToAclDataType(PrimitiveType::F32); // TODO: map from dtype

  // Calculate strides
  std::vector<int64_t> strides(dimensions.size(), 1);
  for (int i = dimensions.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dimensions[i + 1];
  }

  // Create aclTensor
  return aclCreateTensor(
      dimensions.data(),
      dimensions.size(),
      data_type,
      strides.data(),
      0,
      ACL_FORMAT_ND,
      dimensions.data(),
      dimensions.size(),
      const_cast<void*>(buffer.untyped_data()));
}

// Explicit instantiations for common types
template aclTensor* ConvertToAclTensor<DataType::F32, 0>(const Buffer<DataType::F32, 0>&);
template aclTensor* ConvertToAclTensor<DataType::F32, 1>(const Buffer<DataType::F32, 1>&);
template aclTensor* ConvertToAclTensor<DataType::F32, 2>(const Buffer<DataType::F32, 2>&);
template aclTensor* ConvertToAclTensor<DataType::F32, 3>(const Buffer<DataType::F32, 3>&);
template aclTensor* ConvertToAclTensor<DataType::F32, 4>(const Buffer<DataType::F32, 4>&);

// Explicit instantiation for dynamic rank
template aclTensor* ConvertToAclTensor<DataType::F32, std::numeric_limits<size_t>::max()>(const Buffer<DataType::F32, std::numeric_limits<size_t>::max()>&);

aclDataType ConvertToAclDataType(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::F32:
      return ACL_FLOAT;
    case PrimitiveType::F16:
      return ACL_FLOAT16;
    case PrimitiveType::BF16:
      return ACL_BF16;
    case PrimitiveType::S32:
      return ACL_INT32;
    case PrimitiveType::U32:
      return ACL_UINT32;
    case PrimitiveType::S64:
      return ACL_INT64;
    case PrimitiveType::U64:
      return ACL_UINT64;
    default:
      LOG(FATAL) << "Unsupported data type: " << type;
  }
}

absl::Status PrintTensorFirstNElements(aclTensor* tensor, int num_elements, const std::string& tensor_name) {
  if (tensor == nullptr) {
    return absl::InvalidArgumentError("tensor is nullptr");
  }

  // Get tensor data address
  void* tensor_data = nullptr;
  aclError acl_status = aclGetRawTensorAddr(tensor, &tensor_data);
  if (acl_status != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("aclGetRawTensorAddr failed: ", acl_status));
  }

  // Allocate host memory for data
  float* host_data = nullptr;
  acl_status = aclrtMallocHost((void**)&host_data, num_elements * sizeof(float));
  if (acl_status != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("aclrtMallocHost failed: ", acl_status));
  }

  // Copy data from device to host
  acl_status = aclrtMemcpy(host_data, num_elements * sizeof(float),
                          tensor_data, num_elements * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
  if (acl_status != ACL_SUCCESS) {
    aclrtFreeHost(host_data);
    return absl::InternalError(
        absl::StrCat("aclrtMemcpy failed: ", acl_status));
  }

  // Print data
  LOG(INFO) << tensor_name << " (first " << num_elements << " elements):";
  for (int i = 0; i < num_elements; i++) {
    LOG(INFO) << "  " << tensor_name << "[" << i << "] = " << host_data[i];
  }

  // Free host memory
  aclrtFreeHost(host_data);

  return absl::OkStatus();
}

// Test function to copy device memory to host and print values
absl::Status TestDeviceMemoryCopy(void* device_addr, int64_t size_in_bytes) {
  if (device_addr == nullptr) {
    return absl::InvalidArgumentError("device_addr is nullptr");
  }
  
  if (size_in_bytes <= 0) {
    return absl::InvalidArgumentError("size_in_bytes must be positive");
  }
  
  // Calculate number of float elements
  int num_elements = size_in_bytes / sizeof(float);
  if (size_in_bytes % sizeof(float) != 0) {
    LOG(WARNING) << "size_in_bytes is not a multiple of float size, truncating";
  }
  
  if (num_elements == 0) {
    return absl::InvalidArgumentError("size_in_bytes is too small to hold any float elements");
  }
  
  // Allocate host memory for data
  float* host_data = nullptr;
  aclError acl_status = aclrtMallocHost((void**)&host_data, size_in_bytes);
  if (acl_status != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("aclrtMallocHost failed: ", acl_status));
  }
  
  // Copy data from device to host
  acl_status = aclrtMemcpy(host_data, size_in_bytes,
                          device_addr, size_in_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST);
  if (acl_status != ACL_SUCCESS) {
    aclrtFreeHost(host_data);
    return absl::InternalError(
        absl::StrCat("aclrtMemcpy failed: ", acl_status));
  }
  
  // Print data
  LOG(INFO) << "Device memory contents (" << num_elements << " elements):";
  for (int i = 0; i < num_elements; i++) {
    LOG(INFO) << "  data[" << i << "] = " << host_data[i];
  }
  
  // Free host memory
  aclrtFreeHost(host_data);
  
  return absl::OkStatus();
}

}  // namespace xla::ffi
