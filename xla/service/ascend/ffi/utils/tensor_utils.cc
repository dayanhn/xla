#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"

namespace xla::ffi {
  template <DataType dtype, size_t rank>
  aclTensor* ConvertToAclTensor(const Buffer<dtype, rank>& buffer) {
  // Get buffer dimensions
  auto dims = buffer.dimensions();
  std::vector<int64_t> dimensions;
  for (auto dim : dims) {
    dimensions.push_back(dim);
  }

  // Determine Ascend data type using template
  aclDataType data_type = ConvertToAclDataType<dtype>();

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

// Template specialization for ConvertToAclDataType
template <>
aclDataType ConvertToAclDataType<DataType::F32>() {
  return ACL_FLOAT;
}

template <>
aclDataType ConvertToAclDataType<DataType::F16>() {
  return ACL_FLOAT16;
}

template <>
aclDataType ConvertToAclDataType<DataType::BF16>() {
  return ACL_BF16;
}

template <>
aclDataType ConvertToAclDataType<DataType::S32>() {
  return ACL_INT32;
}

template <>
aclDataType ConvertToAclDataType<DataType::U32>() {
  return ACL_UINT32;
}

template <>
aclDataType ConvertToAclDataType<DataType::S64>() {
  return ACL_INT64;
}

template <>
aclDataType ConvertToAclDataType<DataType::U64>() {
  return ACL_UINT64;
}

// Explicit instantiations for common types
template aclTensor* ConvertToAclTensor<DataType::F32, 0>(const Buffer<DataType::F32, 0>&);
template aclTensor* ConvertToAclTensor<DataType::F32, 1>(const Buffer<DataType::F32, 1>&);
template aclTensor* ConvertToAclTensor<DataType::F32, 2>(const Buffer<DataType::F32, 2>&);
template aclTensor* ConvertToAclTensor<DataType::F32, 3>(const Buffer<DataType::F32, 3>&);
template aclTensor* ConvertToAclTensor<DataType::F32, 4>(const Buffer<DataType::F32, 4>&);

template aclTensor* ConvertToAclTensor<DataType::F16, 0>(const Buffer<DataType::F16, 0>&);
template aclTensor* ConvertToAclTensor<DataType::F16, 1>(const Buffer<DataType::F16, 1>&);
template aclTensor* ConvertToAclTensor<DataType::F16, 2>(const Buffer<DataType::F16, 2>&);
template aclTensor* ConvertToAclTensor<DataType::F16, 3>(const Buffer<DataType::F16, 3>&);
template aclTensor* ConvertToAclTensor<DataType::F16, 4>(const Buffer<DataType::F16, 4>&);

template aclTensor* ConvertToAclTensor<DataType::BF16, 0>(const Buffer<DataType::BF16, 0>&);
template aclTensor* ConvertToAclTensor<DataType::BF16, 1>(const Buffer<DataType::BF16, 1>&);
template aclTensor* ConvertToAclTensor<DataType::BF16, 2>(const Buffer<DataType::BF16, 2>&);
template aclTensor* ConvertToAclTensor<DataType::BF16, 3>(const Buffer<DataType::BF16, 3>&);
template aclTensor* ConvertToAclTensor<DataType::BF16, 4>(const Buffer<DataType::BF16, 4>&);

// Explicit instantiation for dynamic rank
template aclTensor* ConvertToAclTensor<DataType::F32, std::numeric_limits<size_t>::max()>(const Buffer<DataType::F32, std::numeric_limits<size_t>::max()>&);
template aclTensor* ConvertToAclTensor<DataType::F16, std::numeric_limits<size_t>::max()>(const Buffer<DataType::F16, std::numeric_limits<size_t>::max()>&);
template aclTensor* ConvertToAclTensor<DataType::BF16, std::numeric_limits<size_t>::max()>(const Buffer<DataType::BF16, std::numeric_limits<size_t>::max()>&);

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

aclDataType ConvertToAclDataType(DataType type) {
  switch (type) {
    case DataType::F32:
      return ACL_FLOAT;
    case DataType::F16:
      return ACL_FLOAT16;
    case DataType::BF16:
      return ACL_BF16;
    case DataType::S32:
      return ACL_INT32;
    case DataType::U32:
      return ACL_UINT32;
    case DataType::S64:
      return ACL_INT64;
    case DataType::U64:
      return ACL_UINT64;
    default:
      LOG(FATAL) << "Unsupported data type: " << static_cast<int>(type);
  }
}

// Template implementation for PrintTensorFirstNElementsImpl
template <typename T>
absl::Status PrintTensorFirstNElementsImpl(aclTensor* tensor, int num_elements, const std::string& tensor_name) {
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
  T* host_data = nullptr;
  acl_status = aclrtMallocHost((void**)&host_data, num_elements * sizeof(T));
  if (acl_status != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("aclrtMallocHost failed: ", acl_status));
  }

  // Copy data from device to host
  acl_status = aclrtMemcpy(host_data, num_elements * sizeof(T),
                          tensor_data, num_elements * sizeof(T),
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

// Explicit instantiations for common types
template absl::Status PrintTensorFirstNElementsImpl<float>(aclTensor* tensor, int num_elements, const std::string& tensor_name);
template absl::Status PrintTensorFirstNElementsImpl<half>(aclTensor* tensor, int num_elements, const std::string& tensor_name);

absl::Status PrintTensorFirstNElements(aclTensor* tensor, int num_elements, const std::string& tensor_name) {
  if (tensor == nullptr) {
    return absl::InvalidArgumentError("tensor is nullptr");
  }

  // Get tensor data type
  aclDataType data_type;
  aclnnStatus acl_status = aclGetDataType(tensor, &data_type);
  if (acl_status != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("aclGetDataType failed: ", acl_status));
  }

  // Dispatch based on data type
  switch (data_type) {
    case ACL_FLOAT:
      return PrintTensorFirstNElementsImpl<float>(tensor, num_elements, tensor_name);
    case ACL_FLOAT16:
      return PrintTensorFirstNElementsImpl<half>(tensor, num_elements, tensor_name);
    case ACL_BF16:
      // For BF16, we'll print as 16-bit values
      return PrintTensorFirstNElementsImpl<uint16_t>(tensor, num_elements, tensor_name);
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported data type for printing: ", data_type));
  }
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
  
  return TestDeviceMemoryCopyImpl<float>(device_addr, num_elements);
}

// Template implementation for TestDeviceMemoryCopyImpl
template <typename T>
absl::Status TestDeviceMemoryCopyImpl(void* device_addr, int64_t num_elements) {
  if (device_addr == nullptr) {
    return absl::InvalidArgumentError("device_addr is nullptr");
  }
  
  if (num_elements <= 0) {
    return absl::InvalidArgumentError("num_elements must be positive");
  }
  
  // Allocate host memory for data
  T* host_data = nullptr;
  aclError acl_status = aclrtMallocHost((void**)&host_data, num_elements * sizeof(T));
  if (acl_status != ACL_SUCCESS) {
    return absl::InternalError(
        absl::StrCat("aclrtMallocHost failed: ", acl_status));
  }
  
  // Copy data from device to host
  acl_status = aclrtMemcpy(host_data, num_elements * sizeof(T),
                          device_addr, num_elements * sizeof(T),
                          ACL_MEMCPY_DEVICE_TO_HOST);
  if (acl_status != ACL_SUCCESS) {
    aclrtFreeHost(host_data);
    return absl::InternalError(
        absl::StrCat("aclrtMemcpy failed: ", acl_status));
  }
  
  // Print data
  LOG(INFO) << "Device memory contents (" << num_elements << " elements):";
  for (int64_t i = 0; i < num_elements; i++) {
    LOG(INFO) << "  data[" << i << "] = " << host_data[i];
  }
  
  // Free host memory
  aclrtFreeHost(host_data);
  
  return absl::OkStatus();
}

// Explicit instantiations for common types
template absl::Status TestDeviceMemoryCopyImpl<float>(void* device_addr, int64_t num_elements);
template absl::Status TestDeviceMemoryCopyImpl<half>(void* device_addr, int64_t num_elements);

}  // namespace xla::ffi
