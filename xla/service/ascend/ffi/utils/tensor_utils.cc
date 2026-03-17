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

}  // namespace xla::ffi
