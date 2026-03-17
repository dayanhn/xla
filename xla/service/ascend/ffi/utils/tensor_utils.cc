
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tensorflow/compiler/xla/literal.h"

namespace xla::ffi {
  aclTensor* ConvertToAclTensor(const BufferBase& buffer) {
  // Get buffer shape and data type
  Shape shape = buffer.shape();
  std::vector<int64_t> dims;
  for (int i = 0; i < shape.rank(); ++i) {
    dims.push_back(shape.dimensions(i));
  }

  // Determine Ascend data type
  aclDataType data_type = ConvertToAclDataType(shape.element_type());

  // Calculate strides
  std::vector<int64_t> strides(dims.size(), 1);
  for (int i = dims.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }

  // Create aclTensor
  return aclCreateTensor(
      dims.data(),
      dims.size(),
      data_type,
      strides.data(),
      0,
      ACL_FORMAT_ND,
      dims.data(),
      dims.size(),
      const_cast<void*>(buffer.data()));
}

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