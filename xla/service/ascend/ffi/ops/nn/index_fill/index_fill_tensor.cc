#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_index_fill_tensor.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// InplaceIndexFillTensor operator FFI handler for F32
// Note: XLA FFI does not support scalar arguments as .Arg<>().
// We use .Attr<T>("name") for scalar parameters (dim and value).
ffi::Error InplaceIndexFillTensorHandlerF32(
    aclrtStream stream, 
    ffi::Buffer<ffi::F32> self,
    int64_t dim,
    ffi::Buffer<ffi::S64> index,
    float value) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* self_tensor = ConvertToAclTensor(self);
  
  // Create aclIntArray from index buffer
  auto index_dims = index.dimensions();
  int64_t index_size = 1;
  for (auto d : index_dims) {
    index_size *= d;
  }
  aclIntArray* index_int_array = aclCreateIntArray(
      static_cast<const int64_t*>(index.untyped_data()), index_size);
  if (index_int_array == nullptr) {
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclIntArray from index buffer");
  }
  
  // Create aclScalar from value
  aclScalar* value_scalar = aclCreateScalar(&value, ACL_FLOAT);
  if (value_scalar == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    return ffi::Error::Internal("Failed to create aclScalar from value");
  }
  
  LOG(INFO) << "InplaceIndexFillTensor F32: dim=" << dim 
            << ", value=" << value 
            << ", index_size=" << index_size;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnInplaceIndexFillTensorGetWorkspaceSize(
      self_tensor, dim, index_int_array, value_scalar, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensorGetWorkspaceSize failed: ", status));
  }

  // Call second stage interface to execute computation
  status = aclnnInplaceIndexFillTensor(
      nullptr,  // workspace is managed by XLA
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensor failed: ", status));
  }

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyIntArray(index_int_array);
  aclDestroyScalar(value_scalar);

  return ffi::Error::Success();
}

// InplaceIndexFillTensor operator FFI handler for F16
// F16 is represented as uint16_t in FFI
ffi::Error InplaceIndexFillTensorHandlerF16(
    aclrtStream stream, 
    ffi::Buffer<ffi::F16> self,
    int64_t dim,
    ffi::Buffer<ffi::S64> index,
    uint16_t value) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* self_tensor = ConvertToAclTensor(self);
  
  // Create aclIntArray from index buffer
  auto index_dims = index.dimensions();
  int64_t index_size = 1;
  for (auto d : index_dims) {
    index_size *= d;
  }
  aclIntArray* index_int_array = aclCreateIntArray(
      static_cast<const int64_t*>(index.untyped_data()), index_size);
  if (index_int_array == nullptr) {
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclIntArray from index buffer");
  }
  
  // Create aclScalar from value (F16)
  aclScalar* value_scalar = aclCreateScalar(&value, ACL_FLOAT16);
  if (value_scalar == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    return ffi::Error::Internal("Failed to create aclScalar from value");
  }
  
  LOG(INFO) << "InplaceIndexFillTensor F16: dim=" << dim 
            << ", value(u16)=" << value 
            << ", index_size=" << index_size;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnInplaceIndexFillTensorGetWorkspaceSize(
      self_tensor, dim, index_int_array, value_scalar, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensorGetWorkspaceSize failed: ", status));
  }

  // Call second stage interface to execute computation
  status = aclnnInplaceIndexFillTensor(
      nullptr,
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensor failed: ", status));
  }

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyIntArray(index_int_array);
  aclDestroyScalar(value_scalar);

  return ffi::Error::Success();
}

// InplaceIndexFillTensor operator FFI handler for BF16
// BF16 is represented as uint16_t in FFI
ffi::Error InplaceIndexFillTensorHandlerBF16(
    aclrtStream stream, 
    ffi::Buffer<ffi::BF16> self,
    int64_t dim,
    ffi::Buffer<ffi::S64> index,
    uint16_t value) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* self_tensor = ConvertToAclTensor(self);
  
  // Create aclIntArray from index buffer
  auto index_dims = index.dimensions();
  int64_t index_size = 1;
  for (auto d : index_dims) {
    index_size *= d;
  }
  aclIntArray* index_int_array = aclCreateIntArray(
      static_cast<const int64_t*>(index.untyped_data()), index_size);
  if (index_int_array == nullptr) {
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclIntArray from index buffer");
  }
  
  // Create aclScalar from value (BF16)
  aclScalar* value_scalar = aclCreateScalar(&value, ACL_BF16);
  if (value_scalar == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    return ffi::Error::Internal("Failed to create aclScalar from value");
  }
  
  LOG(INFO) << "InplaceIndexFillTensor BF16: dim=" << dim 
            << ", value(u16)=" << value 
            << ", index_size=" << index_size;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnInplaceIndexFillTensorGetWorkspaceSize(
      self_tensor, dim, index_int_array, value_scalar, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensorGetWorkspaceSize failed: ", status));
  }

  // Call second stage interface to execute computation
  status = aclnnInplaceIndexFillTensor(
      nullptr,
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensor failed: ", status));
  }

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyIntArray(index_int_array);
  aclDestroyScalar(value_scalar);

  return ffi::Error::Success();
}

// InplaceIndexFillTensor operator FFI handler for S32
ffi::Error InplaceIndexFillTensorHandlerS32(
    aclrtStream stream, 
    ffi::Buffer<ffi::S32> self,
    int64_t dim,
    ffi::Buffer<ffi::S64> index,
    int32_t value) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* self_tensor = ConvertToAclTensor(self);
  
  // Create aclIntArray from index buffer
  auto index_dims = index.dimensions();
  int64_t index_size = 1;
  for (auto d : index_dims) {
    index_size *= d;
  }
  aclIntArray* index_int_array = aclCreateIntArray(
      static_cast<const int64_t*>(index.untyped_data()), index_size);
  if (index_int_array == nullptr) {
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclIntArray from index buffer");
  }
  
  // Create aclScalar from value
  aclScalar* value_scalar = aclCreateScalar(&value, ACL_INT32);
  if (value_scalar == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    return ffi::Error::Internal("Failed to create aclScalar from value");
  }
  
  LOG(INFO) << "InplaceIndexFillTensor S32: dim=" << dim 
            << ", value=" << value 
            << ", index_size=" << index_size;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnInplaceIndexFillTensorGetWorkspaceSize(
      self_tensor, dim, index_int_array, value_scalar, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensorGetWorkspaceSize failed: ", status));
  }

  // Call second stage interface to execute computation
  status = aclnnInplaceIndexFillTensor(
      nullptr,
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensor failed: ", status));
  }

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyIntArray(index_int_array);
  aclDestroyScalar(value_scalar);

  return ffi::Error::Success();
}

// InplaceIndexFillTensor operator FFI handler for S64
ffi::Error InplaceIndexFillTensorHandlerS64(
    aclrtStream stream, 
    ffi::Buffer<ffi::S64> self,
    int64_t dim,
    ffi::Buffer<ffi::S64> index,
    int64_t value) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* self_tensor = ConvertToAclTensor(self);
  
  // Create aclIntArray from index buffer
  auto index_dims = index.dimensions();
  int64_t index_size = 1;
  for (auto d : index_dims) {
    index_size *= d;
  }
  aclIntArray* index_int_array = aclCreateIntArray(
      static_cast<const int64_t*>(index.untyped_data()), index_size);
  if (index_int_array == nullptr) {
    aclDestroyTensor(self_tensor);
    return ffi::Error::Internal("Failed to create aclIntArray from index buffer");
  }
  
  // Create aclScalar from value
  aclScalar* value_scalar = aclCreateScalar(&value, ACL_INT64);
  if (value_scalar == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    return ffi::Error::Internal("Failed to create aclScalar from value");
  }
  
  LOG(INFO) << "InplaceIndexFillTensor S64: dim=" << dim 
            << ", value=" << value 
            << ", index_size=" << index_size;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnInplaceIndexFillTensorGetWorkspaceSize(
      self_tensor, dim, index_int_array, value_scalar, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensorGetWorkspaceSize failed: ", status));
  }

  // Call second stage interface to execute computation
  status = aclnnInplaceIndexFillTensor(
      nullptr,
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyIntArray(index_int_array);
    aclDestroyScalar(value_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnInplaceIndexFillTensor failed: ", status));
  }

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyIntArray(index_int_array);
  aclDestroyScalar(value_scalar);

  return ffi::Error::Success();
}

// Register InplaceIndexFillTensor operator FFI functions for different data types
// Using .Attr<T>("name") for scalar parameters since FFI doesn't support .Arg<basic_type>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendInplaceIndexFillTensorF32,
    InplaceIndexFillTensorHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("dim")
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<float>("value"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendInplaceIndexFillTensorF16,
    InplaceIndexFillTensorHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Attr<int64_t>("dim")
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<uint16_t>("value"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendInplaceIndexFillTensorBF16,
    InplaceIndexFillTensorHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Attr<int64_t>("dim")
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<uint16_t>("value"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendInplaceIndexFillTensorS32,
    InplaceIndexFillTensorHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Attr<int64_t>("dim")
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<int32_t>("value"),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendInplaceIndexFillTensorS64,
    InplaceIndexFillTensorHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<int64_t>("dim")
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<int64_t>("value"),
    {ffi::Traits::kCmdBufferCompatible});

// Generic handler for InplaceIndexFillTensor (default to F32)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendInplaceIndexFillTensor,
    InplaceIndexFillTensorHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("dim")
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<float>("value"),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
