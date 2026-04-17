#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_add.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Add operator FFI handler
template <ffi::DataType DType>
ffi::Error AddHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::Buffer<DType> other, float alpha, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* other_tensor = ConvertToAclTensor(other);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  
  // Create alpha scalar
  aclScalar* alpha_scalar = aclCreateScalar(&alpha, ACL_FLOAT);
  if (alpha_scalar == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(other_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal("Failed to create alpha scalar");
  }
  
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Add operation on stream: " << stream ;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnAddGetWorkspaceSize(
      self_tensor, other_tensor, alpha_scalar, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(other_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyScalar(alpha_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnAddGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnAdd(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(other_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyScalar(alpha_scalar);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnAdd failed: ", status));
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(other_tensor);
  aclDestroyTensor(out_tensor);
  aclDestroyScalar(alpha_scalar);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error AddHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::Buffer<ffi::DataType::F32> other, float alpha, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error AddHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::Buffer<ffi::DataType::F16> other, float alpha, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error AddHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::Buffer<ffi::DataType::BF16> other, float alpha, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error AddHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, ffi::Buffer<ffi::DataType::S32> other, float alpha, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error AddHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, ffi::Buffer<ffi::DataType::S64> other, float alpha, ffi::ResultBuffer<ffi::DataType::S64> out);

// F32 specialization
ffi::Error AddHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::Buffer<ffi::F32> other, float alpha, ffi::ResultBuffer<ffi::F32> out) {
  return AddHandlerImpl<ffi::DataType::F32>(stream, self, other, alpha, out);
}

// F16 specialization
ffi::Error AddHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::Buffer<ffi::F16> other, float alpha, ffi::ResultBuffer<ffi::F16> out) {
  return AddHandlerImpl<ffi::DataType::F16>(stream, self, other, alpha, out);
}

// BF16 specialization
ffi::Error AddHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::Buffer<ffi::BF16> other, float alpha, ffi::ResultBuffer<ffi::BF16> out) {
  return AddHandlerImpl<ffi::DataType::BF16>(stream, self, other, alpha, out);
}

// S32 specialization
ffi::Error AddHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, ffi::Buffer<ffi::S32> other, float alpha, ffi::ResultBuffer<ffi::S32> out) {
  return AddHandlerImpl<ffi::DataType::S32>(stream, self, other, alpha, out);
}

// S64 specialization
ffi::Error AddHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, ffi::Buffer<ffi::S64> other, float alpha, ffi::ResultBuffer<ffi::S64> out) {
  return AddHandlerImpl<ffi::DataType::S64>(stream, self, other, alpha, out);
}

// Register Add operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendAdd,
    AddHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("alpha")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendAddF32,
    AddHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("alpha")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendAddF16,
    AddHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Attr<float>("alpha")
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendAddBF16,
    AddHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Attr<float>("alpha")
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendAddS32,
    AddHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Attr<float>("alpha")
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendAddS64,
    AddHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<float>("alpha")
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi