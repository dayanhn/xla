#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_mul.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include <iostream>

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Muls operator FFI handler
template <ffi::DataType DType>
ffi::Error MulsHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, float other, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  
  // Create other scalar
  aclScalar* other_scalar = aclCreateScalar(&other, ACL_FLOAT);
  if (other_scalar == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal("Failed to create other scalar");
  }
  
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Muls operation on stream: " << stream ;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMulsGetWorkspaceSize(
      self_tensor, other_scalar, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyScalar(other_scalar);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMulsGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnMuls(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyScalar(other_scalar);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnMuls failed: ", status));
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  aclDestroyScalar(other_scalar);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }


  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error MulsHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, float other, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error MulsHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, float other, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error MulsHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, float other, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error MulsHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, float other, ffi::ResultBuffer<ffi::DataType::S32> out);

// F32 specialization
ffi::Error MulsHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, float other, ffi::ResultBuffer<ffi::F32> out) {
  return MulsHandlerImpl<ffi::DataType::F32>(stream, self, other, out);
}

// F16 specialization
ffi::Error MulsHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, float other, ffi::ResultBuffer<ffi::F16> out) {
  return MulsHandlerImpl<ffi::DataType::F16>(stream, self, other, out);
}

// BF16 specialization
ffi::Error MulsHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, float other, ffi::ResultBuffer<ffi::BF16> out) {
  return MulsHandlerImpl<ffi::DataType::BF16>(stream, self, other, out);
}

// S32 specialization
ffi::Error MulsHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, float other, ffi::ResultBuffer<ffi::S32> out) {
  return MulsHandlerImpl<ffi::DataType::S32>(stream, self, other, out);
}

// Register Muls operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMuls,
    MulsHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("other")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMulsF32,
    MulsHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("other")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMulsF16,
    MulsHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Attr<float>("other")
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMulsBF16,
    MulsHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Attr<float>("other")
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMulsS32,
    MulsHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Attr<float>("other")
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
