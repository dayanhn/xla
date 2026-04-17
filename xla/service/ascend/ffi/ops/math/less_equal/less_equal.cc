#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_le_tensor.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of LessEqual operator FFI handler
template <ffi::DataType DType>
ffi::Error LessEqualHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::Buffer<DType> other, ffi::ResultBuffer<ffi::BOOL> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* other_tensor = ConvertToAclTensor(other);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for LessEqual operation on stream: " << stream ;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnLeTensorGetWorkspaceSize(
      self_tensor, other_tensor, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(other_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnLeTensorGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnLeTensor(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(other_tensor);
    aclDestroyTensor(out_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnLeTensor failed: ", status));
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(other_tensor);
  aclDestroyTensor(out_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error LessEqualHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::Buffer<ffi::DataType::F32> other, ffi::ResultBuffer<ffi::BOOL> out);
template ffi::Error LessEqualHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::Buffer<ffi::DataType::F16> other, ffi::ResultBuffer<ffi::BOOL> out);
template ffi::Error LessEqualHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::Buffer<ffi::DataType::BF16> other, ffi::ResultBuffer<ffi::BOOL> out);
template ffi::Error LessEqualHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, ffi::Buffer<ffi::DataType::S32> other, ffi::ResultBuffer<ffi::BOOL> out);
template ffi::Error LessEqualHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, ffi::Buffer<ffi::DataType::S64> other, ffi::ResultBuffer<ffi::BOOL> out);
template ffi::Error LessEqualHandlerImpl<ffi::DataType::U8>(aclrtStream stream, ffi::Buffer<ffi::DataType::U8> self, ffi::Buffer<ffi::DataType::U8> other, ffi::ResultBuffer<ffi::BOOL> out);
template ffi::Error LessEqualHandlerImpl<ffi::DataType::S8>(aclrtStream stream, ffi::Buffer<ffi::DataType::S8> self, ffi::Buffer<ffi::DataType::S8> other, ffi::ResultBuffer<ffi::BOOL> out);
template ffi::Error LessEqualHandlerImpl<ffi::DataType::BOOL>(aclrtStream stream, ffi::Buffer<ffi::DataType::BOOL> self, ffi::Buffer<ffi::DataType::BOOL> other, ffi::ResultBuffer<ffi::BOOL> out);

// F32 specialization
ffi::Error LessEqualHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::Buffer<ffi::F32> other, ffi::ResultBuffer<ffi::BOOL> out) {
  return LessEqualHandlerImpl<ffi::DataType::F32>(stream, self, other, out);
}

// F16 specialization
ffi::Error LessEqualHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::Buffer<ffi::F16> other, ffi::ResultBuffer<ffi::BOOL> out) {
  return LessEqualHandlerImpl<ffi::DataType::F16>(stream, self, other, out);
}

// BF16 specialization
ffi::Error LessEqualHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::Buffer<ffi::BF16> other, ffi::ResultBuffer<ffi::BOOL> out) {
  return LessEqualHandlerImpl<ffi::DataType::BF16>(stream, self, other, out);
}

// S32 specialization
ffi::Error LessEqualHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, ffi::Buffer<ffi::S32> other, ffi::ResultBuffer<ffi::BOOL> out) {
  return LessEqualHandlerImpl<ffi::DataType::S32>(stream, self, other, out);
}

// S64 specialization
ffi::Error LessEqualHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, ffi::Buffer<ffi::S64> other, ffi::ResultBuffer<ffi::BOOL> out) {
  return LessEqualHandlerImpl<ffi::DataType::S64>(stream, self, other, out);
}

// U8 specialization
ffi::Error LessEqualHandlerU8(aclrtStream stream, ffi::Buffer<ffi::U8> self, ffi::Buffer<ffi::U8> other, ffi::ResultBuffer<ffi::BOOL> out) {
  return LessEqualHandlerImpl<ffi::DataType::U8>(stream, self, other, out);
}

// S8 specialization
ffi::Error LessEqualHandlerS8(aclrtStream stream, ffi::Buffer<ffi::S8> self, ffi::Buffer<ffi::S8> other, ffi::ResultBuffer<ffi::BOOL> out) {
  return LessEqualHandlerImpl<ffi::DataType::S8>(stream, self, other, out);
}

// BOOL specialization
ffi::Error LessEqualHandlerBOOL(aclrtStream stream, ffi::Buffer<ffi::BOOL> self, ffi::Buffer<ffi::BOOL> other, ffi::ResultBuffer<ffi::BOOL> out) {
  return LessEqualHandlerImpl<ffi::DataType::BOOL>(stream, self, other, out);
}

// Register LessEqual operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqual,
    LessEqualHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqualF32,
    LessEqualHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqualF16,
    LessEqualHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqualBF16,
    LessEqualHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqualS32,
    LessEqualHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqualS64,
    LessEqualHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqualU8,
    LessEqualHandlerU8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqualS8,
    LessEqualHandlerS8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S8>>()
        .Arg<ffi::Buffer<ffi::S8>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendLessEqualBOOL,
    LessEqualHandlerBOOL,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BOOL>>()
        .Arg<ffi::Buffer<ffi::BOOL>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
