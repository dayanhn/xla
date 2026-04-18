#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_eq_tensor.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Equal operator FFI handler
template <ffi::DataType DType>
ffi::Error EqualHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::Buffer<DType> other, ffi::ResultBuffer<ffi::PRED> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* other_tensor = ConvertToAclTensor(other);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Equal operation on stream: " << stream ;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnEqTensorGetWorkspaceSize(
      self_tensor, other_tensor, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(other_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnEqTensorGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnEqTensor(
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
        absl::StrCat("aclnnEqTensor failed: ", status));
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
template ffi::Error EqualHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::Buffer<ffi::DataType::F32> other, ffi::ResultBuffer<ffi::PRED> out);
template ffi::Error EqualHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::Buffer<ffi::DataType::F16> other, ffi::ResultBuffer<ffi::PRED> out);
template ffi::Error EqualHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::Buffer<ffi::DataType::BF16> other, ffi::ResultBuffer<ffi::PRED> out);
template ffi::Error EqualHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, ffi::Buffer<ffi::DataType::S32> other, ffi::ResultBuffer<ffi::PRED> out);
template ffi::Error EqualHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, ffi::Buffer<ffi::DataType::S64> other, ffi::ResultBuffer<ffi::PRED> out);
template ffi::Error EqualHandlerImpl<ffi::DataType::U8>(aclrtStream stream, ffi::Buffer<ffi::DataType::U8> self, ffi::Buffer<ffi::DataType::U8> other, ffi::ResultBuffer<ffi::PRED> out);
template ffi::Error EqualHandlerImpl<ffi::DataType::S8>(aclrtStream stream, ffi::Buffer<ffi::DataType::S8> self, ffi::Buffer<ffi::DataType::S8> other, ffi::ResultBuffer<ffi::PRED> out);
template ffi::Error EqualHandlerImpl<ffi::DataType::PRED>(aclrtStream stream, ffi::Buffer<ffi::DataType::PRED> self, ffi::Buffer<ffi::DataType::PRED> other, ffi::ResultBuffer<ffi::PRED> out);

// F32 specialization
ffi::Error EqualHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::Buffer<ffi::F32> other, ffi::ResultBuffer<ffi::PRED> out) {
  return EqualHandlerImpl<ffi::DataType::F32>(stream, self, other, out);
}

// F16 specialization
ffi::Error EqualHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::Buffer<ffi::F16> other, ffi::ResultBuffer<ffi::PRED> out) {
  return EqualHandlerImpl<ffi::DataType::F16>(stream, self, other, out);
}

// BF16 specialization
ffi::Error EqualHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::Buffer<ffi::BF16> other, ffi::ResultBuffer<ffi::PRED> out) {
  return EqualHandlerImpl<ffi::DataType::BF16>(stream, self, other, out);
}

// S32 specialization
ffi::Error EqualHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, ffi::Buffer<ffi::S32> other, ffi::ResultBuffer<ffi::PRED> out) {
  return EqualHandlerImpl<ffi::DataType::S32>(stream, self, other, out);
}

// S64 specialization
ffi::Error EqualHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, ffi::Buffer<ffi::S64> other, ffi::ResultBuffer<ffi::PRED> out) {
  return EqualHandlerImpl<ffi::DataType::S64>(stream, self, other, out);
}

// U8 specialization
ffi::Error EqualHandlerU8(aclrtStream stream, ffi::Buffer<ffi::U8> self, ffi::Buffer<ffi::U8> other, ffi::ResultBuffer<ffi::PRED> out) {
  return EqualHandlerImpl<ffi::DataType::U8>(stream, self, other, out);
}

// S8 specialization
ffi::Error EqualHandlerS8(aclrtStream stream, ffi::Buffer<ffi::S8> self, ffi::Buffer<ffi::S8> other, ffi::ResultBuffer<ffi::PRED> out) {
  return EqualHandlerImpl<ffi::DataType::S8>(stream, self, other, out);
}

// PRED specialization
ffi::Error EqualHandlerPRED(aclrtStream stream, ffi::Buffer<ffi::PRED> self, ffi::Buffer<ffi::PRED> other, ffi::ResultBuffer<ffi::PRED> out) {
  return EqualHandlerImpl<ffi::DataType::PRED>(stream, self, other, out);
}

// Register Equal operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqual,
    EqualHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqualF32,
    EqualHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqualF16,
    EqualHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqualBF16,
    EqualHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqualS32,
    EqualHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqualS64,
    EqualHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqualU8,
    EqualHandlerU8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqualS8,
    EqualHandlerS8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S8>>()
        .Arg<ffi::Buffer<ffi::S8>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendEqualPRED,
    EqualHandlerPRED,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
