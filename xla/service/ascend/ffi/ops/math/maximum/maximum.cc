#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_maximum.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Maximum operator FFI handler
template <ffi::DataType DType>
ffi::Error MaximumHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::Buffer<DType> other, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* other_tensor = ConvertToAclTensor(other);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Maximum operation on stream: " << stream ;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMaximumGetWorkspaceSize(
      self_tensor, other_tensor, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(other_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMaximumGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnMaximum(
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
        absl::StrCat("aclnnMaximum failed: ", status));
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
template ffi::Error MaximumHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::Buffer<ffi::DataType::F32> other, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error MaximumHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::Buffer<ffi::DataType::F16> other, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error MaximumHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::Buffer<ffi::DataType::BF16> other, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error MaximumHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, ffi::Buffer<ffi::DataType::S32> other, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error MaximumHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, ffi::Buffer<ffi::DataType::S64> other, ffi::ResultBuffer<ffi::DataType::S64> out);

// F32 specialization
ffi::Error MaximumHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::Buffer<ffi::F32> other, ffi::ResultBuffer<ffi::F32> out) {
  return MaximumHandlerImpl<ffi::DataType::F32>(stream, self, other, out);
}

// F16 specialization
ffi::Error MaximumHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::Buffer<ffi::F16> other, ffi::ResultBuffer<ffi::F16> out) {
  return MaximumHandlerImpl<ffi::DataType::F16>(stream, self, other, out);
}

// BF16 specialization
ffi::Error MaximumHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::Buffer<ffi::BF16> other, ffi::ResultBuffer<ffi::BF16> out) {
  return MaximumHandlerImpl<ffi::DataType::BF16>(stream, self, other, out);
}

// S32 specialization
ffi::Error MaximumHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, ffi::Buffer<ffi::S32> other, ffi::ResultBuffer<ffi::S32> out) {
  return MaximumHandlerImpl<ffi::DataType::S32>(stream, self, other, out);
}

// S64 specialization
ffi::Error MaximumHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, ffi::Buffer<ffi::S64> other, ffi::ResultBuffer<ffi::S64> out) {
  return MaximumHandlerImpl<ffi::DataType::S64>(stream, self, other, out);
}

// Register Maximum operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaximum,
    MaximumHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaximumF32,
    MaximumHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaximumF16,
    MaximumHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaximumBF16,
    MaximumHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaximumS32,
    MaximumHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendMaximumS64,
    MaximumHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
