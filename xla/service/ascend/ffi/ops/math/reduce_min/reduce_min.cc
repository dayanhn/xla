#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_min.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of ReduceMin operator FFI handler
template <ffi::DataType DType>
ffi::Error ReduceMinHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for ReduceMin operation on stream: " << stream ;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMinGetWorkspaceSize(
      self_tensor, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMinGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnMin(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMin failed: ", status));
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  //aclDestroyAclOpExecutor(executor);

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error ReduceMinHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error ReduceMinHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error ReduceMinHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error ReduceMinHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error ReduceMinHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, ffi::ResultBuffer<ffi::DataType::S64> out);

// F32 specialization
ffi::Error ReduceMinHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::ResultBuffer<ffi::F32> out) {
  return ReduceMinHandlerImpl<ffi::DataType::F32>(stream, self, out);
}

// F16 specialization
ffi::Error ReduceMinHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::ResultBuffer<ffi::F16> out) {
  return ReduceMinHandlerImpl<ffi::DataType::F16>(stream, self, out);
}

// BF16 specialization
ffi::Error ReduceMinHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::ResultBuffer<ffi::BF16> out) {
  return ReduceMinHandlerImpl<ffi::DataType::BF16>(stream, self, out);
}

// S32 specialization
ffi::Error ReduceMinHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, ffi::ResultBuffer<ffi::S32> out) {
  return ReduceMinHandlerImpl<ffi::DataType::S32>(stream, self, out);
}

// S64 specialization
ffi::Error ReduceMinHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, ffi::ResultBuffer<ffi::S64> out) {
  return ReduceMinHandlerImpl<ffi::DataType::S64>(stream, self, out);
}

// Register ReduceMin operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMin,
    ReduceMinHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMinF32,
    ReduceMinHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMinF16,
    ReduceMinHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMinBF16,
    ReduceMinHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMinS32,
    ReduceMinHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMinS64,
    ReduceMinHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
