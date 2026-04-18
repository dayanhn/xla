#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_s_where.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Select operator FFI handler
template <ffi::DataType DType>
ffi::Error SelectHandlerImpl(aclrtStream stream, ffi::Buffer<DType> condition, ffi::Buffer<DType> x, ffi::Buffer<DType> y, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* condition_tensor = ConvertToAclTensor(condition);
  aclTensor* x_tensor = ConvertToAclTensor(x);
  aclTensor* y_tensor = ConvertToAclTensor(y);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Select operation on stream: " << stream ;

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnSWhereGetWorkspaceSize(
      condition_tensor, x_tensor, y_tensor, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(condition_tensor);
    aclDestroyTensor(x_tensor);
    aclDestroyTensor(y_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnSWhereGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnSWhere(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(condition_tensor);
    aclDestroyTensor(x_tensor);
    aclDestroyTensor(y_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnSWhere failed: ", status));
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(condition_tensor);
  aclDestroyTensor(x_tensor);
  aclDestroyTensor(y_tensor);
  aclDestroyTensor(out_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyAclOpExecutor(executor);

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error SelectHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> condition, ffi::Buffer<ffi::DataType::F32> x, ffi::Buffer<ffi::DataType::F32> y, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error SelectHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> condition, ffi::Buffer<ffi::DataType::F16> x, ffi::Buffer<ffi::DataType::F16> y, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error SelectHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> condition, ffi::Buffer<ffi::DataType::BF16> x, ffi::Buffer<ffi::DataType::BF16> y, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error SelectHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> condition, ffi::Buffer<ffi::DataType::S32> x, ffi::Buffer<ffi::DataType::S32> y, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error SelectHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> condition, ffi::Buffer<ffi::DataType::S64> x, ffi::Buffer<ffi::DataType::S64> y, ffi::ResultBuffer<ffi::DataType::S64> out);
template ffi::Error SelectHandlerImpl<ffi::DataType::PRED>(aclrtStream stream, ffi::Buffer<ffi::DataType::PRED> condition, ffi::Buffer<ffi::DataType::PRED> x, ffi::Buffer<ffi::DataType::PRED> y, ffi::ResultBuffer<ffi::DataType::PRED> out);

// F32 specialization
ffi::Error SelectHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> condition, ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> y, ffi::ResultBuffer<ffi::F32> out) {
  return SelectHandlerImpl<ffi::DataType::F32>(stream, condition, x, y, out);
}

// F16 specialization
ffi::Error SelectHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> condition, ffi::Buffer<ffi::F16> x, ffi::Buffer<ffi::F16> y, ffi::ResultBuffer<ffi::F16> out) {
  return SelectHandlerImpl<ffi::DataType::F16>(stream, condition, x, y, out);
}

// BF16 specialization
ffi::Error SelectHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> condition, ffi::Buffer<ffi::BF16> x, ffi::Buffer<ffi::BF16> y, ffi::ResultBuffer<ffi::BF16> out) {
  return SelectHandlerImpl<ffi::DataType::BF16>(stream, condition, x, y, out);
}

// S32 specialization
ffi::Error SelectHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> condition, ffi::Buffer<ffi::S32> x, ffi::Buffer<ffi::S32> y, ffi::ResultBuffer<ffi::S32> out) {
  return SelectHandlerImpl<ffi::DataType::S32>(stream, condition, x, y, out);
}

// S64 specialization
ffi::Error SelectHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> condition, ffi::Buffer<ffi::S64> x, ffi::Buffer<ffi::S64> y, ffi::ResultBuffer<ffi::S64> out) {
  return SelectHandlerImpl<ffi::DataType::S64>(stream, condition, x, y, out);
}

// PRED specialization
ffi::Error SelectHandlerPRED(aclrtStream stream, ffi::Buffer<ffi::PRED> condition, ffi::Buffer<ffi::PRED> x, ffi::Buffer<ffi::PRED> y, ffi::ResultBuffer<ffi::PRED> out) {
  return SelectHandlerImpl<ffi::DataType::PRED>(stream, condition, x, y, out);
}

// Register Select operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendSelect,
    SelectHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendSelectF32,
    SelectHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendSelectF16,
    SelectHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendSelectBF16,
    SelectHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendSelectS32,
    SelectHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendSelectS64,
    SelectHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendSelectPRED,
    SelectHandlerPRED,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Arg<ffi::Buffer<ffi::PRED>>()
        .Ret<ffi::Buffer<ffi::PRED>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
