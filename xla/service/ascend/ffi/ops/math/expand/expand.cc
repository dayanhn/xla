#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_expand.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of Expand operator FFI handler
template <ffi::DataType DType>
ffi::Error ExpandHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::Array<int64_t> size, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for Expand operation on stream: " << stream ;

  // Create size array
  aclIntArray* size_array = aclCreateIntArray(size.data(), size.size());
  if (size_array == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal("Failed to create aclIntArray for size");
  }

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnExpandGetWorkspaceSize(
      self_tensor, size_array, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(size_array);
    return ffi::Error::Internal(
        absl::StrCat("aclnnExpandGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnExpand(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(size_array);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnExpand failed: ", status));
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  aclDestroyIntArray(size_array);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error ExpandHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::DataType::S64> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::U8>(aclrtStream stream, ffi::Buffer<ffi::DataType::U8> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::DataType::U8> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::S8>(aclrtStream stream, ffi::Buffer<ffi::DataType::S8> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::DataType::S8> out);
template ffi::Error ExpandHandlerImpl<ffi::DataType::BOOL>(aclrtStream stream, ffi::Buffer<ffi::DataType::BOOL> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::DataType::BOOL> out);

// F32 specialization
ffi::Error ExpandHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::F32> out) {
  return ExpandHandlerImpl<ffi::DataType::F32>(stream, self, size, out);
}

// F16 specialization
ffi::Error ExpandHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::F16> out) {
  return ExpandHandlerImpl<ffi::DataType::F16>(stream, self, size, out);
}

// BF16 specialization
ffi::Error ExpandHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::BF16> out) {
  return ExpandHandlerImpl<ffi::DataType::BF16>(stream, self, size, out);
}

// S32 specialization
ffi::Error ExpandHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::S32> out) {
  return ExpandHandlerImpl<ffi::DataType::S32>(stream, self, size, out);
}

// S64 specialization
ffi::Error ExpandHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::S64> out) {
  return ExpandHandlerImpl<ffi::DataType::S64>(stream, self, size, out);
}

// U8 specialization
ffi::Error ExpandHandlerU8(aclrtStream stream, ffi::Buffer<ffi::U8> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::U8> out) {
  return ExpandHandlerImpl<ffi::DataType::U8>(stream, self, size, out);
}

// S8 specialization
ffi::Error ExpandHandlerS8(aclrtStream stream, ffi::Buffer<ffi::S8> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::S8> out) {
  return ExpandHandlerImpl<ffi::DataType::S8>(stream, self, size, out);
}

// BOOL specialization
ffi::Error ExpandHandlerBOOL(aclrtStream stream, ffi::Buffer<ffi::BOOL> self, ffi::Array<int64_t> size, ffi::ResultBuffer<ffi::BOOL> out) {
  return ExpandHandlerImpl<ffi::DataType::BOOL>(stream, self, size, out);
}

// Register Expand operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpand,
    ExpandHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandF32,
    ExpandHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandF16,
    ExpandHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandBF16,
    ExpandHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandS32,
    ExpandHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandS64,
    ExpandHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandU8,
    ExpandHandlerU8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::U8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandS8,
    ExpandHandlerS8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S8>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::S8>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendExpandBOOL,
    ExpandHandlerBOOL,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BOOL>>()
        .Arg<ffi::Array<int64_t>>()
        .Ret<ffi::Buffer<ffi::BOOL>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
