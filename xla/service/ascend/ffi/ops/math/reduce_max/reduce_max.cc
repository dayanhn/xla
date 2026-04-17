#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_max_v2.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of ReduceMax operator FFI handler
template <ffi::DataType DType>
ffi::Error ReduceMaxHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for ReduceMax operation on stream: " << stream ;

  // Create dims array
  aclIntArray* dims_array = nullptr;
  aclCreateIntArray(dims.size(), dims.data(), &dims_array);

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMaxV2GetWorkspaceSize(
      self_tensor, dims_array, keep_dims, noop_with_empty_dims, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(dims_array);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMaxV2GetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnMaxV2(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(dims_array);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMaxV2 failed: ", status));
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  aclDestroyIntArray(dims_array);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyOpExecutor(executor);

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error ReduceMaxHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error ReduceMaxHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error ReduceMaxHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error ReduceMaxHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error ReduceMaxHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::DataType::S64> out);

// F32 specialization
ffi::Error ReduceMaxHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::F32> out) {
  return ReduceMaxHandlerImpl<ffi::DataType::F32>(stream, self, dims, keep_dims, noop_with_empty_dims, out);
}

// F16 specialization
ffi::Error ReduceMaxHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::F16> out) {
  return ReduceMaxHandlerImpl<ffi::DataType::F16>(stream, self, dims, keep_dims, noop_with_empty_dims, out);
}

// BF16 specialization
ffi::Error ReduceMaxHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::BF16> out) {
  return ReduceMaxHandlerImpl<ffi::DataType::BF16>(stream, self, dims, keep_dims, noop_with_empty_dims, out);
}

// S32 specialization
ffi::Error ReduceMaxHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::S32> out) {
  return ReduceMaxHandlerImpl<ffi::DataType::S32>(stream, self, dims, keep_dims, noop_with_empty_dims, out);
}

// S64 specialization
ffi::Error ReduceMaxHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, std::vector<int64_t> dims, bool keep_dims, bool noop_with_empty_dims, ffi::ResultBuffer<ffi::S64> out) {
  return ReduceMaxHandlerImpl<ffi::DataType::S64>(stream, self, dims, keep_dims, noop_with_empty_dims, out);
}

// Register ReduceMax operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMax,
    ReduceMaxHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMaxF32,
    ReduceMaxHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMaxF16,
    ReduceMaxHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMaxBF16,
    ReduceMaxHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMaxS32,
    ReduceMaxHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMaxS64,
    ReduceMaxHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
