#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_reduce_sum.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of ReduceSum operator FFI handler
template <ffi::DataType DType>
ffi::Error ReduceSumHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for ReduceSum operation on stream: " << stream ;

  // Create dims array
  aclIntArray* dims_array = nullptr;
  aclCreateIntArray(dims.size(), dims.data(), &dims_array);

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnReduceSumGetWorkspaceSize(
      self_tensor, dims_array, keep_dims, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(dims_array);
    return ffi::Error::Internal(
        absl::StrCat("aclnnReduceSumGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnReduceSum(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(dims_array);
    return ffi::Error::Internal(
        absl::StrCat("aclnnReduceSum failed: ", status));
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
template ffi::Error ReduceSumHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error ReduceSumHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error ReduceSumHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error ReduceSumHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error ReduceSumHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::S64> out);

// F32 specialization
ffi::Error ReduceSumHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::F32> out) {
  return ReduceSumHandlerImpl<ffi::DataType::F32>(stream, self, dims, keep_dims, out);
}

// F16 specialization
ffi::Error ReduceSumHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::F16> out) {
  return ReduceSumHandlerImpl<ffi::DataType::F16>(stream, self, dims, keep_dims, out);
}

// BF16 specialization
ffi::Error ReduceSumHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::BF16> out) {
  return ReduceSumHandlerImpl<ffi::DataType::BF16>(stream, self, dims, keep_dims, out);
}

// S32 specialization
ffi::Error ReduceSumHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::S32> out) {
  return ReduceSumHandlerImpl<ffi::DataType::S32>(stream, self, dims, keep_dims, out);
}

// S64 specialization
ffi::Error ReduceSumHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, std::vector<int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::S64> out) {
  return ReduceSumHandlerImpl<ffi::DataType::S64>(stream, self, dims, keep_dims, out);
}

// Register ReduceSum operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceSum,
    ReduceSumHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceSumF32,
    ReduceSumHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceSumF16,
    ReduceSumHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceSumBF16,
    ReduceSumHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceSumS32,
    ReduceSumHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceSumS64,
    ReduceSumHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Arg<ffi::Array<int64_t>>()
        .Arg<ffi::Bool>()
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
