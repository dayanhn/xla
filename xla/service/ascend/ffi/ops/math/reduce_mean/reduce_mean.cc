#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_mean.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of ReduceMean operator FFI handler
template <ffi::DataType DType>
ffi::Error ReduceMeanHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for ReduceMean operation on stream: " << stream ;

  // Create dims array directly from int64_t span (no conversion needed)
  aclIntArray* dims_array = nullptr;
  dims_array = aclCreateIntArray(dims.begin(), dims.size());
  if (dims_array == nullptr) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal("aclCreateIntArray failed");
  }

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnMeanGetWorkspaceSize(
      self_tensor, dims_array, keep_dims, static_cast<aclDataType>(0), out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(dims_array);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMeanGetWorkspaceSize failed: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  status = aclnnMean(
      workspaceAddr, 
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(dims_array);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    aclDestroyAclOpExecutor(executor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnMean failed: ", status));
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  aclDestroyIntArray(dims_array);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  aclDestroyAclOpExecutor(executor);

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error ReduceMeanHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error ReduceMeanHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error ReduceMeanHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::BF16> out);

// F32 specialization
ffi::Error ReduceMeanHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::F32> out) {
  return ReduceMeanHandlerImpl<ffi::DataType::F32>(stream, self, dims, keep_dims, out);
}

// F16 specialization
ffi::Error ReduceMeanHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::F16> out) {
  return ReduceMeanHandlerImpl<ffi::DataType::F16>(stream, self, dims, keep_dims, out);
}

// BF16 specialization
ffi::Error ReduceMeanHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::BF16> out) {
  return ReduceMeanHandlerImpl<ffi::DataType::BF16>(stream, self, dims, keep_dims, out);
}

// Register ReduceMean operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMean,
    ReduceMeanHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMeanF32,
    ReduceMeanHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMeanF16,
    ReduceMeanHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceMeanBF16,
    ReduceMeanHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
