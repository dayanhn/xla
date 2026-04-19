#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_prod.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Template version of ReduceProd operator FFI handler
template <ffi::DataType DType>
ffi::Error ReduceProdHandlerImpl(aclrtStream stream, ffi::Buffer<DType> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<DType> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);
  LOG(INFO) << "Converted XLA buffers to Ascend tensors for ReduceProd operation on stream: " << stream ;

  // Create dims array
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
  aclnnStatus status = ACL_SUCCESS;
  // Note: aclnnProdDimGetWorkspaceSize only supports single dimension reduction
  // For multiple dimensions, we need to call it iteratively
  if (dims.size() == 0) {
    // If no dimensions specified, compute product of all elements
    status = aclnnProdGetWorkspaceSize(
        self_tensor, static_cast<aclDataType>(0), out_tensor, &workspace_size, &executor);
  } else {
    // For single dimension reduction
    int64_t dim = dims[0];
    status = aclnnProdDimGetWorkspaceSize(
        self_tensor, dim, keep_dims, static_cast<aclDataType>(0), out_tensor, &workspace_size, &executor);
  }
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyIntArray(dims_array);
    return ffi::Error::Internal(
        absl::StrCat("Failed to get workspace size: ", status));
  }
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call second stage interface to execute computation
  if (dims.size() == 0) {
    status = aclnnProd(
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
      //aclDestroyAclOpExecutor(executor);
      return ffi::Error::Internal(
          absl::StrCat("aclnnProd failed: ", status));
    }
  } else {
    status = aclnnProdDim(
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
      //aclDestroyAclOpExecutor(executor);
      return ffi::Error::Internal(
          absl::StrCat("aclnnProdDim failed: ", status));
    }
  }
  aclrtSynchronizeStream(stream);

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  aclDestroyIntArray(dims_array);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }
  //aclDestroyAclOpExecutor(executor);

  return ffi::Error::Success();
}

// Explicit instantiations for supported data types
template ffi::Error ReduceProdHandlerImpl<ffi::DataType::F32>(aclrtStream stream, ffi::Buffer<ffi::DataType::F32> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::F32> out);
template ffi::Error ReduceProdHandlerImpl<ffi::DataType::F16>(aclrtStream stream, ffi::Buffer<ffi::DataType::F16> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::F16> out);
template ffi::Error ReduceProdHandlerImpl<ffi::DataType::BF16>(aclrtStream stream, ffi::Buffer<ffi::DataType::BF16> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::BF16> out);
template ffi::Error ReduceProdHandlerImpl<ffi::DataType::S32>(aclrtStream stream, ffi::Buffer<ffi::DataType::S32> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::S32> out);
template ffi::Error ReduceProdHandlerImpl<ffi::DataType::S64>(aclrtStream stream, ffi::Buffer<ffi::DataType::S64> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::DataType::S64> out);

// F32 specialization
ffi::Error ReduceProdHandlerF32(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::F32> out) {
  return ReduceProdHandlerImpl<ffi::DataType::F32>(stream, self, dims, keep_dims, out);
}

// F16 specialization
ffi::Error ReduceProdHandlerF16(aclrtStream stream, ffi::Buffer<ffi::F16> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::F16> out) {
  return ReduceProdHandlerImpl<ffi::DataType::F16>(stream, self, dims, keep_dims, out);
}

// BF16 specialization
ffi::Error ReduceProdHandlerBF16(aclrtStream stream, ffi::Buffer<ffi::BF16> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::BF16> out) {
  return ReduceProdHandlerImpl<ffi::DataType::BF16>(stream, self, dims, keep_dims, out);
}

// S32 specialization
ffi::Error ReduceProdHandlerS32(aclrtStream stream, ffi::Buffer<ffi::S32> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::S32> out) {
  return ReduceProdHandlerImpl<ffi::DataType::S32>(stream, self, dims, keep_dims, out);
}

// S64 specialization
ffi::Error ReduceProdHandlerS64(aclrtStream stream, ffi::Buffer<ffi::S64> self, ffi::Span<const int64_t> dims, bool keep_dims, ffi::ResultBuffer<ffi::S64> out) {
  return ReduceProdHandlerImpl<ffi::DataType::S64>(stream, self, dims, keep_dims, out);
}

// Register ReduceProd operator FFI functions for different data types
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceProd,
    ReduceProdHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceProdF32,
    ReduceProdHandlerF32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceProdF16,
    ReduceProdHandlerF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::F16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceProdBf16,
    ReduceProdHandlerBF16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::BF16>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceProdS32,
    ReduceProdHandlerS32,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendReduceProdS64,
    ReduceProdHandlerS64,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S64>>()
        .Attr<ffi::Span<const int64_t>>("dims")
        .Attr<bool>("keep_dims")
        .Ret<ffi::Buffer<ffi::S64>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
