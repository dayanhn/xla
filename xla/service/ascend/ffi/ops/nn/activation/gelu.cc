#include "xla/ffi/api/ffi.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_gelu.h"
#include "absl/strings/str_cat.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// GELU operator FFI handler
ffi::Error GeluHandler(aclrtStream stream, ffi::Buffer<ffi::F32> self, ffi::ResultBuffer<ffi::F32> out) {
  // Convert XLA Buffer to Ascend Tensor using utility function
  aclTensor* self_tensor = ConvertToAclTensor(self);
  aclTensor* out_tensor = ConvertToAclTensor(*out);

  // Call first stage interface to get workspace size and executor
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnGeluGetWorkspaceSize(
      self_tensor, out_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnGeluGetWorkspaceSize failed: ", status));
  }

  // Call second stage interface to execute computation
  status = aclnnGelu(
      nullptr,  // workspace is managed by XLA
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(self_tensor);
    aclDestroyTensor(out_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnGelu failed: ", status));
  }

  // Release resources
  aclDestroyTensor(self_tensor);
  aclDestroyTensor(out_tensor);
  aclOpExecutorDestroy(executor); 

  return ffi::Error::Success();
}

// Register GELU operator FFI function
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendGelu,
    GeluHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi