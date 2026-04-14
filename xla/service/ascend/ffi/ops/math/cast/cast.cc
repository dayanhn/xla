#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_cast.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Cast operator FFI handler for s32 to u32
ffi::Error CastS32ToU32Handler(
    aclrtStream stream, 
    ffi::Buffer<ffi::S32> input, 
    ffi::ResultBuffer<ffi::U32> output) {

  // Convert XLA Buffer to Ascend Tensor
  aclTensor* input_tensor = ConvertToAclTensor(input);
  aclTensor* output_tensor = ConvertToAclTensor(*output);

  // Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  aclnnStatus status = aclnnCastGetWorkspaceSize(
      input_tensor, ACL_UINT32, output_tensor, &workspace_size, &executor);
  
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnCastGetWorkspaceSize failed: ", status));
  }
  
  // Allocate workspace memory
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclError alloc_status = aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (alloc_status != ACL_SUCCESS) {
      aclDestroyTensor(input_tensor);
      aclDestroyTensor(output_tensor);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed: ", alloc_status));
    }
  }

  // Call aclnnCast
  status = aclnnCast(
      workspaceAddr,
      workspace_size,
      executor,
      stream);
  
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnCast failed: ", status));
  }

  // Synchronize stream
  status = aclrtSynchronizeStream(stream);
  if(status != ACL_SUCCESS){
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyTensor(input_tensor);
  aclDestroyTensor(output_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  return ffi::Error::Success();
}

// Register Cast operator FFI functions

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCastS32ToU32,
    CastS32ToU32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>(),
    {ffi::Traits::kCmdBufferCompatible});

// Generic Cast handler (default to S32 to U32)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCast,
    CastS32ToU32Handler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::U32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
