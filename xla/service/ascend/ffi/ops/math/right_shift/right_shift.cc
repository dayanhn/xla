#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_right_shift.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// RightShift operator FFI handler
ffi::Error RightShiftHandler(
    aclrtStream stream, 
    ffi::Buffer<ffi::S32> input, 
    ffi::Buffer<ffi::S32> shift_bits, 
    ffi::Buffer<ffi::S32> output) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* input_tensor = ConvertToAclTensor(input);
  aclTensor* shift_bits_tensor = ConvertToAclTensor(shift_bits);
  aclTensor* output_tensor = ConvertToAclTensor(output);
  
  LOG(INFO) << "RightShift operation";

  // Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnRightShiftGetWorkspaceSize(
      input_tensor, shift_bits_tensor, output_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(shift_bits_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnRightShiftGetWorkspaceSize failed: ", status));
  }
  
  // Allocate workspace memory
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call aclnnRightShift
  status = aclnnRightShift(
      workspaceAddr,
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(shift_bits_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnRightShift failed: ", status));
  }

  // Synchronize stream
  status = aclrtSynchronizeStream(stream);
  if(status != ACL_SUCCESS){
    aclDestroyTensor(input_tensor);
    aclDestroyTensor(shift_bits_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyTensor(input_tensor);
  aclDestroyTensor(shift_bits_tensor);
  aclDestroyTensor(output_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  return ffi::Error::Success();
}

// Register RightShift operator FFI function

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendRightShift,
    RightShiftHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
