#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_cat.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"

namespace ffi = xla::ffi;

namespace xla::ffi {

// Cat operator FFI handler for two tensors
ffi::Error CatHandler(
    aclrtStream stream, 
    ffi::Buffer<ffi::U32> input1, 
    ffi::Buffer<ffi::U32> input2, 
    int64_t dim, 
    ffi::Buffer<ffi::U32> output) {
  
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* input1_tensor = ConvertToAclTensor(input1);
  aclTensor* input2_tensor = ConvertToAclTensor(input2);
  aclTensor* output_tensor = ConvertToAclTensor(output);
  
  // Create tensor list
  std::vector<aclTensor*> input_tensors{input1_tensor, input2_tensor};
  aclTensorList* input_tensor_list = aclCreateTensorList(input_tensors.data(), input_tensors.size());
  if (input_tensor_list == nullptr) {
    aclDestroyTensor(input1_tensor);
    aclDestroyTensor(input2_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal("Failed to create aclTensorList");
  }
  
  LOG(INFO) << "Cat operation: dim=" << dim;

  // Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnCatGetWorkspaceSize(
      input_tensor_list, dim, output_tensor, &workspace_size, &executor);
  if (status != ACL_SUCCESS) {
    aclDestroyTensorList(input_tensor_list);
    aclDestroyTensor(input1_tensor);
    aclDestroyTensor(input2_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal(
        absl::StrCat("aclnnCatGetWorkspaceSize failed: ", status));
  }
  
  // Allocate workspace memory
  void* workspaceAddr = nullptr;
  if (workspace_size > 0) {
    aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
  }

  // Call aclnnCat
  status = aclnnCat(
      workspaceAddr,
      workspace_size,
      executor,
      stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensorList(input_tensor_list);
    aclDestroyTensor(input1_tensor);
    aclDestroyTensor(input2_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclnnCat failed: ", status));
  }

  // Synchronize stream
  status = aclrtSynchronizeStream(stream);
  if(status != ACL_SUCCESS){
    aclDestroyTensorList(input_tensor_list);
    aclDestroyTensor(input1_tensor);
    aclDestroyTensor(input2_tensor);
    aclDestroyTensor(output_tensor);
    if (workspace_size > 0) {
      aclrtFree(workspaceAddr);
    }
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  // Release resources
  aclDestroyTensorList(input_tensor_list);
  aclDestroyTensor(input1_tensor);
  aclDestroyTensor(input2_tensor);
  aclDestroyTensor(output_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  return ffi::Error::Success();
}

// Register Cat operator FFI function

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendCat,
    CatHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Attr<int64_t>("dim")
        .Arg<ffi::Buffer<ffi::U32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
