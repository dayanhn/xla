#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_cat.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include <vector>

namespace ffi = xla::ffi;

namespace xla::ffi {

// Cat operator FFI handler
ffi::Error CatHandler(
    aclrtStream stream, 
    ffi::Buffer<ffi::U32> input1, 
    ffi::Buffer<ffi::U32> input2, 
    int64_t dim, 
    ffi::ResultBuffer<ffi::U32> output) {
  
  // Special handling for 1D UINT32 tensors with dim=0 on Atlas A2/A3 (UINT32 not supported by aclnnCat)
  bool is_1d_uint32_concat = (input1.dimensions().size() == 1 && 
                               input2.dimensions().size() == 1 && 
                               output->dimensions().size() == 1 &&
                               dim == 0);
  
  if (is_1d_uint32_concat) {
    // Calculate sizes
    size_t input1_size = input1.dimensions()[0] * sizeof(uint32_t);
    size_t input2_size = input2.dimensions()[0] * sizeof(uint32_t);
    
    // Copy input1 to output
    aclError status1 = aclrtMemcpyAsync(
        const_cast<void*>(output->untyped_data()),
        input1_size,
        input1.untyped_data(),
        input1_size,
        ACL_MEMCPY_DEVICE_TO_DEVICE,
        stream);
    
    if (status1 != ACL_SUCCESS) {
      return ffi::Error::Internal(absl::StrCat("aclrtMemcpyAsync for input1 failed: ", status1));
    }
    
    // Copy input2 to output + offset
    void* output_offset = reinterpret_cast<uint8_t*>(const_cast<void*>(output->untyped_data())) + input1_size;
    aclError status2 = aclrtMemcpyAsync(
        output_offset,
        input2_size,
        input2.untyped_data(),
        input2_size,
        ACL_MEMCPY_DEVICE_TO_DEVICE,
        stream);
    
    if (status2 != ACL_SUCCESS) {
      return ffi::Error::Internal(absl::StrCat("aclrtMemcpyAsync for input2 failed: ", status2));
    }
    
    // Synchronize stream
    aclError sync_status = aclrtSynchronizeStream(stream);
    if (sync_status != ACL_SUCCESS) {
      return ffi::Error::Internal(absl::StrCat("aclrtSynchronizeStream failed: ", sync_status));
    }
    
    return ffi::Error::Success();
  }
  
  // Original path using aclnnCat for other cases
  // Convert XLA Buffer to Ascend Tensor
  aclTensor* input1_tensor = ConvertToAclTensor(input1);
  aclTensor* input2_tensor = ConvertToAclTensor(input2);
  aclTensor* output_tensor = ConvertToAclTensor(*output);
  
  // Create tensor list
  std::vector<aclTensor*> input_tensors{input1_tensor, input2_tensor};
  aclTensorList* input_tensor_list = aclCreateTensorList(input_tensors.data(), input_tensors.size());
  
  if (input_tensor_list == nullptr) {
    aclDestroyTensor(input1_tensor);
    aclDestroyTensor(input2_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal("Failed to create aclTensorList");
  }

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
    aclError alloc_status = aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (alloc_status != ACL_SUCCESS) {
      aclDestroyTensorList(input_tensor_list);
      aclDestroyTensor(input1_tensor);
      aclDestroyTensor(input2_tensor);
      aclDestroyTensor(output_tensor);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed: ", alloc_status));
    }
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
  aclDestroyAclOpExecutor(executor);
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
        .Ret<ffi::Buffer<ffi::U32>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi