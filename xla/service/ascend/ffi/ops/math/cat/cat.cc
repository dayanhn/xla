#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_cat.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include <iostream>

namespace ffi = xla::ffi;

namespace xla::ffi {

// Cat operator FFI handler for two tensors
ffi::Error CatHandler(
    aclrtStream stream, 
    ffi::Buffer<ffi::U32> input1, 
    ffi::Buffer<ffi::U32> input2, 
    int64_t dim, 
    ffi::ResultBuffer<ffi::U32> output) {
  
  std::cerr << "[Cat] START - Input1 dimensions count: " << input1.dimensions().size() << ", Input2 dimensions count: " << input2.dimensions().size() << ", Output dimensions count: " << output->dimensions().size() << std::endl;
  
  std::cerr << "[Cat] Input1 shape: [";
  for (size_t i = 0; i < input1.dimensions().size(); ++i) {
    std::cerr << input1.dimensions()[i];
    if (i < input1.dimensions().size() - 1) std::cerr << ", ";
  }
  std::cerr << "]";
  
  std::cerr << ", Input2 shape: [";
  for (size_t i = 0; i < input2.dimensions().size(); ++i) {
    std::cerr << input2.dimensions()[i];
    if (i < input2.dimensions().size() - 1) std::cerr << ", ";
  }
  std::cerr << "]";
  
  std::cerr << ", Output shape: [";
  for (size_t i = 0; i < output->dimensions().size(); ++i) {
    std::cerr << output->dimensions()[i];
    if (i < output->dimensions().size() - 1) std::cerr << ", ";
  }
  std::cerr << "], dim=" << dim << std::endl;
  
  // Special handling for 1D UINT32 tensors with dim=0 on Atlas A2/A3 (UINT32 not supported by aclnnCat)
  bool is_1d_uint32_concat = (input1.dimensions().size() == 1 && 
                               input2.dimensions().size() == 1 && 
                               output->dimensions().size() == 1 &&
                               dim == 0);
  
  if (is_1d_uint32_concat) {
    std::cerr << "[Cat] Using memcpy fallback for 1D UINT32 concat" << std::endl;
    
    // Calculate sizes
    size_t input1_size = input1.dimensions()[0] * sizeof(uint32_t);
    size_t input2_size = input2.dimensions()[0] * sizeof(uint32_t);
    
    std::cerr << "[Cat] Copying input1 (" << input1_size << " bytes) to output" << std::endl;
    aclError status1 = aclrtMemcpyAsync(
        const_cast<void*>(output->untyped_data()),
        input1_size,
        input1.untyped_data(),
        input1_size,
        ACL_MEMCPY_DEVICE_TO_DEVICE,
        stream);
    
    if (status1 != ACL_SUCCESS) {
      std::cerr << "[Cat] ERROR: aclrtMemcpyAsync for input1 failed with status: " << status1 << std::endl;
      return ffi::Error::Internal(absl::StrCat("aclrtMemcpyAsync for input1 failed: ", status1));
    }
    
    std::cerr << "[Cat] Copying input2 (" << input2_size << " bytes) to output + offset" << std::endl;
    void* output_offset = reinterpret_cast<uint8_t*>(const_cast<void*>(output->untyped_data())) + input1_size;
    aclError status2 = aclrtMemcpyAsync(
        output_offset,
        input2_size,
        input2.untyped_data(),
        input2_size,
        ACL_MEMCPY_DEVICE_TO_DEVICE,
        stream);
    
    if (status2 != ACL_SUCCESS) {
      std::cerr << "[Cat] ERROR: aclrtMemcpyAsync for input2 failed with status: " << status2 << std::endl;
      return ffi::Error::Internal(absl::StrCat("aclrtMemcpyAsync for input2 failed: ", status2));
    }
    
    // Synchronize stream
    std::cerr << "[Cat] Synchronizing stream..." << std::endl;
    aclError sync_status = aclrtSynchronizeStream(stream);
    if (sync_status != ACL_SUCCESS) {
      std::cerr << "[Cat] ERROR: aclrtSynchronizeStream failed with status: " << sync_status << std::endl;
      return ffi::Error::Internal(absl::StrCat("aclrtSynchronizeStream failed: ", sync_status));
    }
    
    std::cerr << "[Cat] FINISHED - Success (memcpy fallback)" << std::endl;
    return ffi::Error::Success();
  }
  
  // Original path using aclnnCat for other cases
  std::cerr << "[Cat] Using aclnnCat path" << std::endl;

  // Convert XLA Buffer to Ascend Tensor
  aclTensor* input1_tensor = ConvertToAclTensor(input1);
  aclTensor* input2_tensor = ConvertToAclTensor(input2);
  aclTensor* output_tensor = ConvertToAclTensor(*output);
  
  std::cerr << "[Cat] Tensors created - input1_tensor: " << input1_tensor 
            << ", input2_tensor: " << input2_tensor 
            << ", output_tensor: " << output_tensor << std::endl;
  
  // Debug: Print tensor addresses
  if (input1_tensor) {
    void* addr1 = nullptr;
    aclGetRawTensorAddr(input1_tensor, &addr1);
    std::cerr << "[Cat] input1_tensor addr: " << addr1 << std::endl;
  }
  if (input2_tensor) {
    void* addr2 = nullptr;
    aclGetRawTensorAddr(input2_tensor, &addr2);
    std::cerr << "[Cat] input2_tensor addr: " << addr2 << std::endl;
  }
  if (output_tensor) {
    void* addr_out = nullptr;
    aclGetRawTensorAddr(output_tensor, &addr_out);
    std::cerr << "[Cat] output_tensor addr: " << addr_out << std::endl;
  }
  
  // Create tensor list
  std::vector<aclTensor*> input_tensors{input1_tensor, input2_tensor};
  std::cerr << "[Cat] Creating tensor list with " << input_tensors.size() << " tensors..." << std::endl;
  aclTensorList* input_tensor_list = aclCreateTensorList(input_tensors.data(), input_tensors.size());
  std::cerr << "[Cat] Tensor list created: " << input_tensor_list << std::endl;
  
  if (input_tensor_list == nullptr) {
    std::cerr << "[Cat] ERROR: Failed to create aclTensorList" << std::endl;
    aclDestroyTensor(input1_tensor);
    aclDestroyTensor(input2_tensor);
    aclDestroyTensor(output_tensor);
    return ffi::Error::Internal("Failed to create aclTensorList");
  }
  
  std::cerr << "[Cat] Calling aclnnCatGetWorkspaceSize with dim=" << dim << "..." << std::endl;

  // Get workspace size
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclnnStatus status = aclnnCatGetWorkspaceSize(
      input_tensor_list, dim, output_tensor, &workspace_size, &executor);
  
  std::cerr << "[Cat] aclnnCatGetWorkspaceSize returned - status: " << status 
            << ", workspace_size: " << workspace_size 
            << ", executor: " << executor << std::endl;
  
  if (status != ACL_SUCCESS) {
    std::cerr << "[Cat] ERROR: aclnnCatGetWorkspaceSize failed with status: " << status << std::endl;
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
    std::cerr << "[Cat] Allocating workspace memory, size: " << workspace_size << std::endl;
    aclError alloc_status = aclrtMalloc(&workspaceAddr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    std::cerr << "[Cat] aclrtMalloc for workspace returned - status: " << alloc_status 
              << ", addr: " << workspaceAddr << std::endl;
    if (alloc_status != ACL_SUCCESS) {
      std::cerr << "[Cat] ERROR: aclrtMalloc for workspace failed with status: " << alloc_status << std::endl;
      aclDestroyTensorList(input_tensor_list);
      aclDestroyTensor(input1_tensor);
      aclDestroyTensor(input2_tensor);
      aclDestroyTensor(output_tensor);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed: ", alloc_status));
    }
  } else {
    std::cerr << "[Cat] No workspace memory needed" << std::endl;
  }

  // Call aclnnCat
  std::cerr << "[Cat] Calling aclnnCat..." << std::endl;
  status = aclnnCat(
      workspaceAddr,
      workspace_size,
      executor,
      stream);
  
  std::cerr << "[Cat] aclnnCat returned - status: " << status << std::endl;
  if (status != ACL_SUCCESS) {
    std::cerr << "[Cat] ERROR: aclnnCat failed with status: " << status << std::endl;
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
  std::cerr << "[Cat] Calling aclrtSynchronizeStream..." << std::endl;
  status = aclrtSynchronizeStream(stream);
  std::cerr << "[Cat] aclrtSynchronizeStream returned - status: " << status << std::endl;
  if(status != ACL_SUCCESS){
    std::cerr << "[Cat] ERROR: aclrtSynchronizeStream failed with status: " << status << std::endl;
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
  std::cerr << "[Cat] Releasing resources..." << std::endl;
  aclDestroyTensorList(input_tensor_list);
  aclDestroyTensor(input1_tensor);
  aclDestroyTensor(input2_tensor);
  aclDestroyTensor(output_tensor);
  if (workspace_size > 0) {
    aclrtFree(workspaceAddr);
  }

  std::cerr << "[Cat] FINISHED - Success" << std::endl;
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