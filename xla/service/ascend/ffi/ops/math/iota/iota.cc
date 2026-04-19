#include "xla/ffi/api/ffi.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnnop/aclnn_range.h"
#include "third_party/acl/inc/aclnnop/aclnn_repeat.h"
#include "third_party/acl/inc/aclnnop/aclnn_cast.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status.h"
#include <cstdint>

namespace ffi = xla::ffi;

namespace xla::ffi {

ffi::Error IotaHandlerU8(aclrtStream stream, int64_t iota_dimension, int64_t num_classes, int64_t num_rows, ffi::ResultBuffer<ffi::U8> out) {
  if (iota_dimension != 1) {
    return ffi::Error::Internal(
        absl::StrCat("Iota only supports iota_dimension=1, got: ", iota_dimension));
  }

  std::vector<int64_t> arange_shape = {num_classes};
  std::vector<int64_t> arange_strides(1, 1);

  void* arange_buffer_addr = nullptr;
  aclDataType arange_dtype = ACL_INT32;
  int64_t element_size = 4;
  int64_t arange_size = num_classes * element_size;
  auto ret = aclrtMalloc(&arange_buffer_addr, arange_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    return ffi::Error::Internal(
        absl::StrCat("Failed to allocate arange buffer: ", ret));
  }

  aclTensor* arange_tensor = aclCreateTensor(
      arange_shape.data(), arange_shape.size(), arange_dtype, arange_strides.data(), 0,
      aclFormat::ACL_FORMAT_ND, arange_shape.data(), arange_shape.size(), arange_buffer_addr);

  int32_t start_value = 0;
  int32_t end_value = num_classes;
  int32_t step_value = 1;

  aclScalar* start = aclCreateScalar(&start_value, ACL_INT32);
  aclScalar* end = aclCreateScalar(&end_value, ACL_INT32);
  aclScalar* step = aclCreateScalar(&step_value, ACL_INT32);

  uint64_t workspace_size_arange = 0;
  aclOpExecutor* executor_arange = nullptr;
  aclnnStatus status = aclnnRangeGetWorkspaceSize(
      start, end, step, arange_tensor, &workspace_size_arange, &executor_arange);
  if (status != ACL_SUCCESS) {
    aclDestroyScalar(start);
    aclDestroyScalar(end);
    aclDestroyScalar(step);
    aclDestroyTensor(arange_tensor);
    aclrtFree(arange_buffer_addr);
    return ffi::Error::Internal(
        absl::StrCat("aclnnRangeGetWorkspaceSize failed: ", status));
  }

  void* workspaceAddr_arange = nullptr;
  if (workspace_size_arange > 0) {
    ret = aclrtMalloc(&workspaceAddr_arange, workspace_size_arange, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclDestroyScalar(start);
      aclDestroyScalar(end);
      aclDestroyScalar(step);
      aclDestroyTensor(arange_tensor);
      aclrtFree(arange_buffer_addr);
      return ffi::Error::Internal("Failed to allocate workspace for arange");
    }
  }

  status = aclnnRange(workspaceAddr_arange, workspace_size_arange, executor_arange, stream);
  if (status != ACL_SUCCESS) {
    aclDestroyScalar(start);
    aclDestroyScalar(end);
    aclDestroyScalar(step);
    aclDestroyTensor(arange_tensor);
    if (workspace_size_arange > 0) {
      aclrtFree(workspaceAddr_arange);
    }
    aclrtFree(arange_buffer_addr);
    return ffi::Error::Internal(
        absl::StrCat("aclnnRange failed: ", status));
  }

  aclDestroyScalar(start);
  aclDestroyScalar(end);
  aclDestroyScalar(step);
  if (workspace_size_arange > 0) {
    aclrtFree(workspaceAddr_arange);
  }
  //aclDestroyAclOpExecutor(executor_arange);

  std::vector<int64_t> repeat_out_shape = {num_rows, num_classes};
  std::vector<int64_t> repeat_out_strides(2, 1);
  repeat_out_strides[0] = num_classes;

  void* repeat_buffer_addr = nullptr;
  int64_t repeat_size = num_rows * num_classes * element_size;
  ret = aclrtMalloc(&repeat_buffer_addr, repeat_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    aclDestroyTensor(arange_tensor);
    aclrtFree(arange_buffer_addr);
    return ffi::Error::Internal(
        absl::StrCat("Failed to allocate repeat buffer: ", ret));
  }

  aclTensor* repeat_out_tensor = aclCreateTensor(
      repeat_out_shape.data(), repeat_out_shape.size(), arange_dtype, repeat_out_strides.data(), 0,
      aclFormat::ACL_FORMAT_ND, repeat_out_shape.data(), repeat_out_shape.size(), repeat_buffer_addr);

  std::vector<int64_t> repeats_vec = {num_rows, 1};
  aclIntArray* repeats = aclCreateIntArray(repeats_vec.data(), repeats_vec.size());

  uint64_t workspace_size_repeat = 0;
  aclOpExecutor* executor_repeat = nullptr;
  status = aclnnRepeatGetWorkspaceSize(
      arange_tensor, repeats, repeat_out_tensor, &workspace_size_repeat, &executor_repeat);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(arange_tensor);
    aclDestroyTensor(repeat_out_tensor);
    aclDestroyIntArray(repeats);
    aclrtFree(arange_buffer_addr);
    aclrtFree(repeat_buffer_addr);
    return ffi::Error::Internal(
        absl::StrCat("aclnnRepeatGetWorkspaceSize failed: ", status));
  }

  void* workspaceAddr_repeat = nullptr;
  if (workspace_size_repeat > 0) {
    ret = aclrtMalloc(&workspaceAddr_repeat, workspace_size_repeat, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclDestroyTensor(arange_tensor);
      aclDestroyTensor(repeat_out_tensor);
      aclDestroyIntArray(repeats);
      aclrtFree(arange_buffer_addr);
      aclrtFree(repeat_buffer_addr);
      return ffi::Error::Internal("Failed to allocate workspace for repeat");
    }
  }

  status = aclnnRepeat(workspaceAddr_repeat, workspace_size_repeat, executor_repeat, stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(arange_tensor);
    aclDestroyTensor(repeat_out_tensor);
    aclDestroyIntArray(repeats);
    if (workspace_size_repeat > 0) {
      aclrtFree(workspaceAddr_repeat);
    }
    aclrtFree(arange_buffer_addr);
    aclrtFree(repeat_buffer_addr);
    return ffi::Error::Internal(
        absl::StrCat("aclnnRepeat failed: ", status));
  }

  if (workspace_size_repeat > 0) {
    aclrtFree(workspaceAddr_repeat);
  }
  //aclDestroyAclOpExecutor(executor_repeat);
  aclDestroyTensor(arange_tensor);
  aclDestroyIntArray(repeats);
  aclrtFree(arange_buffer_addr);

  aclTensor* final_out_tensor = ConvertToAclTensor(*out);

  uint64_t workspace_size_cast = 0;
  aclOpExecutor* executor_cast = nullptr;
  status = aclnnCastGetWorkspaceSize(
      repeat_out_tensor, ACL_UINT8, final_out_tensor, &workspace_size_cast, &executor_cast);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(repeat_out_tensor);
    aclDestroyTensor(final_out_tensor);
    aclrtFree(repeat_buffer_addr);
    return ffi::Error::Internal(
        absl::StrCat("aclnnCastGetWorkspaceSize failed: ", status));
  }

  void* workspaceAddr_cast = nullptr;
  if (workspace_size_cast > 0) {
    ret = aclrtMalloc(&workspaceAddr_cast, workspace_size_cast, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      aclDestroyTensor(repeat_out_tensor);
      aclDestroyTensor(final_out_tensor);
      aclrtFree(repeat_buffer_addr);
      return ffi::Error::Internal("Failed to allocate workspace for cast");
    }
  }

  status = aclnnCast(workspaceAddr_cast, workspace_size_cast, executor_cast, stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(repeat_out_tensor);
    aclDestroyTensor(final_out_tensor);
    if (workspace_size_cast > 0) {
      aclrtFree(workspaceAddr_cast);
    }
    aclrtFree(repeat_buffer_addr);
    return ffi::Error::Internal(
        absl::StrCat("aclnnCast failed: ", status));
  }

  status = aclrtSynchronizeStream(stream);
  if (status != ACL_SUCCESS) {
    aclDestroyTensor(repeat_out_tensor);
    aclDestroyTensor(final_out_tensor);
    if (workspace_size_cast > 0) {
      aclrtFree(workspaceAddr_cast);
    }
    aclrtFree(repeat_buffer_addr);
    return ffi::Error::Internal(
        absl::StrCat("aclrtSynchronizeStream failed: ", status));
  }

  aclDestroyTensor(repeat_out_tensor);
  aclDestroyTensor(final_out_tensor);
  if (workspace_size_cast > 0) {
    aclrtFree(workspaceAddr_cast);
  }
  //aclDestroyAclOpExecutor(executor_cast);
  aclrtFree(repeat_buffer_addr);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendIotaU8,
    IotaHandlerU8,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .Attr<int64_t>("iota_dimension")
        .Attr<int64_t>("num_classes")
        .Attr<int64_t>("num_rows")
        .Ret<ffi::Buffer<ffi::U8>>(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi