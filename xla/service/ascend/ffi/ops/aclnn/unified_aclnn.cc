#include "xla/ffi/api/ffi.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnn/acl_meta.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include <dlfcn.h>
#include <vector>

namespace ffi = xla::ffi;

namespace {

void* GetAclnnFuncAddr(const char* apiName) {
  static auto custOpApiHandler = dlopen("libcust_opapi.so", RTLD_LAZY);
  if (custOpApiHandler != nullptr) {
    auto funcAddr = dlsym(custOpApiHandler, apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
  }

  static auto opApiHandler = dlopen("libopapi.so", RTLD_LAZY);
  if (opApiHandler == nullptr) {
    return nullptr;
  }
  return dlsym(opApiHandler, apiName);
}

typedef int (*GenericAclnnFunc)(void*, void*, void*, void*, void*,
                                void*, void*, void*, void*, void*,
                                void*, void*, void*, void*, void*,
                                void*, void*, void*, void*, void*,
                                void*, void*, void*, void*, void*,
                                void*, void*, void*, void*, void*);

typedef int (*ExecuteFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);

constexpr int kMaxParamCount = 30;

void CleanupResources(
    std::vector<aclTensor*>& all_acl_tensors,
    std::vector<aclScalar*>& acl_scalars,
    std::vector<aclIntArray*>& acl_int_arrays,
    std::vector<aclFloatArray*>& acl_float_arrays,
    std::vector<aclBoolArray*>& acl_bool_arrays) {
  for (auto* t : all_acl_tensors) {
    if (t) aclDestroyTensor(t);
  }
  for (auto* s : acl_scalars) {
    if (s) aclDestroyScalar(s);
  }
  for (auto* a : acl_int_arrays) {
    if (a) aclDestroyIntArray(a);
  }
  for (auto* a : acl_float_arrays) {
    if (a) aclDestroyFloatArray(a);
  }
  for (auto* a : acl_bool_arrays) {
    if (a) aclDestroyBoolArray(a);
  }
}

}  // namespace

namespace xla::ffi {

ffi::Error UnifiedAclnnHandler(
    aclrtStream acl_stream,
    ffi::RemainingArgs args,
    ffi::Dictionary attrs,
    ffi::RemainingRets rets) {
  std::vector<aclTensor*> all_acl_tensors;
  std::vector<aclScalar*> acl_scalars;
  std::vector<aclIntArray*> acl_int_arrays;
  std::vector<aclFloatArray*> acl_float_arrays;
  std::vector<aclBoolArray*> acl_bool_arrays;

  auto op_name_result = attrs.get<std::string_view>("op_name");
  if (!op_name_result.has_value()) {
    return ffi::Error::InvalidArgument("Missing 'op_name' attribute");
  }
  std::string op_name(*op_name_result);

  auto num_inputs_result = attrs.get<int64_t>("num_inputs");
  auto num_outputs_result = attrs.get<int64_t>("num_outputs");
  if (!num_inputs_result.has_value() || !num_outputs_result.has_value()) {
    return ffi::Error::InvalidArgument("Missing num_inputs or num_outputs");
  }
  int num_inputs = *num_inputs_result;
  int num_outputs = *num_outputs_result;

  std::vector<aclTensor*> input_tensors;
  for (int i = 0; i < num_inputs; i++) {
    auto buf_result = args.get<ffi::AnyBuffer>(i);
    if (!buf_result.has_value()) {
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(
          absl::StrCat("Failed to get input tensor at index ", i));
    }
    aclTensor* tensor = ConvertAnyBufferToAclTensor(*buf_result);
    if (!tensor) {
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::Internal(
          absl::StrCat("Failed to convert input tensor ", i, " to aclTensor"));
    }
    input_tensors.push_back(tensor);
    all_acl_tensors.push_back(tensor);
  }

  std::vector<aclTensor*> output_tensors;
  for (int i = 0; i < num_outputs; i++) {
    auto ret_result = rets.get<ffi::AnyBuffer>(i);
    if (!ret_result.has_value()) {
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(
          absl::StrCat("Failed to get output tensor at index ", i));
    }
    aclTensor* tensor = ConvertAnyBufferToAclTensor(**ret_result);
    if (!tensor) {
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::Internal(
          absl::StrCat("Failed to convert output tensor ", i, " to aclTensor"));
    }
    output_tensors.push_back(tensor);
    all_acl_tensors.push_back(tensor);
  }

  auto param_count_result = attrs.get<int64_t>("param_count");
  int param_count = param_count_result.has_value() ? *param_count_result : 0;

  std::vector<int64_t> int64_values;
  std::vector<float> float_values;
  std::vector<uint8_t> bool_values;
  std::vector<std::vector<int64_t>> int_array_storage;
  std::vector<std::vector<float>> float_array_storage;
  std::vector<std::vector<uint8_t>> bool_array_storage;

  std::vector<void*> param_ptrs;

  for (auto* t : input_tensors) {
    param_ptrs.push_back(t);
  }

  for (int i = 0; i < param_count; i++) {
    std::string type_key = "param_" + std::to_string(i) + "_type";
    std::string value_key = "param_" + std::to_string(i);

    auto type_result = attrs.get<std::string_view>(type_key);
    if (!type_result.has_value()) {
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(
          absl::StrCat("Missing param type: ", type_key));
    }
    std::string param_type(*type_result);

    if (param_type == "int64") {
      auto val = attrs.get<int64_t>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      int64_values.push_back(*val);
      param_ptrs.push_back(&int64_values.back());
    } else if (param_type == "float") {
      auto val = attrs.get<float>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      float_values.push_back(*val);
      param_ptrs.push_back(&float_values.back());
    } else if (param_type == "bool") {
      auto val = attrs.get<bool>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      bool_values.push_back(*val ? 1 : 0);
      param_ptrs.push_back(&bool_values.back());
    } else if (param_type == "int_array") {
      auto val = attrs.get<Span<const int64_t>>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      int_array_storage.emplace_back(val->begin(), val->end());
      aclIntArray* arr = aclCreateIntArray(
          int_array_storage.back().data(), int_array_storage.back().size());
      if (!arr) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal("Failed to create aclIntArray");
      }
      acl_int_arrays.push_back(arr);
      param_ptrs.push_back(arr);
    } else if (param_type == "float_array") {
      auto val = attrs.get<Span<const float>>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      float_array_storage.emplace_back(val->begin(), val->end());
      aclFloatArray* arr = aclCreateFloatArray(
          float_array_storage.back().data(), float_array_storage.back().size());
      if (!arr) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal("Failed to create aclFloatArray");
      }
      acl_float_arrays.push_back(arr);
      param_ptrs.push_back(arr);
    } else if (param_type == "bool_array") {
      auto val = attrs.get<Span<const uint8_t>>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      bool_array_storage.emplace_back(val->begin(), val->end());
      aclBoolArray* arr = aclCreateBoolArray(
          reinterpret_cast<const bool*>(bool_array_storage.back().data()),
          bool_array_storage.back().size());
      if (!arr) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal("Failed to create aclBoolArray");
      }
      acl_bool_arrays.push_back(arr);
      param_ptrs.push_back(arr);
    } else if (param_type == "scalar_int") {
      auto val = attrs.get<int64_t>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      int64_t v = *val;
      aclScalar* scalar = aclCreateScalar(
          const_cast<void*>(reinterpret_cast<const void*>(&v)), ACL_INT64);
      if (!scalar) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal("Failed to create aclScalar for int");
      }
      acl_scalars.push_back(scalar);
      param_ptrs.push_back(scalar);
    } else if (param_type == "scalar_float") {
      auto val = attrs.get<float>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      float v = *val;
      aclScalar* scalar = aclCreateScalar(
          const_cast<void*>(reinterpret_cast<const void*>(&v)), ACL_FLOAT);
      if (!scalar) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal("Failed to create aclScalar for float");
      }
      acl_scalars.push_back(scalar);
      param_ptrs.push_back(scalar);
    } else {
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(
          absl::StrCat("Unknown param type: ", param_type));
    }
  }

  for (auto* t : output_tensors) {
    param_ptrs.push_back(t);
  }

  std::string get_workspace_name = op_name + "GetWorkspaceSize";
  void* get_workspace_func = GetAclnnFuncAddr(get_workspace_name.c_str());
  void* execute_func = GetAclnnFuncAddr(op_name.c_str());

  if (!get_workspace_func || !execute_func) {
    CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                     acl_float_arrays, acl_bool_arrays);
    return ffi::Error::Internal(
        absl::StrCat("ACLNN function not found: ", op_name));
  }

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  param_ptrs.push_back(&workspace_size);
  param_ptrs.push_back(&executor);

  param_ptrs.resize(kMaxParamCount, nullptr);

  auto get_ws_func = reinterpret_cast<GenericAclnnFunc>(get_workspace_func);

  int ws_status = get_ws_func(
      param_ptrs[0], param_ptrs[1], param_ptrs[2], param_ptrs[3],
      param_ptrs[4], param_ptrs[5], param_ptrs[6], param_ptrs[7],
      param_ptrs[8], param_ptrs[9], param_ptrs[10], param_ptrs[11],
      param_ptrs[12], param_ptrs[13], param_ptrs[14], param_ptrs[15],
      param_ptrs[16], param_ptrs[17], param_ptrs[18], param_ptrs[19],
      param_ptrs[20], param_ptrs[21], param_ptrs[22], param_ptrs[23],
      param_ptrs[24], param_ptrs[25], param_ptrs[26], param_ptrs[27],
      param_ptrs[28], param_ptrs[29]);

  if (ws_status != ACL_SUCCESS) {
    const char* err_msg = aclGetRecentErrMsg();
    std::string error_detail = absl::StrCat(
        "GetWorkspaceSize failed for ", op_name,
        " with code ", ws_status);
    if (err_msg && err_msg[0] != '\0') {
      absl::StrAppend(&error_detail, ", error: ", err_msg);
    }
    LOG(ERROR) << "[ACLNN ERROR] " << error_detail;
    CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                     acl_float_arrays, acl_bool_arrays);
    return ffi::Error::Internal(error_detail);
  }

  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    aclError malloc_status = aclrtMalloc(
        &workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (malloc_status != ACL_SUCCESS) {
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::Internal(
          absl::StrCat("aclrtMalloc failed for workspace: ", malloc_status));
    }
  }

  auto exec_func = reinterpret_cast<ExecuteFunc>(execute_func);
  int exec_status = exec_func(workspace_addr, workspace_size, executor,
                              acl_stream);

  if (exec_status != ACL_SUCCESS) {
    const char* err_msg = aclGetRecentErrMsg();
    std::string error_detail = absl::StrCat(
        "Execute failed for ", op_name,
        " with code ", exec_status);
    if (err_msg && err_msg[0] != '\0') {
      absl::StrAppend(&error_detail, ", error: ", err_msg);
    }
    LOG(ERROR) << "[ACLNN ERROR] " << error_detail;
    if (workspace_size > 0) {
      aclrtFree(workspace_addr);
    }
    CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                     acl_float_arrays, acl_bool_arrays);
    return ffi::Error::Internal(error_detail);
  }

  if (workspace_size > 0) {
    aclrtFree(workspace_addr);
  }
  CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                   acl_float_arrays, acl_bool_arrays);

  return ffi::Error::Success();
}


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AscendUnifiedOp, UnifiedAclnnHandler,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<aclrtStream>>()
        .RemainingArgs()
        .Attrs<ffi::Dictionary>()
        .RemainingRets(),
    {ffi::Traits::kCmdBufferCompatible});

}  // namespace xla::ffi
