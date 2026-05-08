#include "xla/ffi/api/ffi.h"
#include "xla/service/ascend/ffi/utils/tensor_utils.h"
#include "third_party/acl/inc/acl/acl.h"
#include "third_party/acl/inc/aclnn/acl_meta.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include <dlfcn.h>
#include "third_party/libffi/include/ffi.h"
#include <vector>

namespace ffi = xla::ffi;

namespace {

void* GetAclnnFuncAddr(const char* apiName) {
  static auto custOpApiHandler = dlopen("libopapi.so", RTLD_LAZY);
  if (custOpApiHandler != nullptr) {
    auto funcAddr = dlsym(custOpApiHandler, apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
  }

  static auto opApiHandler = dlopen("libcust_opapi.so", RTLD_LAZY);
  if (opApiHandler == nullptr) {
    return nullptr;
  }
  return dlsym(opApiHandler, apiName);
}

typedef int (*ExecuteFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);

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

struct AclnnParam {
  ffi_type* ffi_type;
  void* value_ptr;
};

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
    LOG(ERROR) << "[ACLNN ERROR] Missing 'op_name' attribute";
    return ffi::Error::InvalidArgument("Missing 'op_name' attribute");
  }
  std::string op_name(*op_name_result);

  auto num_inputs_result = attrs.get<int64_t>("num_inputs");
  auto num_outputs_result = attrs.get<int64_t>("num_outputs");
  if (!num_inputs_result.has_value() || !num_outputs_result.has_value()) {
    LOG(ERROR) << "[ACLNN ERROR] Missing num_inputs or num_outputs for op: " << op_name;
    return ffi::Error::InvalidArgument("Missing num_inputs or num_outputs");
  }
  int num_inputs = *num_inputs_result;
  int num_outputs = *num_outputs_result;

  std::vector<AclnnParam> params;
  
  // Reserve space to avoid reallocation which would invalidate pointers
  params.reserve(num_inputs + num_outputs + 10);

  for (int i = 0; i < num_inputs; i++) {
    auto buf_result = args.get<ffi::AnyBuffer>(i);
    if (!buf_result.has_value()) {
      std::string error_msg = absl::StrCat("Failed to get input tensor at index ", i, " for op: ", op_name);
      LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(error_msg);
    }
    aclTensor* tensor = ConvertAnyBufferToAclTensor(*buf_result);
    if (!tensor) {
      std::string error_msg = absl::StrCat("Failed to convert input tensor ", i, " to aclTensor for op: ", op_name);
      LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::Internal(error_msg);
    }
    all_acl_tensors.push_back(tensor);
  }

  for (int i = 0; i < num_outputs; i++) {
    auto ret_result = rets.get<ffi::AnyBuffer>(i);
    if (!ret_result.has_value()) {
      std::string error_msg = absl::StrCat("Failed to get output tensor at index ", i, " for op: ", op_name);
      LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(error_msg);
    }
    aclTensor* tensor = ConvertAnyBufferToAclTensor(**ret_result);
    if (!tensor) {
      std::string error_msg = absl::StrCat("Failed to convert output tensor ", i, " to aclTensor for op: ", op_name);
      LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::Internal(error_msg);
    }
    all_acl_tensors.push_back(tensor);
  }

  for (int i = 0; i < num_inputs; i++) {
    params.push_back({&ffi_type_pointer, &all_acl_tensors[i]});
  }

  auto param_count_result = attrs.get<int64_t>("param_count");
  int param_count = param_count_result.has_value() ? *param_count_result : 0;

  std::vector<int64_t> int64_values;
  std::vector<float> float_values;
  std::vector<int8_t> int8_values;
  std::vector<uint8_t> bool_values;
  std::vector<std::vector<int64_t>> int_array_storage;
  std::vector<std::vector<float>> float_array_storage;
  std::vector<std::vector<uint8_t>> bool_array_storage;

  int64_values.reserve(param_count);
  float_values.reserve(param_count);
  int8_values.reserve(param_count);
  bool_values.reserve(param_count);
  int_array_storage.reserve(param_count);
  float_array_storage.reserve(param_count);
  bool_array_storage.reserve(param_count);
  acl_scalars.reserve(param_count);
  acl_int_arrays.reserve(param_count);
  acl_float_arrays.reserve(param_count);
  acl_bool_arrays.reserve(param_count);

  for (int i = 0; i < param_count; i++) {
    std::string type_key = "param_" + std::to_string(i) + "_type";
    std::string value_key = "param_" + std::to_string(i);

    auto type_result = attrs.get<std::string_view>(type_key);
    if (!type_result.has_value()) {
      std::string error_msg = absl::StrCat("Missing param type: ", type_key, " for op: ", op_name);
      LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(error_msg);
    }
    std::string param_type(*type_result);

    if (param_type == "int64") {
      auto val = attrs.get<int64_t>(value_key);
      if (!val.has_value()) {
        std::string error_msg = absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(error_msg);
      }
      int64_values.push_back(*val);
      params.push_back({&ffi_type_sint64, &int64_values.back()});
    } else if (param_type == "float") {
      auto val = attrs.get<float>(value_key);
      if (!val.has_value()) {
        LOG(ERROR) << "[ACLNN DEBUG] Failed to get value for key: " << value_key
                     << " as float (F32)";
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                          acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name));
      } else {
        float_values.push_back(*val);
        params.push_back({&ffi_type_float, &float_values.back()});
      }
    } else if (param_type == "int8") {
      auto val = attrs.get<int8_t>(value_key);
      if (!val.has_value()) {
        std::string error_msg = absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(error_msg);
      }
      int8_values.push_back(*val);
      params.push_back({&ffi_type_sint8, &int8_values.back()});
    } else if (param_type == "bool") {
      auto val = attrs.get<bool>(value_key);
      if (!val.has_value()) {
        std::string error_msg = absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(error_msg);
      }
      bool_values.push_back(*val ? 1 : 0);
      params.push_back({&ffi_type_uint8, &bool_values.back()});
    } else if (param_type == "int_array") {
      auto val = attrs.get<Span<const int64_t>>(value_key);
      if (!val.has_value()) {
        std::string error_msg = absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(error_msg);
      }
      int_array_storage.emplace_back(val->begin(), val->end());
      aclIntArray* arr = aclCreateIntArray(
          int_array_storage.back().data(), int_array_storage.back().size());
      if (!arr) {
        std::string error_msg = absl::StrCat("Failed to create aclIntArray for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal(error_msg);
      }
      acl_int_arrays.push_back(arr);
      params.push_back({&ffi_type_pointer, &acl_int_arrays.back()});
    } else if (param_type == "float_array") {
      auto val = attrs.get<Span<const float>>(value_key);
      if (!val.has_value()) {
        std::string error_msg = absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(error_msg);
      }
      float_array_storage.emplace_back(val->begin(), val->end());
      aclFloatArray* arr = aclCreateFloatArray(
          float_array_storage.back().data(), float_array_storage.back().size());
      if (!arr) {
        std::string error_msg = absl::StrCat("Failed to create aclFloatArray for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal(error_msg);
      }
      acl_float_arrays.push_back(arr);
      params.push_back({&ffi_type_pointer, &acl_float_arrays.back()});
    } else if (param_type == "bool_array") {
      auto val = attrs.get<Span<const int8_t>>(value_key);
      if (!val.has_value()) {
        std::string error_msg = absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(error_msg);
      }
      bool_array_storage.emplace_back(val->begin(), val->end());
      aclBoolArray* arr = aclCreateBoolArray(
          reinterpret_cast<const bool*>(bool_array_storage.back().data()),
          bool_array_storage.back().size());
      if (!arr) {
        std::string error_msg = absl::StrCat("Failed to create aclBoolArray for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal(error_msg);
      }
      acl_bool_arrays.push_back(arr);
      params.push_back({&ffi_type_pointer, &acl_bool_arrays.back()});
    } else if (param_type == "scalar_int") {
      auto val = attrs.get<int64_t>(value_key);
      if (!val.has_value()) {
        LOG(ERROR) << "[ACLNN DEBUG] Failed to get int64 for key: " << value_key
                   << " for op: " << op_name;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name));
      }
      int64_t v = *val;
      aclScalar* scalar = aclCreateScalar(
          const_cast<void*>(reinterpret_cast<const void*>(&v)), ACL_INT64);
      if (!scalar) {
        std::string error_msg = absl::StrCat("Failed to create aclScalar for int for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal(error_msg);
      }
      acl_scalars.push_back(scalar);
      params.push_back({&ffi_type_pointer, &acl_scalars.back()});
    } else if (param_type == "scalar_float") {
      float v = 0.0f;
      auto val = attrs.get<float>(value_key);
      if (!val.has_value()) {
        auto double_val = attrs.get<double>(value_key);
        if (double_val.has_value()) {
          v = static_cast<float>(*double_val);
        } else {
          LOG(ERROR) << "[ACLNN DEBUG] Failed to get scalar_float as F32 or F64 for key: " << value_key;
          CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                           acl_float_arrays, acl_bool_arrays);
          return ffi::Error::InvalidArgument(
              absl::StrCat("Missing value for param: ", value_key, " for op: ", op_name));
        }
      } else {
        v = *val;
      }
      aclScalar* scalar = aclCreateScalar(
            const_cast<void*>(reinterpret_cast<const void*>(&v)), ACL_FLOAT);
      if (!scalar) {
        std::string error_msg = absl::StrCat("Failed to create aclScalar for float for op: ", op_name);
        LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                          acl_float_arrays, acl_bool_arrays);
        return ffi::Error::Internal(error_msg);
      }
      acl_scalars.push_back(scalar);
      params.push_back({&ffi_type_pointer, &acl_scalars.back()});
    } else {
      std::string error_msg = absl::StrCat("Unknown param type: ", param_type, " for op: ", op_name);
      LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(error_msg);
    }
  }

  for (int i = num_inputs; i < num_inputs + num_outputs; i++) {
    params.push_back({&ffi_type_pointer, &all_acl_tensors[i]});
  }

  std::string get_workspace_name = op_name + "GetWorkspaceSize";
  void* get_workspace_func = GetAclnnFuncAddr(get_workspace_name.c_str());
  void* execute_func = GetAclnnFuncAddr(op_name.c_str());

  if (!get_workspace_func || !execute_func) {
    std::string error_msg = absl::StrCat("ACLNN function not found: ", op_name);
    LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
    CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                     acl_float_arrays, acl_bool_arrays);
    return ffi::Error::Internal(error_msg);
  }

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  void* ws_ptr_storage = &workspace_size;
  void* exec_ptr_storage = &executor;

  params.push_back({&ffi_type_pointer, &ws_ptr_storage});
  params.push_back({&ffi_type_pointer, &exec_ptr_storage});

  std::vector<ffi_type*> arg_types;
  std::vector<void*> arg_values;
  arg_types.reserve(params.size());
  arg_values.reserve(params.size());
  for (auto& p : params) {
    arg_types.push_back(p.ffi_type);
    arg_values.push_back(p.value_ptr);
  }

  ffi_cif cif;
  ffi_status status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI,
                                    static_cast<unsigned int>(arg_types.size()),
                                    &ffi_type_sint, arg_types.data());
  if (status != FFI_OK) {
    std::string error_msg = absl::StrCat("ffi_prep_cif failed for ", op_name,
                     " with status ", static_cast<int>(status));
    LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
    CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                     acl_float_arrays, acl_bool_arrays);
    return ffi::Error::Internal(error_msg);
  }

  int ws_status = 0;
  ffi_call(&cif, reinterpret_cast<void (*)(void)>(get_workspace_func), &ws_status, arg_values.data());

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
      std::string error_msg = absl::StrCat("aclrtMalloc failed for workspace: ", malloc_status, " for op: ", op_name);
      LOG(ERROR) << "[ACLNN ERROR] " << error_msg;
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::Internal(error_msg);
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
