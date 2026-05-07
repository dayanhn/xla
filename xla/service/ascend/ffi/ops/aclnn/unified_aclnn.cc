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

  LOG(ERROR) << "[ACLNN DEBUG] ===== Starting parameter validation =====";
  LOG(ERROR) << "[ACLNN DEBUG] op_name: " << op_name;
  LOG(ERROR) << "[ACLNN DEBUG] num_inputs: " << num_inputs;
  LOG(ERROR) << "[ACLNN DEBUG] num_outputs: " << num_outputs;
  
  std::vector<AclnnParam> params;
  
  // Reserve space to avoid reallocation which would invalidate pointers
  params.reserve(num_inputs + num_outputs + 10);

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
      LOG(ERROR) << "[ACLNN DEBUG] Failed to convert input tensor " << i 
                 << " to aclTensor. Buffer info: element_type=" 
                 << static_cast<int>((*buf_result).element_type())
                 << ", dimensions=[";
      for (auto dim : (*buf_result).dimensions()) {
        LOG(ERROR) << dim << ",";
      }
      LOG(ERROR) << "]";
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::Internal(
          absl::StrCat("Failed to convert input tensor ", i, " to aclTensor"));
    }
    all_acl_tensors.push_back(tensor);
  }

  // Now add all input tensors to params after all pushes to avoid invalidation
  for (int i = 0; i < num_inputs; i++) {
    // For pointer parameters, pass the pointer value directly, not its address
    params.push_back({&ffi_type_pointer, all_acl_tensors[i]});
  }

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
      LOG(ERROR) << "[ACLNN DEBUG] Failed to convert output tensor " << i 
                 << " to aclTensor";
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::Internal(
          absl::StrCat("Failed to convert output tensor ", i, " to aclTensor"));
    }
    all_acl_tensors.push_back(tensor);
  }

  // Add output tensors to params
  for (int i = 0; i < num_outputs; i++) {
    // For pointer parameters, pass the pointer value directly, not its address
    params.push_back({&ffi_type_pointer, all_acl_tensors[num_inputs + i]});
  }

  LOG(ERROR) << "[ACLNN DEBUG] After adding tensors, params.size() = " << params.size();
  for (size_t i = 0; i < params.size(); ++i) {
    LOG(ERROR) << "[ACLNN DEBUG]   params[" << i << "] ffi_type=" << params[i].ffi_type 
               << ", value_ptr=" << params[i].value_ptr;
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
      params.push_back({&ffi_type_sint64, &int64_values.back()});
    } else if (param_type == "float") {
      auto val = attrs.get<float>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      float_values.push_back(*val);
      params.push_back({&ffi_type_float, &float_values.back()});
    } else if (param_type == "int8") {
      auto val = attrs.get<int64_t>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      int8_values.push_back(static_cast<int8_t>(*val));
      params.push_back({&ffi_type_sint8, &int8_values.back()});
    } else if (param_type == "bool") {
      auto val = attrs.get<bool>(value_key);
      if (!val.has_value()) {
        CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                         acl_float_arrays, acl_bool_arrays);
        return ffi::Error::InvalidArgument(
            absl::StrCat("Missing value for param: ", value_key));
      }
      bool_values.push_back(*val ? 1 : 0);
      params.push_back({&ffi_type_uint8, &bool_values.back()});
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
      params.push_back({&ffi_type_pointer, &acl_int_arrays.back()});
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
      params.push_back({&ffi_type_pointer, &acl_float_arrays.back()});
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
      params.push_back({&ffi_type_pointer, &acl_bool_arrays.back()});
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
      params.push_back({&ffi_type_pointer, &acl_scalars.back()});
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
      params.push_back({&ffi_type_pointer, &acl_scalars.back()});
    } else {
      CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                       acl_float_arrays, acl_bool_arrays);
      return ffi::Error::InvalidArgument(
          absl::StrCat("Unknown param type: ", param_type));
    }
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

  LOG(ERROR) << "[ACLNN DEBUG] Calling " << get_workspace_name 
            << " with " << params.size() << " parameters";
  
  // Debug: print tensor addresses and validate them
  for (size_t i = 0; i < all_acl_tensors.size(); ++i) {
    if (all_acl_tensors[i] == nullptr) {
      LOG(ERROR) << "[ACLNN DEBUG] ERROR: Tensor " << i << " is nullptr!";
    } else {
      LOG(ERROR) << "[ACLNN DEBUG] Tensor " << i << " address: " << all_acl_tensors[i]
                 << ", will pass as arg_values[" << i << "] = " << all_acl_tensors[i];
    }
  }

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  // For libffi, we need to pass the pointers directly as argument values
  // The function signature is: aclnnStatus func(..., uint64_t* workspaceSize, aclOpExecutor** executor)
  // So we pass &workspace_size (which is uint64_t*) and &executor (which is aclOpExecutor**)
  params.push_back({&ffi_type_pointer, &workspace_size});
  params.push_back({&ffi_type_pointer, &executor});

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
    CleanupResources(all_acl_tensors, acl_scalars, acl_int_arrays,
                     acl_float_arrays, acl_bool_arrays);
    return ffi::Error::Internal(
        absl::StrCat("ffi_prep_cif failed for ", op_name,
                     " with status ", static_cast<int>(status)));
  }

  int ws_status = 0;
  LOG(ERROR) << "[ACLNN DEBUG] Before ffi_call: workspace_size=" << workspace_size 
            << ", executor=" << executor;
  LOG(ERROR) << "[ACLNN DEBUG] Passing addresses: &workspace_size=" << &workspace_size 
            << ", &executor=" << &executor;
  
  // Print all argument values being passed to libffi
  for (size_t i = 0; i < arg_values.size(); ++i) {
    LOG(ERROR) << "[ACLNN DEBUG] arg_values[" << i << "] = " << arg_values[i];
  }
  
  // Try direct call with correct function signature for testing
  typedef aclnnStatus (*DirectGetWorkspaceFunc)(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
  DirectGetWorkspaceFunc direct_func = reinterpret_cast<DirectGetWorkspaceFunc>(get_workspace_func);
  
  LOG(ERROR) << "[ACLNN DEBUG] ===== Testing direct call =====";
  uint64_t test_workspace_size = 0;
  aclOpExecutor* test_executor = nullptr;
  aclnnStatus direct_status = direct_func(
      all_acl_tensors[0],  // self
      all_acl_tensors[1],  // out
      &test_workspace_size,
      &test_executor
  );
  LOG(ERROR) << "[ACLNN DEBUG] Direct call result: status=" << direct_status 
            << ", workspace_size=" << test_workspace_size 
            << ", executor=" << test_executor;
  LOG(ERROR) << "[ACLNN DEBUG] ===== End direct call test =====";
  
  // Now try libffi call
  LOG(ERROR) << "[ACLNN DEBUG] ===== Starting libffi call =====";
  ffi_call(&cif, reinterpret_cast<void (*)(void)>(get_workspace_func), &ws_status, arg_values.data());
  LOG(ERROR) << "[ACLNN DEBUG] ===== End libffi call =====";

  LOG(ERROR) << "[ACLNN DEBUG] After ffi_call: ws_status=" << ws_status 
            << ", workspace_size=" << workspace_size 
            << ", executor=" << executor;

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
