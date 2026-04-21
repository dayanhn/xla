1. xla/xla/backends/autotuner/backends.proto
```c++
enum Backend {
  //新增ASCEND_DNN后端
  ASCEND_DNN = 16;
}
```
2. xla/xla/pjrt/c/BUILD
```c++
// 新增ASCEND PJRT后端的编译目标
# ==============================================================================
# Ascend PJRT C API Implementation
# ==============================================================================
cc_library(
    name = "pjrt_c_api_ascend_internal",
    srcs = ["pjrt_c_api_ascend_internal.cc"],
    hdrs = ["pjrt_c_api_ascend_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//xla/pjrt/c:pjrt_c_api_ffi_extension_hdrs",
        "//xla/pjrt/npu:se_ascend_pjrt_client",
        "//xla/pjrt/plugin/xla_npu:npu_client_options",
        "//xla/pjrt/plugin/xla_npu:xla_npu_pjrt_client",
        "//xla/service/ascend/ffi:ascend_ffi",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pjrt_c_api_ascend",
    srcs = ["pjrt_c_api_ascend.cc"],
    hdrs = ["pjrt_c_api_ascend.h"],
    deps = [
        ":pjrt_c_api_ascend_internal",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_macros",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:initialize",
        "@tsl//tsl/platform",
    ],
    alwayslink = 1,
)

```

3. xla/xla/service/gpu/build_defs.bzl
```c++
# 新增函数，用于判断Ascend是否被配置(依赖于JAX前端的编译选项)
def if_ascend_is_configured(x):
    """Returns x if Ascend is configured, otherwise returns an empty list.
    
    This function checks if the Ascend backend is enabled in the build configuration.
    It should be used to conditionally include Ascend-specific dependencies.
    
    Args:
      x: A list of items to include if Ascend is configured.
      
    Returns:
      A select statement that returns x when Ascend is enabled, or an empty list otherwise.
    """
    return select({
        "@local_config_ascend//ascend:enable_ascend": x,
        "//conditions:default": [],
    })
```

4. xla/xla/service/gpu/BUILD
```c++
// 加载上一步修改的ASCEND的使能配置
load(
    ":build_defs.bzl",
    "if_ascend_is_configured",
)

// 因为修改了gpu thunk_emitter组件发射thunk的接口，在ascend平台下首先通过ascend的thunk_emitter组件发射thunk，因此
// 需要修改gpu thunk_emitter组件的编译目标，添加ascend的thunk_emitter组件的依赖项
+ if_ascend_is_configured([
        "//xla/service/ascend:thunk_emitter",
    ]),
    defines = if_ascend_is_configured([
        "XLA_ENABLE_ASCEND",
    ]),

// 为gpu_executable增加ascend平台的依赖项
 + if_ascend_is_configured([
        "//xla/stream_executor/ascend:ascend_platform_id",
    ]),
    defines = if_ascend_is_configured([
        "XLA_ENABLE_ASCEND",
    ]),
```

5. xla/xla/service/gpu/gpu_compiler.cc
```c++
    // Ascend平台不添加 AddFusionAutotuningPass pass
    if (gpu_target_config.platform_name != "ASCEND") {
      RETURN_IF_ERROR(AddFusionAutotuningPass(
          &pipeline, hlo_module, options, thread_pool.get_mutable(), stream_exec,
          &gpu_target_config, ShapeSizeBytesFunction(), options.key_value_store));
    }

   // Ascend平台不添加 TreeReductionRewriter pass
   if (gpu_target_config.platform_name != "ASCEND") {
      pipeline.AddPass<HloPassFix<TreeReductionRewriter>>(
          gpu_target_config.device_description);
    }

  // Ascend平台不添加 AddConvAndGemmAutotuningPass pass
  if (gpu_target_config.platform_name != "ASCEND") {          
    TF_RETURN_IF_ERROR(AddConvAndGemmAutotuningPass(
        &pipeline, hlo_module, gpu_version, options, autotune_config, thread_pool,
        stream_exec, &gpu_target_config, options.key_value_store,
        gpu_target_config.device_description.runtime_version(), alias_info,
        debug_options, &mlir_context_, ShapeSizeBytesFunction()));
  }
```

6. xla/xla/service/gpu/gpu_executable.cc
```c++
#ifdef XLA_ENABLE_ASCEND
#include "xla/stream_executor/ascend/ascend_platform_id.h"
#endif


// 在ASCEND平台下不检查兼容性
#ifdef XLA_ENABLE_ASCEND
  else if (platform_id == stream_executor::ascend::kAscendPlatformId) {
    // TODO: Add check.
  } 
#endif
```

7. xla/xla/service/gpu/thunk_emitter.cc
```c++
// 在ASCEND平台下添加ASCEND的thunk_emitter组件
#ifdef XLA_ENABLE_ASCEND
#include "xla/service/ascend/thunk_emitter.h"
#endif

  // 在ASCEND平台下先尝试通过ascend的thunk_emitter组件发射thunk
#ifdef XLA_ENABLE_ASCEND
  {
    auto ascend_result = xla::ascend::TryEmitHloInstructionAscend(
        hlo, ir_emitter_context_, llvm_options_lock_);
    if (ascend_result.ok() && ascend_result->has_value()) {
      // Ascend backend handled this instruction
      return std::move(ascend_result.value().value());
    }
    // If Ascend returned nullopt, continue with GPU implementation
  }
#endif

```