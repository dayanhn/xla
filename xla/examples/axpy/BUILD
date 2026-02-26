load("//xla:xla.default.bzl", "xla_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

xla_cc_test(
    name = "stablehlo_compile_test",
    srcs = ["stablehlo_compile_test.cc"],
    data = ["stablehlo_axpy.mlir"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_cpu",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c_api_client:pjrt_c_api_client",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/platform:path",
    ],
)

cc_binary(
    name = "stablehlo_compiler",
    srcs = ["stablehlo_compiler.cc"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_cpu",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c_api_client:pjrt_c_api_client",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        # Removed test-only deps: literal_test_util, xla_internal_test_main,
        # and status_test_util (these targets are marked testonly and cannot be
        # depended on by non-test binaries).
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/platform:path",
    ],
)

# GPU compiler binary: compile StableHLO to GPU and execute
cc_binary(
    name = "stablehlo_gpu_compiler",
    srcs = ["stablehlo_gpu_compiler.cc"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c:pjrt_c_api_cpu",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c_api_client:pjrt_c_api_client",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        # Ensure the CUDA StreamExecutor platform and PJRT GPU plugin are
        # linked into the binary so their static initializers run and register
        # the "cuda" platform and PJRT GPU plugin.
        "//xla/stream_executor:cuda_platform",
        "//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        # Removed test-only deps: literal_test_util, xla_internal_test_main,
        # and status_test_util (these targets are marked testonly and cannot be
        # depended on by non-test binaries).
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/platform:path",
    ],
)

# Priority Fusion analyzer: analyze Priority Fusion pass processing
cc_binary(
    name = "priority_fusion_analyzer",
    testonly = True,
    srcs = ["priority_fusion_analyzer.cc"],
    deps = [
        # core utilities used by the analyzer executable.
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",

        # we are exercising the PriorityFusion pass itself, so depend on the
        # cc_library that actually compiles the pass.  the original target in
        # this file accidentally referenced the directory rather than the
        # specific library which does not exist and caused the build to fail.
        "//xla/backends/gpu/transforms:priority_fusion",

        # headers pulled in by the analyzer source file.
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:symbolic_map",        # contains symbolic_expr.h
        "//xla/hlo/ir:hlo",                      # core HLO IR types
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:verified_hlo_module",

        # other pieces of the GPU stack that the analyzer touches.
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",

        # retain some of the previously added deps; they are mostly harmless
        # and pull in common utilities used throughout XLA.
        "//xla/hlo/translate:stablehlo",
        "//xla/stream_executor:cuda_platform",
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@stablehlo//:register",
        "@tsl//tsl/platform:path",
    ],
)
