
load("//xla/tsl:tsl.bzl", "if_google")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)
cc_library(
    name = "xla_npu_pjrt_client",
    srcs = [
        "xla_npu_pjrt_client.cc",
    ],
    hdrs = ["xla_npu_pjrt_client.h"],
    deps = [
        ":npu_client_options",
        ":xla_npu_allocator_config",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/plugin:plugin_names",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_callback_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_tpu_topology_extension_hdrs",
        "//xla/pjrt/npu:se_ascend_pjrt_client",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "npu_client_options",
    srcs = [],
    hdrs = ["npu_client_options.h"],
    deps = [
        "//xla/pjrt:host_memory_allocator",
        "//xla/pjrt/distributed:key_value_store_interface",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "xla_npu_allocator_config",
    srcs = [],
    hdrs = ["xla_npu_allocator_config.h"],
    deps = [
        "//xla/tsl/framework:allocator",
    ],
)

cc_binary(
    name = "xla_npu_pjrt_client_test",
    srcs = [
        "xla_npu_pjrt_client_test.cc",
    ],
    deps = [
        ":npu_client_options",
        ":xla_npu_allocator_config",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/npu:se_ascend_pjrt_client",
        ":xla_npu_pjrt_client",
        "@com_google_absl//absl/status:statusor",
        "//xla/stream_executor/ascend:ascend_platform",
    ],
)
