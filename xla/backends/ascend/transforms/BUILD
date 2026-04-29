load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "aclnn_targets",
    srcs = ["aclnn_targets.cc"],
    hdrs = ["aclnn_targets.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "aclnn_config",
    srcs = ["aclnn_config.cc"],
    hdrs = ["aclnn_config.h"],
    deps = [
        ":aclnn_targets",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "aclnn_gemm_rewriter",
    srcs = ["aclnn_gemm_rewriter.cc"],
    hdrs = ["aclnn_gemm_rewriter.h"],
    deps = [
        ":aclnn_config",
        ":aclnn_targets",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/service:algorithm_util",
        "//xla/service:hlo_creation_utils",
        "//xla/service:pattern_matcher",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service:matmul_indexing_utils",
        "//xla/stream_executor:blas_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "aclnn_maxpool_backward_rewriter",
    srcs = ["aclnn_maxpool_backward_rewriter.cc"],
    hdrs = ["aclnn_maxpool_backward_rewriter.h"],
    deps = [
        "//xla/hlo/ir",
        "//xla/hlo/pass",
        "//xla/service",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "aclnn_convolution_rewriter",
    srcs = ["aclnn_convolution_rewriter.cc"],
    hdrs = ["aclnn_convolution_rewriter.h"],
    deps = [
        ":aclnn_config",
        ":aclnn_targets",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/evaluator:hlo_evaluator",
        "//xla/service:algorithm_util",
        "//xla/service:hlo_creation_utils",
        "//xla/service:pattern_matcher",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service:matmul_indexing_utils",
        "//xla/stream_executor:blas_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:ml_dtypes",
    ],
)

cc_library(
    name = "aclnn_convolution_backward_rewriter",
    srcs = ["aclnn_convolution_backward_rewriter.cc"],
    hdrs = ["aclnn_convolution_backward_rewriter.h"],
    deps = [
        "//xla/hlo/ir",
        "//xla/hlo/pass",
        "//xla/service",
        "//xla/util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "aclnn_fusion_pass",
    srcs = ["aclnn_fusion_pass.cc"],
    hdrs = ["aclnn_fusion_pass.h"],
    deps = [
        ":aclnn_gemm_rewriter",
        "//xla/backends/gpu/target_config:target_config",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",

    ],
)

