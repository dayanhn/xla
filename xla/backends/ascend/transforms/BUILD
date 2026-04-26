load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
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