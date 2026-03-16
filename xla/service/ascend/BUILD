# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/tsl:tsl.bzl",
    "internal_visibility",
)

package(
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//xla/...",
    ],
)

cc_library(
    name = "ascend_executable",
    srcs = [
        "ascend_executable.cc",
    ],
    hdrs = [
        "ascend_executable.h",
    ],
    tags = [
        "ascend-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ascend_transfer_manager",
    srcs = [
        "ascend_transfer_manager.cc",
    ],
    hdrs = [
        "ascend_transfer_manager.h",
    ],
    tags = [
        "ascend-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/service:generic_transfer_manager",
        "//xla/service:transfer_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/ascend:ascend_platform_id",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,  # Registers itself with the TransferManager registry.
)

cc_library(
    name = "ascend_compiler",
    srcs = [
        "ascend_compiler.cc",
        "ascend_compiler_registration.cc",
    ],
    hdrs = [
        "ascend_compiler.h",
    ],
    tags = [
        "ascend-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":ascend_executable",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:llvm_compiler",
        "//xla/stream_executor/ascend:ascend_platform_id",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,  # Registers itself with the Compiler registry.
)

cc_library(
    name = "ascend_computation_placer",
    srcs = [
        "ascend_computation_placer_registration.cc",
    ],
    tags = [
        "ascend-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/service:computation_placer",
        "//xla/stream_executor/ascend:ascend_platform_id",
    ],
    alwayslink = 1,  # Registers itself with the ComputationPlacer registry.
)
