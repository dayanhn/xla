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

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "hccl_communicator",
    srcs = ["hccl_communicator.cc"],
    hdrs = ["hccl_communicator.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/acl:acl_headers",
        "//third_party/hccl:hccl_headers",
        "//xla/backends/gpu/collectives:gpu_collectives",
        "//xla/backends/gpu/collectives:gpu_communicator",
        "//xla/backends/gpu/collectives:single_threaded_executor",
        "//xla/core/collectives",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:device_address",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:numbers",
    ],
    copts = select({
        "//conditions:default": ["-Ithird_party/acl/inc"],
    }),
)

cc_library(
    name = "hccl_collectives",
    srcs = ["hccl_collectives.cc"],
    hdrs = ["hccl_collectives.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/acl:acl_headers",
        ":hccl_communicator",
        "//xla/backends/gpu/collectives:gpu_clique_key",
        "//xla/backends/gpu/collectives:cancellation_token",
        "//xla/backends/gpu/collectives:single_threaded_executor",
        "//xla/backends/gpu/collectives:gpu_collectives",
        "//xla/core/collectives",
        "//xla/core/collectives:clique_id",
        "//xla/core/collectives:clique_key",
        "//xla/core/collectives:collectives_registry",
        "//xla/core/collectives:communicator",
        "//xla/core/collectives:rank_id",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/runtime:device_id",
        "//xla/runtime:process_id",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:numbers",
    ],
    copts = select({
        "//conditions:default": ["-Ithird_party/acl/inc"],
    }),
    alwayslink = True,  # registers collectives implementation
)


cc_library(
    name = "collectives",
    deps = [
        ":hccl_collectives",
    ],
    visibility = ["//xla:internal"],
)