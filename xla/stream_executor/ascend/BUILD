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
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
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
    packages = stream_executor_friends(),
)

cc_library(
    name = "ascend_platform_id",
    srcs = ["ascend_platform_id.cc"],
    hdrs = ["ascend_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_binary(
    name = "test_ascend_platform",
    srcs = ["test_ascend_platform.cc"],
    deps = [
        ":ascend_platform",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "@com_google_absl//absl/status",
    ],
)

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:tsl.bzl", "if_oss")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")


cc_library(
    name = "ascend_status",
    srcs = ["ascend_status.cc"],
    hdrs = ["ascend_status.h"],
    deps = [
        "//third_party/acl:acl_headers",
        "@com_google_absl//absl/status",
    ],
    copts = select({
        "//conditions:default": ["-Ithird_party/acl/inc"],
    }),
)

cc_library(
    name = "ascend_event",
    srcs = ["ascend_event.cc"],
    hdrs = ["ascend_event.h"],
    deps = [
        ":ascend_status",
        "//xla/stream_executor:stream_executor_h",
        "//third_party/acl:acl_headers",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:event",
    ],
    copts = select({
        "//conditions:default": ["-Ithird_party/acl/inc"],
    }),
)

cc_library(
    name = "ascend_stream",
    srcs = ["ascend_stream.cc"],
    hdrs = ["ascend_stream.h"],
    deps = [
        ":ascend_event",
        ":ascend_status",
        "//third_party/acl:acl_headers",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
    ],
    copts = select({
        "//conditions:default": ["-Ithird_party/acl/inc"],
    }),
)

cc_library(
    name = "ascend_context_map",
    hdrs = ["ascend_context_map.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "ascend_context",
    srcs = ["ascend_context.cc"],
    hdrs = ["ascend_context.h"],
    deps = [
        ":ascend_status",
        ":ascend_context_map",
        ":scoped_activate_context",
        "//third_party/acl:acl_headers",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:context",
        "@com_google_absl//absl/log:check",
    ],
    copts = select({
        "//conditions:default": ["-Ithird_party/acl/inc"],
    }),
)

cc_library(
    name = "context",
    hdrs = ["context.h"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "scoped_activate_context",
    srcs = ["scoped_activate_context.cc"],
    hdrs = ["scoped_activate_context.h"],
    deps = [
        ":ascend_context_map",
        ":context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/base:core_headers",
        "//third_party/acl:acl_headers",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor:activate_context",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/synchronization",
    ],
    copts = select({
        "//conditions:default": ["-Ithird_party/acl/inc"],
    }),
)

cc_library(
    name = "ascend_executor",
    srcs = [
        "ascend_executor.cc",
    ],
    hdrs = [
        "ascend_executor.h",
    ],
    tags = [
        "ascend-only",
        "gpu",
    ],
    deps = [
        ":ascend_platform_id",
        ":ascend_event",
        ":ascend_stream",
        ":ascend_status",
        ":ascend_context",
        ":scoped_activate_context",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream_executor_common",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//third_party/acl:acl_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
    ],
)


cc_library(
    name = "ascend_platform",
    srcs = [
        "ascend_platform.cc",
    ],
    hdrs = ["ascend_platform.h"],
    tags = [
        "ascend-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":ascend_executor",
        ":ascend_event",
        ":ascend_stream",
        ":ascend_status",
        ":ascend_context",
        ":scoped_activate_context",
        ":ascend_platform_id",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//third_party/acl:acl_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:statusor",
    ],
    alwayslink = 1,  # Registers itself with the PlatformManager.
)
