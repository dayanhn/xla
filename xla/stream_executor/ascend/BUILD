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
        ":ascend_platform_id",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//third_party/acl:acl",
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
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
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
