# Copyright 2024 The OpenXLA Authors.
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
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "aclnn_thunk",
    srcs = ["aclnn_thunk.cc"],
    hdrs = [
        "aclnn_thunk.h",
        "aclnn_api_util.h",
    ],
    deps = [
        "//xla/backends/gpu/runtime:thunk",
        "//xla/service",
        "//xla/service:shaped_slice",
        "//xla/ffi/api:ffi",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor:stream",
        "//third_party/acl:acl_headers",
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
