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
    name = "ascend_ffi",
    srcs = [
        "ascend_ffi.cc",
        "ops/nn/activation/gelu.cc",
        "utils/tensor_utils.cc",
    ],
    hdrs = [
        "ascend_ffi.h",
        "utils/tensor_utils.h",
    ],
    tags = [
        "ascend-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/ffi:ffi_api",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
    ],
    copts = select({
        "//conditions:default": [],
    }),
    linkopts = select({
        "//conditions:default": [],
    }),
)