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


load("//xla:opts.bzl", "xla_common_options")


package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "ascend_compiler",
    srcs = [
        "ascend_compiler.cc",
        "ascend_executable.cc",
        "ascend_compiler_registration.cc",
    ],
    hdrs = [
        "ascend_compiler.h",
        "ascend_executable.h",
    ],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:hlo",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/ascend:ascend_platform_id",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
    copts = xla_common_options(),
)