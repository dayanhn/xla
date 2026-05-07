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
        "ops/nn/matmul/matmul_cublas.cc",
        "ops/nn/matmul/matmul.cc",
        "ops/nn/matmul/matmul_gemm.cc",
        "ops/nn/index_fill/index_fill_tensor.cc",
        "ops/nn/full/full.cc",
        "ops/math/arg_max_with_value/max_dim.cc",
        "ops/math/muls/muls.cc",
        "ops/math/add/add.cc",
        "ops/math/cast/cast.cc",
        "ops/math/divide/divide.cc",
        "ops/math/equal/equal.cc",
        "ops/math/exponential/exponential.cc",
        "ops/math/expand/expand.cc",
        "ops/math/greater/greater.cc",
        "ops/math/greater_equal/greater_equal.cc",
        "ops/math/less/less.cc",
        "ops/math/less_equal/less_equal.cc",
        "ops/math/maximum/maximum.cc",
        "ops/math/multiply/multiply.cc",
        "ops/math/negate/negate.cc",
        "ops/math/not_equal/not_equal.cc",
        "ops/math/reduce_max/reduce_max.cc",
        "ops/math/reduce_mean/reduce_mean.cc",
        "ops/math/reduce_min/reduce_min.cc",
        "ops/math/reduce_prod/reduce_prod.cc",
        "ops/math/select/select.cc",
        "ops/math/subtract/subtract.cc",
        "ops/math/right_shift/right_shift.cc",
        "ops/math/cat/cat.cc",
        "ops/math/reduce_sum/reduce_sum.cc",
        "ops/math/iota/iota.cc",
        "ops/aclnn/unified_aclnn.cc",
        "utils/tensor_utils.cc",
    ],
    hdrs = [
        "ascend_ffi.h",
        "utils/tensor_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/ffi/api:ffi",
        "//xla/ffi",
        "//xla/ffi:attribute_map",
        "//xla/ffi:call_frame",
        "//xla/ffi:execution_context",
        "//xla/ffi:execution_state",
        "//xla/ffi:ffi_api",  # buildcleaner: keep
        "//xla/ffi:ffi_registry",
        "//xla/ffi:invoke",
        "//xla/ffi/api:c_api",
        "//xla:shape_util",
        "//third_party/acl:acl_headers",
        "//third_party/libffi:libffi_headers",
        "@com_google_absl//absl/strings",
    ],
    copts = select({
        "//conditions:default": [
            "-Ithird_party/acl/inc",
        ],
    }),
    linkopts = select({
        "//conditions:default": ["-L/usr/lib/aarch64-linux-gnu", "-lffi"],
    }),
)
