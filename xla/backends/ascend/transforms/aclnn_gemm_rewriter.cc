/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/ascend/transforms/aclnn_gemm_rewriter.h"
#include "xla/backends/ascend/transforms/aclnn_config.h"
#include "xla/backends/ascend/transforms/aclnn_targets.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"
#include "xla/backends/ascend/transforms/aclnn_config.h"

namespace xla {
namespace ascend {
namespace {

namespace m = match;

absl::Status SetName(HloModule* module, HloInstruction* gemm) {
  module->SetAndUniquifyInstrName(gemm, "aclnn-gemm");
  return absl::OkStatus();
}

bool SupportsEpilogueFusion(PrimitiveType type) {
  switch (type) {
    case F16:
    case BF16:
    case F32:
      return true;
    default:
      return false;
  }
}

auto AclnnGemm(HloInstruction** instr) {
  return m::CustomCall(instr, {kAclnnGemmCallTarget});
}

auto BcastConstScalar(HloInstruction** instr, double value) {
  return m::Broadcast(instr, m::ConstantScalar(value));
}

auto BcastConstScalar(double value) { return BcastConstScalar(nullptr, value); }

template <typename Pattern>
auto OptionalSlice(HloInstruction** optional_slice, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Slice(optional_slice, pattern),
                                  std::move(pattern));
}

template <typename Pattern>
auto OptionalConvert(HloInstruction** optional_convert, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Convert(optional_convert, pattern),
                                  std::move(pattern));
}

template <typename Pattern>
auto OptionalBitcast(HloInstruction** optional_bitcast, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Bitcast(optional_bitcast, pattern),
                                  std::move(pattern));
}

absl::StatusOr<bool> IsAclnnSupportedMatMul(const HloInstruction& dot) {
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }

  int num_matrix_operands = 0;
  for (int operand_idx : {0, 1}) {
    TF_ASSIGN_OR_RETURN(auto dims, DotOperandDims::FromDotOperand(&dot, operand_idx));
    if (dims.DimensionCount(DotOperandDims::kContracting) != 1) {
      return false;
    }
    if (dims.DimensionCount(DotOperandDims::kNonContracting) > 1) {
      return false;
    }
    if (dims.DimensionCount(DotOperandDims::kNonContracting) == 1) {
      auto nc_dims = dims.DimensionSizes(DotOperandDims::kNonContracting);
      if (nc_dims[0] != 1) {
        num_matrix_operands += 1;
      }
    }
  }

  if (num_matrix_operands == 0) {
    return false;
  }

  switch (dot.shape().element_type()) {
    case F16:
    case BF16:
    case F32:
      return true;
    default:
      return false;
  }
}

bool IsBiasCompatibleWithOutput(const HloInstruction* bias,
                                const HloInstruction* gemm) {
  const Shape& bias_shape = bias->shape();
  const Shape& gemm_shape = gemm->shape();

  if (bias_shape.dimensions().size() == 0) {
    return true;
  }

  if (bias_shape.dimensions().size() == 1) {
    int64_t contracting_dim =
        gemm_shape.dimensions().size() - 1;
    return bias_shape.dimensions(0) ==
           gemm_shape.dimensions(contracting_dim);
  }

  if (ShapeUtil::Compatible(bias_shape, gemm_shape)) {
    return true;
  }

  return false;
}

bool CanFuseBias(const HloInstruction* gemm, const HloInstruction* bias) {
  if (gemm->user_count() != 1) {
    return false;
  }
  if (!IsBiasCompatibleWithOutput(bias, gemm)) {
    return false;
  }
  return SupportsEpilogueFusion(gemm->shape().element_type());
}

class AclnnGemmRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit AclnnGemmRewriterVisitor(const se::GpuComputeCapability& gpu_version)
      : gpu_version_(gpu_version) {}

  absl::Status HandleDot(HloInstruction* instr) override {
    TF_ASSIGN_OR_RETURN(bool is_supported,
                        IsAclnnSupportedMatMul(*instr));
    if (!is_supported) {
      return absl::OkStatus();
    }

    auto config = std::make_unique<AclnnGemmConfig>();
    config->alpha = 1.0f;
    config->beta = 0.0f;

    HloInstruction* lhs = instr->mutable_operand(0);
    HloInstruction* rhs = instr->mutable_operand(1);

    // Collect reduce axis information and determine transpose flags
    const auto& dot_dims = instr->dot_dimension_numbers();
    int64_t lhs_batch_dims_size = dot_dims.lhs_batch_dimensions_size();
    
    // Get contracting dimensions
    int64_t lhs_contracting_dim = dot_dims.lhs_contracting_dimensions(0);
    int64_t rhs_contracting_dim = dot_dims.rhs_contracting_dimensions(0);
    
    // Determine if we need to transpose
    // For ACLNN GEMM, we need to check if the contracting dimension is the last dimension
    bool is_lhs_vector = lhs->shape().dimensions().size() == lhs_batch_dims_size + 1;
    int64_t lhs_non_contracting_dim = is_lhs_vector ? lhs_batch_dims_size : lhs_batch_dims_size;
    
    bool is_rhs_vector = rhs->shape().dimensions().size() == lhs_batch_dims_size + 1;
    int64_t rhs_non_contracting_dim = is_rhs_vector ? lhs_batch_dims_size : lhs_batch_dims_size + 1;
    
    // Set transpose flags based on contracting dimension position
    config->transpose_a = (lhs_contracting_dim != lhs->shape().dimensions().size() - 1) ? 1 : 0;
    config->transpose_b = (rhs_contracting_dim != rhs->shape().dimensions().size() - 2) ? 1 : 0;

    // Calculate strides
    config->lhs_stride = is_lhs_vector ? lhs->shape().dimensions(lhs_batch_dims_size)
                     : lhs->shape().dimensions(lhs_batch_dims_size) *
                           lhs->shape().dimensions(lhs_batch_dims_size + 1);

    config->rhs_stride = is_rhs_vector ? rhs->shape().dimensions(lhs_batch_dims_size)
                     : rhs->shape().dimensions(lhs_batch_dims_size) *
                           rhs->shape().dimensions(lhs_batch_dims_size + 1);

    Shape output_shape = instr->shape();
    // 确保输出的 layout 与原始指令保持一致，不随 transpose 配置变化
    *output_shape.mutable_layout() = instr->shape().layout();
    
    // 创建操作数布局约束，确保操作数的布局也被保持
    std::vector<Shape> operand_shapes_with_layout;
    operand_shapes_with_layout.push_back(lhs->shape());
    operand_shapes_with_layout.push_back(rhs->shape());
    
    // 使用带有布局约束的 CreateCustomCall 重载，约束操作数和结果的布局
    HloInstruction* gemm_call =
        instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape,
            {lhs, rhs},
            kAclnnGemmCallTarget,
            operand_shapes_with_layout));

    gemm_call->set_raw_backend_config_string(SerializeAclnnConfig(*config));
    
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), gemm_call));
    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, gemm_call));

    return absl::OkStatus();
  }

  absl::Status HandleMultiply(HloInstruction* instr) override {
    HloInstruction *alpha, *existing_gemm;
    if (Match(instr,
              m::MultiplyAnyOrder(
                  AclnnGemm(&existing_gemm).WithOneUser(),
                  m::Broadcast(m::ConstantScalar(&alpha)).WithOneUser()))) {
      std::string config_str = existing_gemm->raw_backend_config_string();
      AclnnGemmConfig config;

      float prev_alpha = config.alpha;
      if (auto val = alpha->literal().GetAsDouble({}); val.has_value()) {
        prev_alpha = static_cast<float>(*val);
      }

      if (config.beta == 0.0f && existing_gemm->user_count() == 1) {
        config.alpha = prev_alpha;
        existing_gemm->set_raw_backend_config_string(config.ToString());
        return ReplaceInstruction(instr, existing_gemm);
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleAdd(HloInstruction* instr) override {
    HloInstruction *bias = nullptr, *existing_gemm = nullptr;
    HloInstruction* optional_slice = nullptr;
    HloInstruction* optional_convert = nullptr;
    HloInstruction* optional_bitcast = nullptr;

    if (Match(instr,
              m::AddAnyOrder(
                  OptionalBitcast(
                      &optional_bitcast,
                      OptionalSlice(
                          &optional_slice,
                          AclnnGemm(&existing_gemm).WithOneUser())
                          .WithOneUser())
                      .WithOneUser(),
                  m::Broadcast(&bias,
                               OptionalConvert(&optional_convert, m::Op()))))) {
      TF_ASSIGN_OR_RETURN(bool was_fused,
                          FuseVectorBiasAdd(instr, bias, existing_gemm,
                                           optional_slice, optional_convert,
                                           optional_bitcast));
      if (was_fused) {
        return absl::OkStatus();
      }
    }

    if (Match(instr,
              m::AddAnyOrder(
                  m::Bitcast(AclnnGemm(&existing_gemm).WithOneUser())
                      .WithOneUser(),
                  m::Broadcast(&bias, m::Op()).WithOneUser()))) {
      TF_ASSIGN_OR_RETURN(bool was_fused,
                          FuseVectorBiasAdd(instr, bias, existing_gemm,
                                           nullptr, nullptr, nullptr));
      if (was_fused) {
        return absl::OkStatus();
      }
    }

    auto is_not_broadcast = HloPredicateIsNotOp<HloOpcode::kBroadcast>;

    if (Match(instr,
              m::AddAnyOrder(
                  m::Bitcast(
                      AclnnGemm(&existing_gemm).WithOneUser())
                      .WithOneUser(),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      TF_ASSIGN_OR_RETURN(bool was_fused,
                          FuseMatrixBiasAdd(instr, bias, existing_gemm));
      if (was_fused) {
        return absl::OkStatus();
      }
    }

    if (Match(instr,
              m::AddAnyOrder(
                  m::AnyOf<HloInstruction>(
                      AclnnGemm(&existing_gemm).WithOneUser(),
                      m::Convert(
                          AclnnGemm(&existing_gemm).WithOneUser())
                          .WithOneUser()),
                  m::Op(&bias).WithPredicate(is_not_broadcast)))) {
      bool types_are_supported =
          SupportsEpilogueFusion(existing_gemm->shape().element_type());

      bool has_no_consumer =
          instr->shape().element_type() ==
              existing_gemm->shape().element_type() ||
          instr->user_count() == 0 ||
          (instr->user_count() == 1 &&
           instr->users()[0]->opcode() == HloOpcode::kTuple &&
           instr->users()[0]->user_count() == 0);

      if (types_are_supported && has_no_consumer) {
        TF_ASSIGN_OR_RETURN(bool was_fused,
                            FuseMatrixBiasAdd(instr, bias, existing_gemm));
        if (was_fused) {
          return absl::OkStatus();
        }
      }
    }

    return absl::OkStatus();
  }

 private:
  absl::StatusOr<bool> FuseVectorBiasAdd(
      HloInstruction* instr, HloInstruction* bias, HloInstruction* gemm,
      HloInstruction* slice, HloInstruction* convert,
      HloInstruction* bitcast) {
    if (!CanFuseBias(gemm, bias)) {
      return false;
    }

    HloInstruction* bias_operand = bias->mutable_operand(0);
    std::vector<HloInstruction*> operands(gemm->operands().begin(),
                                          gemm->operands().end());
    operands.push_back(bias_operand);

    HloComputation* computation = gemm->parent();
    HloInstruction* result = computation->AddInstruction(
        HloInstruction::CreateCustomCall(
            gemm->shape(),
            operands,
            kAclnnGemmCallTarget));

    // Parse original config and modify it
    TF_ASSIGN_OR_RETURN(auto config, ParseAclnnConfig(
        kAclnnGemmCallTarget, gemm->raw_backend_config_string()));
    auto* gemm_config = dynamic_cast<AclnnGemmConfig*>(config.get());
    if (!gemm_config) {
      return absl::InternalError("Failed to cast to AclnnGemmConfig");
    }
    gemm_config->has_bias = true;
    gemm_config->beta = 1.0f;
    result->set_raw_backend_config_string(SerializeAclnnConfig(*config));
    TF_RETURN_IF_ERROR(SetName(gemm->GetModule(), result));

    if (slice) {
      result = computation->AddInstruction(
          slice->CloneWithNewOperands(slice->shape(), {result}));
    }

    if (bitcast) {
      result = computation->AddInstruction(
          bitcast->CloneWithNewOperands(bitcast->shape(), {result}));
    }

    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, result));
    return true;
  }

  absl::StatusOr<bool> FuseMatrixBiasAdd(HloInstruction* instr, HloInstruction* bias,
                                HloInstruction* gemm) {
    TF_RET_CHECK(ShapeUtil::Compatible(bias->shape(), gemm->shape()));

    if (!SupportsEpilogueFusion(gemm->shape().element_type())) {
      return false;
    }

    if (gemm->user_count() != 1) {
      return false;
    }

    std::vector<HloInstruction*> operands(gemm->operands().begin(),
                                          gemm->operands().end());
    operands.push_back(bias);

    // Parse original config and modify it
    TF_ASSIGN_OR_RETURN(auto config, ParseAclnnConfig(
        kAclnnGemmCallTarget, gemm->raw_backend_config_string()));
    auto* gemm_config = dynamic_cast<AclnnGemmConfig*>(config.get());
    if (!gemm_config) {
      return absl::InternalError("Failed to cast to AclnnGemmConfig");
    }
    gemm_config->has_bias = true;
    gemm_config->beta = 1.0f;

    std::unique_ptr<HloInstruction> fused_op =
        HloInstruction::CreateCustomCall(
            gemm->shape(),
            operands,
            kAclnnGemmCallTarget);
    fused_op->mutable_shape()->set_element_type(bias->shape().element_type());
    fused_op->set_raw_backend_config_string(SerializeAclnnConfig(*config));
    TF_RETURN_IF_ERROR(SetName(instr->GetModule(), fused_op.get()));

    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(instr, std::move(fused_op)));
    return true;
  }

  se::GpuComputeCapability gpu_version_;
};

absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
                                     se::GpuComputeCapability gpu_version) {
  AclnnGemmRewriterVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

AclnnGemmRewriter::AclnnGemmRewriter(se::GpuComputeCapability gpu_version)
    : gpu_version_(gpu_version) {}

absl::StatusOr<bool> AclnnGemmRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_));
    changed |= result;
  }
  return changed;
}

}  // namespace ascend
}  // namespace xla
