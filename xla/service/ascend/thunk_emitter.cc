/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/ascend/thunk_emitter.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/ffi/api/c_api.h"  // For CustomCallApiVersion

namespace xla::ascend {
namespace {

// Helper function to check if an instruction should be handled by Ascend backend
bool ShouldHandleByAscend(const HloInstruction* hlo) {
  // For now, we handle all instructions for Ascend backend
  // This can be refined based on specific requirements
  return true;
}

// Helper function to check if a fusion matches the broadcast-constant pattern
// Pattern: fusion { constant -> broadcast }
bool IsBroadcastConstantFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();
  
  // Must have exactly 2 instructions: constant + broadcast
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2) {
    VLOG(4) << "BroadcastConstantFusion: expected 2 instructions, got " 
            << instruction_count;
    return false;
  }
  
  // Find the constant and broadcast instructions
  const HloInstruction* constant_instr = nullptr;
  const HloInstruction* broadcast_instr = nullptr;
  
  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kConstant) {
      constant_instr = instr;
    } else if (instr->opcode() == HloOpcode::kBroadcast) {
      broadcast_instr = instr;
    } else {
      VLOG(4) << "BroadcastConstantFusion: unexpected opcode " 
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }
  
  // Both instructions must be present
  if (!constant_instr || !broadcast_instr) {
    VLOG(4) << "BroadcastConstantFusion: missing constant or broadcast instruction";
    return false;
  }
  
  // Broadcast must take constant as its only operand
  if (broadcast_instr->operand_count() != 1 || 
      broadcast_instr->operand(0) != constant_instr) {
    VLOG(4) << "BroadcastConstantFusion: broadcast does not take constant as operand";
    return false;
  }
  
  // Fusion's root must be the broadcast instruction
  if (computation->root_instruction() != broadcast_instr) {
    VLOG(4) << "BroadcastConstantFusion: fusion root is not broadcast";
    return false;
  }
  
  return true;
}

// Helper function to serialize broadcast-constant fusion metadata
std::string SerializeBroadcastConstantMetadata(
    const HloFusionInstruction* fusion,
    const HloInstruction* broadcast_instr,
    const HloInstruction* constant_instr) {
  // Serialize metadata needed for aclnnBroadcast:
  // - Constant value
  // - Broadcast dimensions
  // - Output shape
  
  std::string metadata;
  
  // Get the constant literal value
  const auto& literal = constant_instr->literal();
  absl::StrAppend(&metadata, "constant_value:", literal.ToString(), ";");
  
  // Get broadcast dimensions
  const auto& broadcast_dims = broadcast_instr->dimensions();
  absl::StrAppend(&metadata, "broadcast_dims:[");
  for (size_t i = 0; i < broadcast_dims.size(); ++i) {
    if (i > 0) absl::StrAppend(&metadata, ",");
    absl::StrAppend(&metadata, broadcast_dims[i]);
  }
  absl::StrAppend(&metadata, "];");
  
  // Output shape
  absl::StrAppend(&metadata, "output_shape:", fusion->shape().ToString(), ";");
  
  return metadata;
}

}  // namespace

ThunkEmitter::ThunkEmitter(
    xla::gpu::IrEmitterContext* absl_nonnull ir_emitter_context,
    xla::llvm_ir::LLVMCommandLineOptionsReleasableLock* absl_nonnull
        llvm_options_lock)
    : ir_emitter_context_(ir_emitter_context),
      send_recv_events_(std::make_shared<xla::gpu::HostSendRecvAsyncEvents>()),
      copy_events_(std::make_shared<xla::gpu::CopyThunk::AsyncEvents>()),
      call_graph_(xla::CallGraph::Build(&ir_emitter_context->hlo_module())),
      constants_module_(ir_emitter_context_->CreateLLVMModule(
          absl::StrCat(ir_emitter_context_->hlo_module().name(), "_consts"))),
      llvm_options_lock_(llvm_options_lock) {}

absl::StatusOr<BufferAllocation::Slice> ThunkEmitter::GetAllocationSliceForHlo(
    const HloInstruction* instr, const ShapeIndex& index) const {
  return ir_emitter_context_->buffer_assignment().GetUniqueSlice(instr, index);
}

absl::StatusOr<xla::ShapedSlice> ThunkEmitter::GetShapedSliceForHlo(
    const xla::HloInstruction* instr, const xla::ShapeIndex& index) const {
  TF_ASSIGN_OR_RETURN(xla::BufferAllocation::Slice slice,
                      GetAllocationSliceForHlo(instr, index));
  TF_ASSIGN_OR_RETURN(
      xla::Shape shape,
      ir_emitter_context_->buffer_assignment().GetShapeForUniqueSlice(instr,
                                                                      index));
  return xla::ShapedSlice{slice, shape};
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitConstant(
    const HloConstantInstruction* instr) {
  // For constants inside fusions, they are handled as part of the fusion's CustomCall
  // For top-level constants, we may need special handling
  VLOG(2) << "EmitConstant for Ascend: " << instr->name();
  
  // Constants are typically embedded in the fusion metadata
  // Return empty sequence - constants will be serialized in fusion metadata
  return xla::gpu::ThunkSequence{};
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitFusion(
    const HloFusionInstruction* fusion) {
  // Only handle kLoop fusion for now
  if (fusion->fusion_kind() != HloInstruction::FusionKind::kLoop) {
    VLOG(3) << "Ascend ThunkEmitter: fusion kind not handled: " 
            << static_cast<int>(fusion->fusion_kind());
    return xla::gpu::ThunkSequence{};
  }
  
  VLOG(2) << "EmitFusion (kLoop) for Ascend: " << fusion->name();
  
  // Try to match and emit specific FFI patterns
  // Pattern 1: broadcast-constant -> ascend.inplace_index_fill_tensor (for memset-like operation)
  if (IsBroadcastConstantFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitBroadcastConstantFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }
  
  // Add more FFI pattern matching here in the future
  // Pattern 2: ...
  
  VLOG(3) << "Ascend ThunkEmitter: no matching FFI pattern found for fusion";
  return xla::gpu::ThunkSequence{};
}

// Helper function to emit broadcast-constant fusion as aclnnBroadcast FFI call
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitBroadcastConstantFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting broadcast-constant fusion as ascend.full: " << fusion->name();
  
  // Get the output buffer allocation (this is the tensor to be filled)
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));
  
  // Extract constant value from the fusion computation
  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* constant_instr = nullptr;
  
  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant) {
      constant_instr = instr;
      break;
    }
  }
  
  if (!constant_instr) {
    return absl::InternalError("No constant instruction found in broadcast-constant fusion");
  }
  
  // Extract the fill value from the constant literal
  // The constant is a scalar (f32[]), so we get the first element
  const auto& literal = constant_instr->literal();
  float fill_value = literal.GetFirstElement<float>();
  
  VLOG(2) << "Broadcast-constant fusion: fill_value=" << fill_value;
  
  // Create operands and results for CustomCallThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  
  // Add the output slice as both operand and result (inplace operation)
  operands.push_back(output_slice);
  results.push_back(output_slice);
  
  // Create attributes map with fill_value
  xla::ffi::AttributesMap attributes;
  attributes["value"] = xla::ffi::Scalar(fill_value);
  
  // Get GPU compute capability
  const se::GpuComputeCapability& gpu_compute_capability = 
      ir_emitter_context_->gpu_compute_capability();
  
  // Create CustomCallThunk
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          "ascend.full.f32",
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));
  
  // Add the thunk to the sequence
  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));
  
  return sequence;
}

absl::StatusOr<std::optional<xla::gpu::ThunkSequence>> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* hlo) {
  
  // Check if this instruction should be handled by Ascend backend
  if (!ShouldHandleByAscend(hlo)) {
    return std::nullopt;
  }
  
  switch (hlo->opcode()) {
    case HloOpcode::kConstant: {
      auto* constant_instr = Cast<HloConstantInstruction>(hlo);
      TF_ASSIGN_OR_RETURN(auto thunks, EmitConstant(constant_instr));
      return thunks;
    }
    
    case HloOpcode::kFusion: {
      auto* fusion = Cast<HloFusionInstruction>(hlo);
      TF_ASSIGN_OR_RETURN(auto thunks, EmitFusion(fusion));
      
      // If EmitFusion returned an empty sequence, it means Ascend doesn't support this fusion
      // Return nullopt to let GPU emitter handle it
      if (thunks.empty()) {
        VLOG(3) << "Ascend ThunkEmitter: fusion not supported, delegating to GPU";
        return std::nullopt;
      }
      
      return thunks;
    }
    
    // Add more cases here as needed for top-level instructions
    // Note: Instructions inside fusions are NOT emitted separately - 
    // they are part of the fusion's CustomCall
    
    default:
      // For unhandled opcodes, return nullopt so GPU emitter can try
      VLOG(3) << "Ascend ThunkEmitter: opcode not handled at top level: " 
              << HloOpcodeString(hlo->opcode());
      return std::nullopt;
  }
}

// Helper function implementation
absl::StatusOr<std::optional<xla::gpu::ThunkSequence>> TryEmitHloInstructionAscend(
    const xla::HloInstruction* hlo,
    xla::gpu::IrEmitterContext* ir_emitter_context,
    xla::llvm_ir::LLVMCommandLineOptionsReleasableLock* llvm_options_lock) {
  
  // Check if we should handle this with Ascend backend
  // For now, check the platform name (case-insensitive)
  if (absl::AsciiStrToLower(ir_emitter_context->platform_name()) != "ascend") {
    return std::nullopt;
  }
  
  // Create an Ascend ThunkEmitter and try to emit
  ThunkEmitter emitter(ir_emitter_context, llvm_options_lock);
  return emitter.EmitHloInstruction(hlo);
}

}  // namespace xla::ascend
