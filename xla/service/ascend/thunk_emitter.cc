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

namespace xla::ascend {
namespace {

// Helper function to check if an instruction should be handled by Ascend backend
bool ShouldHandleByAscend(const HloInstruction* hlo) {
  // For now, we handle all instructions for Ascend backend
  // This can be refined based on specific requirements
  return true;
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
  // TODO: Implement constant emission using FFI
  // For now, return empty sequence as placeholder
  VLOG(2) << "EmitConstant not yet implemented for Ascend: " << instr->name();
  return xla::gpu::ThunkSequence{};
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitBroadcast(
    const xla::HloInstruction* hlo) {
  // TODO: Implement broadcast emission using FFI
  // This will create a CustomCallThunk that calls Ascend's broadcast kernel via FFI
  
  VLOG(2) << "EmitBroadcast for Ascend: " << hlo->name();
  
  // For the initial framework, we'll create a placeholder custom call thunk
  // In the future, this will be replaced with actual Ascend FFI calls
  
  // Get input and output slices
  TF_ASSIGN_OR_RETURN(xla::ShapedSlice input_slice, 
                      GetShapedSliceForHlo(hlo->operand(0)));
  TF_ASSIGN_OR_RETURN(xla::ShapedSlice output_slice, 
                      GetShapedSliceForHlo(hlo));
  
  // Create operands and results vectors for CustomCallThunk
  std::vector<xla::NullableShapedSlice> operands;
  std::vector<xla::NullableShapedSlice> results;
  
  operands.push_back(input_slice);
  results.push_back(output_slice);
  
  // Create a custom call thunk with FFI
  // The target name will be used to look up the FFI handler
  auto thunk = xla::gpu::CustomCallThunk::Create(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(
          hlo, ir_emitter_context_->GetNextThunkId()),
      "ascend_broadcast",  // Custom call target name for FFI lookup
      std::move(operands),
      std::move(results),
      "",  // opaque data (empty for FFI)
      xla::CustomCallApiVersion::API_VERSION_TYPED_FFI,
      ir_emitter_context_->platform_name());
  
  TF_RETURN_IF_ERROR(thunk.status());
  
  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(*thunk));
  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitFusion(
    const HloFusionInstruction* fusion) {
  // Handle fusion operations
  // For kLoop fusion containing broadcast, we need to process the fused computation
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    VLOG(2) << "EmitFusion (kLoop) for Ascend: " << fusion->name();
    // Process the fused computation
    return EmitHloComputation(fusion->fused_instructions_computation());
  }
  
  // For other fusion types, let GPU emitter handle them
  VLOG(3) << "Ascend ThunkEmitter: fusion kind not handled: " 
          << static_cast<int>(fusion->fusion_kind());
  return xla::gpu::ThunkSequence{};
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitHloComputation(
    const HloComputation* computation) {
  xla::gpu::ThunkSequence thunk_sequence;
  
  for (const auto* instruction : computation->MakeInstructionPostOrder()) {
    TF_ASSIGN_OR_RETURN(auto result, EmitHloInstruction(instruction));
    
    // If the instruction was handled, append its thunks
    if (result.has_value()) {
      thunk_sequence.insert(thunk_sequence.end(),
                           std::make_move_iterator(result->begin()),
                           std::make_move_iterator(result->end()));
    }
    // If not handled (nullopt), skip it (it will be handled by GPU emitter)
  }
  
  return thunk_sequence;
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
    
    case HloOpcode::kBroadcast: {
      // Check if this is a broadcast operation that we want to handle
      // For the test case: broadcast(%constant_1_1), dimensions={}
      TF_ASSIGN_OR_RETURN(auto thunks, EmitBroadcast(hlo));
      return thunks;
    }
    
    case HloOpcode::kFusion: {
      auto* fusion = Cast<HloFusionInstruction>(hlo);
      TF_ASSIGN_OR_RETURN(auto thunks, EmitFusion(fusion));
      // Return nullopt if fusion was not handled by Ascend
      if (thunks.empty() && fusion->fusion_kind() != HloInstruction::FusionKind::kLoop) {
        return std::nullopt;
      }
      return thunks;
    }
    
    // Add more cases here as needed
    // case HloOpcode::kAdd:
    // case HloOpcode::kMultiply:
    // etc.
    
    default:
      // For unhandled opcodes, return nullopt so GPU emitter can try
      VLOG(3) << "Ascend ThunkEmitter: opcode not handled: " 
              << HloOpcodeString(hlo->opcode());
      return std::nullopt;
  }
}

}  // namespace xla::ascend

// Helper function implementation
absl::StatusOr<std::optional<xla::gpu::ThunkSequence>> TryEmitHloInstructionAscend(
    const xla::HloInstruction* hlo,
    xla::gpu::IrEmitterContext* ir_emitter_context,
    xla::llvm_ir::LLVMCommandLineOptionsReleasableLock* llvm_options_lock) {
  
  // Check if we should handle this with Ascend backend
  // For now, check the platform name
  if (ir_emitter_context->platform_name() != "ascend") {
    return std::nullopt;
  }
  
  // Create an Ascend ThunkEmitter and try to emit
  xla::ascend::ThunkEmitter emitter(ir_emitter_context, llvm_options_lock);
  return emitter.EmitHloInstruction(hlo);
}
