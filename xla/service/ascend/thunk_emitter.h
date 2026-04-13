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

#ifndef XLA_SERVICE_ASCEND_THUNK_EMITTER_H_
#define XLA_SERVICE_ASCEND_THUNK_EMITTER_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"

namespace xla::ascend {

// Emits Thunks for Ascend backend using FFI mechanism.
class ThunkEmitter {
 public:
  absl::string_view platform_name() const {
    return ir_emitter_context_->platform_name();
  }

  explicit ThunkEmitter(
      xla::gpu::IrEmitterContext* absl_nonnull ir_emitter_context,
      llvm_ir::LLVMCommandLineOptionsReleasableLock* absl_nonnull
          llvm_options_lock);
  ThunkEmitter(const ThunkEmitter&) = delete;
  ThunkEmitter& operator=(const ThunkEmitter&) = delete;

  // Main entry point for emitting thunks for HLO instructions.
  // Returns std::nullopt if the instruction should be handled by another emitter.
  absl::StatusOr<std::optional<xla::gpu::ThunkSequence>> EmitHloInstruction(
      const HloInstruction* hlo);

 private:
  // Helper to get allocation slice for HLO
  absl::StatusOr<BufferAllocation::Slice> GetAllocationSliceForHlo(
      const HloInstruction* instr,
      const ShapeIndex& index = ShapeIndex{}) const;

  // Helper to get shaped slice for HLO
  absl::StatusOr<ShapedSlice> GetShapedSliceForHlo(
      const HloInstruction* instr,
      const ShapeIndex& index = ShapeIndex{}) const;

  // Wraps a thunk into a ThunkSequence
  xla::gpu::ThunkSequence GetThunkSequence(std::unique_ptr<xla::gpu::Thunk> thunk) {
    xla::gpu::ThunkSequence sequence;
    sequence.push_back(std::move(thunk));
    return sequence;
  }

  // Emit handlers for specific HLO opcodes
  absl::StatusOr<xla::gpu::ThunkSequence> EmitConstant(
      const HloConstantInstruction* instr);
      
  absl::StatusOr<xla::gpu::ThunkSequence> EmitFusion(
      const HloFusionInstruction* fusion);
  
  // Emit handlers for specific FFI patterns
  absl::StatusOr<xla::gpu::ThunkSequence> EmitBroadcastConstantFusion(
      const HloFusionInstruction* fusion);

  // Context and state
  xla::gpu::IrEmitterContext* ir_emitter_context_;
  std::shared_ptr<xla::gpu::HostSendRecvAsyncEvents> send_recv_events_;
  std::shared_ptr<xla::gpu::CopyThunk::AsyncEvents> copy_events_;
  std::unique_ptr<xla::CallGraph> call_graph_;
  std::unique_ptr<llvm::Module> constants_module_;
  xla::llvm_ir::LLVMCommandLineOptionsReleasableLock* llvm_options_lock_;
  std::vector<std::unique_ptr<llvm::Module>> kernel_modules_;
};

// Helper function to try emitting HLO instruction with Ascend backend.
// This function is called from GPU's ThunkEmitter to delegate to Ascend when appropriate.
absl::StatusOr<std::optional<xla::gpu::ThunkSequence>> TryEmitHloInstructionAscend(
    const xla::HloInstruction* hlo,
    xla::gpu::IrEmitterContext* ir_emitter_context,
    xla::llvm_ir::LLVMCommandLineOptionsReleasableLock* llvm_options_lock);

}  // namespace xla::ascend

#endif  // XLA_SERVICE_ASCEND_THUNK_EMITTER_H_
