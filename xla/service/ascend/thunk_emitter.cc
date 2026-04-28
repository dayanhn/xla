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
#include "xla/backends/ascend/runtime/aclnn_thunk.h"
#include "xla/backends/ascend/transforms/aclnn_config.h"
#include "xla/backends/ascend/transforms/aclnn_targets.h"
#include "xla/service/gpu/cublas_cudnn.h"
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

// Helper function to check if a fusion is a tensor broadcast fusion (parameter + broadcast)
bool IsTensorBroadcastFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();
  
  // Must have exactly 2 instructions: parameter + broadcast
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2) {
    VLOG(4) << "TensorBroadcastFusion: expected 2 instructions, got " 
            << instruction_count;
    return false;
  }
  
  // Find the parameter and broadcast instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* broadcast_instr = nullptr;
  
  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kBroadcast) {
      broadcast_instr = instr;
    } else {
      VLOG(4) << "TensorBroadcastFusion: unexpected opcode " 
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }
  
  // Both instructions must be present
  if (!param_instr || !broadcast_instr) {
    VLOG(4) << "TensorBroadcastFusion: missing parameter or broadcast instruction";
    return false;
  }
  
  // Broadcast must take parameter as its only operand
  if (broadcast_instr->operand_count() != 1 || 
      broadcast_instr->operand(0) != param_instr) {
    VLOG(4) << "TensorBroadcastFusion: broadcast does not take parameter as operand";
    return false;
  }
  
  // Fusion's root must be the broadcast instruction
  if (computation->root_instruction() != broadcast_instr) {
    VLOG(4) << "TensorBroadcastFusion: fusion root is not broadcast";
    return false;
  }
  
  return true;
}

// Helper function to check if a fusion matches the tanh pattern
// Pattern: fusion { parameter -> tanh }
bool IsTanhFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();
  
  // Must have exactly 2 instructions: parameter + tanh
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2) {
    VLOG(4) << "TanhFusion: expected 2 instructions, got " 
            << instruction_count;
    return false;
  }
  
  // Find the parameter and tanh instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* tanh_instr = nullptr;
  
  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kTanh) {
      tanh_instr = instr;
    } else {
      VLOG(4) << "TanhFusion: unexpected opcode " 
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }
  
  // Both instructions must be present
  if (!param_instr || !tanh_instr) {
    VLOG(4) << "TanhFusion: missing parameter or tanh instruction";
    return false;
  }
  
  // Tanh must take parameter as its only operand
  if (tanh_instr->operand_count() != 1 || 
      tanh_instr->operand(0) != param_instr) {
    VLOG(4) << "TanhFusion: tanh does not take parameter as operand";
    return false;
  }
  
  // Fusion's root must be the tanh instruction
  if (computation->root_instruction() != tanh_instr) {
    VLOG(4) << "TanhFusion: fusion root is not tanh";
    return false;
  }
  
  // Check if it's a supported data type
  PrimitiveType input_type = tanh_instr->operand(0)->shape().element_type();
  PrimitiveType output_type = tanh_instr->shape().element_type();
  
  // Check if input and output types are the same (tanh is element-wise and preserves type)
  if (input_type != output_type) {
    VLOG(4) << "TanhFusion: input and output types must be the same";
    return false;
  }
  
  // Check if it's a supported data type based on aclnnTanh documentation
  bool is_supported = false;
  
  // Supported input/output types for tanh
  if (input_type == PrimitiveType::F32 ||
      input_type == PrimitiveType::F16 ||
      input_type == PrimitiveType::BF16) {
    is_supported = true;
  }
  
  if (!is_supported) {
    VLOG(4) << "TanhFusion: unsupported data type: " 
            << PrimitiveType_Name(input_type);
    return false;
  }
  
  return true;
}

// Helper function to check if a fusion matches the sqrt pattern
// Pattern: fusion { parameter -> sqrt }
bool IsSqrtFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 2 instructions: parameter + sqrt
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2) {
    VLOG(4) << "SqrtFusion: expected 2 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameter and sqrt instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* sqrt_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kSqrt) {
      sqrt_instr = instr;
    } else {
      VLOG(4) << "SqrtFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Both instructions must be present
  if (!param_instr || !sqrt_instr) {
    VLOG(4) << "SqrtFusion: missing parameter or sqrt instruction";
    return false;
  }

  // Sqrt must take parameter as its only operand
  if (sqrt_instr->operand_count() != 1 ||
      sqrt_instr->operand(0) != param_instr) {
    VLOG(4) << "SqrtFusion: sqrt does not take parameter as operand";
    return false;
  }

  // Fusion's root must be the sqrt instruction
  if (computation->root_instruction() != sqrt_instr) {
    VLOG(4) << "SqrtFusion: fusion root is not sqrt";
    return false;
  }

  // Check if it's a supported data type based on aclnnSqrt documentation
  PrimitiveType input_type = sqrt_instr->operand(0)->shape().element_type();
  PrimitiveType output_type = sqrt_instr->shape().element_type();

  // Check if input and output types are the same (sqrt is element-wise and preserves type)
  if (input_type != output_type) {
    VLOG(4) << "SqrtFusion: input and output types must be the same";
    return false;
  }

  // Supported input/output types for sqrt based on aclnnSqrt documentation
  // FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128
  bool is_supported = false;
  if (input_type == PrimitiveType::F32 ||
      input_type == PrimitiveType::F16 ||
      input_type == PrimitiveType::BF16 ||
      input_type == PrimitiveType::F64 ||
      input_type == PrimitiveType::S32 ||
      input_type == PrimitiveType::S64 ||
      input_type == PrimitiveType::S16 ||
      input_type == PrimitiveType::S8 ||
      input_type == PrimitiveType::U8 ||
      input_type == PrimitiveType::PRED) {
    is_supported = true;
  }

  if (!is_supported) {
    VLOG(4) << "SqrtFusion: unsupported data type: "
            << PrimitiveType_Name(input_type);
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the convolution pattern
// Pattern: fusion { parameter(0) -> parameter(1) -> convolution }
bool IsConvolutionFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();
  
  // Must have exactly 3 instructions: parameter(0) + parameter(1) + convolution
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "ConvolutionFusion: expected 3 instructions, got " 
            << instruction_count;
    return false;
  }
  
  // Find the parameter and convolution instructions
  const HloInstruction* param0_instr = nullptr;
  const HloInstruction* param1_instr = nullptr;
  const HloInstruction* conv_instr = nullptr;
  
  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (instr->parameter_number() == 0) {
        param0_instr = instr;
      } else if (instr->parameter_number() == 1) {
        param1_instr = instr;
      } else {
        VLOG(4) << "ConvolutionFusion: unexpected parameter number " 
                << instr->parameter_number();
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kConvolution) {
      conv_instr = instr;
    } else {
      VLOG(4) << "ConvolutionFusion: unexpected opcode " 
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }
  
  // All instructions must be present
  if (!param0_instr || !param1_instr || !conv_instr) {
    VLOG(4) << "ConvolutionFusion: missing parameter or convolution instruction";
    return false;
  }
  
  // Convolution must take both parameters as operands
  if (conv_instr->operand_count() != 2 || 
      conv_instr->operand(0) != param0_instr ||
      conv_instr->operand(1) != param1_instr) {
    VLOG(4) << "ConvolutionFusion: convolution does not take both parameters as operands";
    return false;
  }
  
  // Fusion's root must be the convolution instruction
  if (computation->root_instruction() != conv_instr) {
    VLOG(4) << "ConvolutionFusion: fusion root is not convolution";
    return false;
  }
  
  // Check if it's a supported data type
  PrimitiveType input_type = param0_instr->shape().element_type();
  PrimitiveType weight_type = param1_instr->shape().element_type();
  PrimitiveType output_type = conv_instr->shape().element_type();
  
  // Check if input and weight types are compatible
  bool is_supported = false;
  if ((input_type == PrimitiveType::F32 ||
       input_type == PrimitiveType::F16 ||
       input_type == PrimitiveType::BF16) &&
      (weight_type == PrimitiveType::F32 ||
       weight_type == PrimitiveType::F16 ||
       weight_type == PrimitiveType::BF16)) {
    is_supported = true;
  }
  
  if (!is_supported) {
    VLOG(4) << "ConvolutionFusion: unsupported data types: input=" 
            << PrimitiveType_Name(input_type) << ", weight=" 
            << PrimitiveType_Name(weight_type);
    return false;
  }
  
  return true;
}

// Helper function to check if a fusion matches the convert-element-type pattern
// Pattern: fusion { parameter -> convert-element-type }
bool IsConvertFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();
  
  // Must have exactly 2 instructions: parameter + convert
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2) {
    VLOG(4) << "ConvertFusion: expected 2 instructions, got " 
            << instruction_count;
    return false;
  }
  
  // Find the parameter and convert instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* convert_instr = nullptr;
  
  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kConvert) {
      convert_instr = instr;
    } else {
      VLOG(4) << "ConvertFusion: unexpected opcode " 
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }
  
  // Both instructions must be present
  if (!param_instr || !convert_instr) {
    VLOG(4) << "ConvertFusion: missing parameter or convert instruction";
    return false;
  }
  
  // Convert must take parameter as its only operand
  if (convert_instr->operand_count() != 1 || 
      convert_instr->operand(0) != param_instr) {
    VLOG(4) << "ConvertFusion: convert does not take parameter as operand";
    return false;
  }
  
  // Fusion's root must be the convert instruction
  if (computation->root_instruction() != convert_instr) {
    VLOG(4) << "ConvertFusion: fusion root is not convert";
    return false;
  }
  
  // Check if it's a supported conversion type
  PrimitiveType src_type = convert_instr->operand(0)->shape().element_type();
  PrimitiveType dst_type = convert_instr->shape().element_type();
  
  // Check if it's one of the supported conversion types
  // Supported conversions are based on aclnnCast documentation
  bool is_supported = false;
  
  // Check for supported conversion types
  if ((src_type == PrimitiveType::S32 && dst_type == PrimitiveType::U32) ||
      (src_type == PrimitiveType::U8 && dst_type == PrimitiveType::S32) ||
      (src_type == PrimitiveType::F32 && dst_type == PrimitiveType::F16) ||
      (src_type == PrimitiveType::F16 && dst_type == PrimitiveType::F32) ||
      (src_type == PrimitiveType::F32 && dst_type == PrimitiveType::BF16) ||
      (src_type == PrimitiveType::BF16 && dst_type == PrimitiveType::F32) ||
      (src_type == PrimitiveType::S32 && dst_type == PrimitiveType::F32) ||
      (src_type == PrimitiveType::F32 && dst_type == PrimitiveType::S32) ||
      (src_type == PrimitiveType::S64 && dst_type == PrimitiveType::F32) ||
      (src_type == PrimitiveType::F32 && dst_type == PrimitiveType::S64) ||
      (src_type == PrimitiveType::U32 && dst_type == PrimitiveType::F32) ||
      (src_type == PrimitiveType::F32 && dst_type == PrimitiveType::U32) ||
      (src_type == PrimitiveType::U64 && dst_type == PrimitiveType::F32) ||
      (src_type == PrimitiveType::F32 && dst_type == PrimitiveType::U64) ||
      (src_type == PrimitiveType::S8 && dst_type == PrimitiveType::S32) ||
      (src_type == PrimitiveType::S32 && dst_type == PrimitiveType::S8) ||
      (src_type == PrimitiveType::U8 && dst_type == PrimitiveType::U32) ||
      (src_type == PrimitiveType::U32 && dst_type == PrimitiveType::U8) ||
      (src_type == PrimitiveType::PRED && dst_type == PrimitiveType::S32) ||
      (src_type == PrimitiveType::PRED && dst_type == PrimitiveType::F32) ||
      (src_type == PrimitiveType::PRED && dst_type == PrimitiveType::F16) ||
      (src_type == PrimitiveType::PRED && dst_type == PrimitiveType::BF16) ||
      (src_type == PrimitiveType::S32 && dst_type == PrimitiveType::PRED)) {
    is_supported = true;
  }
  
  if (!is_supported) {
    VLOG(4) << "ConvertFusion: unsupported conversion type: " 
            << PrimitiveType_Name(src_type) << " -> " << PrimitiveType_Name(dst_type);
    return false;
  }
  
  return true;
}

// Helper function to check if a fusion matches the shift-right-logical pattern
// Pattern: fusion { parameter + constant -> shift-right-logical }
bool IsShiftRightFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();
  
  // Must have exactly 3 instructions: parameter + constant + shift-right-logical
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "ShiftRightFusion: expected 3 instructions, got " 
            << instruction_count;
    return false;
  }
  
  // Find the parameter, constant, and shift-right-logical instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* constant_instr = nullptr;
  const HloInstruction* shift_instr = nullptr;
  
  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kConstant) {
      constant_instr = instr;
    } else if (instr->opcode() == HloOpcode::kShiftRightLogical) {
      shift_instr = instr;
    } else {
      VLOG(4) << "ShiftRightFusion: unexpected opcode " 
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }
  
  // All instructions must be present
  if (!param_instr || !constant_instr || !shift_instr) {
    VLOG(4) << "ShiftRightFusion: missing parameter, constant, or shift instruction";
    return false;
  }
  
  // Shift must take parameter and constant as operands
  if (shift_instr->operand_count() != 2 || 
      shift_instr->operand(0) != param_instr ||
      shift_instr->operand(1) != constant_instr) {
    VLOG(4) << "ShiftRightFusion: shift does not take parameter and constant as operands";
    return false;
  }
  
  // Fusion's root must be the shift instruction
  if (computation->root_instruction() != shift_instr) {
    VLOG(4) << "ShiftRightFusion: fusion root is not shift";
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

// Helper function to check if a fusion matches the add pattern
// Pattern: fusion { parameter0 + parameter1 -> add }
bool IsAddFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 3 instructions: parameter0 + parameter1 + add
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "AddFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameters and add instructions
  const HloInstruction* param0_instr = nullptr;
  const HloInstruction* param1_instr = nullptr;
  const HloInstruction* add_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (!param0_instr) {
        param0_instr = instr;
      } else if (!param1_instr) {
        param1_instr = instr;
      } else {
        VLOG(4) << "AddFusion: too many parameter instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kAdd) {
      add_instr = instr;
    } else {
      VLOG(4) << "AddFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // All instructions must be present
  if (!param0_instr || !param1_instr || !add_instr) {
    VLOG(4) << "AddFusion: missing parameter or add instruction";
    return false;
  }

  // Add must take both parameters as operands
  if (add_instr->operand_count() != 2 ||
      (add_instr->operand(0) != param0_instr && add_instr->operand(0) != param1_instr) ||
      (add_instr->operand(1) != param0_instr && add_instr->operand(1) != param1_instr)) {
    VLOG(4) << "AddFusion: add does not take both parameters as operands";
    return false;
  }

  // Fusion's root must be the add instruction
  if (computation->root_instruction() != add_instr) {
    VLOG(4) << "AddFusion: fusion root is not add";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the maximum pattern
// Pattern: fusion { parameter0 + parameter1 -> maximum }
bool IsMaximumFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 3 instructions: parameter0 + parameter1 + maximum
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "MaximumFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameters and maximum instructions
  const HloInstruction* param0_instr = nullptr;
  const HloInstruction* param1_instr = nullptr;
  const HloInstruction* maximum_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (!param0_instr) {
        param0_instr = instr;
      } else if (!param1_instr) {
        param1_instr = instr;
      } else {
        VLOG(4) << "MaximumFusion: too many parameter instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kMaximum) {
      maximum_instr = instr;
    } else {
      VLOG(4) << "MaximumFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // All instructions must be present
  if (!param0_instr || !param1_instr || !maximum_instr) {
    VLOG(4) << "MaximumFusion: missing parameter or maximum instruction";
    return false;
  }

  // Maximum must take both parameters as operands
  if (maximum_instr->operand_count() != 2 ||
      (maximum_instr->operand(0) != param0_instr && maximum_instr->operand(0) != param1_instr) ||
      (maximum_instr->operand(1) != param0_instr && maximum_instr->operand(1) != param1_instr)) {
    VLOG(4) << "MaximumFusion: maximum does not take both parameters as operands";
    return false;
  }

  // Fusion's root must be the maximum instruction
  if (computation->root_instruction() != maximum_instr) {
    VLOG(4) << "MaximumFusion: fusion root is not maximum";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the reduce_max pattern
// Pattern: fusion { parameter -> reduce_max } with maximum reduction function
bool IsArgMaxFusion(const HloFusionInstruction* fusion) {
  // 1. 输出必须是二元组 (f32, s32)
  if (!fusion->shape().IsTuple() || fusion->shape().tuple_shapes().size() != 2) {
    VLOG(4) << "ArgMaxFusion: fusion output is not a tuple with two elements";
    return false;
  }

  // 2. 必须是 2 个输入
  if (fusion->operand_count() != 2) {
    VLOG(4) << "ArgMaxFusion: fusion does not have 2 operands";
    return false;
  }

  // 3. 内部必须有一个 reduce
  const HloInstruction* reduce_instr = nullptr;
  for (const auto* instr : fusion->fused_instructions_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kReduce) {
      reduce_instr = instr;
      break;
    }
  }
  if (!reduce_instr) {
    VLOG(4) << "ArgMaxFusion: no reduce instruction found";
    return false;
  }

  // 4. reduce 必须输出二元组
  if (!reduce_instr->shape().IsTuple() || reduce_instr->shape().tuple_shapes().size() != 2) {
    VLOG(4) << "ArgMaxFusion: reduce output is not a tuple with two elements";
    return false;
  }

  // 5. reduce 必须有 4 个操作数
  if (reduce_instr->operand_count() != 4) {
    VLOG(4) << "ArgMaxFusion: reduce does not have 4 operands";
    return false;
  }

  // 6. 检查 reducer 内部必需算子
  const HloComputation* reduction_computation = reduce_instr->to_apply();
  if (!reduction_computation) {
    VLOG(4) << "ArgMaxFusion: no reduction computation found";
    return false;
  }

  bool has_gt = false, has_lt = false, has_select = false, has_tuple = false;
  for (const auto* instr : reduction_computation->instructions()) {
    if (instr->opcode() == HloOpcode::kCompare) {
      auto direction = instr->comparison_direction();
      if (direction == ComparisonDirection::kGt) {
        has_gt = true;
      } else if (direction == ComparisonDirection::kLt) {
        has_lt = true;
      }
    } else if (instr->opcode() == HloOpcode::kSelect) {
      has_select = true;
    } else if (instr->opcode() == HloOpcode::kTuple) {
      has_tuple = true;
    }
  }

  if (!has_gt || !has_lt || !has_select || !has_tuple) {
    VLOG(4) << "ArgMaxFusion: missing required operations in reduction computation";
    return false;
  }

  return true;
}

bool IsReduceMaxFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 3 instructions: parameter + constant + reduce
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "ReduceMaxFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameter, constant, and reduce instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* constant_instr = nullptr;
  const HloInstruction* reduce_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kConstant) {
      constant_instr = instr;
    } else if (instr->opcode() == HloOpcode::kReduce) {
      reduce_instr = instr;
    } else {
      VLOG(4) << "ReduceMaxFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // All instructions must be present
  if (!param_instr || !constant_instr || !reduce_instr) {
    VLOG(4) << "ReduceMaxFusion: missing parameter, constant, or reduce instruction";
    return false;
  }

  // Reduce must take parameter and constant as operands
  if (reduce_instr->operand_count() != 2 ||
      reduce_instr->operand(0) != param_instr ||
      reduce_instr->operand(1) != constant_instr) {
    VLOG(4) << "ReduceMaxFusion: reduce does not take parameter and constant as operands";
    return false;
  }

  // Check if the reduction function is maximum
  const HloComputation* reduction_computation = reduce_instr->to_apply();
  if (!reduction_computation) {
    VLOG(4) << "ReduceMaxFusion: no reduction computation found";
    return false;
  }

  const HloInstruction* root_instr = reduction_computation->root_instruction();
  if (!root_instr || root_instr->opcode() != HloOpcode::kMaximum) {
    VLOG(4) << "ReduceMaxFusion: reduction function is not maximum";
    return false;
  }

  // Fusion's root must be the reduce instruction
  if (computation->root_instruction() != reduce_instr) {
    VLOG(4) << "ReduceMaxFusion: fusion root is not reduce";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the subtract pattern
// Pattern: fusion { parameter0 + parameter1 -> subtract }
bool IsSubtractFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 3 instructions: parameter0 + parameter1 + subtract
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "SubtractFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameters and subtract instructions
  const HloInstruction* param0_instr = nullptr;
  const HloInstruction* param1_instr = nullptr;
  const HloInstruction* subtract_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (!param0_instr) {
        param0_instr = instr;
      } else if (!param1_instr) {
        param1_instr = instr;
      } else {
        VLOG(4) << "SubtractFusion: too many parameter instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kSubtract) {
      subtract_instr = instr;
    } else {
      VLOG(4) << "SubtractFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // All instructions must be present
  if (!param0_instr || !param1_instr || !subtract_instr) {
    VLOG(4) << "SubtractFusion: missing parameter or subtract instruction";
    return false;
  }

  // Subtract must take both parameters as operands
  if (subtract_instr->operand_count() != 2 ||
      (subtract_instr->operand(0) != param0_instr && subtract_instr->operand(0) != param1_instr) ||
      (subtract_instr->operand(1) != param0_instr && subtract_instr->operand(1) != param1_instr)) {
    VLOG(4) << "SubtractFusion: subtract does not take both parameters as operands";
    return false;
  }

  // Fusion's root must be the subtract instruction
  if (computation->root_instruction() != subtract_instr) {
    VLOG(4) << "SubtractFusion: fusion root is not subtract";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the exponential pattern
// Pattern: fusion { parameter -> exponential }
bool IsExponentialFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 2 instructions: parameter + exponential
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2) {
    VLOG(4) << "ExponentialFusion: expected 2 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameter and exponential instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* exponential_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kExp) {
      exponential_instr = instr;
    } else {
      VLOG(4) << "ExponentialFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Both instructions must be present
  if (!param_instr || !exponential_instr) {
    VLOG(4) << "ExponentialFusion: missing parameter or exponential instruction";
    return false;
  }

  // Exponential must take parameter as its only operand
  if (exponential_instr->operand_count() != 1 ||
      exponential_instr->operand(0) != param_instr) {
    VLOG(4) << "ExponentialFusion: exponential does not take parameter as operand";
    return false;
  }

  // Fusion's root must be the exponential instruction
  if (computation->root_instruction() != exponential_instr) {
    VLOG(4) << "ExponentialFusion: fusion root is not exponential";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the reduce_sum pattern
// Pattern: fusion { parameter + constant -> reduce_sum } with add reduction function
bool IsReduceSumFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Count only direct children instructions (not nested in sub-computations)
  int64_t instruction_count = 0;
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* constant_instr = nullptr;
  const HloInstruction* reduce_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    // Skip instructions that belong to nested computations
    if (instr->parent() != computation) {
      continue;
    }
    
    instruction_count++;
    
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kConstant) {
      constant_instr = instr;
    } else if (instr->opcode() == HloOpcode::kReduce) {
      reduce_instr = instr;
    } else {
      VLOG(4) << "ReduceSumFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Must have exactly 3 instructions: parameter + constant + reduce
  if (instruction_count != 3) {
    VLOG(4) << "ReduceSumFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // All instructions must be present
  if (!param_instr || !constant_instr || !reduce_instr) {
    VLOG(4) << "ReduceSumFusion: missing parameter, constant, or reduce instruction";
    return false;
  }

  // Reduce must take parameter and constant as operands
  if (reduce_instr->operand_count() != 2 ||
      reduce_instr->operand(0) != param_instr ||
      reduce_instr->operand(1) != constant_instr) {
    VLOG(4) << "ReduceSumFusion: reduce does not take parameter and constant as operands";
    return false;
  }

  // Check if the reduction function is add
  const HloComputation* reduction_computation = reduce_instr->to_apply();
  if (!reduction_computation) {
    VLOG(4) << "ReduceSumFusion: no reduction computation found";
    return false;
  }

  const HloInstruction* root_instr = reduction_computation->root_instruction();
  if (!root_instr || root_instr->opcode() != HloOpcode::kAdd) {
    VLOG(4) << "ReduceSumFusion: reduction function is not add";
    return false;
  }

  // Fusion's root must be the reduce instruction
  if (computation->root_instruction() != reduce_instr) {
    VLOG(4) << "ReduceSumFusion: fusion root is not reduce";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the equal pattern
// Pattern: fusion { parameter0 + parameter1 -> compare with EQ direction }
bool IsEqualFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 3 instructions: parameter0 + parameter1 + compare
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "EqualFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameters and compare instructions
  const HloInstruction* param0_instr = nullptr;
  const HloInstruction* param1_instr = nullptr;
  const HloInstruction* compare_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (!param0_instr) {
        param0_instr = instr;
      } else if (!param1_instr) {
        param1_instr = instr;
      } else {
        VLOG(4) << "EqualFusion: too many parameter instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kCompare) {
      compare_instr = instr;
    } else {
      VLOG(4) << "EqualFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // All instructions must be present
  if (!param0_instr || !param1_instr || !compare_instr) {
    VLOG(4) << "EqualFusion: missing parameter or compare instruction";
    return false;
  }

  // Compare must take both parameters as operands
  if (compare_instr->operand_count() != 2 ||
      (compare_instr->operand(0) != param0_instr && compare_instr->operand(0) != param1_instr) ||
      (compare_instr->operand(1) != param0_instr && compare_instr->operand(1) != param1_instr)) {
    VLOG(4) << "EqualFusion: compare does not take both parameters as operands";
    return false;
  }

  // Compare must have EQ direction
  if (compare_instr->comparison_direction() != ComparisonDirection::kEq) {
    VLOG(4) << "EqualFusion: compare direction is not EQ";
    return false;
  }

  // Fusion's root must be the compare instruction
  if (computation->root_instruction() != compare_instr) {
    VLOG(4) << "EqualFusion: fusion root is not compare";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the select pattern
// Pattern: fusion { parameter0 + parameter1 + parameter2 -> select }
bool IsSelectFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 4 instructions: parameter0 + parameter1 + parameter2 + select
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 4) {
    VLOG(4) << "SelectFusion: expected 4 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameters and select instructions
  int param_count = 0;
  const HloInstruction* select_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_count++;
    } else if (instr->opcode() == HloOpcode::kSelect) {
      select_instr = instr;
    } else {
      VLOG(4) << "SelectFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Must have exactly 3 parameters and 1 select
  if (param_count != 3 || !select_instr) {
    VLOG(4) << "SelectFusion: missing parameters or select instruction";
    return false;
  }

  // Select must take 3 operands (condition, x, y)
  if (select_instr->operand_count() != 3) {
    VLOG(4) << "SelectFusion: select does not take 3 operands";
    return false;
  }

  // Fusion's root must be the select instruction
  if (computation->root_instruction() != select_instr) {
    VLOG(4) << "SelectFusion: fusion root is not select";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the negate pattern
// Pattern: fusion { parameter -> negate }
bool IsNegateFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 2 instructions: parameter + negate
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2) {
    VLOG(4) << "NegateFusion: expected 2 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameter and negate instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* negate_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kNegate) {
      negate_instr = instr;
    } else {
      VLOG(4) << "NegateFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Both instructions must be present
  if (!param_instr || !negate_instr) {
    VLOG(4) << "NegateFusion: missing parameter or negate instruction";
    return false;
  }

  // Negate must take parameter as its only operand
  if (negate_instr->operand_count() != 1 ||
      negate_instr->operand(0) != param_instr) {
    VLOG(4) << "NegateFusion: negate does not take parameter as operand";
    return false;
  }

  // Fusion's root must be the negate instruction
  if (computation->root_instruction() != negate_instr) {
    VLOG(4) << "NegateFusion: fusion root is not negate";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the divide pattern
// Pattern: fusion { parameter0 + parameter1 -> divide }
bool IsDivideFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 3 instructions: parameter0 + parameter1 + divide
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "DivideFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameters and divide instructions
  const HloInstruction* param0_instr = nullptr;
  const HloInstruction* param1_instr = nullptr;
  const HloInstruction* divide_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (!param0_instr) {
        param0_instr = instr;
      } else if (!param1_instr) {
        param1_instr = instr;
      } else {
        VLOG(4) << "DivideFusion: too many parameter instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kDivide) {
      divide_instr = instr;
    } else {
      VLOG(4) << "DivideFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // All instructions must be present
  if (!param0_instr || !param1_instr || !divide_instr) {
    VLOG(4) << "DivideFusion: missing parameter or divide instruction";
    return false;
  }

  // Divide must take both parameters as operands
  if (divide_instr->operand_count() != 2 ||
      (divide_instr->operand(0) != param0_instr && divide_instr->operand(0) != param1_instr) ||
      (divide_instr->operand(1) != param0_instr && divide_instr->operand(1) != param1_instr)) {
    VLOG(4) << "DivideFusion: divide does not take both parameters as operands";
    return false;
  }

  // Fusion's root must be the divide instruction
  if (computation->root_instruction() != divide_instr) {
    VLOG(4) << "DivideFusion: fusion root is not divide";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the max pool 2D pattern
// Pattern:
//   fusion { parameter -> reduce-window(parameter, init_value), window={...}, to_apply=max }
// Supports only max pooling with same padding
bool IsMaxPoolFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  const auto& instructions = computation->instructions();

  // Collect all parameters and the reduce-window instruction
  std::vector<const HloInstruction*> params;
  const HloInstruction* reduce_window_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      params.push_back(instr);
    } else if (instr->opcode() == HloOpcode::kReduceWindow) {
      reduce_window_instr = instr;
    } else if (instr->opcode() == HloOpcode::kConstant) {
      // init value constant - allowed
    } else {
      VLOG(4) << "MaxPoolFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Must have exactly 1 parameter
  if (params.size() != 1) {
    VLOG(4) << "MaxPoolFusion: need exactly 1 parameter, got "
            << params.size();
    return false;
  }

  // Must have the reduce-window instruction
  if (!reduce_window_instr) {
    VLOG(4) << "MaxPoolFusion: missing reduce-window instruction";
    return false;
  }

  // Check that reduce-window operand is the parameter
  if (reduce_window_instr->operand(0) != params[0]) {
    VLOG(4) << "MaxPoolFusion: reduce-window does not take parameter as operand";
    return false;
  }

  // Fusion's root must be the reduce-window instruction
  if (computation->root_instruction() != reduce_window_instr) {
    VLOG(4) << "MaxPoolFusion: fusion root is not reduce-window";
    return false;
  }

  // Check data type support
  PrimitiveType dtype = reduce_window_instr->shape().element_type();
  bool is_supported = (dtype == PrimitiveType::F32 ||
                      dtype == PrimitiveType::F16 ||
                      dtype == PrimitiveType::BF16);

  if (!is_supported) {
    VLOG(4) << "MaxPoolFusion: unsupported data type: "
            << PrimitiveType_Name(dtype);
    return false;
  }

  // Check the to_apply computation is maximum
  auto* to_apply = reduce_window_instr->to_apply();
  if (to_apply->root_instruction()->opcode() != HloOpcode::kMaximum) {
    VLOG(4) << "MaxPoolFusion: reduce-window must use maximum, got "
            << HloOpcodeString(to_apply->root_instruction()->opcode());
    return false;
  }

  // Check window dimensions are valid for max pool 2D
  const Window& window = reduce_window_instr->window();
  if (window.dimensions().size() != 4) {
    VLOG(4) << "MaxPoolFusion: only 4D input is supported, got "
            << window.dimensions().size() << "D";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the concatenate pattern
// Pattern:
//   fusion { parameter0, parameter1, ... -> concatenate(parameter0, parameter1, ...), dimensions={N} }
// Supports 2 or more tensors to concatenate
bool IsConcatenateFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  const auto& instructions = computation->instructions();

  // Collect all parameters and the concatenate instruction
  std::vector<const HloInstruction*> params;
  const HloInstruction* concatenate_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      params.push_back(instr);
    } else if (instr->opcode() == HloOpcode::kConcatenate) {
      concatenate_instr = instr;
    } else {
      VLOG(4) << "ConcatenateFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Must have at least 2 parameters (for concatenation)
  if (params.size() < 2) {
    VLOG(4) << "ConcatenateFusion: need at least 2 parameters, got "
            << params.size();
    return false;
  }

  // Must have the concatenate instruction
  if (!concatenate_instr) {
    VLOG(4) << "ConcatenateFusion: missing concatenate instruction";
    return false;
  }

  // Check concatenate operands match parameters
  if (concatenate_instr->operand_count() != params.size()) {
    VLOG(4) << "ConcatenateFusion: concatenate operand count "
            << concatenate_instr->operand_count()
            << " doesn't match parameter count " << params.size();
    return false;
  }

  // Verify each operand of concatenate is a parameter
  for (int i = 0; i < concatenate_instr->operand_count(); ++i) {
    bool found = false;
    for (const auto* param : params) {
      if (concatenate_instr->operand(i) == param) {
        found = true;
        break;
      }
    }
    if (!found) {
      VLOG(4) << "ConcatenateFusion: concatenate operand " << i
              << " is not a parameter";
      return false;
    }
  }

  // Fusion's root must be the concatenate instruction
  if (computation->root_instruction() != concatenate_instr) {
    VLOG(4) << "ConcatenateFusion: fusion root is not concatenate";
    return false;
  }

  // Check data type support (aclnnCat supports multiple types)
  PrimitiveType dtype = concatenate_instr->shape().element_type();
  bool is_supported = (dtype == PrimitiveType::F32 ||
                      dtype == PrimitiveType::F16 ||
                      dtype == PrimitiveType::BF16 ||
                      dtype == PrimitiveType::S32 ||
                      dtype == PrimitiveType::S64 ||
                      dtype == PrimitiveType::U32 ||
                      dtype == PrimitiveType::U64 ||
                      dtype == PrimitiveType::U8 ||
                      dtype == PrimitiveType::S8 ||
                      dtype == PrimitiveType::PRED);

  if (!is_supported) {
    VLOG(4) << "ConcatenateFusion: unsupported data type: "
            << PrimitiveType_Name(dtype);
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the multiply pattern
// Patterns:
// 1. fusion { parameter0 + parameter1 -> multiply } (two different tensors)
// 2. fusion { parameter0 -> multiply(parameter0, parameter0) } (same tensor squared)
bool IsMultiplyFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Check instruction count: either 2 (single parameter squared) or 3 (two parameters)
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2 && instruction_count != 3) {
    VLOG(4) << "MultiplyFusion: expected 2 or 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the instructions
  const HloInstruction* param0_instr = nullptr;
  const HloInstruction* param1_instr = nullptr;
  const HloInstruction* multiply_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (!param0_instr) {
        param0_instr = instr;
      } else if (!param1_instr) {
        param1_instr = instr;
      } else {
        VLOG(4) << "MultiplyFusion: too many parameter instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kMultiply) {
      multiply_instr = instr;
    } else {
      VLOG(4) << "MultiplyFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Must have at least one parameter and the multiply instruction
  if (!param0_instr || !multiply_instr) {
    VLOG(4) << "MultiplyFusion: missing parameter or multiply instruction";
    return false;
  }

  // Check multiply operands
  if (multiply_instr->operand_count() != 2) {
    VLOG(4) << "MultiplyFusion: multiply must have exactly 2 operands";
    return false;
  }

  // Case 1: Two different parameters
  if (instruction_count == 3) {
    if (!param1_instr) {
      VLOG(4) << "MultiplyFusion: missing second parameter";
      return false;
    }
    if ((multiply_instr->operand(0) != param0_instr && multiply_instr->operand(0) != param1_instr) ||
        (multiply_instr->operand(1) != param0_instr && multiply_instr->operand(1) != param1_instr)) {
      VLOG(4) << "MultiplyFusion: multiply does not take both parameters as operands";
      return false;
    }
  } else {
    // Case 2: Same parameter squared
    if (multiply_instr->operand(0) != param0_instr || multiply_instr->operand(1) != param0_instr) {
      VLOG(4) << "MultiplyFusion: multiply does not take the same parameter as both operands";
      return false;
    }
  }

  // Fusion's root must be the multiply instruction
  if (computation->root_instruction() != multiply_instr) {
    VLOG(4) << "MultiplyFusion: fusion root is not multiply";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the scalar multiply pattern
// Pattern: fusion { parameter + constant -> multiply }
bool IsScalarMultiplyFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 3 instructions: parameter + constant + multiply
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "ScalarMultiplyFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* constant_instr = nullptr;
  const HloInstruction* multiply_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (!param_instr) {
        param_instr = instr;
      } else {
        VLOG(4) << "ScalarMultiplyFusion: too many parameter instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kConstant) {
      if (!constant_instr) {
        constant_instr = instr;
      } else {
        VLOG(4) << "ScalarMultiplyFusion: too many constant instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kMultiply) {
      multiply_instr = instr;
    } else {
      VLOG(4) << "ScalarMultiplyFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // All instructions must be present
  if (!param_instr || !constant_instr || !multiply_instr) {
    VLOG(4) << "ScalarMultiplyFusion: missing required instructions";
    return false;
  }

  // Multiply must take param and constant as operands
  if (multiply_instr->operand_count() != 2 ||
      (multiply_instr->operand(0) != param_instr && multiply_instr->operand(0) != constant_instr) ||
      (multiply_instr->operand(1) != param_instr && multiply_instr->operand(1) != constant_instr)) {
    VLOG(4) << "ScalarMultiplyFusion: multiply does not take both operands";
    return false;
  }

  // Fusion's root must be the multiply instruction
  if (computation->root_instruction() != multiply_instr) {
    VLOG(4) << "ScalarMultiplyFusion: fusion root is not multiply";
    return false;
  }

  return true;
}


// Helper function to check if a fusion matches the greater pattern
// Pattern: fusion { parameter0 + parameter1 -> compare with GT direction }
bool IsGreaterFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 3 instructions: parameter0 + parameter1 + compare
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 3) {
    VLOG(4) << "GreaterFusion: expected 3 instructions, got "
            << instruction_count;
    return false;
  }

  // Find the parameters and compare instructions
  const HloInstruction* param0_instr = nullptr;
  const HloInstruction* param1_instr = nullptr;
  const HloInstruction* compare_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (!param0_instr) {
        param0_instr = instr;
      } else if (!param1_instr) {
        param1_instr = instr;
      } else {
        VLOG(4) << "GreaterFusion: too many parameter instructions";
        return false;
      }
    } else if (instr->opcode() == HloOpcode::kCompare) {
      compare_instr = instr;
    } else {
      VLOG(4) << "GreaterFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // All instructions must be present
  if (!param0_instr || !param1_instr || !compare_instr) {
    VLOG(4) << "GreaterFusion: missing parameter or compare instruction";
    return false;
  }

  // Compare must take both parameters as operands
  if (compare_instr->operand_count() != 2 ||
      (compare_instr->operand(0) != param0_instr && compare_instr->operand(0) != param1_instr) ||
      (compare_instr->operand(1) != param0_instr && compare_instr->operand(1) != param1_instr)) {
    VLOG(4) << "GreaterFusion: compare does not take both parameters as operands";
    return false;
  }

  // Compare must have GT direction
  if (compare_instr->comparison_direction() != ComparisonDirection::kGt) {
    VLOG(4) << "GreaterFusion: compare direction is not GT";
    return false;
  }

  // Fusion's root must be the compare instruction
  if (computation->root_instruction() != compare_instr) {
    VLOG(4) << "GreaterFusion: fusion root is not compare";
    return false;
  }

  return true;
}

// Helper function to check if a fusion matches the iota pattern
// Pattern: fusion { -> iota }
bool IsIotaFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Must have exactly 1 instruction: iota
  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 1) {
    VLOG(4) << "IotaFusion: expected 1 instruction, got "
            << instruction_count;
    return false;
  }

  // The instruction must be an iota
  const HloInstruction* iota_instr = nullptr;
  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kIota) {
      iota_instr = instr;
    } else {
      VLOG(4) << "IotaFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  if (!iota_instr) {
    VLOG(4) << "IotaFusion: no iota instruction found";
    return false;
  }

  // Fusion's root must be the iota instruction
  if (computation->root_instruction() != iota_instr) {
    VLOG(4) << "IotaFusion: fusion root is not iota";
    return false;
  }

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

// Overload for fusion instruction inputs - handles cases where operations
// like bitcast are fused inline, sharing memory but changing shape
absl::StatusOr<xla::ShapedSlice> ThunkEmitter::GetInputParamShapedSliceForHlo(
    const xla::HloFusionInstruction* fusion, int64_t operand_index) const {
  // Get slice from operand (memory address is correct)
  TF_ASSIGN_OR_RETURN(xla::BufferAllocation::Slice slice,
                      GetAllocationSliceForHlo(fusion->operand(operand_index)));
  
  // Get shape from fusion internal parameter (handles bitcast fusion)
  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* param_instr = nullptr;
  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kParameter && 
        instr->parameter_number() == operand_index) {
      param_instr = instr;
      break;
    }
  }
  
  // If we found internal parameter, use its shape; otherwise fall back to operand shape
  xla::Shape shape;
  if (param_instr) {
    shape = param_instr->shape();
  } else {
    TF_ASSIGN_OR_RETURN(
        shape,
        ir_emitter_context_->buffer_assignment().GetShapeForUniqueSlice(
            fusion->operand(operand_index), ShapeIndex{}));
  }
  
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
  // Pattern 1: broadcast-constant -> ascend.full.f32 (for memset-like operation)
  if (IsBroadcastConstantFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitBroadcastConstantFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsAddFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitAddFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsScalarMultiplyFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitScalarMultiplyFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsMaximumFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitMaximumFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsReduceMaxFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitReduceMaxFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsSubtractFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitSubtractFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsExponentialFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitExponentialFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsReduceSumFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitReduceSumFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsEqualFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitEqualFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsSelectFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitSelectFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsNegateFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitNegateFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  // Pattern: convolution -> aclnnConvolution
  if (IsConvolutionFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitConvolutionFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsDivideFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitDivideFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsMultiplyFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitMultiplyFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsGreaterFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitGreaterFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  if (IsTensorBroadcastFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitTensorBroadcastFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }
  
  // Pattern 2: convert-element-type -> ascend.cast.s32_to_u32
  if (IsConvertFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitConvertFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }
  
  // Pattern 2.5: tanh -> aclnnTanh
  if (IsTanhFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitTanhFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  // Pattern: sqrt -> aclnnSqrt
  if (IsSqrtFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitSqrtFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  // Pattern 3: shift-right-logical -> ascend.right_shift
  if (IsShiftRightFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitShiftRightFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }
  
  if (IsIotaFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitIotaFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }
  
  if (IsArgMaxFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitArgMaxFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  // Pattern: max pool 2D -> aclnnMaxPool
  if (IsMaxPoolFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitMaxPoolFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  // Pattern: concatenate -> aclnnCat
  if (IsConcatenateFusion(fusion)) {
    TF_ASSIGN_OR_RETURN(auto thunks, EmitConcatenateFusion(fusion));
    if (!thunks.empty()) {
      return thunks;
    }
  }

  // Add more FFI pattern matching here in the future
  // Pattern 5: ...
  
  VLOG(3) << "Ascend ThunkEmitter: no matching FFI pattern found for fusion";
  return xla::gpu::ThunkSequence{};
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitIotaFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting iota fusion as ascend.iota: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* iota_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kIota) {
      iota_instr = instr;
      break;
    }
  }

  if (!iota_instr) {
    return absl::InternalError("No iota instruction found in iota fusion");
  }

  // Extract iota dimension
  int64_t iota_dimension = Cast<HloIotaInstruction>(iota_instr)->iota_dimension();

  // Extract shape information
  const Shape& output_shape = fusion->shape();
  PrimitiveType element_type = output_shape.element_type();
  int64_t num_rows = output_shape.dimensions(0);
  int64_t num_classes = output_shape.dimensions(1);

  VLOG(2) << "Iota fusion: output shape=" << output_shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type)
          << ", iota_dimension=" << iota_dimension
          << ", num_rows=" << num_rows
          << ", num_classes=" << num_classes;

  std::string function_name = "ascend.iota";
  switch (element_type) {
    case U8:
      function_name += ".u8";
      break;
    case S32:
      function_name += ".s32";
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported data type for iota: ", PrimitiveType_Name(element_type)));
  }

  VLOG(2) << "Using iota function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;
  attributes["iota_dimension"] = xla::ffi::Scalar(iota_dimension);
  attributes["num_classes"] = xla::ffi::Scalar(num_classes);
  attributes["num_rows"] = xla::ffi::Scalar(num_rows);

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitArgMaxFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting argmax fusion as ascend.max_dim: " << fusion->name();

  // Get the input buffer allocation for the first operand (the values tensor)
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion,0));
  
  // Handle tuple return value (value + index)
  TF_ASSIGN_OR_RETURN(auto value_slice, GetShapedSliceForHlo(fusion, {0}));
  TF_ASSIGN_OR_RETURN(auto index_slice, GetShapedSliceForHlo(fusion, {1}));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* reduce_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kReduce) {
      reduce_instr = instr;
      break;
    }
  }

  if (!reduce_instr) {
    return absl::InternalError("No reduce instruction found in argmax fusion");
  }

  // Extract reduction dimensions
  absl::Span<const int64_t> reduce_dims = reduce_instr->dimensions();
  if (reduce_dims.empty()) {
    return absl::InternalError("Reduce instruction has no dimensions");
  }
  
  int64_t dim = reduce_dims[0]; // Argmax typically reduces over one dimension
  bool keepdim = false; // Default value, actual keep_dims behavior is handled by the shape

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "ArgMax fusion: input shape=" << input_shape.ToString()
          << ", value output shape=" << value_slice.shape.ToString()
          << ", index output shape=" << index_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type)
          << ", reduce_dim=" << dim
          << ", keepdim=" << keepdim;

  std::string function_name = "ascend.max_dim";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    default:
      VLOG(2) << "Unsupported data type for argmax: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using max_dim function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice);
  results.push_back(value_slice);
  results.push_back(index_slice);

  xla::ffi::AttributesMap attributes;
  attributes["dim"] = xla::ffi::Scalar(dim);
  attributes["keepdim"] = xla::ffi::Scalar(keepdim);

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
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

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitAddFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting add fusion as ascend.add: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice0, GetInputParamShapedSliceForHlo(fusion,0));
  TF_ASSIGN_OR_RETURN(auto input_slice1, GetInputParamShapedSliceForHlo(fusion,1));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* add_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kAdd) {
      add_instr = instr;
      break;
    }
  }

  if (!add_instr) {
    return absl::InternalError("No add instruction found in add fusion");
  }

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Add fusion: input0 shape=" << input_shape.ToString()
          << ", input1 shape=" << fusion->operand(1)->shape().ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.add";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for add: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using add function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice0);
  operands.push_back(input_slice1);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;
  attributes["alpha"] = xla::ffi::Scalar(1.0f);

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitMaximumFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting maximum fusion as ascend.maximum: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice0, GetInputParamShapedSliceForHlo(fusion,0));
  TF_ASSIGN_OR_RETURN(auto input_slice1, GetInputParamShapedSliceForHlo(fusion,1));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* maximum_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kMaximum) {
      maximum_instr = instr;
      break;
    }
  }

  if (!maximum_instr) {
    return absl::InternalError("No maximum instruction found in maximum fusion");
  }

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Maximum fusion: input0 shape=" << input_shape.ToString()
          << ", input1 shape=" << fusion->operand(1)->shape().ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.maximum";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for maximum: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using maximum function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice0);
  operands.push_back(input_slice1);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitReduceMaxFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting reduce_max fusion as ascend.reduce_max: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion,0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* reduce_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kReduce) {
      reduce_instr = instr;
      break;
    }
  }

  if (!reduce_instr) {
    return absl::InternalError("No reduce instruction found in reduce_max fusion");
  }

  // Extract reduction dimensions
  absl::Span<const int64_t> reduce_dims = reduce_instr->dimensions();
  bool keep_dims = false; // Default value, actual keep_dims behavior is handled by the shape
  bool noop_with_empty_dims = false; // Default value

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "ReduceMax fusion: input shape=" << input_shape.ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type)
          << ", reduce_dims=" << absl::StrJoin(reduce_dims, ",")
          << ", keep_dims=" << keep_dims;

  std::string function_name = "ascend.reduce_max";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for reduce_max: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using reduce_max function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;
  std::vector<int64_t> dims_vector(reduce_dims.begin(), reduce_dims.end());
  attributes["dims"] = xla::ffi::Array(dims_vector);
  attributes["keep_dims"] = xla::ffi::Scalar(keep_dims);
  attributes["noop_with_empty_dims"] = xla::ffi::Scalar(noop_with_empty_dims);

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitSubtractFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting subtract fusion as ascend.subtract: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice0, GetInputParamShapedSliceForHlo(fusion,0));
  TF_ASSIGN_OR_RETURN(auto input_slice1, GetInputParamShapedSliceForHlo(fusion,1));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* subtract_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kSubtract) {
      subtract_instr = instr;
      break;
    }
  }

  if (!subtract_instr) {
    return absl::InternalError("No subtract instruction found in subtract fusion");
  }

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Subtract fusion: input0 shape=" << input_shape.ToString()
          << ", input1 shape=" << fusion->operand(1)->shape().ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.subtract";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for subtract: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using subtract function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice0);
  operands.push_back(input_slice1);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;
  attributes["alpha"] = xla::ffi::Scalar(1.0f);

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitExponentialFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting exponential fusion as ascend.exponential: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion,0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* exponential_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kExp) {
      exponential_instr = instr;
      break;
    }
  }

  if (!exponential_instr) {
    return absl::InternalError("No exponential instruction found in exponential fusion");
  }

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Exponential fusion: input shape=" << input_shape.ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.exponential";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for exponential: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using exponential function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitReduceSumFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting reduce_sum fusion as ascend.reduce_sum: " << fusion->name();

  // Use new overload that handles fusion inputs correctly
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* reduce_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kReduce) {
      reduce_instr = instr;
      break;
    }
  }

  if (!reduce_instr) {
    return absl::InternalError("No reduce instruction found in reduce_sum fusion");
  }

  // Extract reduction dimensions
  absl::Span<const int64_t> reduce_dims = reduce_instr->dimensions();
  bool keep_dims = false; // Default value, actual keep_dims behavior is handled by the shape

  const Shape& input_shape = input_slice.shape;
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "ReduceSum fusion: input shape=" << input_shape.ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type)
          << ", reduce_dims=" << absl::StrJoin(reduce_dims, ",")
          << ", keep_dims=" << keep_dims;

  std::string function_name = "ascend.reduce_sum";
  switch (element_type) {
    case F32:
      function_name += "_f32";
      break;
    case F16:
      function_name += "_f16";
      break;
    case BF16:
      function_name += "_bf16";
      break;
    case S32:
      function_name += "_s32";
      break;
    case S64:
      function_name += "_s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for reduce_sum: " << PrimitiveType_Name(element_type);
      function_name += "_f32";
      break;
  }

  VLOG(2) << "Using reduce_sum function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;
  std::vector<int64_t> dims_vector(reduce_dims.begin(), reduce_dims.end());
  attributes["dims"] = xla::ffi::Array(dims_vector);
  attributes["keep_dims"] = xla::ffi::Scalar(keep_dims);

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitEqualFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting equal fusion as ascend.equal: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice0, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto input_slice1, GetInputParamShapedSliceForHlo(fusion, 1));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* compare_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kCompare) {
      compare_instr = instr;
      break;
    }
  }

  if (!compare_instr) {
    return absl::InternalError("No compare instruction found in equal fusion");
  }

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Equal fusion: input0 shape=" << input_shape.ToString()
          << ", input1 shape=" << fusion->operand(1)->shape().ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.equal";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    case U8:
      function_name += ".u8";
      break;
    case S8:
      function_name += ".s8";
      break;
    case PRED:
      function_name += ".bool";
      break;
    default:
      VLOG(2) << "Unsupported data type for equal: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using equal function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice0);
  operands.push_back(input_slice1);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitSelectFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting select fusion as ascend.select: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto condition_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto x_slice, GetInputParamShapedSliceForHlo(fusion, 1));
  TF_ASSIGN_OR_RETURN(auto y_slice, GetInputParamShapedSliceForHlo(fusion, 2));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* select_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kSelect) {
      select_instr = instr;
      break;
    }
  }

  if (!select_instr) {
    return absl::InternalError("No select instruction found in select fusion");
  }

  const Shape& input_shape = fusion->operand(1)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Select fusion: condition shape=" << fusion->operand(0)->shape().ToString()
          << ", x shape=" << input_shape.ToString()
          << ", y shape=" << fusion->operand(2)->shape().ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.select";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    case PRED:
      function_name += ".bool";
      break;
    default:
      VLOG(2) << "Unsupported data type for select: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using select function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(condition_slice);
  operands.push_back(x_slice);
  operands.push_back(y_slice);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitNegateFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting negate fusion as ascend.negate: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* negate_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kNegate) {
      negate_instr = instr;
      break;
    }
  }

  if (!negate_instr) {
    return absl::InternalError("No negate instruction found in negate fusion");
  }

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Negate fusion: input shape=" << input_shape.ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.negate";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for negate: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using negate function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitDivideFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting divide fusion as ascend.divide: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice0, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto input_slice1, GetInputParamShapedSliceForHlo(fusion, 1));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* divide_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kDivide) {
      divide_instr = instr;
      break;
    }
  }

  if (!divide_instr) {
    return absl::InternalError("No divide instruction found in divide fusion");
  }

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Divide fusion: input0 shape=" << input_shape.ToString()
          << ", input1 shape=" << fusion->operand(1)->shape().ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.divide";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for divide: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using divide function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice0);
  operands.push_back(input_slice1);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitMultiplyFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting multiply fusion as ascend.multiply: " << fusion->name();

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* multiply_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kMultiply) {
      multiply_instr = instr;
      break;
    }
  }

  if (!multiply_instr) {
    return absl::InternalError("No multiply instruction found in multiply fusion");
  }

  // Get input slices based on the number of parameters
  TF_ASSIGN_OR_RETURN(auto input_slice0, GetInputParamShapedSliceForHlo(fusion,0));
  NullableShapedSlice input_slice1;
  
  // Check if fusion has two operands or one operand (squared case)
  bool is_squared = (fusion->operand_count() == 1);
  
  if (!is_squared) {
    TF_ASSIGN_OR_RETURN(input_slice1, GetInputParamShapedSliceForHlo(fusion,1));
  } else {
    // For squared case, use the same input slice for both operands
    input_slice1 = input_slice0;
  }
  
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  if (!is_squared) {
    VLOG(2) << "Multiply fusion: input0 shape=" << input_shape.ToString()
            << ", input1 shape=" << fusion->operand(1)->shape().ToString()
            << ", output shape=" << output_slice.shape.ToString()
            << ", element_type=" << PrimitiveType_Name(element_type);
  } else {
    VLOG(2) << "Multiply fusion (squared): input shape=" << input_shape.ToString()
            << ", output shape=" << output_slice.shape.ToString()
            << ", element_type=" << PrimitiveType_Name(element_type);
  }

  std::string function_name = "ascend.multiply";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    default:
      VLOG(2) << "Unsupported data type for multiply: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using multiply function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice0);
  operands.push_back(input_slice1);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitGreaterFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting greater fusion as ascend.greater: " << fusion->name();

  TF_ASSIGN_OR_RETURN(auto input_slice0, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto input_slice1, GetInputParamShapedSliceForHlo(fusion, 1));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* compare_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kCompare) {
      compare_instr = instr;
      break;
    }
  }

  if (!compare_instr) {
    return absl::InternalError("No compare instruction found in greater fusion");
  }

  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  VLOG(2) << "Greater fusion: input0 shape=" << input_shape.ToString()
          << ", input1 shape=" << fusion->operand(1)->shape().ToString()
          << ", output shape=" << output_slice.shape.ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  std::string function_name = "ascend.greater";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    case U8:
      function_name += ".u8";
      break;
    case S8:
      function_name += ".s8";
      break;
    case PRED:
      function_name += ".bool";
      break;
    default:
      VLOG(2) << "Unsupported data type for greater: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using greater function: " << function_name;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  operands.push_back(input_slice0);
  operands.push_back(input_slice1);
  results.push_back(output_slice);

  xla::ffi::AttributesMap attributes;

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

// Helper function to emit tensor broadcast fusion as ascend.expand FFI call
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitTensorBroadcastFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting tensor broadcast fusion as ascend.expand: " << fusion->name();
  
  // Get the input and output buffer allocations
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));
  
  // Extract broadcast dimensions from the fusion computation
  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* broadcast_instr = nullptr;
  
  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kBroadcast) {
      broadcast_instr = instr;
      break;
    }
  }
  
  if (!broadcast_instr) {
    return absl::InternalError("No broadcast instruction found in tensor broadcast fusion");
  }
  
  // Get the broadcast dimensions attribute
  absl::Span<const int64_t> dimensions = broadcast_instr->dimensions();
  
  // The dim parameter for expand FFI: 
  // - If dimensions is empty, it means scalar broadcast (input is scalar, output is any shape)
  // - If dimensions[0] == 0, it means input's dims map to output's starting dims (prepend 1s at end)
  // - If dimensions[0] == input_ndim, it means input's dims map to output's ending dims (prepend 1s at start)
  int64_t dim = 0; // Default to 0 for scalar broadcast
  if (!dimensions.empty()) {
    dim = dimensions[0];
  }
  
  const Shape& input_shape = fusion->operand(0)->shape();
  
  VLOG(2) << "Tensor broadcast fusion: input shape=" << input_shape.ToString() 
          << ", output shape=" << fusion->shape().ToString()
          << ", broadcast dimensions=[" << absl::StrJoin(dimensions, ",") << "]"
          << ", dim=" << dim;
  
  // Create operands and results for CustomCallThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  
  // Add input as operand
  operands.push_back(input_slice);
  // Add output as result
  results.push_back(output_slice);
  
  // Create attributes map with dim parameter
  xla::ffi::AttributesMap attributes;
  attributes["dim"] = xla::ffi::Scalar(dim);
  
  // Get GPU compute capability
  const se::GpuComputeCapability& gpu_compute_capability = 
      ir_emitter_context_->gpu_compute_capability();
  
  // Determine the function name based on data type
  std::string function_name = "ascend.expand";
  PrimitiveType element_type = input_shape.element_type();
  
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    case S64:
      function_name += ".s64";
      break;
    case U8:
      function_name += ".u8";
      break;
    case S8:
      function_name += ".s8";
      break;
    case PRED:
      function_name += ".bool";
      break;
    default:
      VLOG(2) << "Unsupported data type for expand: " << PrimitiveType_Name(element_type);
      function_name += ".f32"; // Default to f32 if type not supported
      break;
  }
  
  VLOG(2) << "Using expand function: " << function_name;
  
  // Create CustomCallThunk
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
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

// Helper function to emit convert-element-type fusion as ascend.cast FFI call
#if 0
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitConvertFusion(
    const HloFusionInstruction* fusion) {
  // Get the input and output buffer allocations
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));
  
  // Determine the conversion type
  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* convert_instr = nullptr;
  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kConvert) {
      convert_instr = instr;
      break;
    }
  }
  
  PrimitiveType src_type = convert_instr->operand(0)->shape().element_type();
  PrimitiveType dst_type = convert_instr->shape().element_type();
  
  // Determine the FFI function name based on conversion type
  std::string function_name = "ascend.cast";
  
  // Map PrimitiveType to string suffix
  auto type_to_suffix = [](PrimitiveType type) {
    switch (type) {
      case PrimitiveType::F32: return "f32";
      case PrimitiveType::F16: return "f16";
      case PrimitiveType::BF16: return "bf16";
      case PrimitiveType::S32: return "s32";
      case PrimitiveType::S64: return "s64";
      case PrimitiveType::U32: return "u32";
      case PrimitiveType::U64: return "u64";
      case PrimitiveType::U8: return "u8";
      case PrimitiveType::S8: return "s8";
      case PrimitiveType::PRED: return "bool";
      default: return "unknown";
    }
  };
  
  std::string src_suffix = type_to_suffix(src_type);
  std::string dst_suffix = type_to_suffix(dst_type);
  
  function_name += "." + src_suffix + "_to_" + dst_suffix;
  VLOG(2) << "Emitting convert-element-type fusion as " << function_name << ": " << fusion->name();
  
  // Create operands and results for CustomCallThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  
  // Add input and output slices
  operands.push_back(input_slice);
  results.push_back(output_slice);
  
  // Create attributes map (empty for cast operation)
  xla::ffi::AttributesMap attributes;
  
  // Get GPU compute capability
  const se::GpuComputeCapability& gpu_compute_capability = 
      ir_emitter_context_->gpu_compute_capability();
  
  // Create CustomCallThunk
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
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
#endif

// Helper function to emit convert-element-type fusion as aclnnCast
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitConvertFusion(
    const HloFusionInstruction* fusion) {
  // Get the input and output buffer allocations
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));
  
  // Create operands and results for AclnnThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  std::vector<xla::ascend::AclnnThunk::Param> params;
  
  // Add input and output slices
  operands.push_back(input_slice);
  results.push_back(output_slice);
  
  VLOG(2) << "Emitting convert-element-type fusion as aclnnCast: " << fusion->name();
  
  // Create AclnnThunk for aclnnCast
  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
      "aclnnCast",
      std::move(operands),
      std::move(results),
      std::move(params));
  
  // Add the thunk to the sequence
  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));
  
  return sequence;
}

// Helper function to emit tanh fusion as aclnnTanh
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitTanhFusion(
    const HloFusionInstruction* fusion) {
  // Get the input and output buffer allocations
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));
  
  // Create operands and results for AclnnThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  std::vector<xla::ascend::AclnnThunk::Param> params;
  
  // Add input and output slices
  operands.push_back(input_slice);
  results.push_back(output_slice);
  
  VLOG(2) << "Emitting tanh fusion as aclnnTanh: " << fusion->name();
  
  // Create AclnnThunk for aclnnTanh
  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
      "aclnnTanh",
      std::move(operands),
      std::move(results),
      std::move(params));
  
  // Add the thunk to the sequence
  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));
  
  return sequence;
}

// Helper function to emit sqrt fusion as aclnnSqrt
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitSqrtFusion(
    const HloFusionInstruction* fusion) {
  // Get the input and output buffer allocations
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  // Create operands and results for AclnnThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  std::vector<xla::ascend::AclnnThunk::Param> params;

  // Add input and output slices
  operands.push_back(input_slice);
  results.push_back(output_slice);

  VLOG(2) << "Emitting sqrt fusion as aclnnSqrt: " << fusion->name();

  // Create AclnnThunk for aclnnSqrt
  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
      "aclnnSqrt",
      std::move(operands),
      std::move(results),
      std::move(params));

  // Add the thunk to the sequence
  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

// Helper function to emit convolution fusion as aclnnConvolution
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitConvolutionFusion(
    const HloFusionInstruction* fusion) {
  // Get the input, weight, and output buffer allocations
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto weight_slice, GetInputParamShapedSliceForHlo(fusion, 1));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));
  
  // Create operands and results for AclnnThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  std::vector<xla::ascend::AclnnThunk::Param> params;
  
  // Add input, weight, and output slices
  operands.push_back(input_slice);
  operands.push_back(weight_slice);
  results.push_back(output_slice);
  
  // Extract convolution parameters from HLO instruction
  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* conv_instr = nullptr;
  
  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kConvolution) {
      conv_instr = instr;
      break;
    }
  }
  
  if (!conv_instr) {
    return absl::InternalError("No convolution instruction found in fusion");
  }
  
  const auto& window = conv_instr->window();
  
  // Extract stride
  std::vector<int64_t> stride;
  for (int i = 0; i < window.dimensions_size(); ++i) {
    stride.push_back(window.dimensions(i).stride());
  }
  
  // Extract padding - use dimensions().padding_low() and padding_high()
  std::vector<int64_t> pad_values;
  for (int i = 0; i < window.dimensions_size(); ++i) {
    pad_values.push_back(window.dimensions(i).padding_low());
    pad_values.push_back(window.dimensions(i).padding_high());
  }
  
  // Extract dilation
  std::vector<int64_t> dilation;
  for (int i = 0; i < window.dimensions_size(); ++i) {
    dilation.push_back(window.dimensions(i).window_dilation());
  }
  
  // Extract transposed
  bool transposed = conv_instr->feature_group_count() > 1 ? false : false; // TODO: Check actual transposed flag
  
  // Extract output padding (default to 0)
  std::vector<int64_t> output_padding(window.dimensions_size(), 0);
  
  // Extract groups
  int64_t groups = conv_instr->feature_group_count();
  
  // Extract cubeMathType (default to 0 - KEEP_DTYPE)
  int8_t cube_math_type = 0;
  
  // Add parameters to params list
  params.push_back(xla::ascend::AclnnThunk::Param{stride});
  params.push_back(xla::ascend::AclnnThunk::Param{pad_values});
  params.push_back(xla::ascend::AclnnThunk::Param{dilation});
  params.push_back(xla::ascend::AclnnThunk::Param{transposed});
  params.push_back(xla::ascend::AclnnThunk::Param{output_padding});
  params.push_back(xla::ascend::AclnnThunk::Param{groups});
  params.push_back(xla::ascend::AclnnThunk::Param{cube_math_type});
  
  VLOG(2) << "Emitting convolution fusion as aclnnConvolution: " << fusion->name();
  VLOG(3) << "Stride: " << absl::StrJoin(stride, ", ");
  VLOG(3) << "Padding: " << absl::StrJoin(pad_values, ", ");
  VLOG(3) << "Dilation: " << absl::StrJoin(dilation, ", ");
  VLOG(3) << "Transposed: " << transposed;
  VLOG(3) << "Groups: " << groups;
  
  // Create AclnnThunk for aclnnConvolution
  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
      "aclnnConvolution",
      std::move(operands),
      std::move(results),
      std::move(params));
  
  // Add the thunk to the sequence
  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));
  
  return sequence;
}

// Helper function to emit shift-right-logical fusion as ascend.right_shift FFI call
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitShiftRightFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting shift-right-logical fusion as ascend.right_shift: " << fusion->name();
  
  // Get the input buffer allocation
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));
  
  // Extract constant value from the fusion computation (shift bits)
  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* constant_instr = nullptr;
  
  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant) {
      constant_instr = instr;
      break;
    }
  }
  
  if (!constant_instr) {
    return absl::InternalError("No constant instruction found in shift-right-logical fusion");
  }
  
  // Extract the shift bits value from the constant literal
  // The constant is a scalar (s32[]), so we get the first element
  const auto& literal = constant_instr->literal();
  int32_t shift_bits_value = literal.GetFirstElement<int32_t>();
  
  VLOG(2) << "Shift-right-logical fusion: shift_bits=" << shift_bits_value;
  
  // Create operands and results for CustomCallThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  
  // Add input slice as operand
  operands.push_back(input_slice);
  results.push_back(output_slice);
  
  // Create attributes map with shift_bits value
  xla::ffi::AttributesMap attributes;
  attributes["shift_bits"] = xla::ffi::Scalar(shift_bits_value);
  
  // Get GPU compute capability
  const se::GpuComputeCapability& gpu_compute_capability = 
      ir_emitter_context_->gpu_compute_capability();
  
  // Create CustomCallThunk
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          "ascend.right_shift",
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


absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitGemmThunk(
    const HloCustomCallInstruction* instr) {
  VLOG(2) << "Emitting matmul as aclnnGemm: " << instr->name();

  TF_ASSIGN_OR_RETURN(auto a_slice, GetShapedSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto b_slice, GetShapedSliceForHlo(instr->operand(1)));
  
  // Handle tuple return value (result + workspace)
  TF_ASSIGN_OR_RETURN(auto c_slice, GetShapedSliceForHlo(instr, {0}));
  std::optional<xla::ShapedSlice> workspace_slice;
  if (instr->shape().IsTuple() && instr->shape().tuple_shapes_size() > 1) {
    TF_ASSIGN_OR_RETURN(auto ws, GetShapedSliceForHlo(instr, {1}));
    workspace_slice = ws;
  }

  const Shape& a_shape = instr->operand(0)->shape();
  PrimitiveType element_type = a_shape.element_type();

  VLOG(2) << "Matmul: a_shape=" << a_shape.ToString()
          << ", b_shape=" << instr->operand(1)->shape().ToString()
          << ", c_shape=" << instr->shape().ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  // Parse GEMM backend config to get transpose information
  int64_t transA = 0;
  int64_t transB = 0;
  float alpha = 1.0f;
  float beta = 0.0f;

  // Try to parse backend config using raw_backend_config_string
  const std::string& backend_config = instr->raw_backend_config_string();
  VLOG(2) << "Backend config: " << backend_config;
  
  // Simple parsing to extract gemm_backend_config
  // This is a simplified parsing, in a real implementation you would use proper JSON parsing
  if (!backend_config.empty()) {
    // Check for rhs_contracting_dimensions to determine if B needs transpose
    // If rhs_contracting_dimensions is ["1"], it means B's contracting dimension is the second dimension,
    // so B needs to be transposed
    size_t rhs_pos = backend_config.find("rhs_contracting_dimensions");
    if (rhs_pos != std::string::npos) {
      // Find the colon after rhs_contracting_dimensions
      size_t colon_pos = backend_config.find(":", rhs_pos);
      if (colon_pos != std::string::npos) {
        // Find the value after the colon, which should be ["1"]
        size_t val_start = backend_config.find("[", colon_pos);
        if (val_start != std::string::npos) {
          size_t val_end = backend_config.find("]", val_start);
          if (val_end != std::string::npos) {
            std::string val = backend_config.substr(val_start, val_end - val_start + 1);
            if (val == "[\"1\"]") {
              transB = 1;
              VLOG(2) << "Setting transB=1 based on rhs_contracting_dimensions=" << val;
            }
          }
        }
      }
    }
    
    // Check for lhs_contracting_dimensions to determine if A needs transpose
    // If lhs_contracting_dimensions is ["0"], it means A's contracting dimension is the first dimension,
    // so A needs to be transposed
    size_t lhs_pos = backend_config.find("lhs_contracting_dimensions");
    if (lhs_pos != std::string::npos) {
      // Find the colon after lhs_contracting_dimensions
      size_t colon_pos = backend_config.find(":", lhs_pos);
      if (colon_pos != std::string::npos) {
        // Find the value after the colon, which should be ["0"]
        size_t val_start = backend_config.find("[", colon_pos);
        if (val_start != std::string::npos) {
          size_t val_end = backend_config.find("]", val_start);
          if (val_end != std::string::npos) {
            std::string val = backend_config.substr(val_start, val_end - val_start + 1);
            if (val == "[\"0\"]") {
              transA = 1;
              VLOG(2) << "Setting transA=1 based on lhs_contracting_dimensions=" << val;
            }
          }
        }
      }
    }
  }

  VLOG(2) << "GEMM parameters: transA=" << transA << ", transB=" << transB << ", alpha=" << alpha << ", beta=" << beta;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  // Create a dummy C tensor (all zeros) since aclnnGemm requires it
  // In a real implementation, you would create a proper zero tensor
  // For simplicity, we'll just pass the same as C for now
  operands.push_back(a_slice);
  operands.push_back(b_slice);
  operands.push_back(c_slice);  // Using c_slice as dummy C
  results.push_back(c_slice);
  
  // Add workspace to results to maintain tuple structure
  if (workspace_slice) {
    results.push_back(*workspace_slice);
  }

  // Create parameters for aclnnGemm
  std::vector<xla::ascend::AclnnThunk::Param> params;
  params.push_back(alpha);
  params.push_back(beta);
  params.push_back(transA);
  params.push_back(transB);

  // Create AclnnThunk
  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(instr, ir_emitter_context_->GetNextThunkId()),
      "aclnnGemm",
      std::move(operands),
      std::move(results),
      std::move(params));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitAclnnGemmThunk(
    const HloCustomCallInstruction* instr) {
  VLOG(2) << "Emitting ACLNN GEMM: " << instr->name();

  TF_ASSIGN_OR_RETURN(auto a_slice, GetShapedSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto b_slice, GetShapedSliceForHlo(instr->operand(1)));
  
  // Handle result
  TF_ASSIGN_OR_RETURN(auto c_slice, GetShapedSliceForHlo(instr));

  const Shape& a_shape = instr->operand(0)->shape();
  PrimitiveType element_type = a_shape.element_type();

  VLOG(2) << "ACLNN GEMM: a_shape=" << a_shape.ToString()
          << ", b_shape=" << instr->operand(1)->shape().ToString()
          << ", c_shape=" << instr->shape().ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  // Parse ACLNN GEMM backend config
  float alpha = 1.0f;
  float beta = 0.0f;
  int64_t transA = 0;
  int64_t transB = 0;
  bool has_bias = false;

  // Parse backend config using the config system
  const std::string& backend_config = instr->raw_backend_config_string();
  VLOG(2) << "Backend config: " << backend_config;
  
  if (!backend_config.empty()) {
    TF_ASSIGN_OR_RETURN(auto config, ParseAclnnConfig(
        instr->custom_call_target(), backend_config));
    auto* gemm_config = dynamic_cast<AclnnGemmConfig*>(config.get());
    if (!gemm_config) {
      return absl::InternalError("Failed to cast to AclnnGemmConfig");
    }
    alpha = gemm_config->alpha;
    beta = gemm_config->beta;
    transA = gemm_config->transpose_a;
    transB = gemm_config->transpose_b;
    has_bias = gemm_config->has_bias;
  }

  VLOG(2) << "ACLNN GEMM parameters: transA=" << transA << ", transB=" << transB 
          << ", alpha=" << alpha << ", beta=" << beta << ", has_bias=" << has_bias;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  // Add operands
  operands.push_back(a_slice);
  operands.push_back(b_slice);
  
  // Add bias if present
  if (has_bias && instr->operand_count() > 2) {
    TF_ASSIGN_OR_RETURN(auto bias_slice, GetShapedSliceForHlo(instr->operand(2)));
    operands.push_back(bias_slice);
  }else{
    beta = 0.0f;
    operands.push_back(c_slice);
  }
  
  // Add result
  results.push_back(c_slice);

  // Create parameters for aclnnGemm
  std::vector<xla::ascend::AclnnThunk::Param> params;
  params.push_back(alpha);
  params.push_back(beta);
  params.push_back(transA);
  params.push_back(transB);

  // Create AclnnThunk
  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(instr, ir_emitter_context_->GetNextThunkId()),
      "aclnnGemm",
      std::move(operands),
      std::move(results),
      std::move(params));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

#if 0
// Original code for CustomCallThunk
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitGemmThunk(
    const HloCustomCallInstruction* instr) {
  VLOG(2) << "Emitting matmul as ascend.gemm: " << instr->name();

  TF_ASSIGN_OR_RETURN(auto a_slice, GetShapedSliceForHlo(instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto b_slice, GetShapedSliceForHlo(instr->operand(1)));
  
  // Handle tuple return value (result + workspace)
  TF_ASSIGN_OR_RETURN(auto c_slice, GetShapedSliceForHlo(instr, {0}));
  std::optional<xla::ShapedSlice> workspace_slice;
  if (instr->shape().IsTuple() && instr->shape().tuple_shapes_size() > 1) {
    TF_ASSIGN_OR_RETURN(auto ws, GetShapedSliceForHlo(instr, {1}));
    workspace_slice = ws;
  }

  const Shape& a_shape = instr->operand(0)->shape();
  PrimitiveType element_type = a_shape.element_type();

  VLOG(2) << "Matmul: a_shape=" << a_shape.ToString()
          << ", b_shape=" << instr->operand(1)->shape().ToString()
          << ", c_shape=" << instr->shape().ToString()
          << ", element_type=" << PrimitiveType_Name(element_type);

  // Parse GEMM backend config to get transpose information
  int64_t transA = 0;
  int64_t transB = 0;
  float alpha = 1.0f;
  float beta = 0.0f;

  // Try to parse backend config using raw_backend_config_string
  const std::string& backend_config = instr->raw_backend_config_string();
  VLOG(2) << "Backend config: " << backend_config;
  
  // Simple parsing to extract gemm_backend_config
  // This is a simplified parsing, in a real implementation you would use proper JSON parsing
  if (!backend_config.empty()) {
    // Check for rhs_contracting_dimensions to determine if B needs transpose
    // If rhs_contracting_dimensions is ["1"], it means B's contracting dimension is the second dimension,
    // so B needs to be transposed
    size_t rhs_pos = backend_config.find("rhs_contracting_dimensions");
    if (rhs_pos != std::string::npos) {
      // Find the colon after rhs_contracting_dimensions
      size_t colon_pos = backend_config.find(":", rhs_pos);
      if (colon_pos != std::string::npos) {
        // Find the value after the colon, which should be ["1"]
        size_t val_start = backend_config.find("[", colon_pos);
        if (val_start != std::string::npos) {
          size_t val_end = backend_config.find("]", val_start);
          if (val_end != std::string::npos) {
            std::string val = backend_config.substr(val_start, val_end - val_start + 1);
            if (val == "[\"1\"]") {
              transB = 1;
              VLOG(2) << "Setting transB=1 based on rhs_contracting_dimensions=" << val;
            }
          }
        }
      }
    }
    
    // Check for lhs_contracting_dimensions to determine if A needs transpose
    // If lhs_contracting_dimensions is ["0"], it means A's contracting dimension is the first dimension,
    // so A needs to be transposed
    size_t lhs_pos = backend_config.find("lhs_contracting_dimensions");
    if (lhs_pos != std::string::npos) {
      // Find the colon after lhs_contracting_dimensions
      size_t colon_pos = backend_config.find(":", lhs_pos);
      if (colon_pos != std::string::npos) {
        // Find the value after the colon, which should be ["0"]
        size_t val_start = backend_config.find("[", colon_pos);
        if (val_start != std::string::npos) {
          size_t val_end = backend_config.find("]", val_start);
          if (val_end != std::string::npos) {
            std::string val = backend_config.substr(val_start, val_end - val_start + 1);
            if (val == "[\"0\"]") {
              transA = 1;
              VLOG(2) << "Setting transA=1 based on lhs_contracting_dimensions=" << val;
            }
          }
        }
      }
    }
  }

  std::string function_name = "ascend.gemm";
  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    default:
      VLOG(2) << "Unsupported data type for matmul: " << PrimitiveType_Name(element_type);
      function_name += ".f32";
      break;
  }

  VLOG(2) << "Using gemm function: " << function_name;
  VLOG(2) << "GEMM parameters: transA=" << transA << ", transB=" << transB << ", alpha=" << alpha << ", beta=" << beta;

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  // Create a dummy C tensor (all zeros) since aclnnGemm requires it
  // In a real implementation, you would create a proper zero tensor
  // For simplicity, we'll just pass the same as C for now
  operands.push_back(a_slice);
  operands.push_back(b_slice);
  operands.push_back(c_slice);  // Using c_slice as dummy C
  results.push_back(c_slice);
  
  // Add workspace to results to maintain tuple structure
  if (workspace_slice) {
    results.push_back(*workspace_slice);
  }

  xla::ffi::AttributesMap attributes;
  attributes["alpha"] = xla::ffi::Scalar(alpha);
  attributes["beta"] = xla::ffi::Scalar(beta);
  attributes["transA"] = xla::ffi::Scalar(transA);
  attributes["transB"] = xla::ffi::Scalar(transB);

  const se::GpuComputeCapability& gpu_compute_capability =
      ir_emitter_context_->gpu_compute_capability();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(instr, ir_emitter_context_->GetNextThunkId()),
          function_name,
          std::move(operands),
          std::move(results),
          std::move(attributes),
          /*called_computation=*/nullptr,
          "ASCEND",
          gpu_compute_capability,
          /*execution_state=*/nullptr));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}
#endif

// Emit scalar multiply fusion using FFI
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitScalarMultiplyFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "EmitScalarMultiplyFusion for Ascend";

  auto* computation = fusion->fused_instructions_computation();
  const auto& instructions = computation->instructions();

  // Find the instructions
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* constant_instr = nullptr;
  const HloInstruction* multiply_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kConstant) {
      constant_instr = instr;
    } else if (instr->opcode() == HloOpcode::kMultiply) {
      multiply_instr = instr;
    }
  }

  // Get the input and output buffer allocations
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  // Extract the constant value
  float constant_value = 0.0f;
  if (constant_instr->shape().element_type() == PrimitiveType::F32) {
    constant_value = constant_instr->literal().Get<float>({});
  } else {
    VLOG(4) << "ScalarMultiplyFusion: unsupported constant type "
            << PrimitiveType_Name(constant_instr->shape().element_type());
    return xla::gpu::ThunkSequence{};
  }

  // Create operands and results for CustomCallThunk
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;

  // Add input as operand
  operands.push_back(input_slice);
  // Add output as result
  results.push_back(output_slice);

  // Create attributes map with the constant value
  xla::ffi::AttributesMap attributes;
  attributes["other"] = xla::ffi::Scalar(constant_value);

  // Get GPU compute capability
  const se::GpuComputeCapability& gpu_compute_capability = 
      ir_emitter_context_->gpu_compute_capability();

  // Determine the function name based on data type
  std::string function_name = "ascend.muls";
  const Shape& input_shape = fusion->operand(0)->shape();
  PrimitiveType element_type = input_shape.element_type();

  switch (element_type) {
    case F32:
      function_name += ".f32";
      break;
    case F16:
      function_name += ".f16";
      break;
    case BF16:
      function_name += ".bf16";
      break;
    case S32:
      function_name += ".s32";
      break;
    default:
      VLOG(4) << "ScalarMultiplyFusion: unsupported data type " << PrimitiveType_Name(element_type);
      function_name += ".f32"; // Default to f32 if type not supported
      break;
  }

  VLOG(2) << "Using muls function: " << function_name;

  // Create CustomCallThunk
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::gpu::CustomCallThunk> thunk,
      xla::gpu::CustomCallThunk::Create(
          xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
          function_name,
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

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitMaxPoolFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting max pool fusion as aclnnMaxPool: " << fusion->name();

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* reduce_window_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kReduceWindow) {
      reduce_window_instr = instr;
      break;
    }
  }

  if (!reduce_window_instr) {
    return absl::InternalError("No reduce-window instruction found in max pool fusion");
  }

  // Get input and output slices
  TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, 0));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  const Window& window = reduce_window_instr->window();

  // Extract spatial dimensions (last 2 dimensions for 4D input)
  // Window format: {size=NxCxHxW, stride=NxCxHxW, pad=N_low_N_highxC_low_C_highxH_low_H_highxW_low_W_high}
  // For aclnnMaxPool, we only use H and W dimensions

  int64_t kernel_h = window.dimensions(2).size();
  int64_t kernel_w = window.dimensions(3).size();

  int64_t stride_h = window.dimensions(2).stride();
  int64_t stride_w = window.dimensions(3).stride();

  // Padding format: N_low, N_high, C_low, C_high, H_low, H_high, W_low, W_high
  int64_t pad_h_low = window.dimensions(2).padding_low();
  int64_t pad_h_high = window.dimensions(2).padding_high();
  int64_t pad_w_low = window.dimensions(3).padding_low();
  int64_t pad_w_high = window.dimensions(3).padding_high();

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  std::vector<xla::ascend::AclnnThunk::Param> params;

  operands.push_back(input_slice);
  results.push_back(output_slice);

  // Add parameters for aclnnMaxPool
  // kernelShape
  params.push_back(xla::ascend::AclnnThunk::Param{std::vector<int64_t>{kernel_h, kernel_w}});

  // strides
  params.push_back(xla::ascend::AclnnThunk::Param{std::vector<int64_t>{stride_h, stride_w}});

  // autoPad (only 0 is supported)
  params.push_back(xla::ascend::AclnnThunk::Param{static_cast<int64_t>(0)});

  // pads (H_low, H_high, W_low, W_high)
  params.push_back(xla::ascend::AclnnThunk::Param{std::vector<int64_t>{pad_h_low, pad_h_high, pad_w_low, pad_w_high}});

  // dilations (only 1 is supported)
  params.push_back(xla::ascend::AclnnThunk::Param{std::vector<int64_t>{1, 1}});

  // ceilMode (0 = false, floor division)
  params.push_back(xla::ascend::AclnnThunk::Param{static_cast<int64_t>(0)});

  VLOG(2) << "Emitting max pool fusion with kernel=[" << kernel_h << "," << kernel_w
          << "], stride=[" << stride_h << "," << stride_w
          << "], pad=[" << pad_h_low << "," << pad_h_high << "," << pad_w_low << "," << pad_w_high << "]";

  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
      "aclnnMaxPool",
      std::move(operands),
      std::move(results),
      std::move(params));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}

absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitConcatenateFusion(
    const HloFusionInstruction* fusion) {
  VLOG(2) << "Emitting concatenate fusion as aclnnCat: " << fusion->name();

  auto* computation = fusion->fused_instructions_computation();
  const HloInstruction* concatenate_instr = nullptr;

  for (const auto* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kConcatenate) {
      concatenate_instr = instr;
      break;
    }
  }

  if (!concatenate_instr) {
    return absl::InternalError("No concatenate instruction found in concatenate fusion");
  }

  // Get the concatenate dimension
  int64_t concat_dim = concatenate_instr->dimensions()[0];

  // Get all input slices
  std::vector<NullableShapedSlice> input_slices;
  for (int i = 0; i < fusion->operand_count(); ++i) {
    TF_ASSIGN_OR_RETURN(auto input_slice, GetInputParamShapedSliceForHlo(fusion, i));
    input_slices.push_back(input_slice);
  }

  // Get output slice
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  std::vector<xla::ascend::AclnnThunk::Param> params;

  // Add input slices
  for (auto& input_slice : input_slices) {
    operands.push_back(input_slice);
  }

  // Add output slice
  results.push_back(output_slice);

  // Add the concatenate dimension as parameter
  params.push_back(xla::ascend::AclnnThunk::Param{concat_dim});

  VLOG(2) << "Emitting concatenate fusion with " << operands.size()
          << " inputs along dimension " << concat_dim;

  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
      "aclnnCat",
      std::move(operands),
      std::move(results),
      std::move(params));

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
    case HloOpcode::kCustomCall: {
      auto* custom_call = Cast<HloCustomCallInstruction>(hlo);
      if (xla::gpu::IsLegacyCublasMatmul(*hlo)) {
        return EmitGemmThunk(custom_call);
      }
      if (xla::ascend::IsAclnnGemmTarget(custom_call->custom_call_target())) {
        return EmitAclnnGemmThunk(custom_call);
      }
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
