---
name: "hlo-to-aclnn-converter"
description: "Converts HLO fusion operators to aclnn operators for Ascend NPU. Invoke when user wants to add support for a new HLO-to-aclnn operator mapping or when implementing kernel offloading for Ascend."
---

# HLO Fusion Operator to ACLNN Operator Conversion Guide

This document serves as a comprehensive guide for AI assistants to implement HLO fusion operator to ACLNN operator conversion for the Ascend NPU backend in JAX/XLA.

## Table of Contents

1. [Overview](#overview)
2. [Conversion Workflow](#conversion-workflow)
3. [Step 1: Implement Pattern Matching Function](#step-1-implement-pattern-matching-function)
4. [Step 2: Implement Thunk Emission Function](#step-2-implement-thunk-emission-function)
5. [Step 3: Register in EmitFusion](#step-3-register-in-emitfusion)
6. [Step 4: Add Executor in AclnnThunk](#step-4-add-executor-in-aclnnthunk)
7. [Complete Example: tanh Conversion](#complete-example-tanh-conversion)
8. [Common Patterns and Reference](#common-patterns-and-reference)

---

## Overview

The conversion process transforms HLO fusion instructions into ACLNN (Ascend Library Neural Network) operator thunks. ACLNN operators follow a two-phase calling convention:

1. **Phase 1**: `xxxGetWorkspaceSize` - Computes workspace size and creates executor
2. **Phase 2**: `xxx` - Executes the actual operation using the workspace

The `EXEC_ACLNN_CMD` macro handles both phases automatically.

---

## Conversion Workflow

For each HLO operator conversion, you must complete **four steps**:

| Step | File | Purpose |
|------|------|---------|
| 1 | `thunk_emitter.cc` | Implement pattern matching function (`IsXxxFusion`) |
| 2 | `thunk_emitter.cc` | Implement thunk emission function (`EmitXxxFusion`) |
| 3 | `thunk_emitter.cc` | Register in `EmitFusion` function |
| 4 | `aclnn_thunk.cc` | Add executor in `kOpExecutors` map |

---

## Step 1: Implement Pattern Matching Function

**File**: `/data3/zhongzhw/code/google/jax/xla/xla/service/ascend/thunk_emitter.cc`

**Location**: Place near other `IsXxxFusion` functions (around line 167)

### Function Signature

```cpp
bool IsXxxFusion(const HloFusionInstruction* fusion);
```

### Purpose

Determines whether a given HLO fusion instruction matches the target pattern.

### Implementation Pattern

```cpp
bool IsXxxFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  // Get all instructions in the fused computation
  const auto& instructions = computation->instructions();

  // Step 1: Check instruction count matches expected pattern
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != EXPECTED_COUNT) {
    VLOG(4) << "XxxFusion: expected " << EXPECTED_COUNT
            << " instructions, got " << instruction_count;
    return false;
  }

  // Step 2: Find and validate each expected instruction
  const HloInstruction* param_instr = nullptr;
  const HloInstruction* target_instr = nullptr;

  for (const auto* instr : instructions) {
    if (instr->opcode() == HloOpcode::kParameter) {
      param_instr = instr;
    } else if (instr->opcode() == HloOpcode::kXxx) {  // Target opcode
      target_instr = instr;
    } else {
      VLOG(4) << "XxxFusion: unexpected opcode "
              << HloOpcodeString(instr->opcode());
      return false;
    }
  }

  // Step 3: Validate both instructions exist
  if (!param_instr || !target_instr) {
    VLOG(4) << "XxxFusion: missing parameter or target instruction";
    return false;
  }

  // Step 4: Validate data flow (operand relationship)
  if (target_instr->operand_count() != EXPECTED_OPERAND_COUNT ||
      target_instr->operand(0) != param_instr) {
    VLOG(4) << "XxxFusion: target does not take parameter as operand";
    return false;
  }

  // Step 5: Validate root instruction
  if (computation->root_instruction() != target_instr) {
    VLOG(4) << "XxxFusion: fusion root is not target";
    return false;
  }

  // Step 6: Validate data type support
  PrimitiveType input_type = target_instr->operand(0)->shape().element_type();
  PrimitiveType output_type = target_instr->shape().element_type();

  if (input_type != output_type) {
    VLOG(4) << "XxxFusion: input and output types must be the same";
    return false;
  }

  // Step 7: Check against supported data types from aclnn documentation
  bool is_supported = (input_type == PrimitiveType::F32 ||
                       input_type == PrimitiveType::F16 ||
                       input_type == PrimitiveType::BF16);

  if (!is_supported) {
    VLOG(4) << "XxxFusion: unsupported data type: "
            << PrimitiveType_Name(input_type);
    return false;
  }

  return true;
}
```

### Key Validation Points

1. **Instruction count**: Verify the exact number of instructions in the fused computation
2. **Instruction types**: Find expected parameter and target instructions
3. **Operand relationship**: Ensure the target takes the parameter as its operand
4. **Root instruction**: Confirm the fusion's root is the target instruction
5. **Data type support**: Check against ACLNN documentation for supported types

---

## Step 2: Implement Thunk Emission Function

**File**: `/data3/zhongzhw/code/google/jax/xla/xla/service/ascend/thunk_emitter.cc`

**Location**: Place near other `EmitXxxFusion` functions (around line 3149)

### Function Signature

```cpp
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitXxxFusion(
    const HloFusionInstruction* fusion);
```

### Purpose

Creates the ACLNN thunk that will execute the operation on the device.

### Implementation Pattern

```cpp
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitXxxFusion(
    const HloFusionInstruction* fusion) {
  // Step 1: Get input and output buffer allocations
  TF_ASSIGN_OR_RETURN(auto input_slice, GetShapedSliceForHlo(fusion->operand(0)));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  // Step 2: Prepare containers for AclnnThunk construction
  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  std::vector<xla::ascend::AclnnThunk::Param> params;

  // Step 3: Add input/output slices
  operands.push_back(input_slice);
  results.push_back(output_slice);

  // Step 4: If operator requires scalar parameters, extract and add them here
  // Example: params.push_back(xla::ascend::AclnnThunk::Param{scalar_value});

  VLOG(2) << "Emitting xxx fusion as aclnnXxx: " << fusion->name();

  // Step 5: Create the AclnnThunk
  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
      "aclnnXxx",  // Must match the key in kOpExecutors
      std::move(operands),
      std::move(results),
      std::move(params));

  // Step 6: Add to sequence and return
  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}
```

### Parameter Extraction Guidelines

| ACLNN Parameter Type | Extraction Method | Param Storage |
|---------------------|-------------------|----------------|
| Tensor (input/output) | `GetShapedSliceForHlo()` | operands/results |
| Scalar (float, int, bool) | Extract from HLO instruction | params list |

### AclnnThunk Constructor Parameters

| Parameter | Description |
|-----------|-------------|
| `ThunkInfo` | Profile annotation with unique thunk ID |
| `op_name` | ACLNN operator name (e.g., `"aclnnTanh"`) |
| `operands` | Vector of input tensor slices |
| `results` | Vector of output tensor slices |
| `params` | Vector of non-tensor parameters (empty if none) |

---

## Step 3: Register in EmitFusion

**File**: `/data3/zhongzhw/code/google/jax/xla/xla/service/ascend/thunk_emitter.cc`

**Location**: Inside `EmitFusion` function (around line 1430)

### Where to Add

Find the section where other patterns are matched and add your pattern:

```cpp
// Inside EmitFusion function, around line 1556

// Pattern 2.5: tanh -> aclnnTanh
if (IsTanhFusion(fusion)) {
  TF_ASSIGN_OR_RETURN(auto thunks, EmitTanhFusion(fusion));
  if (!thunks.empty()) {
    return thunks;
  }
}

// Add your pattern here:
// Pattern X: xxx -> aclnnXxx
if (IsXxxFusion(fusion)) {
  TF_ASSIGN_OR_RETURN(auto thunks, EmitXxxFusion(fusion));
  if (!thunks.empty()) {
    return thunks;
  }
}

// Continue with other patterns...
```

### Pattern Ordering

Place more specific patterns before general patterns. The patterns are checked in order, and the first matching pattern wins.

---

## Step 4: Add Executor in AclnnThunk

**File**: `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/runtime/aclnn_thunk.cc`

**Location**: Inside `kOpExecutors` map (around line 50)

### Executor Function Signature

```cpp
[](const AclnnThunk::ExecuteParams& params,
   se::Stream* stream,
   const std::vector<NullableShapedSlice>& operands,
   const std::vector<NullableShapedSlice>& results,
   const std::vector<AclnnThunk::Param>& params_list,
   const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status
```

### Implementation Pattern

```cpp
{
  "aclnnXxx",
  [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
     const std::vector<NullableShapedSlice>& operands,
     const std::vector<NullableShapedSlice>& results,
     const std::vector<AclnnThunk::Param>& params_list,
     const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {

    // Step 1: Validate operand and result counts
    CHECK(operands.size() == EXPECTED_INPUT_COUNT &&
          results.size() == EXPECTED_OUTPUT_COUNT)
        << "aclnnXxx requires X inputs and Y outputs";

    // Step 2: Extract scalar parameters if needed
    // auto scalar_param = std::get<float>(params_list[0]);

    // Step 3: Call EXEC_ACLNN_CMD with operator-specific arguments
    // The arguments to EXEC_ACLNN_CMD must match xxxGetWorkspaceSize EXCEPT:
    // - workspaceSize and executor are auto-filled by EXEC_ACLNN_CMD
    // - All input tensors use make_triplet()
    // - All output tensors use make_triplet()
    // - Non-tensor parameters are passed directly (or extracted from params_list)

    EXEC_ACLNN_CMD(aclnnXxx, stream,
                   make_triplet(operands[0]),  // input tensor 1
                   // ... more inputs if needed
                   make_triplet(results[0]));   // output tensor 1
                   // ... more outputs if needed
                   // scalar_param1, scalar_param2);  // non-tensor params if any

    return absl::OkStatus();
  }
}
```

### EXEC_ACLNN_CMD Argument Mapping

For an ACLNN operator with signature:

```cpp
aclnnStatus aclnnXxxGetWorkspaceSize(
    const aclTensor* self,    // -> make_triplet(operands[0])
    aclTensor* out,           // -> make_triplet(results[0])
    Scalar other,             // -> params_list[0] value
    uint64_t* workspaceSize, // AUTO - don't pass
    aclOpExecutor** executor  // AUTO - don't pass
)
```

The `EXEC_ACLNN_CMD` call should be:

```cpp
EXEC_ACLNN_CMD(aclnnXxx, stream,
                make_triplet(operands[0]),
                std::get<ScalarType>(params_list[0]),
                make_triplet(results[0]));
```

### TensorTriplet Helper

The `make_triplet` function creates a `TensorTriplet` from a `NullableShapedSlice`:

```cpp
auto make_triplet = [&](const NullableShapedSlice& slice) -> TensorTriplet {
  return TensorTriplet{
    params.buffer_allocations,
    slice.value().slice,
    slice.value().shape
  };
};
```

---

## Complete Example: tanh Conversion

This section documents the complete conversion of the tanh HLO operator to aclnnTanh.

### HLO Fusion Representation

```
%wrapped_tanh = f32[32,32]{1,0} fusion(%get-tuple-element.1), kind=kLoop, calls=
  (param_0.2: f32[32,32]) -> f32[32,32] {
    %param_0.2 = f32[32,32]{1,0} parameter(0)
    ROOT %tanh.1.1 = f32[32,32]{1,0} tanh(%param_0.2),
          metadata={op_name="jit(matmul_with_elementwise)/tanh"}
  }
```

**Analysis**:
- Fusion kind: `kLoop`
- Parameter count: 1 (f32[32,32])
- Operation: Single `tanh` operation
- Data type: f32 (FLOAT)
- Supported: f32, f16, bf16

### ACLNN Interface (aclnnTanh)

From the ACLNN documentation:

```cpp
aclnnStatus aclnnTanhGetWorkspaceSize(
  const aclTensor* self,    // Input tensor
  aclTensor* out,           // Output tensor
  uint64_t* workspaceSize, // AUTO
  aclOpExecutor** executor  // AUTO
)

aclnnStatus aclnnTanh(
  void* workspace,
  uint64_t workspaceSize,
  aclOpExecutor* executor,
  const aclrtStream stream
)
```

### Implementation

#### Step 1: IsTanhFusion (thunk_emitter.cc:167-242)

```cpp
bool IsTanhFusion(const HloFusionInstruction* fusion) {
  auto* computation = fusion->fused_instructions_computation();

  const auto& instructions = computation->instructions();
  int64_t instruction_count = std::distance(instructions.begin(), instructions.end());
  if (instruction_count != 2) {
    VLOG(4) << "TanhFusion: expected 2 instructions, got "
            << instruction_count;
    return false;
  }

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

  if (!param_instr || !tanh_instr) {
    VLOG(4) << "TanhFusion: missing parameter or tanh instruction";
    return false;
  }

  if (tanh_instr->operand_count() != 1 ||
      tanh_instr->operand(0) != param_instr) {
    VLOG(4) << "TanhFusion: tanh does not take parameter as operand";
    return false;
  }

  if (computation->root_instruction() != tanh_instr) {
    VLOG(4) << "TanhFusion: fusion root is not tanh";
    return false;
  }

  PrimitiveType input_type = tanh_instr->operand(0)->shape().element_type();
  PrimitiveType output_type = tanh_instr->shape().element_type();

  if (input_type != output_type) {
    VLOG(4) << "TanhFusion: input and output types must be the same";
    return false;
  }

  bool is_supported = (input_type == PrimitiveType::F32 ||
                       input_type == PrimitiveType::F16 ||
                       input_type == PrimitiveType::BF16);

  if (!is_supported) {
    VLOG(4) << "TanhFusion: unsupported data type: "
            << PrimitiveType_Name(input_type);
    return false;
  }

  return true;
}
```

#### Step 2: EmitTanhFusion (thunk_emitter.cc:3149-3181)

```cpp
absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitTanhFusion(
    const HloFusionInstruction* fusion) {
  TF_ASSIGN_OR_RETURN(auto input_slice, GetShapedSliceForHlo(fusion->operand(0)));
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(fusion));

  std::vector<NullableShapedSlice> operands;
  std::vector<NullableShapedSlice> results;
  std::vector<xla::ascend::AclnnThunk::Param> params;

  operands.push_back(input_slice);
  results.push_back(output_slice);

  VLOG(2) << "Emitting tanh fusion as aclnnTanh: " << fusion->name();

  auto thunk = std::make_unique<xla::ascend::AclnnThunk>(
      xla::gpu::Thunk::ThunkInfo::WithProfileAnnotation(fusion, ir_emitter_context_->GetNextThunkId()),
      "aclnnTanh",
      std::move(operands),
      std::move(results),
      std::move(params));

  xla::gpu::ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  return sequence;
}
```

#### Step 3: Registration in EmitFusion (thunk_emitter.cc:1556-1562)

```cpp
// Pattern 2.5: tanh -> aclnnTanh
if (IsTanhFusion(fusion)) {
  TF_ASSIGN_OR_RETURN(auto thunks, EmitTanhFusion(fusion));
  if (!thunks.empty()) {
    return thunks;
  }
}
```

#### Step 4: Executor in kOpExecutors (aclnn_thunk.cc:66-75)

```cpp
{
  "aclnnTanh",
  [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
     const std::vector<NullableShapedSlice>& operands,
     const std::vector<NullableShapedSlice>& results,
     const std::vector<AclnnThunk::Param>& params_list,
     const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {
    CHECK(operands.size() == 1 && results.size() == 1)
        << "aclnnTanh requires 1 input and 1 output";
    EXEC_ACLNN_CMD(aclnnTanh, stream,
                   make_triplet(operands[0]),
                   make_triplet(results[0]));
    return absl::OkStatus();
  }
}
```

---

## Common Patterns and Reference

### Pattern 1: Simple Element-wise Operations

For operations like `tanh`, `negate`, `exp`:

- **Instruction count**: 2 (parameter + operation)
- **Parameters**: None (params vector empty)
- **Example**: tanh, sigmoid

### Pattern 2: Operations with Scalar Parameters

For operations like `muls` (multiply scalar):

```cpp
// ACLNN: aclnnMulsGetWorkspaceSize(self, other, out, ...)
// HLO: param -> muls(param, scalar)
```

Extract scalar from HLO and store in params:
```cpp
float scalar_value = get_scalar_from_hlo(instr);
params.push_back(xla::ascend::AclnnThunk::Param{scalar_value});
```

### Pattern 3: Operations with Multiple Outputs

For operations like `argmax`:

```cpp
// ACLNN: aclnnMaxDimGetWorkspaceSize(self, dim, keepdim, max, maxIndex, ...)
// Results vector has 2 elements for max and maxIndex
```

### Validation Checklist

Before completing the conversion, verify:

- [ ] Pattern matching correctly identifies the HLO pattern
- [ ] All supported data types from ACLNN docs are handled
- [ ] Unsupported data types are rejected with VLOG message
- [ ] Operand/result counts match ACLNN requirements
- [ ] Scalar parameters are correctly extracted and passed
- [ ] Executor correctly maps operands/results to ACLNN parameters
- [ ] Registration in EmitFusion is in the correct location

### ACLNN Data Type Mapping

| HLO PrimitiveType | ACLNN Data Type |
|------------------|-----------------|
| F32 | ACL_FLOAT |
| F16 | ACL_FLOAT16 |
| BF16 | ACL_BFLOAT16 |
| S32 | ACL_INT32 |
| S64 | ACL_INT64 |
| U8 | ACL_UINT8 |
| U32 | ACL_UINT32 |

---

## Troubleshooting

### Common Issues

1. **"Unsupported aclnn operation" error**
   - The executor key doesn't match the op_name passed to AclnnThunk
   - Check spelling: `"aclnnTanh"` vs `"aclnnTanh "`

2. **"requires X inputs and Y outputs" assertion failure**
   - operands/results counts don't match ACLNN signature
   - Verify GetShapedSliceForHlo is called correctly

3. **Wrong results**
   - Check operand order matches ACLNN parameter order
   - Verify scalar parameter extraction from correct HLO instruction

4. **Type errors**
   - Ensure Param types match what std::get expects
   - Use correct C++ type for ACLNN parameter (float, int64_t, bool)

### Debug Logging

Enable VLOG for debugging:
```bash
GLOG_v=4 ./your_test --tanh_fusion=true
```

Look for VLOG(4) messages from your IsXxxFusion function.
