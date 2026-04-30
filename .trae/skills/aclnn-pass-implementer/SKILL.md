---
name: "aclnn-pass-implementer"
description: "Implements ACLNN optimization passes for Ascend NPU. Invoke when user needs to add support for new ACLNN operators via custom HLO-to-custom-call transformation passes."
---

# ACLNN Optimization Pass Implementer

This document provides a comprehensive guide for implementing ACLNN optimization passes that convert HLO instructions to ACLNN custom-call operations for the Ascend NPU backend.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Analyze ACLNN Operator Documentation](#step-1-analyze-aclnn-operator-documentation)
4. [Step 2: Create Pass Implementation](#step-2-create-pass-implementation)
5. [Step 3: Add Config Serialization](#step-3-add-config-serialization)
6. [Step 4: Update BUILD Configuration](#step-4-update-build-configuration)
7. [Step 5: Register Target Identifier](#step-5-register-target-identifier)
8. [Step 6: Add Thunk Emitter Support](#step-6-add-thunk-emitter-support)
9. [Step 7: Declare Emit Interface](#step-7-declare-emit-interface)
10. [Step 8: Register ACLNN Executor](#step-8-register-aclnn-executor)
11. [Complete Example: aclnnGemm Rewriter](#complete-example-aclnngemm-rewriter)
12. [Validation Checklist](#validation-checklist)

---

## Overview

The ACLNN optimization pass converts specific HLO instructions to `custom_call` operations that invoke ACLNN operators. The complete workflow involves:

1. **Pattern Matching**: Identify HLO instructions that can be converted to ACLNN operators
2. **Custom-Call Generation**: Transform matched HLO instructions to `custom_call` with appropriate backend configuration
3. **Thunk Emission**: Generate device thunks that execute the ACLNN operators
4. **Integration**: Wire everything into the XLA compilation pipeline

---

## Prerequisites

Before starting, you need:
1. ACLNN operator documentation (e.g., `/ascend/ops-nn/matmul/gemm/docs/aclnnGemm.md`)
2. Understanding of the target HLO instruction structure
3. Access to the JAX/XLA codebase with Ascend backend

---

## Step 1: Analyze ACLNN Operator Documentation

**Input**: ACLNN operator documentation (e.g., `aclnnGemm.md`)

### Key Information to Extract

| Item | Description | Example from aclnnGemm |
|------|-------------|------------------------|
| **Supported Functions** | What operations the operator performs | Matrix multiplication with optional bias |
| **Input Parameters** | Tensor inputs and their constraints | `a: f32[m,k]`, `b: f32[k,n]`, `bias: f32[n]` |
| **Output Shape** | Expected output dimensions | `f32[m,n]` |
| **Non-tensor Params** | Scalar parameters like alpha, beta, transpose flags | `transpose_a`, `transpose_b`, `alpha`, `beta` |
| **Data Type Support** | Supported element types | F16, BF16, F32 |
| **Layout Constraints** | Required memory layouts | row-major, column-major |

### Example Analysis for aclnnGemm

```cpp
// ACLNN Interface
aclnnStatus aclnnGemmGetWorkspaceSize(
    const aclTensor* a,           // Input matrix A
    const aclTensor* b,           // Input matrix B
    const aclTensor* bias,        // Optional bias vector
    const float alpha,            // Scaling factor for AB
    const float beta,             // Scaling factor for bias
    const bool transpose_a,       // Transpose A before multiplication
    const bool transpose_b,       // Transpose B before multiplication
    aclTensor* out,               // Output matrix
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

---

## Step 2: Create Pass Implementation

**Files to Create**:
- `xla/backends/ascend/transforms/aclnn_xxx_rewriter.h`
- `xla/backends/ascend/transforms/aclnn_xxx_rewriter.cc`

### 2.1 Header File (`.h`)

```cpp
#ifndef XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_XXX_REWRITER_H_
#define XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_XXX_REWRITER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace ascend {

class AclnnXxxRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "aclnn-xxx-rewriter"; }
  
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_TRANSFORMS_ACLNN_XXX_REWRITER_H_
```

### 2.2 Implementation File (`.cc`)

**Key Components**:

#### A. Pattern Matching Function

```cpp
// Check if an HLO instruction can be converted to ACLNN operator
absl::StatusOr<bool> IsAclnnSupportedXxx(const HloInstruction& instr) {
  // Step 1: Check opcode
  if (instr.opcode() != HloOpcode::kDot) {
    return false;
  }
  
  // Step 2: Validate dimensions
  // Example: Ensure contracting dimensions are in standard positions
  const auto& dot_dims = instr.dot_dimension_numbers();
  if (dot_dims.lhs_contracting_dimensions(0) != 
      instr.operand(0)->shape().dimensions().size() - 1) {
    return false;
  }
  
  // Step 3: Check data type support
  PrimitiveType type = instr.shape().element_type();
  if (type != F32 && type != F16 && type != BF16) {
    return false;
  }
  
  return true;
}
```

#### B. Conversion Logic

```cpp
// Convert HLO to custom_call
absl::Status HandleXxx(HloInstruction* instr) {
  // Step 1: Create config object
  auto config = std::make_unique<AclnnXxxConfig>();
  
  // Step 2: Extract parameters from HLO
  config->alpha = 1.0f;
  config->beta = 0.0f;
  config->transpose_a = /* calculate from HLO */;
  config->transpose_b = /* calculate from HLO */;
  
  // Step 3: Get operands
  HloInstruction* lhs = instr->mutable_operand(0);
  HloInstruction* rhs = instr->mutable_operand(1);
  
  // Step 4: Determine output shape and layout
  Shape output_shape = instr->shape();
  // Adjust layout if necessary based on transpose configuration
  if (config->transpose_a || config->transpose_b) {
    Layout output_layout = output_shape.layout();
    std::vector<int64_t> minor_to_major(
        output_layout.minor_to_major().rbegin(), 
        output_layout.minor_to_major().rend());
    output_layout.set_minor_to_major(minor_to_major);
    output_shape.set_layout(output_layout);
  }
  
  // Step 5: Create custom_call instruction
  HloInstruction* custom_call = instr->AddInstruction(
      HloInstruction::CreateCustomCall(
          output_shape,
          {lhs, rhs},
          kAclnnXxxCallTarget));  // Defined in aclnn_targets.h
  
  // Step 6: Attach backend config
  custom_call->set_raw_backend_config_string(
      SerializeAclnnConfig(*config));
  
  // Step 7: Replace original instruction
  TF_RETURN_IF_ERROR(ReplaceInstruction(instr, custom_call));
  
  return absl::OkStatus();
}
```

#### C. Pass Run Method

```cpp
absl::StatusOr<bool> AclnnXxxRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  
  for (auto* computation : module->computations()) {
    for (auto* instr : computation->MakeInstructionPostOrder()) {
      // Match and convert
      TF_ASSIGN_OR_RETURN(bool is_supported, IsAclnnSupportedXxx(*instr));
      if (is_supported) {
        TF_RETURN_IF_ERROR(HandleXxx(instr));
        changed = true;
      }
      
      // Optional: Handle fusion with already-converted operators
      // Example: fuse bias add with existing gemm custom_call
      if (instr->opcode() == HloOpcode::kAdd) {
        TF_ASSIGN_OR_RETURN(bool fused, FuseXxxWithBias(instr));
        if (fused) changed = true;
      }
    }
  }
  
  return changed;
}
```

#### D. Fusion Handling (Optional)

When fusing subsequent operators with already-converted custom_calls:

```cpp
absl::StatusOr<bool> FuseXxxWithBias(HloInstruction* instr) {
  // Check if this is a bias add pattern
  HloInstruction *gemm, *bias;
  if (!MatchBiasAddPattern(instr, &gemm, &bias)) {
    return false;
  }
  
  // Ensure gemm is already a custom_call
  if (gemm->custom_call_target() != kAclnnXxxCallTarget) {
    return false;
  }
  
  // Parse existing config and modify
  TF_ASSIGN_OR_RETURN(auto config, ParseAclnnConfig(
      kAclnnXxxCallTarget, gemm->raw_backend_config_string()));
  auto* xxx_config = dynamic_cast<AclnnXxxConfig*>(config.get());
  xxx_config->has_bias = true;
  xxx_config->beta = 1.0f;
  
  // Create new custom_call with bias as additional operand
  std::vector<HloInstruction*> operands(gemm->operands().begin(), 
                                        gemm->operands().end());
  operands.push_back(bias->mutable_operand(0));
  
  HloInstruction* fused = instr->AddInstruction(
      HloInstruction::CreateCustomCall(
          instr->shape(),
          operands,
          kAclnnXxxCallTarget));  // Target remains the same
  
  fused->set_raw_backend_config_string(SerializeAclnnConfig(*config));
  
  TF_RETURN_IF_ERROR(ReplaceInstruction(instr, fused));
  return true;
}
```

---

## Step 3: Add Config Serialization

**Files to Modify**:
- `xla/backends/ascend/transforms/aclnn_config.h`
- `xla/backends/ascend/transforms/aclnn_config.cc`

### 3.1 Add Config Struct to Header

```cpp
// In aclnn_config.h
struct AclnnXxxConfig : public AclnnConfigBase {
  float alpha = 1.0f;
  float beta = 0.0f;
  bool transpose_a = false;
  bool transpose_b = false;
  bool has_bias = false;
  
  std::string Serialize() const override;
  static absl::StatusOr<std::unique_ptr<AclnnConfigBase>> Parse(
      const std::string& config_str);
};
```

### 3.2 Implement Serialize/Parse in .cc

```cpp
// In aclnn_config.cc
std::string AclnnXxxConfig::Serialize() const {
  std::stringstream ss;
  ss << "alpha=" << alpha << ","
     << "beta=" << beta << ","
     << "transpose_a=" << transpose_a << ","
     << "transpose_b=" << transpose_b << ","
     << "has_bias=" << has_bias;
  return ss.str();
}

absl::StatusOr<std::unique_ptr<AclnnConfigBase>> AclnnXxxConfig::Parse(
    const std::string& config_str) {
  auto config = std::make_unique<AclnnXxxConfig>();
  
  // Parse config_str and populate fields
  // Example parsing logic
  std::vector<std::string> parts = absl::StrSplit(config_str, ',');
  for (const auto& part : parts) {
    std::vector<std::string> kv = absl::StrSplit(part, '=');
    if (kv[0] == "alpha") config->alpha = std::stof(kv[1]);
    else if (kv[0] == "beta") config->beta = std::stof(kv[1]);
    else if (kv[0] == "transpose_a") config->transpose_a = (kv[1] == "1");
    else if (kv[0] == "transpose_b") config->transpose_b = (kv[1] == "1");
    else if (kv[0] == "has_bias") config->has_bias = (kv[1] == "1");
  }
  
  return config;
}
```

### 3.3 Register Config in Factory

Add to the config parsing factory:

```cpp
// In aclnn_config.cc
std::unordered_map<std::string, ConfigParser> config_parsers = {
    {kAclnnXxxCallTarget, AclnnXxxConfig::Parse},
    // ... other configs
};
```

---

## Step 4: Update BUILD Configuration

**File**: `xla/backends/ascend/transforms/BUILD`

### 4.1 Add New Pass Library

Add new cc_library target with complete dependencies:

```python
cc_library(
    name = "aclnn_xxx_rewriter",
    srcs = ["aclnn_xxx_rewriter.cc"],
    hdrs = ["aclnn_xxx_rewriter.h"],
    deps = [
        ":aclnn_config",
        ":aclnn_targets",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:dfs_hlo_visitor_with_default",
        "//xla/service:hlo_pass_interface",
        "//xla/service:pattern_matcher",
        "//xla:shape_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
```

### 4.2 Add Dependency to aclnn_fusion_pass

Update the `aclnn_fusion_pass` target to include the new pass:

```python
cc_library(
    name = "aclnn_fusion_pass",
    srcs = ["aclnn_fusion_pass.cc"],
    hdrs = ["aclnn_fusion_pass.h"],
    deps = [
        ":aclnn_config",
        ":aclnn_gemm_rewriter",
        ":aclnn_xxx_rewriter",  # Add new pass dependency
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
    alwayslink = True,
)
```

---

## Step 5: Add Pass to Fusion Pipeline

**File**: `xla/backends/ascend/transforms/aclnn_fusion_pass.cc`

Add the new pass to the `RunAclnnFusionPass` function:

```cpp
absl::StatusOr<bool> RunAclnnFusionPass(
    HloModule* module,
    const xla::gpu::GpuTargetConfig& gpu_target_config) {
  HloPassPipeline pipeline("aclnn-fusion");
  
  // Existing passes
  pipeline.AddPass<AclnnGemmRewriter>(gpu_target_config.gpu_compute_capability());
  
  // Add new pass here
  pipeline.AddPass<AclnnXxxRewriter>();
  
  return pipeline.Run(module);
}
```

### Key Notes

1. **Pass Order Matters**: The order of passes in the pipeline affects the transformation results. Place the new pass in an appropriate position.

2. **Dependencies**: Ensure the new pass's header is included:
   ```cpp
   #include "xla/backends/ascend/transforms/aclnn_xxx_rewriter.h"
   ```

3. **Pass Parameters**: If the new pass requires parameters (like `gpu_compute_capability`), pass them to the constructor.

---

## Step 6: Register Target Identifier

**Files to Modify**:
- `xla/backends/ascend/transforms/aclnn_targets.h`
- `xla/backends/ascend/transforms/aclnn_targets.cc`

### 6.1 Add Target Constant to Header

```cpp
// In aclnn_targets.h
constexpr absl::string_view kAclnnXxxCallTarget = "__aclnn$xxx";
```

### 6.2 Add Target Check Function

```cpp
// In aclnn_targets.cc
bool IsAclnnXxxTarget(absl::string_view target) {
  return target == kAclnnXxxCallTarget;
}

// Add to IsAclnnTarget function
bool IsAclnnTarget(absl::string_view target) {
  return IsAclnnGemmTarget(target) || IsAclnnXxxTarget(target);
}
```

---

## Step 7: Add Thunk Emitter Support

**Files to Modify**:
- `xla/service/ascend/thunk_emitter.cc`

### 7.1 Add Target Matching in EmitHloInstruction

```cpp
// Around line 4519-4520
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* instr) {
  // ... existing code ...
  
  if (instr->opcode() == HloOpcode::kCustomCall) {
    const std::string& target = instr->custom_call_target();
    
    // Check for ACLNN targets
    if (ascend::IsAclnnXxxTarget(target)) {
      return EmitAclnnXxxThunk(instr);
    }
    // ... other targets ...
  }
  
  // ... rest of function ...
}
```

### 7.2 Implement Thunk Emission Function

```cpp
// Around line 4037-4121
absl::StatusOr<ThunkSequence> ThunkEmitter::EmitAclnnXxxThunk(
    const HloInstruction* instr) {
  // Step 1: Parse backend config
  TF_ASSIGN_OR_RETURN(auto config, ParseAclnnConfig(
      instr->custom_call_target(), 
      instr->raw_backend_config_string()));
  auto* xxx_config = dynamic_cast<AclnnXxxConfig*>(config.get());
  
  // Step 2: Get operand slices
  std::vector<NullableShapedSlice> operands;
  for (int i = 0; i < instr->operand_count(); ++i) {
    TF_ASSIGN_OR_RETURN(auto slice, GetShapedSliceForHlo(instr->operand(i)));
    operands.push_back(slice);
  }
  
  // Step 3: Get output slice
  TF_ASSIGN_OR_RETURN(auto output_slice, GetShapedSliceForHlo(instr));
  std::vector<NullableShapedSlice> results = {output_slice};
  
  // Step 4: Prepare non-tensor parameters
  std::vector<ascend::AclnnThunk::Param> params;
  params.push_back(ascend::AclnnThunk::Param{xxx_config->alpha});
  params.push_back(ascend::AclnnThunk::Param{xxx_config->beta});
  params.push_back(ascend::AclnnThunk::Param{(int64_t)xxx_config->transpose_a});
  params.push_back(ascend::AclnnThunk::Param{(int64_t)xxx_config->transpose_b});
  params.push_back(ascend::AclnnThunk::Param{(int64_t)xxx_config->has_bias});
  
  // Step 5: Create thunk
  auto thunk = std::make_unique<ascend::AclnnThunk>(
      ThunkInfo::WithProfileAnnotation(instr, ir_emitter_context_->GetNextThunkId()),
      "aclnnXxx",  // Must match executor key
      std::move(operands),
      std::move(results),
      std::move(params));
  
  ThunkSequence sequence;
  sequence.push_back(std::move(thunk));
  return sequence;
}
```

---

## Step 8: Declare Emit Interface

**File**: `xla/service/ascend/thunk_emitter.h`

Add declaration for the new emit function:

```cpp
// Around line 157
class ThunkEmitter {
  // ... existing declarations ...
  
 private:
  // ACLNN thunk emission functions
  absl::StatusOr<ThunkSequence> EmitAclnnXxxThunk(const HloInstruction* instr);
  // ... other emit functions ...
};
```

---

## Step 9: Register ACLNN Executor

**File**: `xla/backends/ascend/runtime/aclnn_thunk.cc`

This step registers the ACLNN operator executor in the `kOpExecutors` map, which is responsible for calling the actual ACLNN runtime functions.

### 9.1 Locate the kOpExecutors Map

Find the `kOpExecutors` map in `aclnn_thunk.cc` (around line 50):

```cpp
const std::unordered_map<std::string, AclnnThunk::OpExecutor> kOpExecutors = {
    // ... existing operators ...
};
```

### 9.2 Add Executor Entry

Add a new entry to the map with the operator name matching the thunk's `op_name` parameter:

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

### 9.3 Implementation Guidelines

#### Parameter Order
The arguments to `EXEC_ACLNN_CMD` must match the `aclnnXxxGetWorkspaceSize` signature **except**:
- `workspaceSize` and `executor` parameters are auto-filled by the macro
- Input tensors use `make_triplet(operands[i])`
- Output tensors use `make_triplet(results[i])`
- Non-tensor parameters are passed directly (or extracted from `params_list`)

#### Example: aclnnGemm Executor

```cpp
{
  "aclnnGemm",
  [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
     const std::vector<NullableShapedSlice>& operands,
     const std::vector<NullableShapedSlice>& results,
     const std::vector<AclnnThunk::Param>& params_list,
     const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {

    CHECK(operands.size() == 3 && results.size() == 1)
        << "aclnnGemm requires 3 inputs and 1 output";

    float alpha = std::get<float>(params_list[0]);
    float beta = std::get<float>(params_list[1]);
    int64_t transA = std::get<int64_t>(params_list[2]);
    int64_t transB = std::get<int64_t>(params_list[3]);

    EXEC_ACLNN_CMD(aclnnGemm, stream,
                   make_triplet(operands[0]),   // input A
                   make_triplet(operands[1]),   // input B
                   make_triplet(operands[2]),   // bias (or output for no bias)
                   alpha, beta,
                   static_cast<bool>(transA),
                   static_cast<bool>(transB),
                   make_triplet(results[0]));   // output

    return absl::OkStatus();
  }
}
```

#### Example: aclnnConvolution Executor

```cpp
{
  "aclnnConvolution",
  [](const AclnnThunk::ExecuteParams& params, se::Stream* stream,
     const std::vector<NullableShapedSlice>& operands,
     const std::vector<NullableShapedSlice>& results,
     const std::vector<AclnnThunk::Param>& params_list,
     const std::function<TensorTriplet(const NullableShapedSlice&)>& make_triplet) -> absl::Status {

    CHECK(operands.size() >= 2 && results.size() == 1)
        << "aclnnConvolution requires at least 2 inputs and 1 output";

    bool has_bias = operands.size() > 2;
    
    std::vector<int64_t> stride = {std::get<int64_t>(params_list[0]), 
                                    std::get<int64_t>(params_list[1])};
    std::vector<int64_t> padding = {std::get<int64_t>(params_list[2]), 
                                     std::get<int64_t>(params_list[3]),
                                     std::get<int64_t>(params_list[4]), 
                                     std::get<int64_t>(params_list[5])};
    std::vector<int64_t> dilation = {std::get<int64_t>(params_list[6]), 
                                      std::get<int64_t>(params_list[7])};
    bool transposed = static_cast<bool>(std::get<int64_t>(params_list[8]));
    std::vector<int64_t> output_padding = {std::get<int64_t>(params_list[9]), 
                                            std::get<int64_t>(params_list[10])};
    int64_t groups = std::get<int64_t>(params_list[11]);
    int8_t cube_math_type = static_cast<int8_t>(std::get<int64_t>(params_list[12]));

    EXEC_ACLNN_CMD(aclnnConvolution, stream,
                   make_triplet(operands[0]),           // input
                   make_triplet(operands[1]),           // weight
                   has_bias ? make_triplet(operands[2]) : nullptr,  // bias (optional)
                   stride.data(), stride.size(),
                   padding.data(), padding.size(),
                   dilation.data(), dilation.size(),
                   transposed,
                   output_padding.data(), output_padding.size(),
                   groups,
                   make_triplet(results[0]),           // output
                   cube_math_type);

    return absl::OkStatus();
  }
}
```

### 9.4 Key Implementation Notes

1. **Operator Name Matching**: The key in `kOpExecutors` must exactly match the `op_name` parameter passed to `AclnnThunk` constructor.

2. **TensorTriplet Creation**: Use `make_triplet()` to convert `NullableShapedSlice` to `TensorTriplet` for all tensor parameters.

3. **Scalar Parameters**: Extract scalar parameters from `params_list` using `std::get<T>()` with the correct type.

4. **Argument Order**: Maintain the exact argument order as specified in the ACLNN operator's `GetWorkspaceSize` function.

5. **Optional Parameters**: For optional parameters (like bias), pass `nullptr` when not provided.

---

## Complete Example: aclnnGemm Rewriter

### Summary of Files Modified/Created

| File | Action | Purpose |
|------|--------|---------|
| `aclnn_gemm_rewriter.h` | Create | Pass declaration |
| `aclnn_gemm_rewriter.cc` | Create | Pass implementation with pattern matching and conversion |
| `aclnn_config.h` | Modify | Add `AclnnGemmConfig` struct |
| `aclnn_config.cc` | Modify | Add Serialize/Parse implementations |
| `aclnn_targets.h` | Modify | Add `kAclnnGemmCallTarget` |
| `aclnn_targets.cc` | Modify | Add `IsAclnnGemmTarget` function |
| `BUILD` | Modify | Add `aclnn_gemm_rewriter` library |
| `BUILD` | Modify | Add new pass dependency to `aclnn_fusion_pass` |
| `aclnn_fusion_pass.cc` | Modify | Add new pass to ACLNN fusion pipeline |
| `thunk_emitter.h` | Modify | Declare `EmitAclnnGemmThunk` |
| `thunk_emitter.cc` | Modify | Add target matching and thunk emission |

### Key Implementation Details

1. **Pattern Matching**: Checks for standard matrix multiplication patterns
2. **Layout Handling**: Adjusts output layout based on transpose configuration
3. **Fusion Support**: Supports fusing bias add operations
4. **Config Serialization**: Encodes all necessary parameters for runtime

---

## Validation Checklist

Before completing the implementation, verify:

- [ ] ACLNN operator documentation has been thoroughly analyzed
- [ ] Pattern matching correctly identifies supported HLO instructions
- [ ] Custom-call target matches the identifier in `aclnn_targets.h`
- [ ] Backend config includes all required parameters
- [ ] Output shape and layout are correctly computed
- [ ] BUILD file includes all necessary dependencies
- [ ] Target check function is implemented and registered
- [ ] Thunk emitter correctly parses config and creates thunk
- [ ] Header declares all new functions
- [ ] ACLNN executor is registered in `kOpExecutors` map
- [ ] Executor key matches the `op_name` passed to `AclnnThunk`
- [ ] Executor correctly calls `EXEC_ACLNN_CMD` with proper arguments
- [ ] Code compiles without errors
- [ ] Unit tests pass

---

## Troubleshooting

### Common Issues

1. **Custom-call not recognized**: Ensure target string matches exactly (check for typos)
2. **Incorrect output**: Verify layout handling and transpose flags
3. **Config parsing errors**: Check serialization/deserialization format
4. **Build failures**: Verify BUILD dependencies are complete
5. **Runtime errors**: Check ACLNN executor registration in `aclnn_thunk.cc`

### Debug Tips

- Enable VLOG to trace pattern matching: `GLOG_v=4 ./your_test`
- Check HLO dumps to verify custom-call generation
- Inspect backend_config in generated HLO IR
- Validate ACLNN executor registration in `kOpExecutors`
