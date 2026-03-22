HloModule jit_test_ffi_matmul_then_gelu, entry_computation_layout={(f32[16,16]{1,0}, f32[16,16]{1,0})->f32[16,16]{1,0}}

ENTRY main.1 {
  A.1 = f32[16,16]{1,0} parameter(0)
  B.1 = f32[16,16]{1,0} parameter(1)
  ffi_call.2 = f32[16,16]{1,0} custom-call(A.1, B.1), custom_call_target="ascend.matmul", operand_layout_constraints={f32[16,16]{1,0}, f32[16,16]{1,0}}, api_version=API_VERSION_TYPED_FFI, backend_config={}
  ROOT ffi_call.3 = f32[16,16]{1,0} custom-call(ffi_call.2), custom_call_target="ascend.gelu", operand_layout_constraints={f32[16,16]{1,0}}, api_version=API_VERSION_TYPED_FFI, backend_config={}
}