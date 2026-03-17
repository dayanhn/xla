HloModule jit_test_ffi_gelu_fwd, entry_computation_layout={(f32[1024,1024]{1,0})->f32[1024,1024]{1,0}}

ENTRY main.1 {
  x.1 = f32[1024,1024]{1,0} parameter(0)
  ROOT ffi_call.1 = f32[1024,1024]{1,0} custom-call(x.1), custom_call_target="ascend.gelu", operand_layout_constraints={f32[1024,1024]{1,0}}, api_version=API_VERSION_TYPED_FFI, backend_config={}
}