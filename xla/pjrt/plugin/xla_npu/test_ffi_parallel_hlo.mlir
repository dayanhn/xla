HloModule jit__fun, entry_computation_layout={(f32[8,256,2048]{2,1,0}, f32[8,2048,2048]{2,1,0})->f32[8,256,2048]{2,1,0}}, frontend_attributes={xla.sdy.meshes={mesh = #sdy.mesh<["<axis 0xfffeaa4e91c0>"=8]>}}

xla.sdy.manual_computation_body.1 {
  shard_map.5 = f32[1,256,2048]{2,1,0} parameter(0)
  squeeze.2 = f32[256,2048]{1,0} reshape(shard_map.5)
  shard_map.6 = f32[1,2048,2048]{2,1,0} parameter(1)
  squeeze.3 = f32[2048,2048]{1,0} reshape(shard_map.6)
  ffi_call.1 = f32[256,2048]{1,0} custom-call(squeeze.2, squeeze.3), custom_call_target="ascend.matmul", operand_layout_constraints={f32[256,2048]{1,0}, f32[2048,2048]{1,0}}, api_version=API_VERSION_TYPED_FFI, backend_config={}
  ROOT broadcast_in_dim.1 = f32[1,256,2048]{2,1,0} reshape(ffi_call.1)
}

ENTRY main.2 {
  flat_args_0_.1 = f32[8,256,2048]{2,1,0} parameter(0), sharding={devices=[8,1,1]<=[8]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<@mesh, [{\"<axis 0xfffeaa4e91c0>\"}, {}, {}]>"}
  flat_args_1_.1 = f32[8,2048,2048]{2,1,0} parameter(1), sharding={devices=[8,1,1]<=[8]}, frontend_attributes={xla.sdy.sharding="#sdy.sharding<@mesh, [{\"<axis 0xfffeaa4e91c0>\"}, {}, {}]>"}
  shard_map.10 = (f32[1,256,2048]{2,1,0}, f32[1,2048,2048]{2,1,0}) custom-call(flat_args_0_.1, flat_args_1_.1), custom_call_target="xla.sdy.GlobalToLocalShape", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.in_shardings="#sdy.sharding_per_value<[<@mesh, [{\"<axis 0xfffeaa4e91c0>\"}, {}, {}]>, <@mesh, [{\"<axis 0xfffeaa4e91c0>\"}, {}, {}]>]>",xla.sdy.manual_axes="#sdy<manual_axes{\"<axis 0xfffeaa4e91c0>\"}>"}
  shard_map.11 = f32[1,256,2048]{2,1,0} get-tuple-element(shard_map.10), index=0
  shard_map.12 = f32[1,2048,2048]{2,1,0} get-tuple-element(shard_map.10), index=1
  shard_map.13 = f32[1,256,2048]{2,1,0} call(shard_map.11, shard_map.12), to_apply=xla.sdy.manual_computation_body.1, frontend_attributes={inlineable="false"}
  shard_map.14 = f32[8,256,2048]{2,1,0} custom-call(shard_map.13), custom_call_target="xla.sdy.LocalToGlobalShape", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.manual_axes="#sdy<manual_axes{\"<axis 0xfffeaa4e91c0>\"}>",xla.sdy.out_shardings="#sdy.sharding_per_value<[<@mesh, [{\"<axis 0xfffeaa4e91c0>\"}, {}, {}]>]>"}
  shard_map.15 = f32[8,256,2048]{2,1,0} custom-call(shard_map.14), custom_call_target="xla.sdy.FuncResultSharding", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<@mesh, [{\"<axis 0xfffeaa4e91c0>\"}, {}, {}]>]>"}
  tuple.1 = (f32[8,256,2048]{2,1,0}) tuple(shard_map.15)
  ROOT get-tuple-element.1 = f32[8,256,2048]{2,1,0} get-tuple-element(tuple.1), index=0, sharding={devices=[8,1,1]<=[8]}
}