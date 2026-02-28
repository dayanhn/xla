HloModule jit_matmul_with_elementwise, entry_computation_layout={(f32[32,32]{1,0}, f32[32,32]{1,0}, f32[3,3,1,8]{3,2,1,0})->f32[32,32]{1,0}}

%parameter_0 (parameter_0.1: f32[32,32]) -> f32[32,32] {
  ROOT %parameter_0.1 = f32[32,32]{1,0} parameter(0)
}

%parameter_1 (parameter_0.2: f32[32,32]) -> f32[32,32] {
  ROOT %parameter_0.2 = f32[32,32]{1,0} parameter(0)
}

%gemm_fusion_dot.1_computation (parameter_0: f32[32,32], parameter_1: f32[32,32]) -> f32[32,32] {
  %parameter_0 = f32[32,32]{1,0} parameter(0)
  %block_fusion = f32[32,32]{1,0} fusion(%parameter_0), kind=kCustom, calls=%parameter_0, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"num_warps":"4","output_tiles":[{"sizes":["32","32"]}],"num_ctas":1,"num_stages":1,"is_tma_allowed":false,"is_warp_specialization_allowed":false}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
  %parameter_1 = f32[32,32]{1,0} parameter(1)
  %block_fusion.1 = f32[32,32]{1,0} fusion(%parameter_1), kind=kCustom, calls=%parameter_1, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"num_warps":"4","output_tiles":[{"sizes":["32","16"]}],"num_ctas":1,"num_stages":1,"is_tma_allowed":false,"is_warp_specialization_allowed":false}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
  ROOT %dot.0 = f32[32,32]{1,0} dot(%block_fusion, %block_fusion.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={source_file="-" source_line=12 source_end_line=12 source_column=10 source_end_column=10}
}

%region_0.1 (Arg_0.2: f32[], Arg_1.2: f32[]) -> f32[] {
  %Arg_0.2 = f32[] parameter(0)
  %Arg_1.2 = f32[] parameter(1)
  ROOT %add.1.0 = f32[] add(%Arg_0.2, %Arg_1.2), metadata={source_file="-" source_line=7 source_end_line=7 source_column=26 source_end_column=26}
}

%fused_computation.4 (param_0.21: f32[32,32], param_1.27: f32[1,32,32,8]) -> f32[32,32] {
  %param_1.27 = f32[1,32,32,8]{3,2,1,0} parameter(1)
  %tanh.2.1 = f32[1,32,32,8]{3,2,1,0} tanh(%param_1.27), metadata={source_file="-" source_line=5 source_end_line=5 source_column=10 source_end_column=10}
  %bitcast.12.5 = f32[32,32,8]{2,1,0} bitcast(%tanh.2.1)
  %constant_9_1 = f32[] constant(0)
  %reduce.7 = f32[32,32]{1,0} reduce(%bitcast.12.5, %constant_9_1), dimensions={2}, to_apply=%region_0.1, metadata={source_file="-" source_line=7 source_end_line=7 source_column=26 source_end_column=26}
  %bitcast.13.3 = f32[1,32,32]{2,1,0} bitcast(%reduce.7)
  %constant_12 = f32[] constant(0.125)
  %broadcast.8 = f32[1,32,32]{2,1,0} broadcast(%constant_12), dimensions={}, metadata={source_file="-" source_line=9 source_end_line=9 source_column=10 source_end_column=10}
  %multiply.6 = f32[1,32,32]{2,1,0} multiply(%bitcast.13.3, %broadcast.8)
  %bitcast.1.3 = f32[32,32]{1,0} bitcast(%multiply.6)
  %param_0.21 = f32[32,32]{1,0} parameter(0)
  %tanh.3.3 = f32[32,32]{1,0} tanh(%param_0.21), metadata={source_file="-" source_line=13 source_end_line=13 source_column=10 source_end_column=10}
  %constant_7_1 = f32[] constant(2)
  %broadcast.6.3 = f32[32,32]{1,0} broadcast(%constant_7_1), dimensions={}, metadata={source_file="-" source_line=15 source_end_line=15 source_column=10 source_end_column=10}
  %multiply.2.3 = f32[32,32]{1,0} multiply(%tanh.3.3, %broadcast.6.3), metadata={source_file="-" source_line=16 source_end_line=16 source_column=11 source_end_column=11}
  %constant_6_1 = f32[] constant(0.1)
  %broadcast.5.5 = f32[32,32]{1,0} broadcast(%constant_6_1), dimensions={}, metadata={source_file="-" source_line=18 source_end_line=18 source_column=11 source_end_column=11}
  %add.4.5 = f32[32,32]{1,0} add(%multiply.2.3, %broadcast.5.5), metadata={source_file="-" source_line=19 source_end_line=19 source_column=11 source_end_column=11}
  %add.5.3 = f32[32,32]{1,0} add(%bitcast.1.3, %add.4.5), metadata={source_file="-" source_line=20 source_end_line=20 source_column=11 source_end_column=11}
  %constant_5_1 = f32[] constant(0.5)
  %broadcast.4.1 = f32[32,32]{1,0} broadcast(%constant_5_1), dimensions={}, metadata={source_file="-" source_line=22 source_end_line=22 source_column=11 source_end_column=11}
  ROOT %multiply.3.1 = f32[32,32]{1,0} multiply(%add.5.3, %broadcast.4.1), metadata={source_file="-" source_line=23 source_end_line=23 source_column=11 source_end_column=11}
}

%wrapped_transpose_computation (param_0.22: f32[3,3,1,8]) -> f32[8,3,3,1] {
  %param_0.22 = f32[3,3,1,8]{3,2,1,0} parameter(0)
  ROOT %transpose.2 = f32[8,3,3,1]{3,2,1,0} transpose(%param_0.22), dimensions={3,0,1,2}
}

ENTRY %main.2 (Arg_0.3: f32[32,32], Arg_1.3: f32[32,32], Arg_2.1: f32[3,3,1,8]) -> f32[32,32] {
  %Arg_0.3 = f32[32,32]{1,0} parameter(0)
  %Arg_1.3 = f32[32,32]{1,0} parameter(1)
  %gemm_fusion_dot.1 = f32[32,32]{1,0} fusion(%Arg_0.3, %Arg_1.3), kind=kCustom, calls=%gemm_fusion_dot.1_computation, metadata={source_file="-" source_line=12 source_end_line=12 source_column=10 source_end_column=10}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"num_warps":"4","output_tiles":[{"sizes":["32","16"]}],"num_ctas":1,"num_stages":1,"is_tma_allowed":false,"is_warp_specialization_allowed":false}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
  %bitcast.3.0 = f32[1,32,32,1]{3,2,1,0} bitcast(%Arg_0.3)
  %Arg_2.1 = f32[3,3,1,8]{3,2,1,0} parameter(2)
  %wrapped_transpose = f32[8,3,3,1]{3,2,1,0} fusion(%Arg_2.1), kind=kInput, calls=%wrapped_transpose_computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID","native_emitter_backend_config":{}}
  %cudnn-conv.1 = (f32[1,32,32,8]{3,2,1,0}, u8[0]{0}) custom-call(%bitcast.3.0, %wrapped_transpose), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward", metadata={source_file="-" source_line=4 source_end_line=4 source_column=10 source_end_column=10}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"side_input_scale":0,"algorithm":{"algo_id":"28","math_type":"DEFAULT_MATH","tuning_knobs":{"3":"0","2":"3"},"is_cudnn_frontend":false,"workspace_size":"0"},"leakyrelu_alpha":0},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
  %get-tuple-element.3 = f32[1,32,32,8]{3,2,1,0} get-tuple-element(%cudnn-conv.1), index=0
  ROOT %fusion.7 = f32[32,32]{1,0} fusion(%gemm_fusion_dot.1, %get-tuple-element.3), kind=kCustom, calls=%fused_computation.4, metadata={source_file="-" source_line=23 source_end_line=23 source_column=11 source_end_column=11}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton","block_level_fusion_config":{"num_warps":"1","output_tiles":[{"sizes":["1","4"]}],"num_ctas":1,"num_stages":1,"is_tma_allowed":false,"is_warp_specialization_allowed":false}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
}

