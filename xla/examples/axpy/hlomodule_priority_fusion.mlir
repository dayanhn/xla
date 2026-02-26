HloModule jit_matmul_with_elementwise, entry_computation_layout={(f32[32,32]{1,0}, f32[32,32]{1,0}, f32[3,3,1,8]{3,2,1,0})->f32[32,32]{1,0}}

%parameter_0 (parameter_0.1: f32[32,32]) -> f32[32,32] {
  ROOT %parameter_0.1 = f32[32,32]{1,0} parameter(0)
}

%parameter_1 (parameter_0.2: f32[32,32]) -> f32[32,32] {
  ROOT %parameter_0.2 = f32[32,32]{1,0} parameter(0)
}

%gemm_fusion_dot.1_computation (parameter_0: f32[32,32], parameter_1: f32[32,32]) -> f32[32,32] {
  %parameter_0 = f32[32,32]{1,0} parameter(0)
  %block_fusion = f32[32,32]{1,0} fusion(%parameter_0), kind=kCustom, calls=%parameter_0, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"num_warps":"2","output_tiles":[{"sizes":["16","16"]}],"num_ctas":1,"num_stages":1,"is_tma_allowed":false,"is_warp_specialization_allowed":false}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
  %parameter_1 = f32[32,32]{1,0} parameter(1)
  %block_fusion.1 = f32[32,32]{1,0} fusion(%parameter_1), kind=kCustom, calls=%parameter_1, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"num_warps":"2","output_tiles":[{"sizes":["16","16"]}],"num_ctas":1,"num_stages":1,"is_tma_allowed":false,"is_warp_specialization_allowed":false}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
  ROOT %dot.0 = f32[32,32]{1,0} dot(%block_fusion, %block_fusion.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={source_file="-" source_line=12 source_end_line=12 source_column=10 source_end_column=10}
}

%region_0.1 (Arg_0.2: f32[], Arg_1.2: f32[]) -> f32[] {
  %Arg_0.2 = f32[] parameter(0)
  %Arg_1.2 = f32[] parameter(1)
  ROOT %add.1.0 = f32[] add(%Arg_0.2, %Arg_1.2), metadata={source_file="-" source_line=7 source_end_line=7 source_column=26 source_end_column=26}
}

ENTRY %main.2 (Arg_0.3: f32[32,32], Arg_1.3: f32[32,32], Arg_2.1: f32[3,3,1,8]) -> f32[32,32] {
  %Arg_0.3 = f32[32,32]{1,0} parameter(0)
  %bitcast.3.0 = f32[1,32,32,1]{3,2,1,0} bitcast(%Arg_0.3)
  %Arg_2.1 = f32[3,3,1,8]{3,2,1,0} parameter(2)
  %transpose.1 = f32[8,3,3,1]{3,2,1,0} transpose(%Arg_2.1), dimensions={3,0,1,2}
  %cudnn-conv.1 = (f32[1,32,32,8]{3,2,1,0}, u8[0]{0}) custom-call(%bitcast.3.0, %transpose.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward", metadata={source_file="-" source_line=4 source_end_line=4 source_column=10 source_end_column=10}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"cudnn_conv_backend_config":{"activation_mode":"kNone","conv_result_scale":1,"side_input_scale":0,"algorithm":{"algo_id":"28","math_type":"DEFAULT_MATH","tuning_knobs":{"3":"0","2":"3"},"is_cudnn_frontend":false,"workspace_size":"0"},"leakyrelu_alpha":0},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
  %get-tuple-element.3 = f32[1,32,32,8]{3,2,1,0} get-tuple-element(%cudnn-conv.1), index=0
  %tanh.2.0 = f32[1,32,32,8]{3,2,1,0} tanh(%get-tuple-element.3), metadata={source_file="-" source_line=5 source_end_line=5 source_column=10 source_end_column=10}
  %bitcast.12.0 = f32[32,32,8]{2,1,0} bitcast(%tanh.2.0)
  %constant.9.0 = f32[] constant(0)
  %reduce.2 = f32[32,32]{1,0} reduce(%bitcast.12.0, %constant.9.0), dimensions={2}, to_apply=%region_0.1, metadata={source_file="-" source_line=7 source_end_line=7 source_column=26 source_end_column=26}
  %bitcast.13.0 = f32[1,32,32]{2,1,0} bitcast(%reduce.2)
  %constant.1 = f32[] constant(0.125)
  %broadcast.1 = f32[1,32,32]{2,1,0} broadcast(%constant.1), dimensions={}, metadata={source_file="-" source_line=9 source_end_line=9 source_column=10 source_end_column=10}
  %multiply.1 = f32[1,32,32]{2,1,0} multiply(%bitcast.13.0, %broadcast.1)
  %bitcast.1.0 = f32[32,32]{1,0} bitcast(%multiply.1)
  %Arg_1.3 = f32[32,32]{1,0} parameter(1)
  %gemm_fusion_dot.1 = f32[32,32]{1,0} fusion(%Arg_0.3, %Arg_1.3), kind=kCustom, calls=%gemm_fusion_dot.1_computation, metadata={source_file="-" source_line=12 source_end_line=12 source_column=10 source_end_column=10}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion","block_level_fusion_config":{"num_warps":"2","output_tiles":[{"sizes":["16","16"]}],"num_ctas":1,"num_stages":1,"is_tma_allowed":false,"is_warp_specialization_allowed":false}},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
  %tanh.3.0 = f32[32,32]{1,0} tanh(%gemm_fusion_dot.1), metadata={source_file="-" source_line=13 source_end_line=13 source_column=10 source_end_column=10}
  %constant.7.0 = f32[] constant(2)
  %broadcast.6.0 = f32[32,32]{1,0} broadcast(%constant.7.0), dimensions={}, metadata={source_file="-" source_line=15 source_end_line=15 source_column=10 source_end_column=10}
  %multiply.2.0 = f32[32,32]{1,0} multiply(%tanh.3.0, %broadcast.6.0), metadata={source_file="-" source_line=16 source_end_line=16 source_column=11 source_end_column=11}
  %constant.6.0 = f32[] constant(0.1)
  %broadcast.5.0 = f32[32,32]{1,0} broadcast(%constant.6.0), dimensions={}, metadata={source_file="-" source_line=18 source_end_line=18 source_column=11 source_end_column=11}
  %add.4.0 = f32[32,32]{1,0} add(%multiply.2.0, %broadcast.5.0), metadata={source_file="-" source_line=19 source_end_line=19 source_column=11 source_end_column=11}
  %add.5.0 = f32[32,32]{1,0} add(%bitcast.1.0, %add.4.0), metadata={source_file="-" source_line=20 source_end_line=20 source_column=11 source_end_column=11}
  %constant.5.0 = f32[] constant(0.5)
  %broadcast.4.0 = f32[32,32]{1,0} broadcast(%constant.5.0), dimensions={}, metadata={source_file="-" source_line=22 source_end_line=22 source_column=11 source_end_column=11}
  ROOT %multiply.3.0 = f32[32,32]{1,0} multiply(%add.5.0, %broadcast.4.0), metadata={source_file="-" source_line=23 source_end_line=23 source_column=11 source_end_column=11}
}

