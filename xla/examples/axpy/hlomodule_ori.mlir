HloModule jit_matmul_with_elementwise, entry_computation_layout={(f32[32,32]{1,0}, f32[32,32]{1,0}, f32[3,3,1,8]{3,2,1,0})->f32[32,32]{1,0}}

%region_0.1 (Arg_0.2: f32[], Arg_1.2: f32[]) -> f32[] {
  %Arg_0.2 = f32[] parameter(0)
  %Arg_1.2 = f32[] parameter(1)
  ROOT %add.1 = f32[] add(%Arg_0.2, %Arg_1.2), metadata={source_file="-" source_line=7 source_end_line=7 source_column=26 source_end_column=26}
}

ENTRY %main.2 (Arg_0.3: f32[32,32], Arg_1.3: f32[32,32], Arg_2.1: f32[3,3,1,8]) -> f32[32,32] {
  %Arg_0.3 = f32[32,32]{1,0} parameter(0)
  %reshape.2 = f32[1,32,32,1]{3,2,1,0} reshape(%Arg_0.3), metadata={source_file="-" source_line=3 source_end_line=3 source_column=10 source_end_column=10}
  %Arg_2.1 = f32[3,3,1,8]{3,2,1,0} parameter(2)
  %convolution.1 = f32[1,32,32,8]{3,2,1,0} convolution(%reshape.2, %Arg_2.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={source_file="-" source_line=4 source_end_line=4 source_column=10 source_end_column=10}
  %tanh.2 = f32[1,32,32,8]{3,2,1,0} tanh(%convolution.1), metadata={source_file="-" source_line=5 source_end_line=5 source_column=10 source_end_column=10}
  %constant.9 = f32[] constant(0)
  %reduce.1 = f32[1,32,32]{2,1,0} reduce(%tanh.2, %constant.9), dimensions={3}, to_apply=%region_0.1, metadata={source_file="-" source_line=7 source_end_line=7 source_column=26 source_end_column=26}
  %constant.8 = f32[] constant(8)
  %broadcast.7 = f32[1,32,32]{2,1,0} broadcast(%constant.8), dimensions={}, metadata={source_file="-" source_line=9 source_end_line=9 source_column=10 source_end_column=10}
  %divide.1 = f32[1,32,32]{2,1,0} divide(%reduce.1, %broadcast.7), metadata={source_file="-" source_line=10 source_end_line=10 source_column=10 source_end_column=10}
  %reshape.3 = f32[32,32]{1,0} reshape(%divide.1), metadata={source_file="-" source_line=11 source_end_line=11 source_column=10 source_end_column=10}
  %Arg_1.3 = f32[32,32]{1,0} parameter(1)
  %dot.1 = f32[32,32]{1,0} dot(%Arg_0.3, %Arg_1.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={source_file="-" source_line=12 source_end_line=12 source_column=10 source_end_column=10}
  %tanh.3 = f32[32,32]{1,0} tanh(%dot.1), metadata={source_file="-" source_line=13 source_end_line=13 source_column=10 source_end_column=10}
  %constant.7 = f32[] constant(2)
  %broadcast.6 = f32[32,32]{1,0} broadcast(%constant.7), dimensions={}, metadata={source_file="-" source_line=15 source_end_line=15 source_column=10 source_end_column=10}
  %multiply.2 = f32[32,32]{1,0} multiply(%tanh.3, %broadcast.6), metadata={source_file="-" source_line=16 source_end_line=16 source_column=11 source_end_column=11}
  %constant.6 = f32[] constant(0.1)
  %broadcast.5 = f32[32,32]{1,0} broadcast(%constant.6), dimensions={}, metadata={source_file="-" source_line=18 source_end_line=18 source_column=11 source_end_column=11}
  %add.4 = f32[32,32]{1,0} add(%multiply.2, %broadcast.5), metadata={source_file="-" source_line=19 source_end_line=19 source_column=11 source_end_column=11}
  %add.5 = f32[32,32]{1,0} add(%reshape.3, %add.4), metadata={source_file="-" source_line=20 source_end_line=20 source_column=11 source_end_column=11}
  %constant.5 = f32[] constant(0.5)
  %broadcast.4 = f32[32,32]{1,0} broadcast(%constant.5), dimensions={}, metadata={source_file="-" source_line=22 source_end_line=22 source_column=11 source_end_column=11}
  ROOT %multiply.3 = f32[32,32]{1,0} multiply(%add.5, %broadcast.4), metadata={source_file="-" source_line=23 source_end_line=23 source_column=11 source_end_column=11}
}
