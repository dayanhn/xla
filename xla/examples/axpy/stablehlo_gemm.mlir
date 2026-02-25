module @jit_matmul_with_elementwise attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<3x3x1x8xf32>) -> (tensor<32x32xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<32x32xf32>) -> tensor<1x32x32x1xf32>
    %1 = stablehlo.convolution(%0, %arg2) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x32x32x1xf32>, tensor<3x3x1x8xf32>) -> tensor<1x32x32x8xf32>
    %2 = stablehlo.tanh %1 : tensor<1x32x32x8xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x32x32x8xf32>, tensor<f32>) -> tensor<1x32x32xf32>
    %cst_0 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x32x32xf32>
    %5 = stablehlo.divide %3, %4 : tensor<1x32x32xf32>
    %6 = stablehlo.reshape %5 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %7 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %8 = stablehlo.tanh %7 : tensor<32x32xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<32x32xf32>
    %cst_2 = stablehlo.constant dense<1.000000e-01> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %12 = stablehlo.add %10, %11 : tensor<32x32xf32>
    %13 = stablehlo.add %6, %12 : tensor<32x32xf32>
    %cst_3 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %15 = stablehlo.multiply %13, %14 : tensor<32x32xf32>
    return %15 : tensor<32x32xf32>
  }
}

