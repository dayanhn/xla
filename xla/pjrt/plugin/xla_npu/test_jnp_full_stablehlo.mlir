module @jit_create_full_matrix attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<128x128xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}