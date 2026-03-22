module @jit_test_ffi_matmul_then_gelu attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<16x16xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @"ascend.matmul"(%arg0, %arg1) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %1 = stablehlo.custom_call @"ascend.gelu"(%0) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }
}