module @jit_test_ffi_gelu_fwd attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024xf32>) -> (tensor<1024x1024xf32> {jax.result_info = "result[0]"}) {
    %0 = stablehlo.custom_call @"ascend.gelu"(%arg0) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }
}