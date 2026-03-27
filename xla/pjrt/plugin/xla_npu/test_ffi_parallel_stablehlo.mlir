module @jit__fun attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["<axis 0x78c896710ea0>"=8]>
  func.func public @main(%arg0: tensor<8x256x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"<axis 0x78c896710ea0>"}, {}, {}]>}, %arg1: tensor<8x2048x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"<axis 0x78c896710ea0>"}, {}, {}]>}) -> (tensor<8x256x2048xf32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"<axis 0x78c896710ea0>"}, {}, {}]>}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"<axis 0x78c896710ea0>"}, {}, {}]>, <@mesh, [{"<axis 0x78c896710ea0>"}, {}, {}]>] out_shardings=[<@mesh, [{"<axis 0x78c896710ea0>"}, {}, {}]>] manual_axes={"<axis 0x78c896710ea0>"} (%arg2: tensor<1x256x2048xf32>, %arg3: tensor<1x2048x2048xf32>) {
      %1 = stablehlo.reshape %arg2 : (tensor<1x256x2048xf32>) -> tensor<256x2048xf32>
      %2 = stablehlo.reshape %arg3 : (tensor<1x2048x2048xf32>) -> tensor<2048x2048xf32>
      %3 = stablehlo.custom_call @matmul(%1, %2) {backend_config = "", mhlo.backend_config = {}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<256x2048xf32>, tensor<2048x2048xf32>) -> tensor<256x2048xf32>
      %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<256x2048xf32>) -> tensor<1x256x2048xf32>
      sdy.return %4 : tensor<1x256x2048xf32>
    } : (tensor<8x256x2048xf32>, tensor<8x2048x2048xf32>) -> tensor<8x256x2048xf32>
    return %0 : tensor<8x256x2048xf32>
  }
}