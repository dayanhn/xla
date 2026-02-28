import os
os.system("clear")

# 只使用 6、7 号 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


#os.system("clear")
os.system("rm -rf ./tmp/xla_dump/*")
os.system("rm -rf ./tmp/jax_dump/*")
os.environ["JAX_DUMP_IR_MODES"] = "jaxpr,stablehlo"
os.environ["JAX_DUMP_IR_TO"] = "/tmp/jax_dump"

dump_out = False

if dump_out:
    # Ensure C++/Abseil logging flags are set before any C++ extensions are loaded.
    # Use TF_CPP_* env vars which logging_initializer.cc reads and applies.
    # Increase max vlog so XLA_VLOG_LINES(3, ...) will print
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "3"


    # Enable verbose logging for HLO-related sources and MLIR conversion passes.
    # Use basenames (file without extension), e.g. hlo_module for hlo_module.cc.
    os.environ["TF_CPP_VMODULE"] = ("hlo_module=3,hlo_instruction=3,hlo_schedule=3,defuser=2,mlir_to_hlo=5,mlir_hlo_to_hlo=5,stablehlo=5")

# Move XLA_FLAGS here so they take effect before importing JAX/C++ backends.
# --xla_gpu_force_compilation_parallelism=1 禁用多线程编译，方便调试
'''
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_experimental_enable_triton_heroless_priority_fusion=true '
    '--xla_dump_to=./tmp/xla_dump '
    '--xla_gpu_force_compilation_parallelism=1 '
    '--xla_dump_hlo_as_text=true '
    '--xla_dump_hlo_as_proto=true '
    '--xla_dump_hlo_pass_re=.* '
    '--xla_dump_hlo_module_re=.*')
'''
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_experimental_enable_triton_heroless_priority_fusion=true '
    '--xla_dump_to=./tmp/xla_dump '
    '--xla_gpu_force_compilation_parallelism=1 '
    '--xla_dump_hlo_as_text=true '
    '--xla_dump_hlo_as_proto=false '
    '--xla_dump_hlo_pass_re=.* '
    '--xla_dump_hlo_module_re=jit_matmul_with_elementwise '
    '--xla_dump_emitter_re=triton-fusion '
    # 自动调优日志配置
    '--xla_gpu_dump_autotune_logs_to=./tmp/autotune_logs.txt '
    )


print("pid = ",os.getpid())

import jax
import jax.numpy as jnp
from jax import jit, pmap, make_jaxpr

print("Testing JAX operator fusion with Triton...")
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# 示例：在同一个 jit 函数里同时包含卷积（CuDNN 候选）和矩阵乘（Triton 候选）
@jit
def matmul_with_elementwise(x, y, filters):
    """在单个 JIT 函数内同时执行：
    - 使用 reshape 后的输入做 2D 卷积 (conv + bias + activation)，更容易走 cuDNN 路径；
    - 同时计算原始的矩阵乘并做 element-wise 操作，作为 Triton 候选。

    函数仍然保持原始签名 (x, y)，以减少对调用处的改动。
    """
    # 1) 卷积部分：把矩阵视为单通道图像
    # 假设 x 形状为 (H, W)，把它视为 (N=1, H, W, C=1)
    H, W = x.shape
    x_img = x.reshape((1, H, W, 1))


    # 卷积（NHWC, HWIO, NHWC），padding SAME 保持空间尺寸
    conv = jax.lax.conv_general_dilated(
        x_img,
        filters,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    conv_activated = jnp.tanh(conv)
    # 把多个通道聚合取均值，得到 (n,H,W) -> squeeze 到 (H,W)
    conv_mat = jnp.mean(conv_activated, axis=-1).squeeze(0)


    # 2) 矩阵乘部分（保留原先的 element-wise 流程）
    matmul_result = jnp.matmul(x, y)
    matmul_activated = jnp.tanh(matmul_result)
    matmul_scaled = matmul_activated * 2.0 + 0.1

    # 3) 合并两条路径的结果（例如相加并取平均），保持形状一致
    combined = (conv_mat + matmul_scaled) * 0.5
    return combined


# 创建输入
m, n, k = 32, 32, 32
x = jnp.ones((m, k))
y = jnp.ones((k, n))

out_channels = 8
in_channels = 1
filters = jnp.ones((3, 3, in_channels, out_channels))  # HWIO 卷积核

# 1. 获取 Traced 对象
traced_obj = matmul_with_elementwise.trace(x, y, filters)
#print(traced_obj.jaxpr)
lowerd_obj = traced_obj.lower()
stablehlo = lowerd_obj.as_text('stablehlo')
#print(stablehlo)
#hlo = lowerd_obj.as_text('hlo')
#print(hlo)
compiled = lowerd_obj.compile()
#txt = compiled.as_text()
#print(txt)
z = compiled(x, y, filters)
print(z)
