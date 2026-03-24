# JAX Ascend PJRT 快速参考指南

## 环境准备

### 1. 安装 CANN Toolkit
```bash
# 下载并安装 CANN 8.5.0 或更高版本
# 参考：https://www.hiascend.com/document
```

### 2. 配置环境变量
```bash
# 每次使用前执行
source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh

# 验证环境变量
echo $ASCEND_TOOLKIT_HOME
# 应该输出：/data3/zhongzhw/Ascend8.5REL/cann-8.5.0 (或类似路径)
```

### 3. 验证硬件访问
```bash
# 检查 NPU 设备
npu-smi info

# 应该显示所有可用的 Ascend NPU 设备
```

## 编译方法

### 方法一：使用 build.py（推荐）

```bash
# 在项目根目录执行
python build/build.py build \
  --wheels=jax-ascend-pjrt \
  --editable \
  --bazel_options=--compilation_mode=dbg \
  --bazel_options=--copt=-g \
  --bazel_options=--copt=-O0 \
  --bazel_options=--strip=never \
  --bazel_options=--override_repository=xla=$(pwd)/xla \
  --local_xla_path=$(pwd)/xla \
  --ascend_path=$ASCEND_TOOLKIT_HOME

# 编译产物位置
ls -lh dist/jax_ascend910_pjrt-*.whl
```

### 方法二：手动 Bazel 编译

```bash
# 1. 编译核心库
bazel build //xla/pjrt/c:pjrt_c_api_ascend

# 2. 编译插件共享库
bazel build //jax_plugins/ascend:pjrt_c_api_ascend_plugin.so

# 3. 编译 Wheel 包
bazel build //jaxlib/tools:jax_ascend_pjrt_wheel

# 产物位置
ls -lh bazel-bin/jaxlib/tools/dist/jax_ascend910_pjrt-*.whl
```

## 安装方法

### 从 Wheel 安装
```bash
pip install dist/jax_ascend910_pjrt-*.whl
```

### 可编辑模式安装（开发用）
```bash
pip install -e dist/jax_ascend910_pjrt_editable/
```

## 使用方法

### 基本使用
```python
import jax
import jax.numpy as jnp
from jax import jit, grad

# 自动使用 Ascend 设备
@jit
def compute(x):
    return jnp.sum(x ** 2)

x = jnp.ones((100, 100))
result = compute(x)
print(f"Result: {result}")
```

### 显式设备选择
```python
import jax

# 获取 Ascend 设备
ascend_devices = [d for d in jax.local_devices() if d.platform == 'ascend']
device = ascend_devices[0]

# 在指定设备上执行
with jax.default_device(device):
    x = jnp.ones((4, 4))
    y = x @ x.T
    print(f"Computed on {y.device()}")
```

### 多设备并行
```python
from jax.experimental import multihost_gpu
import jax

# 使用多个 Ascend 设备
devices = jax.local_devices()[:4]  # 使用前 4 个设备

# 数据并行
sharded_x = jax.device_put(jnp.ones((8, 8)), devices)
```

## 调试技巧

### 1. 启用详细日志
```bash
export JAX_LOG_LEVEL=DEBUG
export TF_CPP_MIN_LOG_LEVEL=0
```

### 2. 跳过自动初始化
```bash
export JAX_SKIP_ASCEND_INIT=1
# 然后手动初始化
```

### 3. 查看 PJRT 插件信息
```python
import jax
print(f"Available platforms: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")
```

### 4. 性能分析
```python
from jax.profiler import start_trace, stop_trace

start_trace("/tmp/jax_trace")
# ... 你的代码 ...
stop_trace()

# 使用 TensorBoard 查看
tensorboard --logdir=/tmp/jax_trace
```

## 常见问题排查

### 问题 1: 找不到 Ascend 设备
```bash
# 检查环境变量
echo $ASCEND_TOOLKIT_HOME

# 检查驱动
lsmod | grep ascend

# 重启服务
sudo systemctl restart ascend-driver
```

### 问题 2: 链接库错误
```bash
# 检查库文件是否存在
ls -lh $ASCEND_TOOLKIT_HOME/lib64/libascendcl.so

# 添加到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$LD_LIBRARY_PATH
```

### 问题 3: 编译失败
```bash
# 清理构建缓存
bazel clean

# 重新配置
python build/build.py configure_only \
  --ascend_path=$ASCEND_TOOLKIT_HOME

# 查看详细错误
bazel build //... --verbose_failures
```

### 问题 4: 运行时错误
```python
# 捕获详细错误
import traceback
try:
    # 你的代码
    pass
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
```

## 测试命令

### 运行 Python 测试
```bash
python test_ascend_pjrt.py
```

### 运行 C++ 测试
```bash
# 先编译测试
bazel build //xla/pjrt/plugin/xla_npu:xla_npu_pjrt_client_test

# 执行测试
./bazel-bin/xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client_test \
    /path/to/test_ffi_matmul_gelu_stablehlo.mlir
```

### 运行 JAX 测试套件
```bash
# 基础测试
python -m pytest tests/lax_test.py -v

# 特定后端测试
python -m pytest tests/pjit_test.py -v -k "ascend"
```

## 性能调优

### 1. 内存管理
```python
# 设置内存池大小
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

# 预分配内存
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
```

### 2. 编译器优化
```python
# 设置优化级别
os.environ['XLA_OPTIMIZE_LEVEL'] = '3'

# 启用 XLA 调试
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dumps'
```

### 3. 批处理优化
```python
from jax import vmap

# 使用 vmap 替代循环
@vmap
def process_single(x):
    return model(x)

results = process_single(batch_data)
```

## 参考资源

### 文档
- [JAX 官方文档](https://jax.readthedocs.io/)
- [XLA 开发者指南](https://www.tensorflow.org/xla/)
- [PJRT 插件开发](https://github.com/google/jax/tree/main/jax_plugins)
- [Ascend CANN 文档](https://www.hiascend.com/document)

### 代码示例
- JAX examples: `examples/` 目录
- 测试代码：`tests/` 目录
- 本实现文档：`xla/mydoc/JAX_Ascend_PJRT_实现总结.md`

### 社区支持
- GitHub Issues: https://github.com/jax-ml/jax/issues
- JAX 讨论组：jax-dev@googlegroups.com

## 版本信息

| 组件 | 版本 |
|------|------|
| JAX | 0.9.1 |
| XLA | 内置 |
| CANN | 8.5.0+ |
| Python | 3.11+ |
| Bazel | 7.4.1+ |

---
最后更新：2024-03-24
