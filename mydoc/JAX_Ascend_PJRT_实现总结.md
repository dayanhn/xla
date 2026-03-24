# JAX Ascend PJRT 实现总结

## 已完成的工作

### 1. XLA PJRT C API 层实现

#### 文件创建：
- ✅ `xla/xla/pjrt/c/pjrt_c_api_ascend.h` - PJRT C API 头文件
- ✅ `xla/xla/pjrt/c/pjrt_c_api_ascend.cc` - PJRT C API 入口实现
- ✅ `xla/xla/pjrt/c/pjrt_c_api_ascend_internal.h` - 内部实现头文件
- ✅ `xla/xla/pjrt/c/pjrt_c_api_ascend_internal.cc` - 内部实现核心逻辑

#### BUILD 配置更新：
- ✅ `xla/xla/pjrt/c/BUILD` - 添加了 `pjrt_c_api_ascend` 和 `pjrt_c_api_ascend_internal` 目标

#### 功能特性：
- 集成现有的 `xla_npu_pjrt_client`
- 注册 Ascend FFI handlers
- 支持 visible_devices 配置选项
- 提供完整的 PJRT C API 桥接

### 2. JAX 插件层完善

#### 文件更新：
- ✅ `jaxlib/ascend/ascend_plugin_extension.cc` - 完善 Python 扩展模块
  - 添加 `initialize_ascend()` 函数
  - 添加 `register_custom_type()` 函数
  - 添加 `register_custom_call_target()` 函数

#### 已有文件（无需修改）：
- ✅ `jax_plugins/ascend/BUILD.bazel` - 已正确定义插件构建目标
- ✅ `jax_plugins/ascend/__init__.py` - 已实现完整的初始化逻辑
- ✅ `jax_plugins/ascend/ascend_version_script.lds` - 版本脚本
- ✅ `jax_plugins/ascend/plugin_pyproject.toml` - PyPI 配置
- ✅ `jax_plugins/ascend/plugin_setup.py` - 插件安装配置
- ✅ `jax_plugins/ascend/pyproject.toml` - 项目配置
- ✅ `jax_plugins/ascend/setup.py` - Wheel 打包配置
- ✅ `jaxlib/ascend/BUILD` - 已定义 ascend_support 目标
- ✅ `jaxlib/ascend/ascend_configure.bzl` - Ascend 配置脚本

### 3. 构建系统集成

#### 已有配置（无需修改）：
- ✅ `build/build.py` - 已支持 `--wheels=jax-ascend-pjrt`
- ✅ `jaxlib/tools/BUILD.bazel` - 已定义 `jax_ascend_pjrt_wheel` 目标
- ✅ `jaxlib/jax.bzl` - 已定义 `if_ascend_is_configured` 宏

## 编译测试方法

### 使用 build.py 脚本编译（推荐）

```bash
# 初始化 Ascend 环境
source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh

# 编译 jax-ascend-pjrt wheel
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
```

### 手动 Bazel 编译（用于调试）

```bash
# 初始化环境
source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh

# 单独编译 PJRT C API Ascend 库
./bazel-7.4.1-linux-arm64 build \
    --compilation_mode=dbg \
    --copt=-g \
    --copt=-O0 \
    --strip=never \
    --action_env=ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME \
    --linkopt=-L$ASCEND_TOOLKIT_HOME/lib64 \
    --linkopt=-Wl,-rpath,$ASCEND_TOOLKIT_HOME/lib64 \
    --linkopt=-lascendcl \
    --linkopt=-lnnopbase \
    --linkopt=-lopapi_nn \
    --linkopt=-lhccl \
    --linkopt=-lhcomm \
    //xla/pjrt/c:pjrt_c_api_ascend

# 编译插件共享库
./bazel-7.4.1-linux-arm64 build \
    //jax_plugins/ascend:pjrt_c_api_ascend_plugin.so

# 运行测试
./bazel-bin/xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client_test \
     /path/to/test_ffi_matmul_gelu_stablehlo.mlir
```

## 架构流程

```
JAX Python Code
    ↓
jax_plugins/ascend/__init__.py (initialize())
    ↓
加载 pjrt_c_api_ascend_plugin.so
    ↓
调用 GetPjrtApi() → pjrt_c_api_ascend.cc
    ↓
pjrt::ascend_plugin::GetAscendPjrtApi() → pjrt_c_api_ascend_internal.cc
    ↓
CreateAscendClient() → xla::GetXlaPjrtNpuClient()
    ↓
xla/pjrt/npu/se_ascend_pjrt_client.cc
    ↓
xla/service/ascend/* (编译器、执行器)
    ↓
stream_executor/ascend/* (硬件接口)
    ↓
Ascend CANN Runtime (libascendcl.so 等)
```

## 关键依赖关系

### XLA 层依赖
```
//xla/pjrt/c:pjrt_c_api_ascend
  ├── //xla/pjrt/c:pjrt_c_api_ascend_internal
  │   ├── //xla/pjrt/npu:se_ascend_pjrt_client
  │   ├── //xla/pjrt/plugin/xla_npu:xla_npu_pjrt_client
  │   ├── //xla/pjrt/plugin/xla_npu:npu_client_options
  │   └── //xla/service/ascend/ffi:ascend_ffi
  └── //xla/pjrt/c:pjrt_c_api_wrapper_impl
```

### JAX 插件层依赖
```
//jax_plugins/ascend:pjrt_c_api_ascend_plugin.so
  ├── //xla/pjrt/c:pjrt_c_api_ascend
  ├── //xla/service:ascend_plugin (待实现)
  ├── //xla/stream_executor:ascend_platform
  └── //jaxlib/ascend:ascend_plugin_extension
```

## 还需要完成的工作

### 高优先级

1. **XLA Service 层 Ascend Plugin** (可选优化)
   - 当前直接使用 `xla_npu_pjrt_client`，可以进一步封装
   - 文件：`xla/xla/service/ascend_plugin.cc/h`
   - BUILD 目标：`//xla/xla/service:ascend_plugin`

2. **测试验证**
   - 创建 Python 层的端到端测试
   - 验证 JAX 基本操作在 Ascend 上的执行
   - 示例测试代码：
     ```python
     import jax
     import jax.numpy as jnp
     
     # 测试基本操作
     @jax.jit
     def test_gelu(x):
       return jax.nn.gelu(x)
     
     x = jnp.ones((32, 32))
     result = test_gelu(x)
     print(result)
     ```

3. **环境变量配置优化**
   - 在 `.bazelrc` 中完善 Ascend 配置段
   - 确保 `ASCEND_TOOLKIT_HOME` 正确传递到链接器

### 中优先级

4. **错误处理增强**
   - 完善 Ascend 错误码到 absl::Status 的转换
   - 添加详细的错误日志

5. **性能优化**
   - 内存池配置调优
   - 异步执行优化
   - 多设备通信优化

6. **文档完善**
   - 用户安装指南
   - 开发者贡献指南
   - API 参考文档

### 低优先级

7. **高级特性**
   - 分布式训练支持
   - 自动并行化
   - SPMD 分区

## 已知问题和注意事项

### 1. 路径依赖
- Ascend CANN toolkit 必须正确安装
- `ASCEND_TOOLKIT_HOME` 环境变量必须设置
- 动态库路径必须包含 `-L$ASCEND_TOOLKIT_HOME/lib64`

### 2. 版本兼容性
- 当前基于 JAX 0.9.1
- XLA 版本需要与 JAX 匹配
- CANN 版本建议 8.5.0+

### 3. 平台限制
- 仅支持 Linux aarch64 (ARM64)
- 需要 Ascend 910/310 系列 NPU

## 参考资料

- CUDA PJRT 实现：`xla/xla/pjrt/c/pjrt_c_api_gpu*`
- ROCm PJRT 实现：`xla/xla/pjrt/c/pjrt_c_api_rocm*`
- TPU PJRT 实现：`xla/xla/pjrt/c/pjrt_c_api_tpu*`
- JAX 插件机制：`jax_plugins/cuda/`
- XLA PJRT 文档：`xla/xla/pjrt/README.md`

## 联系与支持

如有问题，请参考：
1. JAX 官方文档：https://jax.readthedocs.io/
2. XLA 开发者指南：https://www.tensorflow.org/xla/
3. Ascend CANN 文档：https://www.hiascend.com/document
