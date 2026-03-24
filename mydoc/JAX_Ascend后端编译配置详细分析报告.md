# JAX Ascend后端编译配置详细分析报告

## 1. CUDA编译配置分析

### 1.1 构建系统架构

JAX使用`build/build.py`作为构建入口，支持构建多种wheel包，包括：

- `jaxlib`：核心库
- `jax-cuda-plugin`：CUDA插件
- `jax-cuda-pjrt`：CUDA PJRT插件
- `jax-rocm-plugin`：ROCm插件
- `jax-rocm-pjrt`：ROCm PJRT插件

### 1.2 构建命令解析

以支持CUDA的构建命令为例：

```bash
python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt --editable --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla
```

**关键参数解析**：

- `--wheels`：指定要构建的wheel包，多个包用逗号分隔
- `--editable`：使用可编辑模式构建，修改源码后无需重新安装
- `--bazel_options`：传递给Bazel的构建选项
- `--override_repository=xla`：指定本地XLA仓库路径
- `--local_xla_path`：指定本地XLA路径

### 1.3 CUDA插件构建流程

1. **插件目录结构**：`jax_plugins/cuda/`
   - `__init__.py`：插件初始化代码
   - `BUILD.bazel`：Bazel构建规则
   - `setup.py`：Python包配置

2. **构建脚本**：`jaxlib/tools/build_gpu_plugin_wheel.py`
   - 负责组装wheel源码树
   - 复制必要的文件
   - 构建wheel包

3. **插件初始化流程**：
   - 加载NVIDIA库
   - 导入CUDA扩展
   - 获取库路径
   - 检查CUDA版本
   - 注册插件到JAX

4. **关键代码**：

```python
# 注册插件
def initialize():
  _load_nvidia_libraries()
  _import_extensions()
  path = _get_library_path()
  if path is None:
    return

  if not os.getenv("JAX_SKIP_CUDA_CONSTRAINTS_CHECK"):
    _check_cuda_versions(raise_on_first_error=True)

  options = xla_client.generate_pjrt_gpu_plugin_options()
  c_api = xb.register_plugin(
      'cuda', priority=500, library_path=str(path), options=options
  )
  # 注册自定义类型和调用处理程序
  if cuda_plugin_extension:
    xla_client.register_custom_type_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_type, c_api
        ),
    )
    # 更多注册...
```

### 1.4 Bazel构建配置

- `jax_plugins/cuda/BUILD.bazel`：定义CUDA插件的构建规则
- 依赖XLA的CUDA相关组件：
  - `@xla//xla/pjrt/c:pjrt_c_api_gpu`
  - `@xla//xla/service:gpu_plugin`
  - `@xla//xla/stream_executor:cuda_platform`

## 2. Ascend后端编译配置方案

### 2.1 目录结构设计

参考CUDA插件的结构，为Ascend后端创建以下目录：

```
jax_plugins/
└── ascend/
    ├── __init__.py          # 插件初始化代码
    ├── BUILD.bazel          # Bazel构建规则
    ├── setup.py             # Python包配置
    ├── pyproject.toml       # Python项目配置
    └── plugin_setup.py      # 插件设置脚本
```

### 2.2 构建系统修改

#### 2.2.1 修改 `build/build.py`

1. **添加Ascend构建目标**：

```python
WHEEL_BUILD_TARGET_DICT = {
    # 现有目标...
    "jax-ascend-plugin": "//jaxlib/tools:jax_ascend_plugin_wheel",
    "jax-ascend-plugin_editable": "//jaxlib/tools:jax_ascend_plugin_wheel_editable",
    "jax-ascend-pjrt": "//jaxlib/tools:jax_ascend_pjrt_wheel",
    "jax-ascend-pjrt_editable": "//jaxlib/tools:jax_ascend_pjrt_wheel_editable",
}
```

2. **添加Ascend构建选项**：

```python
# Ascend Options
ascend_group = parser.add_argument_group('Ascend Options')
ascend_group.add_argument(
    "--ascend_version",
    type=str,
    default="910",
    help="Ascend version to use",
)

ascend_group.add_argument(
    "--ascend_path",
    type=str,
    default="",
    help="Path to the Ascend toolkit.",
)
```

3. **添加Ascend构建逻辑**：

```python
if "ascend" in args.wheels:
  wheel_build_command_base.append("--config=ascend")
  if args.ascend_path:
    logging.debug("Ascend toolkit path: %s", args.ascend_path)
    wheel_build_command_base.append(f"--action_env=ASCEND_PATH=\"{args.ascend_path}\"")
```

#### 2.2.2 创建 `jaxlib/tools/build_ascend_plugin_wheel.py`

参考 `build_gpu_plugin_wheel.py`，创建Ascend插件的构建脚本。

#### 2.2.3 修改 `jaxlib/tools/BUILD.bazel`

添加Ascend插件的构建规则。

### 2.3 Ascend插件实现

#### 2.3.1 创建 `jax_plugins/ascend/__init__.py`

```python
import ctypes
import functools
import importlib
import logging
import os
import pathlib
import traceback
from typing import Any

from jax._src.lib import triton
from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

ascend_plugin_extension = None
ascend_versions = None

def _import_extensions():
  global ascend_plugin_extension
  global ascend_versions

  # 尝试导入Ascend扩展
  for pkg_name in ['jax_ascend_plugin', 'jaxlib.ascend']:
    try:
      ascend_plugin_extension = importlib.import_module(
          f'{pkg_name}.ascend_plugin_extension'
      )
      ascend_versions = importlib.import_module(
          f'{pkg_name}._versions'
      )
    except ImportError:
      ascend_plugin_extension = None
      ascend_versions = None
    else:
      break

logger = logging.getLogger(__name__)

def _get_library_path():
  installed_path = (
      pathlib.Path(__file__).resolve().parent / 'xla_ascend_plugin.so'
  )
  if installed_path.exists():
    return installed_path

  local_path = os.path.join(
      os.path.dirname(__file__), 'pjrt_c_api_ascend_plugin.so'
  )
  if not os.path.exists(local_path):
    runfiles_dir = os.getenv('RUNFILES_DIR', None)
    if runfiles_dir:
      local_path = os.path.join(
          runfiles_dir, '__main__/jax_plugins/ascend/pjrt_c_api_ascend_plugin.so'
      )

  if os.path.exists(local_path):
    logger.debug(
        'Native library %s does not exist. This most likely indicates an issue'
        ' with how %s was built or installed. Fallback to local test'
        ' library %s',
        installed_path,
        __package__,
        local_path,
    )
    return local_path

  logger.debug(
      'WARNING: Native library %s and local test library path %s do not'
      ' exist. This most likely indicates an issue with how %s was built or'
      ' installed or missing src files.',
      installed_path,
      local_path,
      __package__,
  )
  return None

def _load_ascend_libraries():
  """尝试加载Ascend的库"""
  # 加载Ascend相关库
  # 这里需要根据实际的Ascend库名称进行修改
  pass

def _check_ascend_versions(raise_on_first_error: bool = False,
                         debug: bool = False):
  # 检查Ascend版本
  pass

def initialize():
  _load_ascend_libraries()
  _import_extensions()
  path = _get_library_path()
  if path is None:
    return

  if not os.getenv("JAX_SKIP_ASCEND_CONSTRAINTS_CHECK"):
    _check_ascend_versions(raise_on_first_error=True)
  else:
    logger.debug('Skipped Ascend versions constraints check due to the '
                'JAX_SKIP_ASCEND_CONSTRAINTS_CHECK env var being set.')

  options = xla_client.generate_pjrt_ascend_plugin_options()
  c_api = xb.register_plugin(
      'ascend', priority=500, library_path=str(path), options=options
  )
  if ascend_plugin_extension:
    xla_client.register_custom_type_handler(
        "ASCEND",
        functools.partial(
            ascend_plugin_extension.register_custom_type, c_api
        ),
    )
    xla_client.register_custom_call_handler(
        "ASCEND",
        functools.partial(
            ascend_plugin_extension.register_custom_call_target, c_api
        ),
    )
    for _name, _value in ascend_plugin_extension.ffi_types().items():
      xla_client.register_custom_type(
          _name, _value, platform='ASCEND'
      )
    for _name, _value in ascend_plugin_extension.ffi_handlers().items():
      xla_client.register_custom_call_target(
          _name, _value, platform='ASCEND', api_version=1
      )
  else:
    logger.warning('ascend_plugin_extension is not found.')
```

#### 2.3.2 创建 `jax_plugins/ascend/BUILD.bazel`

```python
# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//jaxlib:jax.bzl",
    "py_library_providing_imports_info",
    "pytype_library",
)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

exports_files([
    "__init__.py",
    "plugin_pyproject.toml",
    "plugin_setup.py",
    "pyproject.toml",
    "setup.py",
])

cc_binary(
    name = "pjrt_c_api_ascend_plugin.so",
    features = ["asan_runtime_closure"],
    linkopts = [
        "-Wl,--version-script,$(location :ascend_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    deps = [
        ":ascend_version_script.lds",
        "@xla//xla/pjrt/c:pjrt_c_api_ascend",
        "@xla//xla/service:ascend_plugin",
        "@xla//xla/stream_executor:ascend_platform",
    ],
)

py_library_providing_imports_info(
    name = "ascend_plugin",
    srcs = [
        "__init__.py",
    ],
    data = [":pjrt_c_api_ascend_plugin.so"],
    lib_rule = pytype_library,
)
```

#### 2.3.3 创建 `jax_plugins/ascend/setup.py`

```python
# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup

package_name = "jax_plugins.xla_ascend"

setup(
    name=f"xla_ascend_plugin",
    version="0.9.1.dev0+selfbuilt",
    description="JAX Ascend plugin",
    packages=[package_name],
    package_data={
        package_name: ["xla_ascend_plugin.so"],
    },
    zip_safe=False,
    entry_points={
        "jax_plugins": [
            f"xla_ascend = {package_name}",
        ],
    },
)
```

### 2.4 XLA Ascend后端完善

#### 2.4.1 PJRT API实现

确保XLA Ascend后端实现了完整的PJRT API，包括：

- `se_ascend_pjrt_client.cc/h`：基于StreamExecutor的Ascend PJRT客户端
- `xla_npu_pjrt_client.cc/h`：XLA NPU的PJRT客户端

#### 2.4.2 编译配置文件

创建或修改以下配置文件：

- `.bazelrc`：添加Ascend相关配置
- `jaxlib/jax.bzl`：添加Ascend构建规则
- `jaxlib/tools/BUILD.bazel`：添加Ascend插件构建目标

### 2.5 编译命令

#### 2.5.1 构建命令

```bash
# 构建Ascend插件
python build/build.py build --wheels=jaxlib,jax-ascend-plugin,jax-ascend-pjrt --editable --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla

# 指定Ascend路径
python build/build.py build --wheels=jaxlib,jax-ascend-plugin,jax-ascend-pjrt --editable --ascend_path=/usr/local/Ascend --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla
```

#### 2.5.2 环境变量

- `ASCEND_PATH`：Ascend toolkit路径
- `JAX_SKIP_ASCEND_CONSTRAINTS_CHECK`：跳过Ascend版本检查

## 3. 代码修改清单

### 3.1 需要创建的文件

1. **jax_plugins/ascend/** 目录
   - `__init__.py`：插件初始化代码
   - `BUILD.bazel`：Bazel构建规则
   - `setup.py`：Python包配置
   - `pyproject.toml`：Python项目配置
   - `plugin_setup.py`：插件设置脚本
   - `ascend_version_script.lds`：版本脚本

2. **jaxlib/ascend/** 目录
   - `BUILD`：Bazel构建规则
   - `ascend_plugin_extension.cc`：Ascend插件扩展
   - `versions.cc`：版本管理
   - `versions_helpers.cc`：版本管理辅助函数
   - `versions_helpers.h`：版本管理辅助函数头文件

3. **jaxlib/tools/** 目录
   - `build_ascend_plugin_wheel.py`：Ascend插件构建脚本

### 3.2 需要修改的文件

1. **build/build.py**
   - 添加Ascend构建目标
   - 添加Ascend构建选项
   - 添加Ascend构建逻辑

2. **jaxlib/tools/BUILD.bazel**
   - 添加Ascend插件构建规则

3. **jaxlib/jax.bzl**
   - 添加Ascend构建规则

4. **.bazelrc**
   - 添加Ascend相关配置

5. **jax/_src/xla_bridge.py**
   - 添加Ascend后端支持

### 3.3 XLA后端需要完善的文件

1. **xla/pjrt/c/** 目录
   - `pjrt_c_api_ascend.cc`：Ascend PJRT C API实现
   - `pjrt_c_api_ascend_internal.cc`：Ascend PJRT C API内部实现

2. **xla/service/** 目录
   - `ascend/ascend_compiler.cc/h`：Ascend编译器实现
   - `ascend/ascend_compiler_registration.cc`：注册Ascend编译器

3. **xla/stream_executor/ascend/** 目录
   - `ascend_platform.cc/h`：Ascend平台实现
   - `ascend_executor.cc/h`：Ascend执行器实现

## 4. 编译和测试流程

### 4.1 环境准备

1. 安装Ascend toolkit
2. 安装依赖项
3. 配置环境变量

### 4.2 编译步骤

1. 克隆JAX和XLA代码库
2. 进入JAX目录
3. 执行构建命令
4. 安装生成的wheel包

### 4.3 测试步骤

1. 运行简单的测试代码：

```python
import jax
print(f"JAX devices: {jax.devices()}")
```

2. 运行更复杂的测试：

```python
import jax
import jax.numpy as jnp

# 测试矩阵乘法
def test_matmul():
    a = jnp.ones((1024, 1024))
    b = jnp.ones((1024, 1024))
    c = jax.jit(lambda x, y: jnp.dot(x, y))(a, b)
    print(f"Matmul result shape: {c.shape}")
    print(f"Matmul result sum: {c.sum()}")

if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    test_matmul()
```

## 5. 总结

本报告详细分析了JAX CUDA编译配置体系，并为Ascend后端提供了完整的编译配置方案。通过参考CUDA插件的实现，我们可以为Ascend后端创建类似的插件结构，包括：

1. 创建Ascend插件目录结构
2. 修改构建系统以支持Ascend
3. 实现Ascend插件的初始化代码
4. 完善XLA Ascend后端的实现
5. 提供编译和测试指南

通过这些步骤，我们可以为JAX添加对Ascend后端的支持，使JAX能够在Ascend硬件上高效运行。

## 6. 注意事项

1. **版本兼容性**：确保Ascend toolkit版本与实现兼容
2. **路径配置**：正确设置Ascend toolkit路径
3. **依赖项**：确保所有依赖项都已安装
4. **测试**：在不同环境下进行测试，确保稳定性
5. **性能优化**：根据Ascend硬件特性进行性能优化

## 7. 后续工作

1. **完善算子支持**：添加更多深度学习算子的支持
2. **性能优化**：进一步优化执行性能
3. **分布式训练**：完善分布式训练支持
4. **工具链集成**：与更多工具链集成
5. **生态系统建设**：构建更完善的生态系统

通过不断完善和优化，JAX Ascend后端将为Ascend硬件的深度学习加速提供强大的支持，为用户提供更好的使用体验和更高的性能。