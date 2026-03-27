# 总的编译配置流程

```
┌──────────────────────────────────────────────────────┐
│ Phase 1: WORKSPACE 解析阶段                           │
│ (Bazel 启动时执行)                                    │
│                                                      │
│ ┌──────────────────────────────────────────────────┐ │
│ │ ascend_configure(name="local_config_ascend")     │ │
│ │ ↓                                                │ │
│ │ 读取环境变量：                                   │ │
│ │   - TF_NEED_ASCEND (来自 --repo_env)             │ │
│ │   - ASCEND_TOOLKIT_HOME (来自 --repo_env)        │ │
│ │ ↓                                                │ │
│ │ 生成 @local_config_ascend 仓库                    │ │
│ │   - ascend/BUILD                                 │ │
│ │   - ascend/ascend/ascend_config.h                │ │
│ └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────┐
│ Phase 2: .bazelrc 解析与命令注入                      │
│ (用户执行 bazel build --config=ascend)               │
│                                                      │
│ common:ascend                                        │
│   --repo_env TF_NEED_ASCEND=1 → 传递给 Repository   │
│   --action_env ASCEND_TOOLKIT_HOME → 传递给 Build   │
│   --define=using_ascend=true → 全局标志             │
└──────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────┐
│ Phase 3: BUILD 文件解析与条件编译                     │
│                                                      │
│ //jaxlib/ascend/BUILD:                               │
│   if_ascend_is_configured([...])                     │
│   = select({                                         │
│       "@local_config_ascend//ascend:enable_ascend":  │
│         [...],                                       │
│       "//conditions:default": []                     │
│     })                                               │
└──────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────┐
│ Phase 4: 编译/链接动作执行                            │
│                                                      │
│ clang -L$ASCEND_TOOLKIT_HOME/lib64 \                 │
│       -lascendcl \                                   │
│       jax_ascend_pjrt.so                             │
└──────────────────────────────────────────────────────┘
```


| 配置层级 | 文件 | 关键配置 | 依赖的工作流 |
|---------|------|---------|-------------|
| **L1: 仓库初始化** | [WORKSPACE](file:///data3/zhongzhw/code/google/jax/WORKSPACE#L175-L180) | `ascend_configure(name="...")` | 定义 Repository Rule 实例 |
| **L2: 环境激活** | `.bazelrc` | `--repo_env TF_NEED_ASCEND=1` | 向 Repository Rule 传递参数 |
| **L3: 配置生成** | [ascend_configure.bzl](file:///data3/zhongzhw/code/google/jax/jaxlib/ascend/ascend_configure.bzl) | 生成 `@local_config_ascend` | 基于环境变量创建配置文件 |
| **L4: 条件编译** | [jaxlib/ascend/BUILD](file:///data3/zhongzhw/code/google/jax/jaxlib/ascend/BUILD) | `if_ascend_is_configured()` | 引用生成的 `config_setting` |

# WORKSPACE
```
load(
    "//jaxlib/ascend:ascend_configure.bzl",
    "ascend_configure",
)

ascend_configure(name = "local_config_ascend")
```
## WORKSPACE 文件作用
**WORKSPACE = Bazel 项目的"依赖管理与仓库初始化配置文件"**
- **作用域**：定义外部依赖（External Repositories）、工具链注册、Repository Rule 绑定
- **执行时机**：Bazel 构建启动时**最先解析**的文件（优先级高于 [BUILD](file:///data3/zhongzhw/code/google/jax/jaxlib/BUILD)、`.bazelrc`）
- **核心职能**：
  1. 声明并拉取第三方依赖（如 XLA、rules_ml_toolchain、Python 包等）
  2. 注册工具链（Toolchains）供编译时使用
  3. 初始化 Repository Rule，生成平台特定的配置仓库（如 `@local_config_cuda`、`@local_config_ascend`）


## Ascend 配置

| 代码片段 | 技术含义 | 作用机制 |
|---------|---------|---------|
| **[load("//jaxlib/ascend:ascend_configure.bzl", "ascend_configure")](file:///data3/zhongzhw/code/google/jax/jax/_src/pallas/mosaic/primitives.py#L1187-L1200)** | 从 [[jaxlib/ascend/ascend_configure.bzl](file:///data3/zhongzhw/code/google/jax/jaxlib/ascend/ascend_configure.bzl)](file:///data3/zhongzhw/code/google/jax/jaxlib/ascend/ascend_configure.bzl) 导入 `ascend_configure` Repository Rule | - **Repository Rule**: Bazel 的元编程机制，用于在构建前动态生成仓库结构<br>- **加载路径**: `//jaxlib/ascend:` 表示相对于工作区根目录的 `.bzl` 文件 |
| **`ascend_configure(name = "local_config_ascend")`** | 实例化 Repository Rule，生成名为 `@local_config_ascend` 的外部仓库 | - **执行阶段**: WORKSPACE 解析阶段（Build 之前）<br>- **输入**: 通过 `--repo_env` 传递的环境变量（`TF_NEED_ASCEND`, `ASCEND_TOOLKIT_HOME`）<br>- **输出**: 生成 `@local_config_ascend` 仓库，包含：<br>  &nbsp;&nbsp;✓ [ascend/BUILD](file:///data3/zhongzhw/code/google/jax/jaxlib/ascend/BUILD)（含 `config_setting(name="enable_ascend")`）<br>  &nbsp;&nbsp;✓ `ascend/ascend/ascend_config.h`（C++ 头文件，定义 `ASCEND_TOOLKIT_PATH`）<br>  &nbsp;&nbsp;✓ `ascend/ascend/ascend_config.py`（Python 配置模块） |


# .bazelrc
```
# Configs for Ascend
common:ascend --repo_env TF_NEED_ASCEND=1
common:ascend --repo_env ASCEND_TOOLKIT_HOME
common:ascend --action_env ASCEND_TOOLKIT_HOME
common:ascend --define=using_ascend=true
```
## `.bazelrc` 文件作用
**`.bazelrc` = Bazel 构建的"配置文件"**
- **定位**：定义 Bazel 如何编译项目的"总开关"
- **作用**：批量声明编译选项（编译器选择、库路径、依赖、优化级别等）
- **类比**：类似 CMake 的 [CMakeLists.txt](file:///data3/zhongzhw/code/google/jax/docs/ffi/CMakeLists.txt) 或 Makefile 的变量定义部分
- **核心机制**：通过**配置段**（如 `common:ascend`）组织不同场景的选项，按需激活

## Ascend 配置
**"Ascend后端构建的总开关"** —— 当激活 `--config=ascend` 时，这 4 行配置会联动完成三件事：

1. **触发 Repository Rule** → 生成 Ascend 配置文件
2. **传递环境变量** → 让构建脚本知道 CANN toolkit 在哪
3. **定义全局标志** → 告诉所有 BUILD 文件"现在要编译 Ascend 代码"



