# 编译命令：
```
 python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt --editable --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla
 ```

# plugin介绍
1. 编写plugin
- 在jax_plugins目录下模仿cuda创建npu插件，通过定义插件里面的内容，最终编译打包成一个可安装的whl文件，如cuda插件会生成：jax_cuda12_plugin-0.9.1.dev0+selfbuilt-cp313-cp313-manylinux_2_27_x86_64.whl
- jax前端首先会加载相关的插件，比如在cuda下会加载jax_cuda12_plugin
- 对应的python代码
```python
import jax
print(f"JAX devices: {jax.devices()}")
```

jax.devices()触发以下调用链：
```python
load_pjrt_plugin_dynamically (\home\zzw\miniconda3\lib\python3.13\site-packages\jaxlib\xla_client.py:117)
register_plugin (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:615)
initialize (\home\zzw\miniconda3\lib\python3.13\site-packages\jax_plugins\xla_cuda12\__init__.py:334)
discover_pjrt_plugins (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:478)
_discover_and_register_pjrt_plugins (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:667)
backends (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:771)
_get_backend_uncached (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:916)
get_backend (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:937)
devices (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:1003)
<module> (\home\zzw\code\compiler\jax\test_jax_build.py:9)
```
- cuda插件的初使化:initialize,会注册插件，这个插件会通过load_pjrt_plugin_dynamically来加载pjrt,参考[pjrt学习](./pjrt学习.md)
在上面注册插件的过程中，backends会返回`jaxlib._jax.Client`类型的列表，包括cpu,cuda，Client的介绍可参考[Client类在JAX生态中的地位和作用.md](./jax/jaxlib/_jax/Client类在JAX生态中的地位和作用.md)。
- jaxlib._jax.Client对应的属性和方法在`jaxlib/py_client.cc`中定义，包括devices()方法：
```c++
.def("devices", &PyClient::Devices)
```

PyClient::Devices方法定义如下：
```c++
std::vector<nb_class_ptr<PyDevice>> PyClient::Devices() {
  std::vector<nb_class_ptr<PyDevice>> devices;
  auto span = ifrt_client_->devices();
  devices.reserve(span.size());
  for (ifrt::Device* device : span) {
    devices.push_back(GetPyDevice(device));
  }
  return devices;
}
```
通过前缀nb_class_ptr可以知道PyDevice也是通过nanobind绑定到了python侧的Device类。

# gpu后端的PJRT加载
## 测试python代码
```python
import jax
print(f"JAX devices: {jax.devices()}")
```

## cuda plugin代码结构
我们编译jax会生成jax_cuda12_pjrt-0.9.1.dev0+selfbuilt-py3-none-manylinux_2_27_x86_64.whl然后进行安装。这个包对应的源码在为：jax_plugins/cuda
jax_plugins/cuda代码有几个关键处理。
首先是jax_plugins/cuda/setup.py中对于包数据和入口点的配置：
```python
    package_data={
        package_name: ["xla_cuda_plugin.so"],
    },
    zip_safe=False,
    entry_points={
        "jax_plugins": [
            f"xla_cuda{cuda_version} = {package_name}",
        ],
    },    
```
这是JAX CUDA插件的`setup.py`文件中关于包数据和入口点的配置部分，用于定义插件的安装和发现机制。

- `package_data`配置，指定需要包含在安装包中的非Python文件
  - **`package_name`**：插件包的名称（动态生成）
  - **`["xla_cuda_plugin.so"]`**：需要包含的文件列表，这里是CUDA插件的核心库文件
    - `xla_cuda_plugin.so`：编译后的CUDA插件共享库，包含了实际的CUDA硬件加速实现

- `entry_points`配置：注册插件的入口点，使JAX能够自动发现和加载该插件
- **`"jax_plugins"`**：入口点组名，对应JAX插件发现机制中的`entry_points(group="jax_plugins")`
- **`f"xla_cuda{cuda_version} = {package_name}"`**：具体的入口点定义
  - **左侧**：插件名称，格式为`xla_cuda{cuda_version}`（如`xla_cuda12`）
  - **右侧**：插件模块名称，JAX通过这个模块名导入插件
  - **`cuda_version`**：CUDA版本号（动态生成），确保不同CUDA版本的插件可以共存

从后面可以看到，jax/_src/xla_bridge.py的discover_pjrt_plugins()函数中自动发现插件时，会从`entry_points(group="jax_plugins")`自动获取到cuda pjrt这个插件。
```python
# jax/_src/xla_bridge.py
def discover_pjrt_plugins() -> None:
  from importlib.metadata import entry_points

  for entry_point in entry_points(group="jax_plugins"):
    plugin_modules.add(entry_point.value)

  # Now load and initialize them all.
  for plugin_module_name in plugin_modules:
    plugin_module = importlib.import_module(plugin_module_name)
    plugin_module.initialize()
```
discover_pjrt_plugins()发现这个插件后会导入这个插件，然后调用插件的initialize()函数进行初始化。
因此在`jax_plugins/cuda/__init__.py`中initialize()的处理如下：
```python
from jax._src.lib import triton
from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

def initialize():
  _load_nvidia_libraries()
  _import_extensions()
  c_api = xb.register_plugin(
      'cuda', priority=500, library_path=str(path), options=options
  )
  if cuda_plugin_extension:
    xla_client.register_custom_type_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_type, c_api
        ),
    )
    xla_client.register_custom_call_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_call_target, c_api
        ),
    )
    for _name, _value in cuda_plugin_extension.ffi_types().items():
      xla_client.register_custom_type(
          _name, _value, platform='CUDA'
      )
    for _name, _value in cuda_plugin_extension.ffi_handlers().items():
      xla_client.register_custom_call_target(
          _name, _value, platform='CUDA', api_version=1
      )
    triton.register_compilation_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.compile_triton_to_asm, c_api
        ),
    )
```
`initialize`函数的处理如下：
1. 加载cuda相关的库。
2. 导入cuda_plugin_extension模块，这个模块对应的是编译结果`jax_cuda12_plugin-0.9.1.dev0+selfbuilt-cp313-cp313-manylinux_2_27_x86_64.whl`,它包含了cuda相关的扩展，比如triton等编译接口(可以直接调用外部triton编译器)，对应的核心源文件是:jaxlib/gpu/gpu_plugin_extension.cc,它通过nanobind了很多的C++接口。
3. 调用xb.register_plugin来加载cuda pjrt插件，并返回c_api。
4. 通过导入jaxlib模块，给jaxlib模块注册cuda相关的扩展接口。



## python前端初使化
接下来我们看python前端的初使化过程。
jax.devices()触发以下调用链：
```python
backends (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:771)
_get_backend_uncached (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:916)
get_backend (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:937)
devices (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:1003)
<module> (\home\zzw\code\compiler\jax\test_jax_build.py:9)

```

backends()由两部分处理组成：
```python
def backends() -> dict[str, xla_client.Client]:

  _discover_and_register_pjrt_plugins()
  for platform, priority, fail_quietly in platform_registrations:
      try:
        if platform == "cuda" and not hardware_utils.has_visible_nvidia_gpu():
          continue

        backend = _init_backend(platform)
        _backends[platform] = backend
```
第一部分为自动发现和注册pjrt插件，第二部分为根据注册的插件初始化后端。

### PJRT_Api初始化
我们首先看第一部分`_discover_and_register_pjrt_plugins`的调用链：
```python
register_plugin (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:615)
initialize (\home\zzw\miniconda3\lib\python3.13\site-packages\jax_plugins\xla_cuda12\__init__.py:334)
discover_pjrt_plugins (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:478)
_discover_and_register_pjrt_plugins (\home\zzw\code\compiler\jax\jax\_src\xla_bridge.py:667)
```
结合前面的介绍，我们知道这里会自动发现jax_plugins.xla_cuda12这个插件，并调用其initialize方法，再终调用xb.register_plugin来加载cuda pjrt插件，并返回c_api。
`register_plugin`的处理如下：
```python
def register_plugin(...):
  ...
  c_api = xla_client.load_pjrt_plugin_dynamically(plugin_name, library_path)
  ...
  register_backend_factory(plugin_name, factory, priority=priority,
                           fail_quietly=False, experimental=experimental,
                           make_topology=make_topology, c_api=c_api)
```

xla_client.load_pjrt_plugin_dynamically对应的代码片段：
```python
from jaxlib import _jax as _xla
def load_pjrt_plugin_dynamically(plugin_name: str, library_path: str) -> Any:
  return _xla.load_pjrt_plugin(plugin_name, library_path, c_api=None)
```

_jax模块的load_pjrt_plugin是jaxlib/jax.cc注册的属性：
```c++
m.def(
      "load_pjrt_plugin",
      [](std::string platform_name, std::optional<std::string> library_path,
         std::optional<nb::capsule> c_api) -> nb::capsule {
        if (library_path.has_value()) {
          const PJRT_Api* api = xla::ValueOrThrow(
              pjrt::LoadPjrtPlugin(platform_name, *library_path));
          return nb::capsule(absl::bit_cast<void*>(api), "pjrt_c_api");
        }
        if (std::string_view(c_api->name()) != "pjrt_c_api") {
          throw nb::value_error(
              "c_api argument to load_pjrt_plugin is not a pjrt_c_api "
              "capsule.");
        }
        xla::ThrowIfError(pjrt::SetPjrtApi(
            platform_name, static_cast<const PJRT_Api*>(c_api->data())));
        return *c_api;
      },
      nb::arg("platform_name"), nb::arg("library_path").none() = std::nullopt,
      nb::arg("c_api").none() = std::nullopt);
```
因些_xla.load_pjrt_plugin的调用会执行pjrt::LoadPjrtPlugin函数

```c++
pjrt::CreatePjrtApi(PJRT_Error* (*)(PJRT_Client_Create_Args*), PJRT_Error* (*)(PJRT_ExecuteContext_Create_Args*), PJRT_Error* (*)(PJRT_TopologyDescription_Create_Args*), PJRT_Error* (*)(PJRT_Plugin_Initialize_Args*), PJRT_Extension_Base*, PJRT_Error* (*)(PJRT_Plugin_Attributes_Args*)) (/home/zzw/code/compiler/jax/xla/xla/pjrt/c/pjrt_c_api_wrapper_impl.cc:3163)
pjrt::gpu_plugin::GetGpuPjrtApi() (/home/zzw/code/compiler/jax/xla/xla/pjrt/c/pjrt_c_api_gpu_internal.cc:516)
::GetPjrtApi() (/home/zzw/code/compiler/jax/xla/xla/pjrt/c/pjrt_c_api_gpu.cc:30)
pjrt::LoadPjrtPlugin(std::basic_string_view<char, std::char_traits<char>>, std::basic_string_view<char, std::char_traits<char>>) (/home/zzw/code/compiler/jax/xla/xla/pjrt/pjrt_api.cc:120)
```
`pjrt::gpu_plugin::GetGpuPjrtApi()`的定义如下：
```c++
const PJRT_Api* GetGpuPjrtApi() {
  static PJRT_Gpu_Custom_Call custom_call{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_Gpu_Custom_Call_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call,
          /*next=*/&stream.base,
      },
      /*custom_call=*/PJRT_Gpu_Register_Custom_Call,
  };

  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(&custom_call.base);

  static PJRT_FFI_Extension ffi_extension =
      pjrt::CreateFfiExtension(&layouts_extension.base);

  static PJRT_MemoryDescriptions_Extension memory_descriptions_extension =
      pjrt::CreateMemoryDescriptionsExtension(&ffi_extension.base);

  static PJRT_Triton_Extension triton_extension =
      pjrt::CreateTritonExtension(&memory_descriptions_extension.base);

  static PJRT_CrossHostTransfers_Extension cross_host_transfers_extension =
      pjrt::CreateCrossHostTransfersExtension(&triton_extension.base);

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      pjrt::gpu_plugin::PJRT_Client_Create,
      pjrt::gpu_plugin::PJRT_ExecuteContext_Create,
      pjrt::gpu_plugin::PJRT_GpuDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &cross_host_transfers_extension.base,
      pjrt::gpu_plugin::PJRT_Plugin_Attributes_Gpu);

  return &pjrt_api;
}
```
`GetGpuPjrtApi()`函数通过**扩展链表机制**构建了完整的GPU平台PJRT API。这种设计允许API功能的模块化组织和灵活扩展，每个扩展专注于特定功能领域（如自定义调用、布局管理、FFI接口、内存描述、**Triton支持**、跨主机传输等）。
扩展以单链表形式组织，每个扩展都包含指向下一个扩展的指针，最终构建出完整的功能链。详细请参考[PJRT_GPU_API构建与扩展机制分析.md](./xla/xla/pjrt/c/PJRT_GPU_API构建与扩展机制分析.md)
PJRT_Plugin_Initialize这个接口API被赋值为：PJRT_Plugin_Initialize_NoOp

在这里我们可以重点关注triton扩展，它定义了triton编译的接口：
```c++
inline PJRT_Triton_Extension CreateTritonExtension(PJRT_Extension_Base* next) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_Triton_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Triton,
          /*next=*/next,
      },
      /*compile=*/PJRT_Triton_Compile,
  };
}
```
其中`PJRT_Triton_Compile`可以完成triton编译。

返回PJRT_API后进行注册，后面就可以根据其PJRT_Plugin_Initialize这个接口(PJRT_Plugin_Initialize_NoOp)完成CLIENT的创建。

### PjRtClient初使化
```python
make_pjrt_c_api_client (/home/zzw/code/compiler/jax/jax/_src/xla_bridge.py:536)
_init_backend (/home/zzw/code/compiler/jax/jax/_src/xla_bridge.py:889)
backends (/home/zzw/code/compiler/jax/jax/_src/xla_bridge.py:805)
```
在上一步中register_backend_factory会将CUDA Pjrt的初使化接口关联：make_pjrt_c_api_client，其代码处理如下：
```python
def make_pjrt_c_api_client():
  xla_client.initialize_pjrt_plugin(plugin_name)
  xla_client.make_c_api_client(plugin_name, updated_options, None)
```
注意此处xla_client包，它是由以下来导入：
```python
from jax._src.lib import xla_client
```
我们在jax/_src/lib/__init__.py又可以看到：
```python
try:
  import jaxlib as jaxlib
except ModuleNotFoundError as err:
  raise ModuleNotFoundError(
    'jax requires jaxlib to be installed. See '
    'https://github.com/jax-ml/jax#installation for installation instructions.'
    ) from err

import jaxlib.xla_client as xla_client  # noqa: F401    
```
从这里可以看到jax导入了jaxlib包，通过jaxlib包中的xla_client来和后端的c++代码进行交互.
initialize_pjrt_plugin,make_c_api_client的定义如下：
```python
from jaxlib import _jax as _xla
def initialize_pjrt_plugin(plugin_name: str) -> None:
  _xla.initialize_pjrt_plugin(plugin_name)

def make_c_api_client(
    plugin_name: str,
    options: _NameValueMapping | None = None,
    distributed_client: _xla.DistributedRuntimeClient | None = None,
    transfer_server_factory: _xla.TransferServerInterfaceFactory | None = None,
    force_dcn_cross_host_transfers: bool = False,
):
  """Creates a PJRT C API client for a PJRT plugin.
  """
  return _xla.get_c_api_client(
      plugin_name,
      options,
      distributed_client,
      transfer_server_factory,
      force_dcn_cross_host_transfers,
  )
```
从上述代码可以看到_xla实质是_jax这个模块，_jax这个模块的属性由jax.cc中进行的绑定。


initialize_pjrt_plugin会调用C++后端PJRT_API的PJRT_Plugin_Initialize接口(并没有实行性处理)
make_c_api_client则会调用xla::GetCApiClient接口，其调用链如下：
```c++
xla::PjRtClient::PjRtClient() (/home/zzw/code/compiler/jax/xla/xla/pjrt/pjrt_client.h:513)
xla::CommonPjRtClient::CommonPjRtClient() (/home/zzw/code/compiler/jax/xla/xla/pjrt/common_pjrt_client.h:55)
xla::PjRtStreamExecutorClient::PjRtStreamExecutorClient() (/home/zzw/code/compiler/jax/xla/xla/pjrt/pjrt_stream_executor_client.cc:277)
xla::StreamExecutorGpuClient::StreamExecutorGpuClient() (/home/zzw/code/compiler/jax/xla/xla/pjrt/gpu/se_gpu_pjrt_client.cc:219)
xla::GetStreamExecutorGpuClient(xla::GpuClientOptions const&) (/home/zzw/code/compiler/jax/xla/xla/pjrt/gpu/se_gpu_pjrt_client.cc:1856)
xla::GetXlaPjrtGpuClient(xla::GpuClientOptions) (/home/zzw/code/compiler/jax/xla/xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.cc:33)
pjrt::gpu_plugin::PJRT_Client_Create(PJRT_Client_Create_Args*) (/home/zzw/code/compiler/jax/xla/xla/pjrt/c/pjrt_c_api_gpu_internal.cc:203)
xla::WrapClientAroundCApi() (/home/zzw/code/compiler/jax/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc:4145)
xla::GetCApiClient() (/home/zzw/code/compiler/jax/xla/xla/pjrt/c_api_client/pjrt_c_api_client.cc:4119)
```
最终创建了xla::StreamExecutorGpuClient，并且封装到了xla::ifrt::PjRtClient，最后封装成jax::PyClient返回给前端。PJRTClient的创建过程参考[PJRT各个类的关系.md](./xla/xla/pjrt/技术调研/PJRT各个类的关系.md)


