参考：jax_plugins/cuda/__init__.py，注册ffi:
```
    for _name, _value in cuda_plugin_extension.ffi_handlers().items():
      xla_client.register_custom_call_target(
          _name, _value, platform='CUDA', api_version=1
      )
```