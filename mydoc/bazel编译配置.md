# 总的目标配置
build/build.py
```python
# Define the build target for each wheel.
WHEEL_BUILD_TARGET_DICT = {
    "jax": "//:jax_wheel",
    "jax_editable": "//:jax_wheel_editable",
    "jax_source_package": "//:jax_source_package",
    "jaxlib": "//jaxlib/tools:jaxlib_wheel",
    "jaxlib_editable": "//jaxlib/tools:jaxlib_wheel_editable",
    "jax-cuda-plugin": "//jaxlib/tools:jax_cuda{cuda_major_version}_plugin_wheel",
    "jax-cuda-plugin_editable": "//jaxlib/tools:jax_cuda{cuda_major_version}_plugin_wheel_editable",
    "jax-cuda-pjrt": "//jaxlib/tools:jax_cuda{cuda_major_version}_pjrt_wheel",
    "jax-cuda-pjrt_editable": "//jaxlib/tools:jax_cuda{cuda_major_version}_pjrt_wheel_editable",
    "jax-rocm-plugin": "//jaxlib/tools:jax_rocm_plugin_wheel",
    "jax-rocm-pjrt": "//jaxlib/tools:jax_rocm_pjrt_wheel",
    "jax-ascend-plugin": "//jaxlib/tools:jax_ascend_plugin_wheel",
    "jax-ascend-plugin_editable": "//jaxlib/tools:jax_ascend_plugin_wheel_editable",
    "jax-ascend-pjrt": "//jaxlib/tools:jax_ascend_pjrt_wheel",
    "jax-ascend-pjrt_editable": "//jaxlib/tools:jax_ascend_pjrt_wheel_editable",
    "mosaic-gpu-cuda": "//jaxlib/tools:mosaic_gpu_wheel_cuda{cuda_major_version}",
}
```

# 配置jaxlib的编译
配置文件：jaxlib/tools/BUILD.bazel
目标：//jaxlib/tools:jaxlib_wheel
```python
wheel_sources(
    name = "jaxlib_sources",
    data_srcs = [
        "//jaxlib",
        "//jaxlib:jaxlib_binaries",
        "//jaxlib:_jax",
    ],
    hdr_srcs = [
        "@xla//xla/ffi/api:ffi",
    ],
    py_srcs = [
        "//jaxlib",
    ],
    static_srcs = [
        "//jaxlib:README.md",
        "LICENSE.txt",
        "//jaxlib:setup.py",
        "//jaxlib:xla_client.py",
    ],
    symlink_data_srcs = [
        "//jaxlib",
    ],
)

jax_wheel(
    name = "jaxlib_wheel",
    no_abi = False,
    source_files = [":jaxlib_sources"],
    wheel_binary = ":build_wheel_tool",
    wheel_name = "jaxlib",
)
```
