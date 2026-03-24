我在为jax,xla增加ascend后端，当前已经完成了xla下部分代码的添加，这部分内容可以参考 ：xla/mydoc/XLA_Ascend后端代码结构报告.md， xla下编译测试ascend功能代码为： xla/xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client_test.cc ，编译脚本为： xla/build_ascend_client.sh ，对于Ascend要根据CANN的安装位置来初使化环境，比如CANN安装在： ~/Ascend8.5REL路径，此时用以下命令来初使化环境变量：source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh,这条命令会创建以下环境变量：ASCEND_TOOLKIT_HOME=/data3/zhongzhw/Ascend8.5REL/cann-8.5.0，然后链接的库在：$ASCEND_TOOLKIT_HOME/lib64下。我当前ascend后端用到的库有：--linkopt=-lascendcl --linkopt=-lnnopbase --linkopt=-lopapi_nn  --linkopt=-lhccl --linkopt=-lhcomm。
现在我还需要为jax增加ascend后端，主要是参考cuda后端的代码，编译支持cuda的命令为：
```shell
    python build/build.py build --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt --editable --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla
```
同样的，我也想通过类似的命令为jax增加ascend后端编译：
```shell
python build/build.py build --wheels=jaxlib,jax-ascend-plugin,jax-ascend-pjrt --editable --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla

```
首先请你帮我结合jax,xla的编译配置分析下，编译jaxlib这个wheel包时，会不会编译cuda或者ascend的相关代码？还是完全和后端无关？

python build/build.py build --wheels=jax-cuda-plugin --editable --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla

我现在参考cuda增加jax的ascend后端，但是相关的代码和配置文件都还存在问题，请你首先分析wheels目标为jax-cuda-plugin的完整编译过程，然后帮我解决以下的编译问题：

python build/build.py build --wheels=jax-ascend-pjrt --editable --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla




我在为jax,xla增加ascend后端，xla完成的工作如下： XLA_Ascend后端代码结构报告.md ，目前能独立前端Jax进行编译测试， xla下编译测试ascend功能代码为： xla/xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client_test.cc ，编译脚本为： xla/build_ascend_client.sh ，对于Ascend要根据CANN的安装位置来初使化环境，比如CANN安装在： ~/Ascend8.5REL路径，此时用以下命令来初使化环境变量：source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh,这条命令会创建以下环境变量：ASCEND_TOOLKIT_HOME=/data3/zhongzhw/Ascend8.5REL/cann-8.5.0，然后链接的库在：$ASCEND_TOOLKIT_HOME/lib64下。我当前ascend后端用到的库有：--linkopt=-lascendcl --linkopt=-lnnopbase --linkopt=-lopapi_nn  --linkopt=-lhccl --linkopt=-lhcomm。pjrt这部分功能还很不完备。
jax也只增加了一部分配置，jax和xla还没有打通。现在我想参考编译jax-cuda-pjrt来编译jax-ascend-pjrt,
jax-cuda-pjrt的编译命令为：
    ```shell
    python build/build.py build --wheels=jax-cuda-pjrt --editable --bazel_options=--compilation_mode=dbg --bazel_options=--copt=-g --bazel_options=--copt=-O0 --bazel_options=--strip=never --bazel_options=--override_repository=xla=$(pwd)/xla --local_xla_path=$(pwd)/xla
    ```
首先请你结合代码深入分析下jax-cuda-pjrt是如何编译的

请你帮我完成缺失的