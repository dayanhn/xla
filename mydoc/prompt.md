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

我在为jax,xla增加ascend后端,目前完成了部分工作，为了尽可能的复用原来的代码，我选择将ascend相关的组件都继承了gpu的实现.
我计划从ThunkEmitter::EmitHloInstruction函数入口开始，根据hlo指令的opcode值来转换为通过ffi机制来调用ascend的接口。
首先实现以下Hlo指令的转换，打通整个执行路径：
ENTRY %main.1 () -> f32[128,128] {
  ROOT %loop_broadcast_fusion = f32[128,128]{1,0} fusion(), kind=kLoop, calls=
  () -> f32[128,128] {
    %constant_1_1 = f32[] constant(2)
    ROOT %broadcast_in_dim.1.1 = f32[128,128]{1,0} broadcast(%constant_1_1), dimensions={}, metadata={op_name="jit(create_full_matrix)/broadcast_in_dim" stack_frame_id=8}
  }, metadata={op_name="jit(create_full_matrix)/broadcast_in_dim" stack_frame_id=8}
}
现在实现的xla/service/ascend/thunk_emitter.cc并没有很好的完成这个模式匹配功能，无法正确的返回一个ffi类型的CustomCallThunk，请你帮我分析评估下，这个文件可以从哪些文件完善改进

在实现这个功能，需要你帮我完成以下代码框架的搭建（接口的实现细节可以分多次迭代完成，本次主要完成整体框架的搭建）：
1、参考xla/xla/service/gpu/thunk_emitter.cc，xla/xla/service/gpu/thunk_emitter.h，在xla/xla/service/ascend目录下创建类似的文件。
2、也实现ThunkEmitter::EmitHloInstruction这样的接口，然后进行模式匹配，匹配出其实是一个广播操作后，需要参考case HloOpcode::kCustomCall: {
      auto* custom_call = Cast<HloCustomCallInstruction>(hlo);
      ...
      return EmitCustomCallThunk(custom_call);
}
的处理以及ThunkEmitter::EmitCustomCallThunk函数的处理，返回一个ffi类型的CustomCallThunk。对于匹配的指令，返回一个已处理标志，否则返回未处理标志。
3、在gpu/thunk_emitter.cc的EmitHloInstruction函数入口处，判断instr->custom_call_target()是否为ascend后端，如果是则调用ascend/thunk_emitter.cc的EmitHloInstructionb函数，如果返回未处理标志，则继续调用gpu/thunk_emitter.cc的EmitHloInstruction函数，否则直接返回获取到的thunk。
4、修改相应的BUILD文件。

我在为jax,xla增加ascend后端，第一步就是能先通过ffi机制调用aclnn的算子，我已经完成了relu,matmul两个aclnn算子的ffi修饰。对于matmul要作的改动有：
ffi修饰：
google/xla/xla/service/ascend/ffi/ops/nn/matmul/matmul.cc，
xla侧算子注册：
google/xla/xla/service/ascend/ffi/ascend_ffi.cc，google/xla/xla/service/ascend/ffi/ascend_ffi.h,
jax侧注册：
google/jax/jax_plugins/ascend/ffi_ops.py
google/jax/jax_plugins/ascend/__init__.py
现在需要你参考Matmul的实现，帮我完成aclnnInplaceIndexFillTensor算子，通过ffi能够在jax中调用，
aclnnInplaceIndexFillTensor使用参考：ascend/ops-nn/docs/zh/context/两段式接口.md，使用示例：ascend/ops-nn/index/index_fill_d/examples/test_aclnn_inplace_index_fill_tensor.cpp

我在为jax,xla增加ascend后端，第一步就是能先通过ffi机制调用aclnn的算子，我已经完成了relu,matmul两个aclnn算子的ffi修饰。对于matmul要作的改动有：
ffi修饰：
jax/xla/xla/service/ascend/ffi/ops/nn/matmul/matmul.cc，
xla侧算子注册：
jax/xla/xla/service/ascend/ffi/ascend_ffi.cc，jax/xla/xla/service/ascend/ffi/ascend_ffi.h,
编译配置文件:
jax/xla/xla/service/ascend/ffi/BUILD
现在需要你参考Matmul的实现，帮我完成aclnnExpand算子，通过ffi能够在jax中调用，
aclnnExpand使用参考：ascend/ops-nn/docs/zh/context/两段式接口.md，使用说明：ascend/ops-math/math/expand/docs/aclnnExpand.md


------------
我在为jax,xla增加ascend后端，第一步就是能先通过ffi机制调用aclnn的算子，我已经完成了relu,matmul等部分aclnn算子的ffi修饰。对于matmul要作的改动有：
ffi修饰：
google/xla/xla/service/ascend/ffi/ops/nn/matmul/matmul.cc，
xla侧算子注册：
google/xla/xla/service/ascend/ffi/ascend_ffi.cc，google/xla/xla/service/ascend/ffi/ascend_ffi.h,
编译配置文件：
google/xla/xla/service/ascend/ffi/BUILD

现在需要你参考Matmul或者google/xla/xla/service/ascend/ffi/ops/nn/full/full.cc的实现，帮我完成aclnnCast 、 aclnnRightShift 、和 aclnnCat等算子，通过ffi能够在jax中调用，
aclnn算子的使用参考：ascend/ops-nn/docs/zh/context/两段式接口.md，
每个api的使用示例：
ascend/ops-math/math/cast/docs/aclnnCast.md
ascend/ops-math/math/right_shift/docs/aclnnRightShift.md
ascend/ops-math/conversion/concat_d/docs/aclnnCat.md

-------------
`\data3\zhongzhw\code\uni_ai\google\xla\xla\service\ascend\thunk_emitter.cc#L197-202` 实现了将以下的HLO算子：
  ROOT %loop_broadcast_fusion = f32[128,128]{1,0} fusion(), kind=kLoop, calls=
  () -> f32[128,128] {
    %constant_1_1 = f32[] constant(2)
    ROOT %broadcast_in_dim.1.1 = f32[128,128]{1,0} broadcast(%constant_1_1), dimensions={}, metadata={op_name="jit(create_full_matrix)/broadcast_in_dim" stack_frame_id=8}
  }, metadata={op_name="jit(create_full_matrix)/broadcast_in_dim" stack_frame_id=8}
转换为调用ascend.full.f32算子，此算子的定义和接口注册参考： `/data3/zhongzhw/code/uni_ai/google/xla/xla/service/ascend/ffi/ops/nn/full/full.cc` `\data3\zhongzhw\code\uni_ai\google\xla\xla\service\ascend\ffi\ascend_ffi.cc#L70-75` ，现在我需要你参考其实现再修改函数 `\data3\zhongzhw\code\uni_ai\google\xla\xla\service\ascend\thunk_emitter.cc#L184-184` ，支持识别以下的hlo算子：
%wrapped_convert.1 = u32[] fusion(%seed.1), kind=kLoop, calls=
  (param_0.2: s32[]) -> u32[] {
    %param_0.2 = s32[] parameter(0)
    ROOT %convert_element_type.3.1 = u32[] convert(%param_0.2), metadata={op_name="jit(_threefry_seed)/convert_element_type" stack_frame_id=8}
  }, metadata={op_name="jit(_threefry_seed)/convert_element_type" stack_frame_id=8}然后将其转换为调用 `\data3\zhongzhw\code\uni_ai\google\xla\xla\service\ascend\ffi\ascend_ffi.cc#L131-135` ，其定义在 `/data3/zhongzhw/code/uni_ai/google/xla/xla/service/ascend/ffi/ops/math/cast/cast.cc`
  
  --------
我在为jax,xla增加ascend后端，第一步就是能先通过ffi机制调用aclnn的算子，我已经完成了relu,matmul等部分aclnn算子的ffi修饰。对于matmul要作的改动有：
ffi修饰：
jax/xla/xla/service/ascend/ffi/ops/nn/matmul/matmul.cc，
xla侧算子注册：
jax/xla/xla/service/ascend/ffi/ascend_ffi.cc，jax/xla/xla/service/ascend/ffi/ascend_ffi.h,
编译配置文件：
jax/xla/xla/service/ascend/ffi/BUILD

现在需要你参考Matmul或者jax/xla/xla/service/ascend/ffi/ops/nn/full/full.cc或者jax/xla/xla/service/ascend/ffi/ops/math/expand/expand.cc的实现，帮我完成以下算子的实现：
 `\data3\zhongzhw\code\google\jax\xla\mydoc\hlo_analysis_result.txt#L27-28` `\data3\zhongzhw\code\google\jax\xla\mydoc\hlo_analysis_result.txt#L36-37` `\data3\zhongzhw\code\google\jax\xla\mydoc\hlo_analysis_result.txt#L45-46`

 ----
 `/home/zzw/code/uni_ai/xla/xla/service/ascend/thunk_emitter.cc#L422-429` 首先匹配Hlo融合算子，实现了将以下的HLO算子：
  ROOT %loop_broadcast_fusion = f32[128,128]{1,0} fusion(), kind=kLoop, calls=
  () -> f32[128,128] {
    %constant_1_1 = f32[] constant(2)
    ROOT %broadcast_in_dim.1.1 = f32[128,128]{1,0} broadcast(%constant_1_1), dimensions={}, metadata={op_name="jit(create_full_matrix)/broadcast_in_dim" stack_frame_id=8}
  }, metadata={op_name="jit(create_full_matrix)/broadcast_in_dim" stack_frame_id=8}识别出为一个常量广播，然后通过 `/home/zzw/code/uni_ai/xla/xla/service/ascend/thunk_emitter.cc#L470-535` 
转换为调用ascend.full.f32算子： `/home/zzw/code/uni_ai/xla/xla/service/ascend/ffi/ascend_ffi.cc#L70-75` ，此算子的定义和接口注册参考： `/home/zzw/code/uni_ai/xla/xla/service/ascend/ffi/ops/nn/full/full.cc` ，现在我需要你参考其实现再修改函数 thunk_emitter.cc中的absl::StatusOr<xla::gpu::ThunkSequence> ThunkEmitter::EmitFusion函数，识别出以下的Hlo融合指令：
 `/home/zzw/code/uni_ai/xla/mydoc/hlo_main.txt#L20-25` ，然后将其转换为 `/home/zzw/code/uni_ai/xla/xla/service/ascend/ffi/ops/math/add/add.cc` 中接口，其接口的定义在 `/home/zzw/code/uni_ai/xla/xla/service/ascend/ffi/ascend_ffi.cc#L145-234` ，注意根据数据类型调用相匹配的接口，可以参考 `/home/zzw/code/uni_ai/xla/xla/service/ascend/thunk_emitter.cc#L537-538` 函数