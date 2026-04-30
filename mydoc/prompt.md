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


 ------------
 请根据hlo-to-aclnn-converter技能，帮我完成卷积算子的转换。对应的hlo算子信息为： 
 %wrapped_convolution = f32[128,32,32,64]{3,2,1,0} fusion(%x.1, %params__conv1____conv____W__.1), kind=kLoop, calls= 
 (param_0.50: f32[128,32,32,3], param_1.33: f32[3,3,3,64]) -> f32[128,32,32,64] { 
   %param_0.50 = f32[128,32,32,3]{3,2,1,0} parameter(0) 
   %param_1.33 = f32[3,3,3,64]{3,2,1,0} parameter(1) 
   ROOT %conv_general_dilated.8.1 = f32[128,32,32,64]{3,2,1,0} convolution(%param_0.50, %param_1.33), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_name="jit(compute_grads)/jvp()/conv_general_dilated" stack_frame_id=20} 
 }, metadata={op_name="jit(compute_grads)/jvp()/conv_general_dilated" stack_frame_id=20} 
 需要转换为aclnnConvolution算子，对应的接口说明文档为： `/home/zzw/code/google/ascend/ops-nn/conv/convolution_forward/docs/aclnnConvolution.md` ，你需要注意aclnnConvolution接口的参数相对要复杂一点，有些参数需要从Hlo算子中提取信息，比如步长，padding，然后传入EXEC_ACLNN_CMD，由EXEC_ACLNN_CMD去将这些参数转换为aclIntArray，你需要关注aclIntArray是否已经有了这种自动转换能力
 ------------

 ascend提供的卷积算子接口如下: `/home/zzw/code/google/ascend/ops-nn/conv/convolution_forward/docs/aclnnConvolution.md` 我现在需要你研究该接口，同时参考xla下Pass的写法，比如 `/home/zzw/code/google/jax/xla/xla/backends/gpu/transforms/gemm_rewriter.cc` ,帮我在jax/xla/xla/backends/ascend/transforms目录下写一个aclnn_convolution  pass,该PASS能将卷积,+bias等指令进行融合，重写为一个custom_call指令，该指令方便后续转换为aclnn_thunk来直接调用aclnnConvolution算子。以下是一个带bias的conv hlo ir示例：
 `/home/zzw/code/google/jax/tmp/xla_dump_conv_bias_net/module_0003.jit_convnet_forward_and_backward.0000.annotate-host-compute.after_pipeline-start.before_hlo_host_device_type_call_wrapper.txt#L132-140` ，要求PASS支持带bias,也不支持不带bias的情况。并且能提取出准确全面的信息用于构造custom_call指令
 ------------
 之前我们实现了一个在ascend上执行的aclnn_gemm_rewriter的pass，（输入包括要转换的hlo算子信息，aclnn算子接口说明手册），其主要步骤如下：
1、分析输入的算子接口说明手册： `/data3/zhongzhw/code/google/ascend/ops-nn/matmul/gemm/docs/aclnnGemm.md` ，识别并理解它所支持功能，接口参数等关键信息。
2、编写相应的PASS实现接口： `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_gemm_rewriter.cc` `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_gemm_rewriter.h` ，在实现这些接口时有几个关键的要点：a. 根据输入的aclnn算子功能去匹配hlo 算子，当匹配成功后，将相应的Hlo指令转换为customcall指令： `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_gemm_rewriter.cc#L237-251` ，这里要特别注意输出的布局约束，相应的customcall target在 `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_targets.h` 中定义。然后还要封装相应的backend_config信息，这些信息主要用来传递后续调用aclnn算子所需要的参数信息。l在转换时，可能要对一些Hlo算子做融合，要注意后续的算子融合前面已经转换完成的算子时依然要重新一个新的custom_call算子，且target保持不变。
3、在 `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_config.cc` `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_config.h` 中增加相应算子的congig的定义，parse,serialize接口。
4、修改 `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/BUILD` ，增加新增的PASS的编译配置。
5、修改 `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_targets.cc` `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_targets.h` ，增加识别新增的target的接口
6、在 `/data3/zhongzhw/code/google/jax/xla/xla/service/ascend/thunk_emitter.cc#L4519-4520` ThunkEmitter::EmitHloInstruction接口增加target匹配的判断，识别出新增的custom_call指令，并返回aclnnthunk: `/data3/zhongzhw/code/google/jax/xla/xla/service/ascend/thunk_emitter.cc#L4552-4554` 。
7、为相应的custom_call指令编写发射aclnnthunk的接口： `/data3/zhongzhw/code/google/jax/xla/xla/service/ascend/thunk_emitter.cc#L4037-4121` ， 这里主要是从backend_config解析出调用aclnn算子的参数信息，然后将算子的操作数，返回值，非张量参数等信息构造一个AclnnThunk返回。
8、在 `/data3/zhongzhw/code/google/jax/xla/xla/service/ascend/thunk_emitter.h` 中声明新加的emit接口。
现在我需要你对上述步骤转换成一个技能，指导下次类似功能的实现，当我提供Hlo算子输入，aclnn算子接口说明手册后，能正确的实现Pass转换，指令发射等功能。
---------------
`/data3/zhongzhw/code/google/jax/tmp/a.txt` 是需要优化的hlo ir,现在我需要匹配里面的卷积算子，然后可以转换成调用： `/data3/zhongzhw/code/google/ascend/ops-nn/conv/convolution_forward/docs/aclnnConvolution.md` ，请你根据aclnn-pass-implementer这个技能帮我完成功能，原来已经实现了部分功能，比如： `/data3/zhongzhw/code/google/jax/xla/xla/backends/ascend/transforms/aclnn_convolution_rewriter.cc` ，但存在很多的错误，请你重新依据技能帮我实现一遍。
修改完成后需要你执行 `/data3/zhongzhw/code/google/jax/build.sh` 这个编译脚本，直到编译通过。在编译过程中如果出现错误，请认真解决，遵循功能正确，而不是 为了解决错误，直接裁剪功能。另外每次修改完成后要在同一个终端里执行build.sh，另外如果另外开一个终端编译会导致jax,xla又全量重编，需要大量的时间。