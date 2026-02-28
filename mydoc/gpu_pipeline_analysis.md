# GPU 编译流水线各阶段说明（以 stablehlo_gemm.mlir 为例）

本文档记录了在 `stablehlo_gemm.mlir` 上运行
``./gpu_compiler.sh 1`` 并在 XLA/HLO 编译流程的多个阶段导出
IR 时的观察结果。目的是识别哪个 pass 做了**关键**变换，
特别关注用户提到的五个 pass：

# * `conv-rewriter`
# * `triton-gemm-rewriter`
# * `autotuner`
# * `nest_gemm_fusion`
# * `priority-fusion`

In addition to those, a number of other passes also modify the module in
important ways; they are listed below with brief descriptions and
examples taken from the dumped files under `tmp/ir`.

---

## 1. 早期代数简化

**dump 文件里显示为：**`0006.simplification.after_algsimp`

在进行任何 GPU 专用优化之前，通用的 HLO 代数简化器
(`AlgebraicSimplifier`)会重写表达式、折叠常量的 `add`/`multiply`、
规范化卷积的窗口等。在我们的模块里，它把卷积的
`window={size=3x3 pad=1_1x1_1}` 规范化，并消掉了
reduce 操作中多余的加法，但并不修改控制流或数据布局。

*示例 diff：*

```diff
%add.1 = f32[] add(%Arg_0.2, %Arg_1.2)
// 常量合并或删除，语义不变
```

虽然改动很小，这些简化会减小 AST 大小，并有时为后续
融合打开机会。

## 2. Reshape 分解

**pass 名称：**`0012.reshape-decomposer`（出现在 layout 正规化之前）

`ReshapeDecomposer` 将任意 reshape 拆成一系列
`bitcast`、`transpose`、`copy`，便于后续布局和融合
pass 理解。在我们的 trace 中，`%Arg_0` 的 reshape 仅
保持为简单 bitcast，因为形状本来就兼容。

> 这个 pass 主要是结构性的；在更大的模块中它可能插入
> 隐式拷贝或转置，而后续的布局分配可能把这些拷贝
> 提前或消除。

data layouts appropriate for each operation (row‑major, column‑major,
vectorized, etc.).  In our trace it adds a `copy` to convert the filter
operand `%Arg_2.1` from `{3,2,1,0}` to `{2,1,0,3}` which matches the
CUDNN GEMM implementation.
## 3. 布局分配与规范化

**pass 名称：**`0016.layout_assignment`、
`0018.layout_normalization`、`0020.move_copy_to_users` 等。

当 HLO 形状确认后，`LayoutAssignment` 会为每个操作选择
适当的物理布局（行主、列主、向量化等）。在本例中，它
在滤波器操作后插入了一个 `copy`，将 `%Arg_2.1` 从
`{3,2,1,0}` 转为 `{2,1,0,3}`，以匹配 CUDNN 的要求：

```
%copy = f32[3,3,1,8]{2,1,0,3} copy(%Arg_2.1)
```

layout_normalization 随后会压缩或移动这些拷贝；
`MoveCopyToUsers` 将 copy 移动到了卷积之前，但因 conv
custom-call 需要该布局而未被移除。（后续的 `priority-fusion`
可能因 tiling 元数据变化再生成拷贝。）

这些转换对 GPU 性能至关重要，因为内存访问通常是性能
瓶颈。

## 4. 填充 tiling 元数据与拷贝移动

`add_tiling_metadata` 和 `move_copy_to_users` 等 pass 并不改变计算
语义；它们只是给 HLO 指令添加 tile 大小的元数据，并尽量
缩小布局转换拷贝的范围。在 dump 文件中唯一可见的效果是
backend_config 字段中插入了 tiling metadata 字符串，以及
前面提到的 copy 的位置变动。

## 5. Conv 重写器 (`conv-rewriter`)

**出现在 dump 名称中：**`0031.conv-rewriter`。

该 GPU 专用 pass 将一个 `conv` HLO 替换为 CUDNN custom call。
它检查卷积的维度号和窗口参数，提取 stride/ dilation 信息，
然后生成带有序列化 `cudnn_conv_backend_config` 的
`__cudnn$convForward` 调用。IR 从普通的 `conv` 变为
`custom-call` 并且对滤波器做了转置以符合 CUDNN 布局。

```
%transpose = ...  // conv-rewriter 插入
%cudnn-conv = (f32[...], u8[0]) custom-call(...,
             custom_call_target="__cudnn$convForward", backend_config=...)
```

这个 pass 还会为后续的自动调优注册该操作。

## 6. Triton GEMM 重写器 (`triton-gemm-rewriter`)

**出现于：**`0039.triton-gemm-rewriter` 以及更早的
`0023.after_triton-gemm-rewriter`。

`GemmFusion` 寻找 `dot` 操作，当满足特定条件（例如两个
操作数都来自参数或常量）时，将该 dot 替换为一个带有
Triton 注释的 fusion 节点（`__triton_gemm`）。
它为融合 kernel 构建一个小计算体，并设置 tile 和
`triton_gemm_config` 选项（例如选用 K=4/8/16）。
这是把 dot 转为 GPU 代码的第一个主要步骤。

## 7. 自动调优（`autotuner`）和融合发射器

融合完成后，`AutotunerPass`（及其助手
`GemmFusionAutotuner`）遍历带有 backend_config 的 fusion
指令，并执行搜索来选择最优的 tile 大小/算法。在 dump
里该过程出现在 `0051.autotune-fusion-emitters`，配置字段
被填入选定参数。pass 会把决策记录到缓存中，并相应地
重写 fusion 的 `triton_gemm_config`。

## 8. 嵌套 GEMM 融合 (`nest_gemm_fusion`)

在 `0031.post-layout_assignment.after_nest_gemm_fusion` 出现。
当一个 Triton GEMM 融合的输出被若干逐元素操作消费时，
该 pass 会把它们合并到同一个 fusion 节点以减少内存通信。
我们的简例中并没有实际嵌套 GEMM，但该 pass 仍然执行了，
只是没有修改 fusion 节点。

## 9. 优先级驱动的融合 (`priority-fusion`)

出现在 CSE 之前（`0039.fusion.after_priority-fusion`）。
这个调度器基于成本模型（预估 FLOP 和字节数）重新排序
fusion，使最高成本的放在最前，并在 fusion 节点上添加
`tile_sizes` 字段，为后续的循环融合提供指导。我们的例子
中只有一个 GEMM，因此它获得了唯一的优先级，后续
CSE、多输出融合等都在这个排序上进行。

## 10. 其它融合 / 非融合 Pass

下列 pass 在 dump 名称中可见，但对于本模块要么无改动，
要么只是机械式重写——不过在更复杂的计算中它们依然很重要：

* **generic-fusion** / **loop-fusion** – 合并相邻的逐元素操作或
  循环。我们的模块包含一些 add、multiply 和 broadcast；
  generic-fusion 会在早期生成一个小的 kElemWise fusion。
  loop-fusion（`0037.pre-fusion`）将多个 fusion 组合在一起，
  减少 kernel 启动开销。
* **nested-dot-fusion** – 将 `A·B·C` 这样的 dot 链合并为单个
  kernel。本例中只有一个 dot，因此无改动。
* **dot-decomposer** – 把一个大型 dot 拆成更小的块（用于 tiling
  或当形状超过上限时）。未触发。
* **call-inliner** – 内联函数，减少调用开销；我们看到它将
  reduce 内部的 `region_0.1` 加法内联了。
* **scatter-expander**, **rng-expander**, **logits** – 将特殊的 HLO
  展开为更原始的操作，或者为硬件重写；我们的运行中没有
  这些操作，因此无变化。
* **no-op-elimination**, **constant-folding** – 清理冗余操作并折叠
  编译时常量。elimination pass 运行了两次，但只删除了 dot
  融合后多余的 bitcast。
* **lower-collective**, **host-memory-transfer-asyncifier** 等 – 这些
  属于通用流水线，与我们这个简单计算无关。

Although several of these passes don’t change the IR in this particular
example, they are still “key” to the overall pipeline: they keep the
module small, canonical, and tractable for later GPU‑specific
optimizations.  In larger models they can have substantial impact on
performance and memory usage.

---

### 总结

`tmp/ir` 下的 IR dump 为编译器的变换过程提供了一条面包屑。
通过将文件名与源代码（`conv_rewriter.cc`、`gemm_fusion.cc`、
`autotuner_pass.cc` 等）对应起来，可以清楚地看到每个重写何时
发生以及为何发生。对于 stablehlo_gemm 例子，用户最初强调的
五个 pass 确实是最戏剧性的，它们把高层的卷积和 dot 转换为
融合的 GPU kernel。然而，环绕它们的那些基础工作——代数
简化、布局分配、融合调度以及各种清理——在实践中同样重要。
若无这些前奏，kernel 生成器要么产生错误代码，要么表现欠佳。

流水线很长；上述 dump 名称列表可以作为检查其它模块时的
一个清单。

*由 Copilot 在研究 XLA GPU 优化时生成。*
