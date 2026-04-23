1.aclnn算子的调用
参考vllm-ascend，csrc/aclnn_torch_adapter/op_api_common.h
通过定义：#define EXEC_NPU_CMD(aclnn_api, ...)  这样的宏来统一调用aclnn算子。