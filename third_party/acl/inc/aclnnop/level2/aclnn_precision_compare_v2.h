/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_PRECISION_COMPARE_V2_H_
#define OP_API_INC_PRECISION_COMPARE_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnPrecisionCompareV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 比较计算两个Tensor是否相同，返回算子执行的状态码:
 *
 * @param [in] golden: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT数据类型，
 * golden与realdata数据类型一致
 * @param [in] realdata: npu device侧的aclTensor，
 * @param [in] detect_type: host侧的整数，数据类型支持UINT32。
 * 数据类型支持FLOAT16,FLOAT数据类型，
 * realdata与golden数据类型一致
 * @param [in] out: 输出一个数据类型为UINT32类型的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus aclnnPrecisionCompareV2GetWorkspaceSize(const aclTensor* golden, const aclTensor* realdata,
                                                              uint32_t detect_type, aclTensor* out,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnPrecisionCompareV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnPrecisionCompareV2GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus aclnnPrecisionCompareV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif  // OP_API_INC_PRECISION_COMPARE_V2_H_