
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_FUSED_TOPK_H_
#define ACLNN_INNER_MOE_FUSED_TOPK_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeFusedTopkGetWorkspaceSize
 * parameters :
 * x : required
 * addNum : required
 * mappingNumOptional : optional
 * mappingTableOptional : optional
 * groupNum : required
 * groupTopk : required
 * topN : required
 * topK : required
 * activateType : required
 * isNorm : required
 * scale : required
 * enableExpertMapping : required
 * yOut : required
 * indicesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeFusedTopkGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *addNum,
    const aclTensor *mappingNumOptional,
    const aclTensor *mappingTableOptional,
    int64_t groupNum,
    int64_t groupTopk,
    int64_t topN,
    int64_t topK,
    int64_t activateType,
    bool isNorm,
    double scale,
    bool enableExpertMapping,
    const aclTensor *yOut,
    const aclTensor *indicesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeFusedTopk
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeFusedTopk(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
