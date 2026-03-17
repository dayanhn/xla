
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_ALLTO_ALL_ALL_GATHER_BATCH_MAT_MUL_H_
#define ACLNN_INNER_ALLTO_ALL_ALL_GATHER_BATCH_MAT_MUL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerAlltoAllAllGatherBatchMatMulGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * biasOptional : optional
 * groupEp : required
 * groupTp : required
 * epWorldSize : required
 * tpWorldSize : required
 * xShardType : optional
 * actType : optional
 * transposeWeight : optional
 * outputY2Flag : optional
 * outputY3Flag : optional
 * y1Out : required
 * y2OutOptional : optional
 * y3OutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerAlltoAllAllGatherBatchMatMulGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *biasOptional,
    char *groupEp,
    char *groupTp,
    int64_t epWorldSize,
    int64_t tpWorldSize,
    int64_t xShardType,
    int64_t actType,
    bool transposeWeight,
    bool outputY2Flag,
    bool outputY3Flag,
    const aclTensor *y1Out,
    const aclTensor *y2OutOptional,
    const aclTensor *y3OutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerAlltoAllAllGatherBatchMatMul
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerAlltoAllAllGatherBatchMatMul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
