
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_GROUPED_MAT_MUL_ALLTO_ALLV_H_
#define ACLNN_INNER_GROUPED_MAT_MUL_ALLTO_ALLV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize
 * parameters :
 * gmmX : required
 * gmmWeight : required
 * sendCountsTensorOptional : optional
 * recvCountsTensorOptional : optional
 * mmXOptional : optional
 * mmWeightOptional : optional
 * group : required
 * epWorldSize : required
 * sendCounts : required
 * recvCounts : required
 * transGmmWeight : optional
 * transMmWeight : optional
 * yOut : required
 * mmYOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor *gmmX,
    const aclTensor *gmmWeight,
    const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional,
    const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional,
    char *group,
    int64_t epWorldSize,
    const aclIntArray *sendCounts,
    const aclIntArray *recvCounts,
    bool transGmmWeight,
    bool transMmWeight,
    const aclTensor *yOut,
    const aclTensor *mmYOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerGroupedMatMulAlltoAllv
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerGroupedMatMulAlltoAllv(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
