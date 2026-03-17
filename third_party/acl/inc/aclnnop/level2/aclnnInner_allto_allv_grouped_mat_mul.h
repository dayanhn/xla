
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_ALLTO_ALLV_GROUPED_MAT_MUL_H_
#define ACLNN_INNER_ALLTO_ALLV_GROUPED_MAT_MUL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize
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
 * permuteOutFlag : optional
 * gmmYOut : required
 * mmYOutOptional : optional
 * permuteOutOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
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
    bool permuteOutFlag,
    const aclTensor *gmmYOut,
    const aclTensor *mmYOutOptional,
    const aclTensor *permuteOutOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerAlltoAllvGroupedMatMul
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerAlltoAllvGroupedMatMul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
