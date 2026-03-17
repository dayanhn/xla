
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_DISTRIBUTE_DISPATCH_V2_H_
#define ACLNN_INNER_MOE_DISTRIBUTE_DISPATCH_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize
 * parameters :
 * x : required
 * expertIds : required
 * scalesOptional : optional
 * xActiveMaskOptional : optional
 * expertScalesOptional : optional
 * elasticInfoOptional : optional
 * performanceInfoOptional : optional
 * groupEp : required
 * epWorldSize : required
 * epRankId : required
 * moeExpertNum : required
 * groupTpOptional : optional
 * tpWorldSize : optional
 * tpRankId : optional
 * expertShardType : optional
 * sharedExpertNum : optional
 * sharedExpertRankNum : optional
 * quantMode : optional
 * globalBs : optional
 * expertTokenNumsType : optional
 * commAlgOptional : optional
 * zeroExpertNum : optional
 * copyExpertNum : optional
 * constExpertNum : optional
 * expandXOut : required
 * dynamicScalesOut : required
 * assistInfoForCombineOut : required
 * expertTokenNumsOut : required
 * epRecvCountOut : required
 * tpRecvCountOut : required
 * expandScalesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIds,
    const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional,
    const aclTensor *expertScalesOptional,
    const aclTensor *elasticInfoOptional,
    const aclTensor *performanceInfoOptional,
    char *groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    char *groupTpOptional,
    int64_t tpWorldSize,
    int64_t tpRankId,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t quantMode,
    int64_t globalBs,
    int64_t expertTokenNumsType,
    char *commAlgOptional,
    int64_t zeroExpertNum,
    int64_t copyExpertNum,
    int64_t constExpertNum,
    const aclTensor *expandXOut,
    const aclTensor *dynamicScalesOut,
    const aclTensor *assistInfoForCombineOut,
    const aclTensor *expertTokenNumsOut,
    const aclTensor *epRecvCountOut,
    const aclTensor *tpRecvCountOut,
    const aclTensor *expandScalesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeDistributeDispatchV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeDistributeDispatchV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
