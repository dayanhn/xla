
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_NORM_ROPE_CONCAT_H_
#define ACLNN_INNER_NORM_ROPE_CONCAT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerNormRopeConcatGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * encoderQueryOptional : optional
 * encoderKeyOptional : optional
 * encoderValueOptional : optional
 * normQueryWeightOptional : optional
 * normQueryBiasOptional : optional
 * normKeyWeightOptional : optional
 * normKeyBiasOptional : optional
 * normAddedQueryWeightOptional : optional
 * normAddedQueryBiasOptional : optional
 * normAddedKeyWeightOptional : optional
 * normAddedKeyBiasOptional : optional
 * ropeSinOptional : optional
 * ropeCosOptional : optional
 * normType : optional
 * normAddedType : optional
 * ropeType : optional
 * concatOrder : optional
 * eps : optional
 * isTraining : optional
 * queryOutputOut : required
 * keyOutputOut : required
 * valueOutputOut : required
 * normQueryMeanOutOptional : optional
 * normQueryRstdOutOptional : optional
 * normKeyMeanOutOptional : optional
 * normKeyRstdOutOptional : optional
 * normAddedQueryMeanOutOptional : optional
 * normAddedQueryRstdOutOptional : optional
 * normAddedKeyMeanOutOptional : optional
 * normAddedKeyRstdOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerNormRopeConcatGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *encoderQueryOptional,
    const aclTensor *encoderKeyOptional,
    const aclTensor *encoderValueOptional,
    const aclTensor *normQueryWeightOptional,
    const aclTensor *normQueryBiasOptional,
    const aclTensor *normKeyWeightOptional,
    const aclTensor *normKeyBiasOptional,
    const aclTensor *normAddedQueryWeightOptional,
    const aclTensor *normAddedQueryBiasOptional,
    const aclTensor *normAddedKeyWeightOptional,
    const aclTensor *normAddedKeyBiasOptional,
    const aclTensor *ropeSinOptional,
    const aclTensor *ropeCosOptional,
    int64_t normType,
    int64_t normAddedType,
    int64_t ropeType,
    int64_t concatOrder,
    double eps,
    bool isTraining,
    const aclTensor *queryOutputOut,
    const aclTensor *keyOutputOut,
    const aclTensor *valueOutputOut,
    const aclTensor *normQueryMeanOutOptional,
    const aclTensor *normQueryRstdOutOptional,
    const aclTensor *normKeyMeanOutOptional,
    const aclTensor *normKeyRstdOutOptional,
    const aclTensor *normAddedQueryMeanOutOptional,
    const aclTensor *normAddedQueryRstdOutOptional,
    const aclTensor *normAddedKeyMeanOutOptional,
    const aclTensor *normAddedKeyRstdOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerNormRopeConcat
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerNormRopeConcat(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
