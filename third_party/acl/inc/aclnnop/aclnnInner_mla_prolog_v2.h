
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MLA_PROLOG_V2_H_
#define ACLNN_INNER_MLA_PROLOG_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMlaPrologV2GetWorkspaceSize
 * parameters :
 * tokenX : required
 * weightDq : required
 * weightUqQr : required
 * weightUk : required
 * weightDkvKr : required
 * rmsnormGammaCq : required
 * rmsnormGammaCkv : required
 * ropeSin : required
 * ropeCos : required
 * cacheIndex : required
 * kvCacheRef : required
 * krCacheRef : required
 * dequantScaleXOptional : optional
 * dequantScaleWDqOptional : optional
 * dequantScaleWUqQrOptional : optional
 * dequantScaleWDkvKrOptional : optional
 * quantScaleCkvOptional : optional
 * quantScaleCkrOptional : optional
 * smoothScalesCqOptional : optional
 * rmsnormEpsilonCq : optional
 * rmsnormEpsilonCkv : optional
 * cacheModeOptional : optional
 * queryOut : required
 * queryRopeOut : required
 * kvCacheRef : required
 * krCacheRef : required
 * dequantScaleQNopeOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMlaPrologV2GetWorkspaceSize(
    const aclTensor *tokenX,
    const aclTensor *weightDq,
    const aclTensor *weightUqQr,
    const aclTensor *weightUk,
    const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq,
    const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin,
    const aclTensor *ropeCos,
    const aclTensor *cacheIndex,
    aclTensor *kvCacheRef,
    aclTensor *krCacheRef,
    const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional,
    double rmsnormEpsilonCq,
    double rmsnormEpsilonCkv,
    char *cacheModeOptional,
    const aclTensor *queryOut,
    const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMlaPrologV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMlaPrologV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
