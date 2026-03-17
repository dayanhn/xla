
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ROPE_QUANT_KVCACHE_H_
#define ACLNN_ROPE_QUANT_KVCACHE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRopeQuantKvcacheGetWorkspaceSize
 * parameters :
 * qkv : required
 * cos : required
 * sin : required
 * quantScale : required
 * quantOffset : required
 * kCacheRef : required
 * vCacheRef : required
 * indice : required
 * sizeSplitsOptional : optional
 * layoutOptional : optional
 * kvOutput : optional
 * qOut : required
 * kOut : required
 * vOut : required
 * kCacheRef : required
 * vCacheRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRopeQuantKvcacheGetWorkspaceSize(
    const aclTensor *qkv,
    const aclTensor *cos,
    const aclTensor *sin,
    const aclTensor *quantScale,
    const aclTensor *quantOffset,
    aclTensor *kCacheRef,
    aclTensor *vCacheRef,
    const aclTensor *indice,
    const aclIntArray *sizeSplitsOptional,
    char *layoutOptional,
    bool kvOutput,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRopeQuantKvcache
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRopeQuantKvcache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
