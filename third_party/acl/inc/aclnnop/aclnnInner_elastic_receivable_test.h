
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_ELASTIC_RECEIVABLE_TEST_H_
#define ACLNN_INNER_ELASTIC_RECEIVABLE_TEST_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerElasticReceivableTestGetWorkspaceSize
 * parameters :
 * dstRank : required
 * group : required
 * worldSize : required
 * rankNum : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerElasticReceivableTestGetWorkspaceSize(
    const aclTensor *dstRank,
    char *group,
    int64_t worldSize,
    int64_t rankNum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerElasticReceivableTest
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerElasticReceivableTest(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
