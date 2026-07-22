package service

import "github.com/msk-mind/local-ai-broker/broker/pkg/types"

const (
	taskParamStrictRetrievalQuality  = "strict_retrieval_quality"
	taskParamExcludeDirs             = "_broker_exclude_dirs"
	taskParamRetryAction             = "_broker_retry_action"
	taskParamForcedRelease           = "_broker_forced_release"
	taskParamDispatchChunk           = "_broker_dispatch_chunk"
	taskParamDependencyBackendRunIDs = "_dependency_backend_run_ids"
	taskParamChildJobIDs             = "child_job_ids"
	taskParamRootJobID               = "root_job_id"
	taskParamRetryOfJobID            = "_broker_retry_recommended_of_job_id"
	taskParamRetryBackendPreference  = "_broker_retry_backend_preference"
	taskParamRetryTierPreference     = "_broker_retry_tier_preference"
	taskParamRetryQOS                = "_broker_retry_qos"
	taskParamRetryNodeList           = "_broker_retry_nodelist"
	taskParamRetryConstraint         = "_broker_retry_constraint"
	taskParamRetryPreemptible        = "_broker_retry_preemptible"
	taskParamRetryRationale          = "_broker_retry_rationale"
)

func ensureTaskParams(job *types.Job) map[string]any {
	job.Request.TaskParams = cloneTaskParams(job.Request.TaskParams)
	return job.Request.TaskParams
}

func setTaskParam(taskParams map[string]any, key string, value any) map[string]any {
	if taskParams == nil {
		taskParams = make(map[string]any)
	}
	taskParams[key] = value
	return taskParams
}
