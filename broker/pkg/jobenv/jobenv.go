package jobenv

import "github.com/msk-mind/local-ai-broker/broker/pkg/types"

const (
	TaskParamRunRoot  = "_broker_run_root"
	TaskParamRepoRoot = "_broker_repo_root"
)

func RunRoot(job types.Job) string {
	if job.Request.TaskParams != nil {
		if value, ok := job.Request.TaskParams[TaskParamRunRoot].(string); ok && value != "" {
			return value
		}
	}
	return ".broker/runs"
}

func RepoRoot(job types.Job) string {
	if job.Request.TaskParams != nil {
		if value, ok := job.Request.TaskParams[TaskParamRepoRoot].(string); ok && value != "" {
			return value
		}
	}
	return "."
}
