package backends

import (
	"context"
	"fmt"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type Backend interface {
	Name() string
	SubmitRun(context.Context, types.Job) (SubmitResponse, error)
	GetRun(context.Context, string) (RunStatus, error)
	CancelRun(context.Context, string) error
}

type BatchBackend interface {
	SubmitRunBatch(context.Context, []types.Job) ([]SubmitResponse, error)
}

type ExecutionProfileResolver interface {
	ResolveExecutionProfile(context.Context, types.SubmitJobRequest) (types.ExecutionProfile, error)
}

type SubmitResponse struct {
	BackendKind  string
	BackendRunID string
	InitialState types.JobState
}

type InlineExecutionBundle struct {
	JobSpec       map[string]any
	ExecutionPlan map[string]any
	InputManifest map[string]any
}

type InlineInspectRepoWarmSubmitter interface {
	SubmitWarmInspectRepoRun(context.Context, types.Job, InlineExecutionBundle) (SubmitResponse, bool, error)
}

type LocalInspectRepoResultWaiter interface {
	AwaitLocalInspectRepoResult(context.Context, string, time.Duration) bool
}

type RunStatus struct {
	BackendRunID string
	State        types.JobState
	RawState     string
	ExitCode     string
	Diagnostics  map[string]any
}

func StubSubmitResponse(backendKind, runID string) SubmitResponse {
	return SubmitResponse{
		BackendKind:  backendKind,
		BackendRunID: runID,
		InitialState: types.JobStateQueued,
	}
}

func StubRunStatus(runID string) RunStatus {
	return RunStatus{
		BackendRunID: runID,
		State:        types.JobStateQueued,
		RawState:     "STUB",
	}
}

func IndexedStubResponses(backendKind, prefix string, jobCount int, nextID func() uint64) []SubmitResponse {
	responses := make([]SubmitResponse, 0, jobCount)
	for range jobCount {
		runID := fmt.Sprintf("%s-%06d", prefix, nextID())
		responses = append(responses, StubSubmitResponse(backendKind, runID))
	}
	return responses
}
