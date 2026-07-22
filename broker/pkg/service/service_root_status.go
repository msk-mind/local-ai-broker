package service

import "github.com/msk-mind/local-ai-broker/broker/pkg/types"

type rootSummaryState struct {
	jobs                []types.Job
	usage               rootActionUsageSummary
	dispatchingChildren int
	pendingChildren     int
	activeChunks        int
	pendingChunks       int
	effectiveChildren   []types.Job
	effectiveReducers   []types.Job
}

func buildRootSummaryState(jobs []types.Job, includeCancelled bool) rootSummaryState {
	dispatchingChildren, pendingChildren, activeChunks, pendingChunks := dispatchObservability(jobs)
	return rootSummaryState{
		jobs:                jobs,
		usage:               rootActionUsage(jobs),
		dispatchingChildren: dispatchingChildren,
		pendingChildren:     pendingChildren,
		activeChunks:        activeChunks,
		pendingChunks:       pendingChunks,
		effectiveChildren:   effectiveShardJobs(jobs, includeCancelled),
		effectiveReducers:   effectiveReducerJobs(jobs),
	}
}

func (s *Service) rootJobStatusFromJobs(rootJobID string, jobs []types.Job) types.RootJobStatus {
	return rootJobStatusFromState(rootJobID, buildRootSummaryState(jobs, false))
}

func rootJobStatusFromState(rootJobID string, state rootSummaryState) types.RootJobStatus {
	status := types.RootJobStatus{
		RootJobID: rootJobID,
		State:     types.JobStateQueued,
	}
	status.DispatchingChildren = state.dispatchingChildren
	status.PendingChildren = state.pendingChildren
	status.ActiveChunks = state.activeChunks
	status.PendingChunks = state.pendingChunks
	status.ForcedReleasedChunks = state.usage.ForcedReleasedChunks
	status.RetriedShardActions = state.usage.RetriedShardActions

	var reducer *types.Job
	aggregationKeys := map[string]struct{}{}

	for i := range state.effectiveChildren {
		applyRootStatusJob(&status, state.effectiveChildren[i], aggregationKeys, true)
	}
	if len(state.effectiveReducers) > 0 {
		reducer = &state.effectiveReducers[0]
		applyRootStatusJob(&status, *reducer, aggregationKeys, false)
	}
	if reducer != nil {
		status.ReducerJobID = reducer.ID
		status.ReducerState = reducer.State
		status.ReducerDeferred = reducer.State == types.JobStateDispatching && reducer.BackendRunID == ""
		applyReducerMetrics(&status, reducer.Result)
	}
	status.AggregationKeys = setKeys(aggregationKeys)
	status.State = aggregateRootState(status)
	return status
}

func applyRootStatusJob(status *types.RootJobStatus, job types.Job, aggregationKeys map[string]struct{}, includeChildID bool) {
	if status == nil {
		return
	}
	status.TotalJobs++
	switch job.State {
	case types.JobStateAccepted, types.JobStateQueued, types.JobStateDispatching:
		status.QueuedJobs++
	case types.JobStateRunning:
		status.RunningJobs++
	case types.JobStateSucceeded:
		status.SucceededJobs++
	case types.JobStateFailed, types.JobStatePreempted, types.JobStateTimedOut:
		status.FailedJobs++
	case types.JobStateCancelled:
		status.CancelledJobs++
	}
	if includeChildID {
		status.ChildJobIDs = append(status.ChildJobIDs, job.ID)
	}
	if job.Orchestration != nil && job.Orchestration.AggregationKey != "" {
		aggregationKeys[job.Orchestration.AggregationKey] = struct{}{}
	}
	if job.ResultError != "" && status.RepresentativeError == "" {
		status.RepresentativeError = job.ResultError
	}
	if progressNewer(job.Progress, status.Progress) {
		status.Progress = job.Progress
	}
}

func setKeys(values map[string]struct{}) []string {
	if len(values) == 0 {
		return nil
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	return keys
}

func aggregateRootState(status types.RootJobStatus) types.JobState {
	if status.ReducerJobID != "" && status.ReducerState != "" {
		return status.ReducerState
	}
	if status.RunningJobs > 0 {
		return types.JobStateRunning
	}
	if status.FailedJobs > 0 {
		return types.JobStateFailed
	}
	if status.QueuedJobs > 0 {
		return types.JobStateQueued
	}
	if status.CancelledJobs == status.TotalJobs {
		return types.JobStateCancelled
	}
	if status.SucceededJobs == status.TotalJobs {
		return types.JobStateSucceeded
	}
	return types.JobStateQueued
}

func applyReducerMetrics(status *types.RootJobStatus, result *types.Result) {
	if status == nil || result == nil || result.Payload == nil {
		return
	}
	raw, ok := result.Payload["aggregate_metrics"].(map[string]any)
	if !ok {
		return
	}
	status.ChildrenTotal = intFromAny(raw["children_total"])
	status.ChildrenSucceeded = intFromAny(raw["children_succeeded"])
	status.ChildrenFailed = intFromAny(raw["children_failed"])
	status.CoverageFraction = floatFromAny(raw["coverage_fraction"])
}
