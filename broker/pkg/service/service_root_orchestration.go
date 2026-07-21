package service

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func (s *Service) submitRetriedReducer(ctx context.Context, reducer types.Job, effectiveChildren []types.Job) (*types.SubmitJobResponse, error) {
	childJobIDs, childBackendRunIDs := reducerDependencies(effectiveChildren)
	reducerReq := retrySubmitRequest(reducer)
	reducerReq.TaskParams = setTaskParam(reducerReq.TaskParams, taskParamChildJobIDs, append([]string(nil), childJobIDs...))
	reducerReq.TaskParams = setTaskParam(reducerReq.TaskParams, taskParamRootJobID, reducer.RootJobID)
	reducerReq.TaskParams = setTaskParam(reducerReq.TaskParams, taskParamDependencyBackendRunIDs, append([]string(nil), childBackendRunIDs...))

	resp, err := s.SubmitJob(ctx, reducerReq)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (s *Service) submitStoredReducer(ctx context.Context, reducer types.Job, effectiveChildren []types.Job) error {
	childJobIDs, childBackendRunIDs := reducerDependencies(effectiveChildren)
	taskParams := ensureTaskParams(&reducer)
	taskParams[taskParamChildJobIDs] = append([]string(nil), childJobIDs...)
	taskParams[taskParamRootJobID] = reducer.RootJobID
	taskParams[taskParamDependencyBackendRunIDs] = append([]string(nil), childBackendRunIDs...)
	if err := s.stageExecutionBundle(ctx, &reducer); err != nil {
		return err
	}
	submitResp, err := s.backend.SubmitRun(ctx, reducer)
	if err != nil {
		return err
	}
	reducer.State = submitResp.InitialState
	reducer.BackendKind = submitResp.BackendKind
	reducer.BackendRunID = submitResp.BackendRunID
	reducer.UpdatedAt = time.Now().UTC()
	return s.store.UpdateJob(ctx, reducer)
}

func (s *Service) createDeferredReducer(ctx context.Context, req types.SubmitJobRequest) (*types.SubmitJobResponse, error) {
	job := s.newJob(ctx, req, types.JobStateDispatching, "", "")
	if err := s.stageExecutionBundle(ctx, &job); err != nil {
		return nil, err
	}
	if err := s.store.CreateJob(ctx, job); err != nil {
		return nil, err
	}
	s.audit(ctx, "job.submit", "success", &job, map[string]any{
		"backend_kind": job.BackendKind,
	})
	resp := submitJobResponse(job)
	return &resp, nil
}

func reducerDependencies(children []types.Job) ([]string, []string) {
	childJobIDs := make([]string, 0, len(children))
	childBackendRunIDs := make([]string, 0, len(children))
	for _, child := range children {
		childJobIDs = append(childJobIDs, child.ID)
		if child.BackendRunID != "" {
			childBackendRunIDs = append(childBackendRunIDs, child.BackendRunID)
		}
	}
	return childJobIDs, childBackendRunIDs
}

type dispatchReleaseResult struct {
	ReleasedChunks   int
	ReleasedChildren int
	ReducerReleased  bool
}

type rootDispatchState struct {
	jobs              []types.Job
	activeChunkCount  int
	pendingByChunk    map[int][]types.Job
	pendingChunkOrder []int
	effectiveChildren []types.Job
	effectiveReducers []types.Job
}

func (s *Service) releaseDispatchingRootChildren(ctx context.Context, rootJobID string, maxAdditionalBatches int, forced bool) (dispatchReleaseResult, error) {
	result := dispatchReleaseResult{}
	if strings.TrimSpace(rootJobID) == "" {
		return result, nil
	}

	state, err := s.loadRootDispatchState(ctx, rootJobID)
	if err != nil {
		return result, err
	}
	if len(state.jobs) == 0 {
		return result, nil
	}

	slots := availableRootBatchSlots(s.options.ParallelMaxActiveBatches, state.activeChunkCount)
	if maxAdditionalBatches > 0 {
		slots = maxAdditionalBatches
	}
	if slots > 0 {
		for _, chunkIndex := range state.pendingChunkOrder {
			if slots == 0 {
				break
			}
			chunkJobs := state.pendingByChunk[chunkIndex]
			if len(chunkJobs) == 0 {
				continue
			}
			if err := s.submitStoredChunk(ctx, chunkJobs, forced); err != nil {
				return result, err
			}
			result.ReleasedChunks++
			result.ReleasedChildren += len(chunkJobs)
			slots--
		}
	}

	if len(state.pendingByChunk) > 0 {
		return result, nil
	}

	for _, reducer := range state.effectiveReducers {
		if reducer.State == types.JobStateDispatching && reducer.BackendRunID == "" {
			if err := s.submitStoredReducer(ctx, reducer, state.effectiveChildren); err != nil {
				return result, err
			}
			result.ReducerReleased = true
			return result, nil
		}
	}
	return result, nil
}

func (s *Service) loadRootDispatchState(ctx context.Context, rootJobID string) (rootDispatchState, error) {
	jobs, err := s.loadRootJobsForDispatch(ctx, rootJobID)
	if err != nil {
		return rootDispatchState{}, err
	}
	return buildRootDispatchState(jobs), nil
}

func (s *Service) loadRootJobsForDispatch(ctx context.Context, rootJobID string) ([]types.Job, error) {
	rootJobs, err := s.loadRootJobsByID(ctx, rootJobID)
	if err != nil || len(rootJobs) == 0 {
		return rootJobs, err
	}
	if err := s.refreshRootJobsForDispatch(ctx, rootJobs); err != nil {
		return nil, err
	}
	return s.loadRootJobsByID(ctx, rootJobID)
}

func (s *Service) loadRootJobsByID(ctx context.Context, rootJobID string) ([]types.Job, error) {
	jobs, err := s.store.ListJobs(ctx)
	if err != nil {
		return nil, err
	}
	rootJobs := make([]types.Job, 0, len(jobs))
	for _, job := range jobs {
		if job.RootJobID == rootJobID {
			rootJobs = append(rootJobs, job)
		}
	}
	return rootJobs, nil
}

func buildRootDispatchState(jobs []types.Job) rootDispatchState {
	pendingByChunk := dispatchingChildrenByChunk(jobs)
	return rootDispatchState{
		jobs:              jobs,
		activeChunkCount:  len(activeDispatchChunks(jobs)),
		pendingByChunk:    pendingByChunk,
		pendingChunkOrder: sortedChunkIndexes(pendingByChunk),
		effectiveChildren: effectiveShardJobs(jobs, false),
		effectiveReducers: effectiveReducerJobs(jobs),
	}
}

func (s *Service) refreshRootJobsForDispatch(ctx context.Context, jobs []types.Job) error {
	for _, job := range jobs {
		if job.BackendRunID == "" || isTerminal(job.State) {
			continue
		}
		runStatus, err := s.backend.GetRun(ctx, job.BackendRunID)
		if err != nil || runStatus.State == "" || runStatus.State == job.State {
			continue
		}
		job.State = runStatus.State
		job.BackendState = runStatus.RawState
		job.BackendExitCode = runStatus.ExitCode
		job = mergeBackendRunDiagnostics(job, runStatus)
		if runStatus.State == types.JobStateFailed && strings.TrimSpace(job.ResultError) == "" {
			job.ResultError = "worker_failed_before_result"
		}
		now := time.Now().UTC()
		job.UpdatedAt = now
		if runStatus.State == types.JobStateRunning && job.StartedAt == nil {
			job.StartedAt = &now
		}
		if isTerminal(runStatus.State) {
			job.CompletedAt = &now
		}
		if err := s.store.UpdateJob(ctx, job); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) submitStoredChunk(ctx context.Context, chunkJobs []types.Job, forced bool) error {
	if len(chunkJobs) == 0 {
		return nil
	}
	if len(chunkJobs) == 1 {
		if forced {
			taskParams := ensureTaskParams(&chunkJobs[0])
			taskParams[taskParamForcedRelease] = true
		}
		submitResp, err := s.backend.SubmitRun(ctx, chunkJobs[0])
		if err != nil {
			return fmt.Errorf("submit child shard %d: %w", shardIndexOf(chunkJobs[0]), err)
		}
		return s.applyStoredSubmission(ctx, chunkJobs[0], submitResp)
	}
	if batchBackend, ok := s.backend.(backends.BatchBackend); ok {
		if forced {
			for i := range chunkJobs {
				taskParams := ensureTaskParams(&chunkJobs[i])
				taskParams[taskParamForcedRelease] = true
			}
		}
		submitResps, err := batchBackend.SubmitRunBatch(ctx, chunkJobs)
		if err != nil {
			return fmt.Errorf("submit child batch chunk %d: %w", dispatchChunkIndex(chunkJobs[0]), err)
		}
		if len(submitResps) != len(chunkJobs) {
			return fmt.Errorf("submit child batch chunk %d: expected %d responses, got %d", dispatchChunkIndex(chunkJobs[0]), len(chunkJobs), len(submitResps))
		}
		for i := range chunkJobs {
			if err := s.applyStoredSubmission(ctx, chunkJobs[i], submitResps[i]); err != nil {
				return err
			}
		}
		return nil
	}
	for _, job := range chunkJobs {
		if forced {
			job.Request.TaskParams = setTaskParam(job.Request.TaskParams, taskParamForcedRelease, true)
		}
		submitResp, err := s.backend.SubmitRun(ctx, job)
		if err != nil {
			return fmt.Errorf("submit child shard %d: %w", shardIndexOf(job), err)
		}
		if err := s.applyStoredSubmission(ctx, job, submitResp); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) applyStoredSubmission(ctx context.Context, job types.Job, submitResp backends.SubmitResponse) error {
	job.State = submitResp.InitialState
	job.BackendKind = submitResp.BackendKind
	job.BackendRunID = submitResp.BackendRunID
	job.UpdatedAt = time.Now().UTC()
	return s.store.UpdateJob(ctx, job)
}

func effectiveShardJobs(jobs []types.Job, includeCancelled bool) []types.Job {
	grouped := make(map[string][]types.Job)
	for _, job := range jobs {
		if orchestrationStrategyOf(job) == "aggregator" {
			continue
		}
		grouped[shardAttemptKey(job)] = append(grouped[shardAttemptKey(job)], job)
	}
	effective := make([]types.Job, 0, len(grouped))
	for _, attempts := range grouped {
		effective = append(effective, selectEffectiveAttempt(attempts, includeCancelled))
	}
	sort.SliceStable(effective, func(i, j int) bool {
		if shardIndexOf(effective[i]) != shardIndexOf(effective[j]) {
			return shardIndexOf(effective[i]) < shardIndexOf(effective[j])
		}
		if shardKeyOf(effective[i]) != shardKeyOf(effective[j]) {
			return shardKeyOf(effective[i]) < shardKeyOf(effective[j])
		}
		return jobMoreRecent(effective[i], effective[j])
	})
	return effective
}

func effectiveReducerJobs(jobs []types.Job) []types.Job {
	grouped := make(map[string][]types.Job)
	for _, job := range jobs {
		if orchestrationStrategyOf(job) != "aggregator" {
			continue
		}
		grouped[reducerAttemptKey(job)] = append(grouped[reducerAttemptKey(job)], job)
	}
	effective := make([]types.Job, 0, len(grouped))
	for _, attempts := range grouped {
		effective = append(effective, selectEffectiveAttempt(attempts, false))
	}
	sort.SliceStable(effective, func(i, j int) bool {
		if aggregationKeyOf(effective[i]) != aggregationKeyOf(effective[j]) {
			return aggregationKeyOf(effective[i]) < aggregationKeyOf(effective[j])
		}
		return jobMoreRecent(effective[i], effective[j])
	})
	return effective
}

func activeDispatchChunks(jobs []types.Job) map[int]struct{} {
	active := make(map[int]struct{})
	for _, job := range jobs {
		if orchestrationStrategyOf(job) == "aggregator" {
			continue
		}
		if job.BackendRunID == "" {
			continue
		}
		if isTerminal(job.State) {
			continue
		}
		active[dispatchChunkIndex(job)] = struct{}{}
	}
	return active
}

func dispatchObservability(jobs []types.Job) (dispatchingChildren, pendingChildren, activeChunks, pendingChunks int) {
	active := activeDispatchChunks(jobs)
	pending := dispatchingChildrenByChunk(jobs)
	for _, job := range jobs {
		if orchestrationStrategyOf(job) == "aggregator" {
			continue
		}
		if job.State == types.JobStateDispatching {
			dispatchingChildren++
			if job.BackendRunID == "" {
				pendingChildren++
			}
		}
	}
	return dispatchingChildren, pendingChildren, len(active), len(pending)
}

type rootActionUsageSummary struct {
	ForcedReleasedChunks int
	RetriedShardActions  int
}

func rootActionUsage(jobs []types.Job) rootActionUsageSummary {
	usage := rootActionUsageSummary{}
	forcedChunks := make(map[int]struct{})
	for _, job := range jobs {
		if taskParamBool(job, taskParamRetryAction) {
			usage.RetriedShardActions++
		}
		if taskParamBool(job, taskParamForcedRelease) {
			forcedChunks[dispatchChunkIndex(job)] = struct{}{}
		}
	}
	usage.ForcedReleasedChunks = len(forcedChunks)
	return usage
}

func availableRootBatchSlots(maxActiveBatches, currentActive int) int {
	if maxActiveBatches <= 0 {
		return int(^uint(0) >> 1)
	}
	if currentActive >= maxActiveBatches {
		return 0
	}
	return maxActiveBatches - currentActive
}

func dispatchingChildrenByChunk(jobs []types.Job) map[int][]types.Job {
	grouped := make(map[int][]types.Job)
	for _, job := range jobs {
		if orchestrationStrategyOf(job) == "aggregator" {
			continue
		}
		if job.State != types.JobStateDispatching || job.BackendRunID != "" {
			continue
		}
		grouped[dispatchChunkIndex(job)] = append(grouped[dispatchChunkIndex(job)], job)
	}
	return grouped
}

func sortedChunkIndexes(grouped map[int][]types.Job) []int {
	indexes := make([]int, 0, len(grouped))
	for index := range grouped {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	return indexes
}

func hasDispatchingChildrenForJobs(jobs []types.Job) bool {
	for _, job := range jobs {
		if orchestrationStrategyOf(job) == "aggregator" {
			continue
		}
		if job.State == types.JobStateDispatching && job.BackendRunID == "" {
			return true
		}
	}
	return false
}

func hasDispatchingChildrenForJobIDs(ctx context.Context, jobStore store.JobStore, jobIDs []string) bool {
	for _, jobID := range jobIDs {
		job, err := jobStore.GetJob(ctx, jobID)
		if err != nil {
			continue
		}
		if orchestrationStrategyOf(job) == "aggregator" {
			continue
		}
		if job.State == types.JobStateDispatching && job.BackendRunID == "" {
			return true
		}
	}
	return false
}

func selectEffectiveAttempt(attempts []types.Job, includeCancelled bool) types.Job {
	ordered := append([]types.Job(nil), attempts...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return jobMoreRecent(ordered[i], ordered[j])
	})
	for _, job := range ordered {
		if !isTerminal(job.State) {
			return job
		}
	}
	for _, job := range ordered {
		if job.State == types.JobStateSucceeded {
			return job
		}
	}
	for _, job := range ordered {
		if shouldRetryShard(job, includeCancelled) || job.State == types.JobStateCancelled {
			return job
		}
	}
	return ordered[0]
}

func shouldRetryShard(job types.Job, includeCancelled bool) bool {
	switch job.State {
	case types.JobStateFailed, types.JobStatePreempted, types.JobStateTimedOut:
		return true
	case types.JobStateCancelled:
		return includeCancelled
	default:
		return false
	}
}

func retrySkipReason(job types.Job, includeCancelled bool) string {
	switch {
	case !isTerminal(job.State):
		return "in_progress"
	case job.State == types.JobStateSucceeded:
		return "already_succeeded"
	case job.State == types.JobStateCancelled && !includeCancelled:
		return "cancelled_excluded"
	case shouldRetryShard(job, includeCancelled):
		return "not_retried"
	default:
		return "not_retryable"
	}
}

func dispatchChunkIndex(job types.Job) int {
	if job.Request.TaskParams == nil {
		return 0
	}
	switch typed := job.Request.TaskParams[taskParamDispatchChunk].(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
}

func taskParamBool(job types.Job, key string) bool {
	if job.Request.TaskParams == nil {
		return false
	}
	value, ok := job.Request.TaskParams[key]
	if !ok {
		return false
	}
	typed, ok := value.(bool)
	return ok && typed
}

func shardAttemptKey(job types.Job) string {
	return strings.Join([]string{
		job.TaskType,
		orchestrationStrategyOf(job),
		shardKeyOf(job),
		fmt.Sprintf("%d", shardIndexOf(job)),
		fmt.Sprintf("%d", shardCountOf(job)),
		aggregationKeyOf(job),
	}, "|")
}

func reducerAttemptKey(job types.Job) string {
	return strings.Join([]string{
		job.TaskType,
		orchestrationStrategyOf(job),
		aggregationKeyOf(job),
	}, "|")
}

func orchestrationStrategyOf(job types.Job) string {
	if job.Orchestration == nil {
		return ""
	}
	return strings.TrimSpace(job.Orchestration.Strategy)
}

func shardKeyOf(job types.Job) string {
	if job.Orchestration == nil {
		return ""
	}
	return strings.TrimSpace(job.Orchestration.ShardKey)
}

func shardIndexOf(job types.Job) int {
	if job.Orchestration == nil {
		return 0
	}
	return job.Orchestration.ShardIndex
}

func shardCountOf(job types.Job) int {
	if job.Orchestration == nil {
		return 0
	}
	return job.Orchestration.ShardCount
}

func aggregationKeyOf(job types.Job) string {
	if job.Orchestration == nil {
		return ""
	}
	return strings.TrimSpace(job.Orchestration.AggregationKey)
}

func dependsOnJobIDsOf(job types.Job) []string {
	if job.Orchestration == nil {
		return nil
	}
	return append([]string(nil), job.Orchestration.DependsOnJobIDs...)
}

func jobMoreRecent(a, b types.Job) bool {
	if !a.CreatedAt.Equal(b.CreatedAt) {
		return a.CreatedAt.After(b.CreatedAt)
	}
	if !a.UpdatedAt.Equal(b.UpdatedAt) {
		return a.UpdatedAt.After(b.UpdatedAt)
	}
	return a.ID > b.ID
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func appendIfNonEmpty(in []string, value string) []string {
	if strings.TrimSpace(value) == "" {
		return in
	}
	return append(in, strings.TrimSpace(value))
}
