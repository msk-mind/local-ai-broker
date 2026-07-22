package service

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/cache"
	"github.com/msk-mind/local-ai-broker/broker/pkg/jobenv"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/tasks"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

const (
	localInlineSubmitReleaseProbeWindow             = 120 * time.Millisecond
	localInlineSubmitDirectWorkerReleaseProbeWindow = 48 * time.Millisecond
	localInlineSubmitWarmQueuedReleaseProbeWindow   = 40 * time.Millisecond
	localInflightAliasReleaseProbeWindow            = 160 * time.Millisecond
	localInlineReleaseProbeInterval                 = 2 * time.Millisecond
	localInlineStateRefreshInterval                 = 32 * time.Millisecond
)

type cacheLookupResult struct {
	key                string
	cacheable          bool
	lookupMS           int64
	computeKeyMS       int64
	contentHash        string
	dirtyPaths         []string
	cleanWorktreeFiles []string
	keyTimingMS        map[string]int64
	job                *types.Job
}

func attachCacheKeySubtimings(timings map[string]any, keyTimingMS map[string]int64) map[string]any {
	if timings == nil {
		timings = map[string]any{}
	}
	for key, value := range keyTimingMS {
		if strings.TrimSpace(key) == "" {
			continue
		}
		timings[key] = value
	}
	return timings
}

func attachInspectRepoBrokerTimings(job *types.Job, brokerResultSource string, timings map[string]any) {
	if job == nil || job.TaskType != "inspect_repo" {
		return
	}
	runtimeDiagnostics := cloneMap(job.RuntimeDiagnostics)
	if runtimeDiagnostics == nil {
		runtimeDiagnostics = map[string]any{}
	}
	if brokerResultSource = strings.TrimSpace(brokerResultSource); brokerResultSource != "" {
		runtimeDiagnostics["broker_result_source"] = brokerResultSource
	}
	existingTimings := cloneMap(mapValue(runtimeDiagnostics["broker_phase_timings_ms"]))
	if existingTimings == nil {
		existingTimings = map[string]any{}
	}
	for key, value := range timings {
		switch typed := value.(type) {
		case int:
			existingTimings[key] = float64(typed)
		case int64:
			existingTimings[key] = float64(typed)
		case float64:
			existingTimings[key] = typed
		case string:
			if strings.TrimSpace(typed) != "" {
				existingTimings[key] = typed
			}
		case bool:
			existingTimings[key] = typed
		}
	}
	if len(existingTimings) > 0 {
		runtimeDiagnostics["broker_phase_timings_ms"] = existingTimings
	}
	job.RuntimeDiagnostics = runtimeDiagnostics
}

func isReusableInFlightState(state types.JobState) bool {
	switch state {
	case types.JobStateAccepted, types.JobStateQueued, types.JobStateRunning, types.JobStateDispatching:
		return true
	default:
		return false
	}
}

func (s *Service) enrichSubmitRequest(ctx context.Context, req types.SubmitJobRequest) (types.SubmitJobRequest, error) {
	req.InputRefs = normalizeInputRefs(req.InputRefs)
	if req.TaskType == "inspect_repo" {
		// Inspection request workers only discover/chunk files and perform
		// lexical fallback. Model inference is served by independent GPU service
		// leases, so the request job must never allocate or start a model GPU.
		req.ExecutionProfile.Tier = "cpu-rag-indexing"
		req.ExecutionProfile.Runtime = "deterministic"
		req.ExecutionProfile.Model = ""
		req.ExecutionProfile.Accelerator = ""
		req.ExecutionProfile.GPUCount = 0
		req.ExecutionProfile.NodeList = ""
		req.ExecutionProfile.Constraint = ""
	}
	req.ExecutionProfile = s.applyExecutionProfileDefaults(req.ExecutionProfile)
	if resolver, ok := s.backend.(backends.ExecutionProfileResolver); ok {
		resolved, err := resolver.ResolveExecutionProfile(ctx, req)
		if err != nil {
			return types.SubmitJobRequest{}, fmt.Errorf("resolve execution profile: %w", err)
		}
		req.ExecutionProfile = s.applyExecutionProfileDefaults(normalizeResolvedProfile(req.ExecutionProfile, resolved))
	}
	taskParams := cloneTaskParams(req.TaskParams)
	taskParams[jobenv.TaskParamRunRoot] = s.runRoot
	taskParams[jobenv.TaskParamRepoRoot] = s.repoRoot
	req.TaskParams = taskParams
	return req, nil
}

func normalizeInputRefs(inputRefs []types.InputRef) []types.InputRef {
	if len(inputRefs) == 0 {
		return inputRefs
	}
	out := make([]types.InputRef, len(inputRefs))
	copy(out, inputRefs)
	for i := range out {
		uri := strings.TrimSpace(out[i].URI)
		if uri == "" {
			continue
		}
		if strings.HasPrefix(uri, "artifact:") {
			continue
		}
		if normalized, ok := normalizeLocalInputURI(uri); ok {
			out[i].URI = normalized
			continue
		}
		if filepath.IsAbs(uri) {
			out[i].URI = (&url.URL{Scheme: "file", Path: filepath.Clean(uri)}).String()
		}
	}
	return out
}

func normalizeLocalInputURI(raw string) (string, bool) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme == "" {
		return "", false
	}
	switch parsed.Scheme {
	case "file":
		if parsed.Path == "" {
			return raw, false
		}
		return (&url.URL{Scheme: "file", Path: filepath.Clean(parsed.Path)}).String(), true
	case "repo", "directory", "log", "document":
		localPath := parsed.Path
		if parsed.Host != "" {
			localPath = path.Join("/", parsed.Host, parsed.Path)
		}
		if localPath == "" {
			return raw, false
		}
		return (&url.URL{Scheme: "file", Path: filepath.Clean(localPath)}).String(), true
	default:
		return "", false
	}
}

func normalizeResolvedProfile(original, resolved types.ExecutionProfile) types.ExecutionProfile {
	normalized := resolved
	if strings.TrimSpace(normalized.Tier) != strings.TrimSpace(original.Tier) {
		if strings.TrimSpace(normalized.Model) == strings.TrimSpace(original.Model) {
			normalized.Model = ""
		}
		if strings.TrimSpace(normalized.Accelerator) == strings.TrimSpace(original.Accelerator) {
			normalized.Accelerator = ""
		}
	}
	return normalized
}

func (s *Service) lookupCompletedCacheJob(ctx context.Context, req types.SubmitJobRequest) (cacheLookupResult, error) {
	keyStartedAt := time.Now()
	keyDetails, err := cache.KeyDetailsForRequest(req)
	keyDurationMS := durationMS(keyStartedAt)
	if err != nil {
		return cacheLookupResult{}, fmt.Errorf("compute cache key: %w", err)
	}

	result := cacheLookupResult{
		key:                keyDetails.Key,
		cacheable:          keyDetails.Cacheable,
		computeKeyMS:       keyDurationMS,
		contentHash:        strings.TrimSpace(keyDetails.ContentHash),
		dirtyPaths:         append([]string(nil), keyDetails.DirtyPaths...),
		cleanWorktreeFiles: append([]string(nil), keyDetails.CleanWorktreeFiles...),
		keyTimingMS:        maps.Clone(keyDetails.TimingsMS),
	}
	if !keyDetails.Cacheable {
		return result, nil
	}

	lookupStartedAt := time.Now()
	cachedJob, err := cache.FindCompletedJobByCacheKey(ctx, s.store, keyDetails.Key)
	result.lookupMS = durationMS(lookupStartedAt)
	if err != nil {
		return cacheLookupResult{}, fmt.Errorf("lookup cache: %w", err)
	}
	if cachedJob != nil && !reusableCachedJobForRequest(req, *cachedJob) {
		cachedJob = nil
	}
	result.job = cachedJob
	return result, nil
}

func maybeAttachInspectRepoFingerprintHint(req *types.SubmitJobRequest, contentHash string, dirtyPaths []string, cleanWorktreeFiles []string) {
	if req == nil || req.TaskType != "inspect_repo" {
		return
	}
	if strings.TrimSpace(contentHash) == "" {
		return
	}
	if hasInspectRepoCustomExclusions(req.TaskParams) {
		return
	}
	if len(req.InputRefs) != 1 {
		return
	}
	input := req.InputRefs[0]
	switch input.Type {
	case "repo", "directory":
	default:
		return
	}
	if strings.TrimSpace(input.ContentHash) != "" {
		return
	}
	req.InputRefs[0].ContentHash = strings.TrimSpace(contentHash)
	taskParamsUpdated := false
	taskParams := req.TaskParams
	if len(dirtyPaths) > 0 {
		taskParams = cloneTaskParams(taskParams)
		taskParams["_broker_touched_paths"] = append([]string(nil), dirtyPaths...)
		taskParamsUpdated = true
	} else if len(cleanWorktreeFiles) > 0 {
		taskParams = cloneTaskParams(taskParams)
		taskParams["_broker_clean_worktree_files"] = append([]string(nil), cleanWorktreeFiles...)
		taskParamsUpdated = true
	}
	if taskParamsUpdated {
		req.TaskParams = taskParams
	}
}

func hasInspectRepoCustomExclusions(taskParams map[string]any) bool {
	if len(taskParams) == 0 {
		return false
	}
	for _, key := range []string{"excluded_dir_names", "exclude_dirs", "excluded_paths"} {
		value, ok := taskParams[key]
		if !ok || value == nil {
			continue
		}
		switch typed := value.(type) {
		case []string:
			if len(typed) > 0 {
				return true
			}
		case []any:
			if len(typed) > 0 {
				return true
			}
		case string:
			if strings.TrimSpace(typed) != "" {
				return true
			}
		}
	}
	return false
}

func reusableCachedJobForRequest(req types.SubmitJobRequest, cachedJob types.Job) bool {
	if req.TaskType != "inspect_repo" {
		return true
	}
	return reusableInspectRepoCachedResult(req, cachedJob.Result)
}

func (s *Service) lookupReusableInflightJob(ctx context.Context, req types.SubmitJobRequest, cacheKey string) (*types.Job, error) {
	if strings.TrimSpace(cacheKey) == "" {
		return nil, nil
	}
	jobs := []types.Job(nil)
	if finder, ok := s.store.(store.JobsByCacheKeyLookup); ok {
		var err error
		jobs, err = finder.ListJobsByCacheKey(ctx, cacheKey)
		if err != nil {
			return nil, err
		}
	} else {
		var err error
		jobs, err = s.store.ListJobs(ctx)
		if err != nil {
			return nil, err
		}
	}
	slices.SortFunc(jobs, func(a, b types.Job) int {
		if a.SubmittedAt.Equal(b.SubmittedAt) {
			if a.ID < b.ID {
				return -1
			}
			if a.ID > b.ID {
				return 1
			}
			return 0
		}
		if a.SubmittedAt.Before(b.SubmittedAt) {
			return -1
		}
		return 1
	})
	for i := len(jobs) - 1; i >= 0; i-- {
		job := jobs[i]
		if job.ID == "" || job.CacheKey != cacheKey || job.TaskType != req.TaskType {
			continue
		}
		if job.CacheStatus == "hit" && strings.TrimSpace(job.CacheSourceJobID) != "" {
			continue
		}
		if job.State == types.JobStateSucceeded {
			if reusableCachedJobForRequest(req, job) {
				candidate := job
				return &candidate, nil
			}
			continue
		}
		if isReusableInFlightState(job.State) {
			candidate := job
			return &candidate, nil
		}
	}
	return nil, nil
}

func reusableInspectRepoCachedResult(req types.SubmitJobRequest, result *types.Result) bool {
	mode := strings.ToLower(strings.TrimSpace(stringValue(req.TaskParams["mode"])))
	if result == nil || result.SchemaName != "repo_inspection_v2" {
		return false
	}
	quality, _ := result.Payload["quality"].(map[string]any)
	switch mode {
	case "evidence":
		return stringValue(quality["result"]) == "evidence_only"
	case "answer":
		if stringValue(quality["result"]) != "answer_ready" {
			return false
		}
		if stringValue(quality["retrieval"]) != "gpu" || stringValue(quality["reranking"]) != "gpu" || stringValue(quality["synthesis"]) != "gpu" {
			return false
		}
		if !boolValue(quality["answer_ready"]) {
			return false
		}
		if strings.TrimSpace(stringValue(result.Payload["answer"])) == "" {
			return false
		}
		findings, ok := result.Payload["findings"].([]any)
		return ok && len(findings) > 0
	default:
		return false
	}
}

func (s *Service) createCacheAliasJob(ctx context.Context, req types.SubmitJobRequest, cacheKey string, sourceJob types.Job) (types.Job, error) {
	now := time.Now().UTC()
	principal := auth.PrincipalFromContext(ctx)
	state := sourceJob.State
	backendState := "CACHE_HIT"
	startedAt := &now
	completedAt := &now
	if state != types.JobStateSucceeded {
		backendState = "CACHE_ALIAS"
		startedAt = nil
		completedAt = nil
	}
	job := types.Job{
		ID:               newJobID(),
		TaskType:         req.TaskType,
		State:            state,
		SubmittedBy:      principal.Actor,
		Request:          req,
		CreatedAt:        now,
		UpdatedAt:        now,
		SubmittedAt:      now,
		StartedAt:        startedAt,
		CompletedAt:      completedAt,
		CacheKey:         cacheKey,
		CacheStatus:      "hit",
		CacheSourceJobID: sourceJob.ID,
		BackendKind:      "cache",
		BackendState:     backendState,
	}
	if err := s.store.CreateJob(ctx, job); err != nil {
		return types.Job{}, fmt.Errorf("store cached job: %w", err)
	}
	return job, nil
}

func cacheHitJobFromSource(job types.Job, source types.Job) types.Job {
	if source.Result != nil {
		result := cloneResult(source.Result)
		if job.TaskType == "inspect_repo" {
			result = inspectRepoCacheHitResult(result)
		}
		job.Result = result
	}
	job.RuntimeDiagnostics = mergeRuntimeDiagnostics(
		cloneMap(source.RuntimeDiagnostics),
		cloneMap(job.RuntimeDiagnostics),
	)
	job.ExecutionQuality = source.ExecutionQuality
	job.DegradedLocalExecution = source.DegradedLocalExecution
	job.RetryRecommended = source.RetryRecommended
	job.Artifacts = cloneArtifacts(source.Artifacts)
	if job.ResultError == "" {
		job.ResultError = source.ResultError
	}
	return job
}

func cloneResult(result *types.Result) *types.Result {
	if result == nil {
		return nil
	}
	cloned := *result
	if len(result.Payload) > 0 {
		cloned.Payload = cloneMap(result.Payload)
	}
	return &cloned
}

func inspectRepoCacheHitResult(result *types.Result) *types.Result {
	if result == nil || result.SchemaName != "repo_inspection_v2" {
		return result
	}
	payload := cloneMap(result.Payload)
	retrieval := cloneMap(mapValue(payload["retrieval"]))
	retrieval["query_stage_cache_hit"] = true
	retrieval["lexical_candidates"] = 0
	retrieval["semantic_candidates"] = 0
	retrieval["reranked_candidates"] = 0
	retrieval["chunk_cache_reused_files"] = 0
	retrieval["chunk_cache_rebuilt_files"] = 0
	retrieval["lexical_index_cache_hit"] = false
	retrieval["lexical_index_working_cache_hit"] = false
	retrieval["lexical_index_updated_files"] = 0
	retrieval["lexical_index_removed_files"] = 0
	retrieval["lexical_index_inserted_chunks"] = 0
	for _, key := range []string{
		"chunk_manifest_restore_ms",
		"chunk_shared_manifest_load_ms",
		"chunk_snapshot_local_load_ms",
		"chunk_snapshot_shared_load_ms",
		"lexical_index_working_manifest_load_ms",
		"lexical_index_working_check_ms",
		"lexical_index_shared_restore_ms",
		"lexical_index_sqlite_update_ms",
		"lexical_index_sqlite_rebuild_ms",
	} {
		if _, ok := retrieval[key]; ok {
			retrieval[key] = 0.0
		}
	}
	for _, key := range []string{
		"chunk_build_substage_timings_ms",
		"setup_timings_ms",
		"stage_timings_ms",
		"tail_timings_ms",
	} {
		if timings := cloneMap(mapValue(retrieval[key])); len(timings) > 0 {
			for timingKey := range timings {
				timings[timingKey] = 0.0
			}
			retrieval[key] = timings
		}
	}
	payload["retrieval"] = retrieval
	runtime := cloneMap(mapValue(payload["runtime"]))
	if attempts, ok := runtime["attempts"].([]any); ok {
		runtime["attempts"] = sanitizeRuntimeAttempts(attempts, 12)
	}
	workerPhaseTimings := cloneMap(mapValue(runtime["worker_phase_timings_ms"]))
	if len(workerPhaseTimings) > 0 {
		for _, key := range []string{
			"parse_args",
			"load_job_inputs",
			"import_validate_request",
			"validate_request",
			"discover_inputs",
			"import_prefetch_helpers",
			"prefetch_cache_context",
			"cached_probe",
			"import_run_inspection",
			"run_inspection",
			"write_artifacts",
			"finalize",
			"total",
		} {
			workerPhaseTimings[key] = 0.0
		}
		workerPhaseTimings["cache_hit"] = true
		runtime["worker_phase_timings_ms"] = workerPhaseTimings
	}
	runtime["result_source"] = "broker_cache_hit"
	payload["runtime"] = runtime
	cloned := *result
	cloned.Payload = payload
	return &cloned
}

func (s *Service) SubmitJob(ctx context.Context, req types.SubmitJobRequest) (types.SubmitJobResponse, error) {
	submitStartedAt := time.Now()
	req, cacheLookup, err := s.prepareSubmit(ctx, req)
	if err != nil {
		s.logger.Printf("submit failed task_type=%s stage=cache cache_key_ms=%d cache_lookup_ms=%d err=%v", req.TaskType, cacheLookup.computeKeyMS, cacheLookup.lookupMS, err)
		return types.SubmitJobResponse{}, err
	}
	if cacheLookup.job != nil {
		job, err := s.createCacheAliasJob(ctx, req, cacheLookup.key, *cacheLookup.job)
		if err != nil {
			return types.SubmitJobResponse{}, err
		}
		totalDurationMS := durationMS(submitStartedAt)
		attachInspectRepoBrokerTimings(&job, "cache_hit", attachCacheKeySubtimings(map[string]any{
			"cache_key_ms":    cacheLookup.computeKeyMS,
			"cache_lookup_ms": cacheLookup.lookupMS,
			"total_submit_ms": totalDurationMS,
		}, cacheLookup.keyTimingMS))
		s.logger.Printf("cache hit job=%s source_job=%s task_type=%s cache_key_ms=%d cache_lookup_ms=%d total_submit_ms=%d", job.ID, cacheLookup.job.ID, job.TaskType, cacheLookup.computeKeyMS, cacheLookup.lookupMS, totalDurationMS)
		s.audit(ctx, "job.submit", "success", &job, map[string]any{
			"cache_status":    "hit",
			"backend_kind":    job.BackendKind,
			"cache_key_ms":    cacheLookup.computeKeyMS,
			"cache_lookup_ms": cacheLookup.lookupMS,
			"total_submit_ms": totalDurationMS,
			"cacheable":       cacheLookup.cacheable,
		})
		resolved := job
		if cacheLookup.job.Result != nil {
			resolved = cacheHitJobFromSource(job, *cacheLookup.job)
		} else {
			resolved, err = s.resolveCacheHitJob(job)
			if err != nil {
				return types.SubmitJobResponse{}, err
			}
		}
		release, err := buildReleasedResult(resolved)
		if err != nil {
			return types.SubmitJobResponse{}, err
		}
		return submitJobResponseWithRelease(job, &release), nil
	}
	inflightJob, err := s.lookupReusableInflightJob(ctx, req, cacheLookup.key)
	if err != nil {
		return types.SubmitJobResponse{}, err
	}
	if inflightJob != nil {
		if releaseJob, _, ok, err := s.awaitLocalInspectRepoRelease(ctx, *inflightJob, localInflightAliasReleaseProbeWindow); err != nil {
			return types.SubmitJobResponse{}, err
		} else if ok {
			job, err := s.createCacheAliasJob(ctx, req, cacheLookup.key, releaseJob)
			if err != nil {
				return types.SubmitJobResponse{}, err
			}
			attachInspectRepoBrokerTimings(&job, "inflight_alias_inline_release", attachCacheKeySubtimings(map[string]any{
				"cache_key_ms":    cacheLookup.computeKeyMS,
				"cache_lookup_ms": cacheLookup.lookupMS,
				"total_submit_ms": durationMS(submitStartedAt),
			}, cacheLookup.keyTimingMS))
			resolved := cacheHitJobFromSource(job, releaseJob)
			release, err := buildReleasedResult(resolved)
			if err != nil {
				return types.SubmitJobResponse{}, err
			}
			return submitJobResponseWithRelease(job, &release), nil
		}
		job, err := s.createCacheAliasJob(ctx, req, cacheLookup.key, *inflightJob)
		if err != nil {
			return types.SubmitJobResponse{}, err
		}
		totalDurationMS := durationMS(submitStartedAt)
		attachInspectRepoBrokerTimings(&job, "inflight_alias", attachCacheKeySubtimings(map[string]any{
			"cache_key_ms":    cacheLookup.computeKeyMS,
			"cache_lookup_ms": cacheLookup.lookupMS,
			"total_submit_ms": totalDurationMS,
		}, cacheLookup.keyTimingMS))
		s.logger.Printf("cache alias job=%s source_job=%s task_type=%s cache_key_ms=%d cache_lookup_ms=%d total_submit_ms=%d", job.ID, inflightJob.ID, job.TaskType, cacheLookup.computeKeyMS, cacheLookup.lookupMS, totalDurationMS)
		s.audit(ctx, "job.submit", "success", &job, map[string]any{
			"cache_status":    "hit",
			"backend_kind":    job.BackendKind,
			"cache_key_ms":    cacheLookup.computeKeyMS,
			"cache_lookup_ms": cacheLookup.lookupMS,
			"total_submit_ms": totalDurationMS,
			"cacheable":       cacheLookup.cacheable,
			"cache_source":    "inflight_alias",
		})
		return submitJobResponse(job), nil
	}
	maybeAttachInspectRepoFingerprintHint(&req, cacheLookup.contentHash, cacheLookup.dirtyPaths, cacheLookup.cleanWorktreeFiles)

	job := s.newJob(ctx, req, types.JobStateAccepted, cacheLookup.key, "miss")
	attachInspectRepoBrokerTimings(&job, "submit_start", attachCacheKeySubtimings(map[string]any{
		"cache_key_ms":    cacheLookup.computeKeyMS,
		"cache_lookup_ms": cacheLookup.lookupMS,
	}, cacheLookup.keyTimingMS))
	if warmSubmitter, ok := s.backend.(backends.InlineInspectRepoWarmSubmitter); ok {
		bundleStartedAt := time.Now()
		bundle, err := s.executionBundle(ctx, &job)
		if err != nil {
			s.logger.Printf("submit failed task_type=%s stage=build_inline_bundle cache_key_ms=%d build_inline_bundle_ms=%d err=%v", req.TaskType, cacheLookup.computeKeyMS, durationMS(bundleStartedAt), err)
			return types.SubmitJobResponse{}, fmt.Errorf("build inline execution bundle: %w", err)
		}
		bundleDurationMS := durationMS(bundleStartedAt)
		backendSubmitStartedAt := time.Now()
		if submitResp, accepted, err := warmSubmitter.SubmitWarmInspectRepoRun(ctx, job, bundle); err != nil {
			s.logger.Printf("submit failed task_type=%s stage=inline_backend_submit cache_key_ms=%d build_inline_bundle_ms=%d backend_submit_ms=%d err=%v", req.TaskType, cacheLookup.computeKeyMS, bundleDurationMS, durationMS(backendSubmitStartedAt), err)
			return types.SubmitJobResponse{}, fmt.Errorf("submit warm inspect_repo run: %w", err)
		} else if accepted {
			backendSubmitDurationMS := durationMS(backendSubmitStartedAt)
			job.State = submitResp.InitialState
			job.BackendKind = submitResp.BackendKind
			job.BackendRunID = submitResp.BackendRunID

			storeCtx := ctx
			if job.TaskType == "inspect_repo" && submitResp.BackendKind == "local" {
				storeCtx = store.WithNonDurableWrite(storeCtx)
			}
			storeJobStartedAt := time.Now()
			if err := s.store.CreateJob(storeCtx, job); err != nil {
				return types.SubmitJobResponse{}, fmt.Errorf("store job: %w", err)
			}
			storeJobDurationMS := durationMS(storeJobStartedAt)
			waitWindow, hasWaitWindow := s.opportunisticInspectRepoSubmitReleaseWindow(job)
			if !preferInlineLocalRelease(ctx) {
				if hasWaitWindow {
					if release, ok, err := s.tryInlineLocalSubmitReleaseWithWindow(ctx, job, waitWindow); err != nil {
						return types.SubmitJobResponse{}, err
					} else if ok {
						timings := attachCacheKeySubtimings(map[string]any{
							"cache_key_ms":           cacheLookup.computeKeyMS,
							"build_inline_bundle_ms": bundleDurationMS,
							"backend_submit_ms":      backendSubmitDurationMS,
							"store_create_job_ms":    storeJobDurationMS,
							"total_submit_ms":        durationMS(submitStartedAt),
						}, cacheLookup.keyTimingMS)
						attachInspectRepoBrokerTimings(&job, "inline_release", timings)
						attachInspectRepoBrokerRuntimeToRelease(&release, "inline_release", timings)
						return submitJobResponseWithRelease(job, &release), nil
					}
				} else if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
					return types.SubmitJobResponse{}, err
				} else if ok {
					attachInspectRepoBrokerTimings(&releaseJob, "direct_run_files_inline_release", attachCacheKeySubtimings(map[string]any{
						"cache_key_ms":           cacheLookup.computeKeyMS,
						"build_inline_bundle_ms": bundleDurationMS,
						"backend_submit_ms":      backendSubmitDurationMS,
						"store_create_job_ms":    storeJobDurationMS,
						"total_submit_ms":        durationMS(submitStartedAt),
					}, cacheLookup.keyTimingMS))
					release, err := buildReleasedResult(releaseJob)
					if err != nil {
						return types.SubmitJobResponse{}, err
					}
					return submitJobResponseWithRelease(job, &release), nil
				}
			}
			if preferInlineLocalRelease(ctx) {
				if release, ok, err := s.tryInlineLocalSubmitRelease(ctx, job); err != nil {
					return types.SubmitJobResponse{}, err
				} else if ok {
					timings := attachCacheKeySubtimings(map[string]any{
						"cache_key_ms":           cacheLookup.computeKeyMS,
						"build_inline_bundle_ms": bundleDurationMS,
						"backend_submit_ms":      backendSubmitDurationMS,
						"store_create_job_ms":    storeJobDurationMS,
						"total_submit_ms":        durationMS(submitStartedAt),
					}, cacheLookup.keyTimingMS)
					attachInspectRepoBrokerTimings(&job, "preferred_inline_release", timings)
					attachInspectRepoBrokerRuntimeToRelease(&release, "preferred_inline_release", timings)
					return submitJobResponseWithRelease(job, &release), nil
				}
			}
			totalDurationMS := durationMS(submitStartedAt)
			s.logger.Printf("submitted job=%s task_type=%s backend_run_id=%s cache_key_ms=%d build_inline_bundle_ms=%d backend_submit_ms=%d total_submit_ms=%d", job.ID, job.TaskType, job.BackendRunID, cacheLookup.computeKeyMS, bundleDurationMS, backendSubmitDurationMS, totalDurationMS)
			s.audit(ctx, "job.submit", "success", &job, map[string]any{
				"cache_status":           job.CacheStatus,
				"backend_kind":           job.BackendKind,
				"cache_key_ms":           cacheLookup.computeKeyMS,
				"build_inline_bundle_ms": bundleDurationMS,
				"backend_submit_ms":      backendSubmitDurationMS,
				"store_create_job_ms":    storeJobDurationMS,
				"total_submit_ms":        totalDurationMS,
				"cacheable":              cacheLookup.cacheable,
				"submission_path":        "inline_warm_inspect_repo",
			})
			return submitJobResponse(job), nil
		}
	}
	stageBundleStartedAt := time.Now()
	if err := s.stageExecutionBundle(ctx, &job); err != nil {
		s.logger.Printf("submit failed task_type=%s stage=stage_bundle cache_key_ms=%d stage_bundle_ms=%d err=%v", req.TaskType, cacheLookup.computeKeyMS, durationMS(stageBundleStartedAt), err)
		return types.SubmitJobResponse{}, fmt.Errorf("stage execution bundle: %w", err)
	}
	stageBundleDurationMS := durationMS(stageBundleStartedAt)

	backendSubmitStartedAt := time.Now()
	submitResp, err := s.backend.SubmitRun(ctx, job)
	backendSubmitDurationMS := durationMS(backendSubmitStartedAt)
	if err != nil {
		s.logger.Printf("submit failed task_type=%s stage=backend_submit cache_key_ms=%d stage_bundle_ms=%d backend_submit_ms=%d err=%v", req.TaskType, cacheLookup.computeKeyMS, stageBundleDurationMS, backendSubmitDurationMS, err)
		return types.SubmitJobResponse{}, fmt.Errorf("submit backend run: %w", err)
	}
	job.State = submitResp.InitialState
	job.BackendKind = submitResp.BackendKind
	job.BackendRunID = submitResp.BackendRunID

	storeCtx := ctx
	if job.TaskType == "inspect_repo" && submitResp.BackendKind == "local" {
		storeCtx = store.WithNonDurableWrite(storeCtx)
	}
	storeJobStartedAt := time.Now()
	if err := s.store.CreateJob(storeCtx, job); err != nil {
		return types.SubmitJobResponse{}, fmt.Errorf("store job: %w", err)
	}
	storeJobDurationMS := durationMS(storeJobStartedAt)
	waitWindow, hasWaitWindow := s.opportunisticInspectRepoSubmitReleaseWindow(job)
	if !preferInlineLocalRelease(ctx) {
		if hasWaitWindow {
			if release, ok, err := s.tryInlineLocalSubmitReleaseWithWindow(ctx, job, waitWindow); err != nil {
				return types.SubmitJobResponse{}, err
			} else if ok {
				timings := attachCacheKeySubtimings(map[string]any{
					"cache_key_ms":        cacheLookup.computeKeyMS,
					"stage_bundle_ms":     stageBundleDurationMS,
					"backend_submit_ms":   backendSubmitDurationMS,
					"store_create_job_ms": storeJobDurationMS,
					"total_submit_ms":     durationMS(submitStartedAt),
				}, cacheLookup.keyTimingMS)
				attachInspectRepoBrokerTimings(&job, "inline_release", timings)
				attachInspectRepoBrokerRuntimeToRelease(&release, "inline_release", timings)
				return submitJobResponseWithRelease(job, &release), nil
			}
		} else if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
			return types.SubmitJobResponse{}, err
		} else if ok {
			attachInspectRepoBrokerTimings(&releaseJob, "direct_run_files_inline_release", attachCacheKeySubtimings(map[string]any{
				"cache_key_ms":        cacheLookup.computeKeyMS,
				"stage_bundle_ms":     stageBundleDurationMS,
				"backend_submit_ms":   backendSubmitDurationMS,
				"store_create_job_ms": storeJobDurationMS,
				"total_submit_ms":     durationMS(submitStartedAt),
			}, cacheLookup.keyTimingMS))
			release, err := buildReleasedResult(releaseJob)
			if err != nil {
				return types.SubmitJobResponse{}, err
			}
			return submitJobResponseWithRelease(job, &release), nil
		} else {
			if releaseJob, ok, err := s.awaitDirectInspectRepoReleasedResult(job, localInspectRepoResultProbeWindow); err != nil {
				return types.SubmitJobResponse{}, err
			} else if ok {
				attachInspectRepoBrokerTimings(&releaseJob, "direct_run_files_probe_release", attachCacheKeySubtimings(map[string]any{
					"cache_key_ms":        cacheLookup.computeKeyMS,
					"stage_bundle_ms":     stageBundleDurationMS,
					"backend_submit_ms":   backendSubmitDurationMS,
					"store_create_job_ms": storeJobDurationMS,
					"total_submit_ms":     durationMS(submitStartedAt),
				}, cacheLookup.keyTimingMS))
				release, err := buildReleasedResult(releaseJob)
				if err != nil {
					return types.SubmitJobResponse{}, err
				}
				return submitJobResponseWithRelease(job, &release), nil
			}
		}
	}
	if preferInlineLocalRelease(ctx) {
		if release, ok, err := s.tryInlineLocalSubmitRelease(ctx, job); err != nil {
			return types.SubmitJobResponse{}, err
		} else if ok {
			timings := attachCacheKeySubtimings(map[string]any{
				"cache_key_ms":        cacheLookup.computeKeyMS,
				"stage_bundle_ms":     stageBundleDurationMS,
				"backend_submit_ms":   backendSubmitDurationMS,
				"store_create_job_ms": storeJobDurationMS,
				"total_submit_ms":     durationMS(submitStartedAt),
			}, cacheLookup.keyTimingMS)
			attachInspectRepoBrokerTimings(&job, "preferred_inline_release", timings)
			attachInspectRepoBrokerRuntimeToRelease(&release, "preferred_inline_release", timings)
			return submitJobResponseWithRelease(job, &release), nil
		}
	}

	totalDurationMS := durationMS(submitStartedAt)
	s.logger.Printf("submitted job=%s task_type=%s backend_run_id=%s cache_key_ms=%d stage_bundle_ms=%d backend_submit_ms=%d total_submit_ms=%d", job.ID, job.TaskType, job.BackendRunID, cacheLookup.computeKeyMS, stageBundleDurationMS, backendSubmitDurationMS, totalDurationMS)
	s.audit(ctx, "job.submit", "success", &job, map[string]any{
		"cache_status":        job.CacheStatus,
		"backend_kind":        job.BackendKind,
		"cache_key_ms":        cacheLookup.computeKeyMS,
		"stage_bundle_ms":     stageBundleDurationMS,
		"backend_submit_ms":   backendSubmitDurationMS,
		"store_create_job_ms": storeJobDurationMS,
		"total_submit_ms":     totalDurationMS,
		"cacheable":           cacheLookup.cacheable,
	})
	return submitJobResponse(job), nil
}

// prepareSubmit owns request normalization and all cache-key inputs. Keeping
// this boundary separate makes the submission state machine below about job
// aliases/backend work only, without changing any cache or validation rules.
func (s *Service) prepareSubmit(ctx context.Context, req types.SubmitJobRequest) (types.SubmitJobRequest, cacheLookupResult, error) {
	if req.TaskType == "" {
		return types.SubmitJobRequest{}, cacheLookupResult{}, errors.New("task_type is required")
	}
	if req.OutputSchema.Name == "" {
		return types.SubmitJobRequest{}, cacheLookupResult{}, errors.New("output_schema.name is required")
	}
	req = tasks.NormalizeSubmitRequest(req)
	if err := tasks.ValidateSubmitRequest(req); err != nil {
		return types.SubmitJobRequest{}, cacheLookupResult{}, err
	}
	var err error
	if req, err = s.enrichSubmitRequest(ctx, req); err != nil {
		return types.SubmitJobRequest{}, cacheLookupResult{}, err
	}
	if req, err = s.resolveRequestInputRefs(ctx, req); err != nil {
		return types.SubmitJobRequest{}, cacheLookupResult{}, fmt.Errorf("resolve input refs: %w", err)
	}
	lookup, err := s.lookupCompletedCacheJob(ctx, req)
	return req, lookup, err
}

func (s *Service) shouldOpportunisticallyAwaitDirectWorkerInspectRepoRelease(job types.Job) bool {
	if job.BackendKind != "local" || job.TaskType != "inspect_repo" || strings.TrimSpace(job.BackendRunID) == "" {
		return false
	}
	if strings.TrimSpace(s.runRoot) == "" {
		return true
	}
	markerPath := filepath.Join(s.runRoot, job.ID, "warm-request.marker")
	_, err := os.Stat(markerPath)
	return errors.Is(err, os.ErrNotExist)
}

func (s *Service) opportunisticInspectRepoSubmitReleaseWindow(job types.Job) (time.Duration, bool) {
	if job.BackendKind != "local" || job.TaskType != "inspect_repo" || strings.TrimSpace(job.BackendRunID) == "" {
		return 0, false
	}
	if s.shouldOpportunisticallyAwaitDirectWorkerInspectRepoRelease(job) {
		return localInlineSubmitDirectWorkerReleaseProbeWindow, true
	}
	if strings.TrimSpace(s.runRoot) == "" {
		return 0, false
	}
	markerPath := filepath.Join(s.runRoot, job.ID, "warm-request.marker")
	if _, err := os.Stat(markerPath); err == nil {
		return localInlineSubmitWarmQueuedReleaseProbeWindow, true
	}
	return 0, false
}

func (s *Service) tryInlineLocalSubmitRelease(ctx context.Context, job types.Job) (types.JobResultRelease, bool, error) {
	return s.tryInlineLocalSubmitReleaseWithWindow(ctx, job, localInlineSubmitReleaseProbeWindow)
}

func (s *Service) tryInlineLocalSubmitReleaseWithWindow(ctx context.Context, job types.Job, waitWindow time.Duration) (types.JobResultRelease, bool, error) {
	release, ok, err := s.awaitLocalInspectRepoSubmitRelease(job, waitWindow)
	return release, ok, err
}

func attachInlineSubmitReleaseTimings(release *types.JobResultRelease, timings map[string]any) {
	if release == nil || release.Result == nil || release.Result.SchemaName != "repo_inspection_v2" {
		return
	}
	runtimeDiagnostics := cloneMap(release.RuntimeDiagnostics)
	if runtimeDiagnostics == nil {
		runtimeDiagnostics = map[string]any{}
	}
	existingTimings := cloneMap(mapValue(runtimeDiagnostics["broker_phase_timings_ms"]))
	if existingTimings == nil {
		existingTimings = map[string]any{}
	}
	for key, value := range timings {
		switch typed := value.(type) {
		case int:
			existingTimings[key] = float64(typed)
		case int64:
			existingTimings[key] = float64(typed)
		case float64:
			existingTimings[key] = typed
		}
	}
	runtimeDiagnostics["broker_phase_timings_ms"] = existingTimings
	release.RuntimeDiagnostics = runtimeDiagnostics
	release.Result = inspectRepoReleasedResultWithBrokerRuntime(types.Job{
		TaskType:           "inspect_repo",
		RuntimeDiagnostics: runtimeDiagnostics,
	}, release.Result)
}

func elapsedMSFloat(start time.Time) float64 {
	return float64(time.Since(start).Microseconds()) / 1000.0
}

func (s *Service) awaitLocalInspectRepoSubmitRelease(job types.Job, waitWindow time.Duration) (types.JobResultRelease, bool, error) {
	if job.BackendKind != "local" || job.TaskType != "inspect_repo" || strings.TrimSpace(job.BackendRunID) == "" || waitWindow <= 0 {
		return types.JobResultRelease{}, false, nil
	}
	startedAt := time.Now()
	if waiter, ok := s.backend.(backends.LocalInspectRepoResultWaiter); ok {
		initialProbeStartedAt := time.Now()
		if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
			return types.JobResultRelease{}, false, err
		} else if ok {
			release, err := buildReleasedResult(releaseJob)
			if err != nil {
				return types.JobResultRelease{}, false, err
			}
			attachInlineSubmitReleaseTimings(&release, map[string]any{
				"inline_release_initial_probe_ms": elapsedMSFloat(initialProbeStartedAt),
				"inline_release_total_ms":         elapsedMSFloat(startedAt),
			})
			return release, true, nil
		}
		waitStartedAt := time.Now()
		waited := waiter.AwaitLocalInspectRepoResult(context.Background(), job.BackendRunID, waitWindow)
		waitMS := elapsedMSFloat(waitStartedAt)
		if waited {
			postWaitStartedAt := time.Now()
			if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
				return types.JobResultRelease{}, false, err
			} else if ok {
				release, err := buildReleasedResult(releaseJob)
				if err != nil {
					return types.JobResultRelease{}, false, err
				}
				attachInlineSubmitReleaseTimings(&release, map[string]any{
					"inline_release_initial_probe_ms":           elapsedMSFloat(initialProbeStartedAt),
					"inline_release_waiter_wait_ms":             waitMS,
					"inline_release_post_wait_release_build_ms": elapsedMSFloat(postWaitStartedAt),
					"inline_release_total_ms":                   elapsedMSFloat(startedAt),
				})
				return release, true, nil
			}
		}
		return types.JobResultRelease{}, false, nil
	}
	deadline := time.Now().Add(waitWindow)
	current := job
	for {
		if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(current); err != nil {
			return types.JobResultRelease{}, false, err
		} else if ok {
			release, err := buildReleasedResult(releaseJob)
			if err != nil {
				return types.JobResultRelease{}, false, err
			}
			return release, true, nil
		}
		if time.Now().After(deadline) {
			return types.JobResultRelease{}, false, nil
		}
		time.Sleep(localInlineReleaseProbeInterval)
	}
}

func (s *Service) tryInlineLocalInspectRepoRelease(ctx context.Context, job types.Job) (types.Job, types.JobResultRelease, bool, error) {
	if job.BackendKind != "local" || job.TaskType != "inspect_repo" || strings.TrimSpace(job.BackendRunID) == "" {
		return job, types.JobResultRelease{}, false, nil
	}
	return s.awaitLocalInspectRepoRelease(ctx, job, localInflightAliasReleaseProbeWindow)
}

func (s *Service) awaitLocalInspectRepoRelease(ctx context.Context, job types.Job, waitWindow time.Duration) (types.Job, types.JobResultRelease, bool, error) {
	if job.BackendKind != "local" || job.TaskType != "inspect_repo" || strings.TrimSpace(job.BackendRunID) == "" || waitWindow <= 0 {
		return job, types.JobResultRelease{}, false, nil
	}
	if waiter, ok := s.backend.(backends.LocalInspectRepoResultWaiter); ok {
		if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
			return job, types.JobResultRelease{}, false, err
		} else if ok {
			release, err := buildReleasedResult(releaseJob)
			if err != nil {
				return job, types.JobResultRelease{}, false, err
			}
			return releaseJob, release, true, nil
		}
		if waiter.AwaitLocalInspectRepoResult(ctx, job.BackendRunID, waitWindow) {
			if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
				return job, types.JobResultRelease{}, false, err
			} else if ok {
				release, err := buildReleasedResult(releaseJob)
				if err != nil {
					return job, types.JobResultRelease{}, false, err
				}
				return releaseJob, release, true, nil
			}
		}
		return job, types.JobResultRelease{}, false, nil
	}
	deadline := time.Now().Add(waitWindow)
	current := job
	lastRefresh := time.Now()
	for {
		if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(current); err != nil {
			return current, types.JobResultRelease{}, false, err
		} else if ok {
			release, err := buildReleasedResult(releaseJob)
			if err != nil {
				return current, types.JobResultRelease{}, false, err
			}
			return releaseJob, release, true, nil
		}
		now := time.Now()
		if lastRefresh.IsZero() || now.Sub(lastRefresh) >= localInlineStateRefreshInterval || now.After(deadline) {
			refreshed, err := s.refreshJobState(ctx, current)
			if err == nil {
				current = refreshed
			}
			lastRefresh = now
		}
		if current.Result != nil && isTerminal(current.State) {
			release, err := buildReleasedResult(current)
			if err != nil {
				return current, types.JobResultRelease{}, false, err
			}
			return current, release, true, nil
		}
		if time.Now().After(deadline) {
			return current, types.JobResultRelease{}, false, nil
		}
		time.Sleep(localInlineReleaseProbeInterval)
	}
}

func (s *Service) SubmitParallelJobs(ctx context.Context, req types.SubmitParallelJobsRequest) (types.SubmitParallelJobsResponse, error) {
	if req.TaskType == "" {
		return types.SubmitParallelJobsResponse{}, errors.New("task_type is required")
	}
	if req.OutputSchema.Name == "" {
		return types.SubmitParallelJobsResponse{}, errors.New("output_schema.name is required")
	}
	if len(req.Children) == 0 {
		return types.SubmitParallelJobsResponse{}, errors.New("children is required")
	}

	rootJobID := strings.TrimSpace(req.RootJobID)
	if rootJobID == "" {
		rootJobID = newRootJobID()
	}
	strategy := strings.TrimSpace(req.Strategy)
	if strategy == "" {
		strategy = "fanout_child"
	}

	children := make([]types.ParallelChildSubmission, len(req.Children))
	childJobIDs := make([]string, len(req.Children))
	childBackendRunIDs := make([]string, 0, len(req.Children))
	pending := make([]pendingChild, 0, len(req.Children))

	for index, child := range req.Children {
		submitReq := s.parallelChildSubmitRequest(req, child, rootJobID, strategy)
		submitReq = tasks.NormalizeSubmitRequest(submitReq)
		if err := tasks.ValidateSubmitRequest(submitReq); err != nil {
			return types.SubmitParallelJobsResponse{}, fmt.Errorf("invalid child shard %d: %w", child.ShardIndex, err)
		}
		submitReq, err := s.enrichSubmitRequest(ctx, submitReq)
		if err != nil {
			return types.SubmitParallelJobsResponse{}, fmt.Errorf("resolve child shard %d execution profile: %w", child.ShardIndex, err)
		}
		submitReq, err = s.resolveRequestInputRefs(ctx, submitReq)
		if err != nil {
			return types.SubmitParallelJobsResponse{}, fmt.Errorf("resolve child shard %d input refs: %w", child.ShardIndex, err)
		}

		cacheLookup, err := s.lookupCompletedCacheJob(ctx, submitReq)
		if err != nil {
			return types.SubmitParallelJobsResponse{}, fmt.Errorf("lookup cache for child shard %d: %w", child.ShardIndex, err)
		}
		if cacheLookup.job != nil {
			resp, err := s.SubmitJob(ctx, submitReq)
			if err != nil {
				return types.SubmitParallelJobsResponse{}, fmt.Errorf("submit cached child shard %d: %w", child.ShardIndex, err)
			}
			children[index] = parallelChildSubmissionFromResponse(resp, child)
			childJobIDs[index] = resp.JobID
			continue
		}
		maybeAttachInspectRepoFingerprintHint(&submitReq, cacheLookup.contentHash, cacheLookup.dirtyPaths, cacheLookup.cleanWorktreeFiles)

		job := s.newJob(ctx, submitReq, types.JobStateDispatching, cacheLookup.key, "miss")
		if err := s.stageExecutionBundle(ctx, &job); err != nil {
			return types.SubmitParallelJobsResponse{}, fmt.Errorf("stage child shard %d execution bundle: %w", child.ShardIndex, err)
		}
		pending = append(pending, pendingChild{
			index: index,
			child: child,
			job:   job,
			chunk: len(pending) / s.options.ParallelMaxBatchSize,
		})
	}

	if len(pending) > 0 {
		if err := s.storePendingChildren(ctx, pending, children, childJobIDs); err != nil {
			return types.SubmitParallelJobsResponse{}, err
		}
		if _, err := s.releaseDispatchingRootChildren(ctx, rootJobID, 0, false); err != nil {
			return types.SubmitParallelJobsResponse{}, fmt.Errorf("release initial child batches: %w", err)
		}
		childBackendRunIDs = s.refreshParallelChildStates(ctx, children)
	}

	reducerResp, err := s.submitParallelReducer(ctx, req, rootJobID, childJobIDs, childBackendRunIDs)
	if err != nil {
		return types.SubmitParallelJobsResponse{}, err
	}

	return types.SubmitParallelJobsResponse{
		RootJobID:   rootJobID,
		ParentJobID: req.ParentJobID,
		Strategy:    strategy,
		ChildCount:  len(children),
		Children:    children,
		ReducerJob:  reducerResp,
	}, nil
}

type pendingChild struct {
	index int
	child types.ParallelChildRequest
	job   types.Job
	chunk int
}

func (s *Service) parallelChildSubmitRequest(req types.SubmitParallelJobsRequest, child types.ParallelChildRequest, rootJobID, strategy string) types.SubmitJobRequest {
	taskParams := cloneTaskParams(req.TaskParams)
	for k, v := range child.TaskParams {
		taskParams[k] = v
	}
	return types.SubmitJobRequest{
		TaskType:         req.TaskType,
		InputRefs:        child.InputRefs,
		TaskParams:       taskParams,
		Constraints:      req.Constraints,
		ExecutionProfile: s.applyExecutionProfileDefaults(req.ExecutionProfile),
		OutputSchema:     req.OutputSchema,
		Orchestration: types.OrchestrationRequest{
			ParentJobID:     req.ParentJobID,
			RootJobID:       rootJobID,
			Strategy:        strategy,
			ShardKey:        child.ShardKey,
			ShardIndex:      child.ShardIndex,
			ShardCount:      child.ShardCount,
			AggregationKey:  firstNonEmpty(child.AggregationKey, ""),
			DependsOnJobIDs: append([]string(nil), child.DependsOnJobIDs...),
		},
	}
}

func (s *Service) newJob(ctx context.Context, req types.SubmitJobRequest, state types.JobState, cacheKey, cacheStatus string) types.Job {
	now := time.Now().UTC()
	jobID := newJobID()
	orchestration := normalizeOrchestration(jobID, req.Orchestration)
	return types.Job{
		ID:            jobID,
		TaskType:      req.TaskType,
		State:         state,
		SubmittedBy:   auth.PrincipalFromContext(ctx).Actor,
		Request:       req,
		CreatedAt:     now,
		UpdatedAt:     now,
		SubmittedAt:   now,
		CacheKey:      cacheKey,
		CacheStatus:   cacheStatus,
		ParentJobID:   orchestration.ParentJobID,
		RootJobID:     orchestration.RootJobID,
		Orchestration: orchestration,
	}
}

func submitJobResponse(job types.Job) types.SubmitJobResponse {
	return types.SubmitJobResponse{
		JobID:     job.ID,
		State:     job.State,
		Cache:     types.CacheStatus{Status: job.CacheStatus},
		StatusURL: "/v1/jobs/" + job.ID,
	}
}

func submitJobResponseWithRelease(job types.Job, release *types.JobResultRelease) types.SubmitJobResponse {
	if release != nil && release.Result != nil && job.TaskType == "inspect_repo" && release.Result.SchemaName == "repo_inspection_v2" {
		payload := cloneMap(release.Result.Payload)
		runtime := cloneMap(mapValue(payload["runtime"]))
		lifecycle := cloneMap(mapValue(runtime["broker_lifecycle"]))
		lifecycle["broker_submit_response_ready_unix_ns"] = time.Now().UnixNano()
		runtime["broker_lifecycle"] = lifecycle
		payload["runtime"] = runtime
		result := *release.Result
		result.Payload = payload
		release.Result = &result
	}
	resp := submitJobResponse(job)
	resp.ReleasedResult = release
	return resp
}

func parallelChildSubmissionFromResponse(resp types.SubmitJobResponse, child types.ParallelChildRequest) types.ParallelChildSubmission {
	return types.ParallelChildSubmission{
		JobID:          resp.JobID,
		State:          resp.State,
		Cache:          resp.Cache,
		StatusURL:      resp.StatusURL,
		ShardKey:       child.ShardKey,
		ShardIndex:     child.ShardIndex,
		ShardCount:     child.ShardCount,
		AggregationKey: child.AggregationKey,
	}
}

func parallelChildSubmissionFromJob(job types.Job, child types.ParallelChildRequest) types.ParallelChildSubmission {
	return types.ParallelChildSubmission{
		JobID:          job.ID,
		State:          job.State,
		Cache:          types.CacheStatus{Status: job.CacheStatus},
		StatusURL:      "/v1/jobs/" + job.ID,
		ShardKey:       child.ShardKey,
		ShardIndex:     child.ShardIndex,
		ShardCount:     child.ShardCount,
		AggregationKey: child.AggregationKey,
	}
}

func (s *Service) storePendingChildren(ctx context.Context, pending []pendingChild, children []types.ParallelChildSubmission, childJobIDs []string) error {
	for i := range pending {
		pending[i].job.Request.TaskParams[taskParamDispatchChunk] = pending[i].chunk
		if err := s.store.CreateJob(ctx, pending[i].job); err != nil {
			return fmt.Errorf("store child shard %d: %w", pending[i].child.ShardIndex, err)
		}
		s.logger.Printf("created child job=%s task_type=%s root_job_id=%s state=%s", pending[i].job.ID, pending[i].job.TaskType, pending[i].job.RootJobID, pending[i].job.State)
		s.audit(ctx, "job.submit", "success", &pending[i].job, map[string]any{
			"cache_status": pending[i].job.CacheStatus,
			"backend_kind": pending[i].job.BackendKind,
		})
		children[pending[i].index] = parallelChildSubmissionFromJob(pending[i].job, pending[i].child)
		childJobIDs[pending[i].index] = pending[i].job.ID
	}
	return nil
}

func (s *Service) refreshParallelChildStates(ctx context.Context, children []types.ParallelChildSubmission) []string {
	childBackendRunIDs := make([]string, 0, len(children))
	for i, childResp := range children {
		job, err := s.store.GetJob(ctx, childResp.JobID)
		if err == nil {
			children[i].State = job.State
			childBackendRunIDs = appendIfNonEmpty(childBackendRunIDs, job.BackendRunID)
		}
	}
	return childBackendRunIDs
}

func (s *Service) submitParallelReducer(ctx context.Context, req types.SubmitParallelJobsRequest, rootJobID string, childJobIDs, childBackendRunIDs []string) (*types.SubmitJobResponse, error) {
	if req.Reducer == nil {
		return nil, nil
	}
	reducerSubmitReq := s.buildParallelReducerRequest(req, rootJobID, childJobIDs, childBackendRunIDs)
	reducerChildJobIDs := compactNonEmptyStrings(childJobIDs)
	if hasDispatchingChildrenForJobIDs(ctx, s.store, reducerChildJobIDs) {
		resp, err := s.createDeferredReducer(ctx, reducerSubmitReq)
		if err != nil {
			return nil, fmt.Errorf("create deferred reducer: %w", err)
		}
		return resp, nil
	}
	resp, err := s.SubmitJob(ctx, reducerSubmitReq)
	if err != nil {
		return nil, fmt.Errorf("submit reducer: %w", err)
	}
	return &resp, nil
}

func normalizeOrchestration(jobID string, req types.OrchestrationRequest) *types.OrchestrationInfo {
	rootJobID := strings.TrimSpace(req.RootJobID)
	parentJobID := strings.TrimSpace(req.ParentJobID)
	if rootJobID == "" {
		if parentJobID != "" {
			rootJobID = parentJobID
		} else {
			rootJobID = jobID
		}
	}
	strategy := strings.TrimSpace(req.Strategy)
	if strategy == "" {
		if parentJobID != "" {
			strategy = "fanout_child"
		} else {
			strategy = "standalone"
		}
	}
	return &types.OrchestrationInfo{
		ParentJobID:     parentJobID,
		RootJobID:       rootJobID,
		Strategy:        strategy,
		ShardKey:        strings.TrimSpace(req.ShardKey),
		ShardIndex:      req.ShardIndex,
		ShardCount:      req.ShardCount,
		AggregationKey:  strings.TrimSpace(req.AggregationKey),
		DependsOnJobIDs: append([]string(nil), req.DependsOnJobIDs...),
	}
}
