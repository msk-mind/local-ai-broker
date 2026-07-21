package service

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/authz"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/policy"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

const (
	localInspectRepoResultProbeWindow   = 24 * time.Millisecond
	localInspectRepoResultProbeInterval = 2 * time.Millisecond
)

type Service struct {
	store       store.JobStore
	backend     backends.Backend
	logger      *log.Logger
	auditLogger audit.Logger
	runRoot     string
	repoRoot    string
	models      modelProfiles
	runtimes    runtimeProfiles
	gpuServices gpuServiceSettings
	options     Options
}

type modelProfiles struct {
	cpu  string
	p40  string
	a100 string
}

type runtimeProfiles struct {
	llamaCPP runtimeConnection
	vllm     runtimeConnection
	sglang   runtimeConnection
}

type runtimeConnection struct {
	BaseURL        string
	TimeoutSeconds int
}

type gpuServiceSettings struct {
	Enabled               bool
	RegistryPath          string
	ControlRequestPath    string
	ControlToken          string
	HealthIntervalSeconds int
	StartupTimeoutSeconds int
}

type Options struct {
	ParallelMaxBatchSize           int
	ParallelMaxActiveBatches       int
	RootActionMaxAdditionalBatches int
	RootActionMaxRetriedShards     int
}

type Params struct {
	JobStore    store.JobStore
	Backend     backends.Backend
	Logger      *log.Logger
	AuditLogger audit.Logger
	RunRoot     string
	RepoRoot    string
	Options     Options
	Config      *config.Config
}

func New(jobStore store.JobStore, backend backends.Backend, logger *log.Logger, runRoot, repoRoot string) *Service {
	return NewWithParams(Params{
		JobStore: jobStore,
		Backend:  backend,
		Logger:   logger,
		RunRoot:  runRoot,
		RepoRoot: repoRoot,
	})
}

func NewWithAudit(jobStore store.JobStore, backend backends.Backend, logger *log.Logger, auditLogger audit.Logger, runRoot, repoRoot string) *Service {
	return NewWithParams(Params{
		JobStore:    jobStore,
		Backend:     backend,
		Logger:      logger,
		AuditLogger: auditLogger,
		RunRoot:     runRoot,
		RepoRoot:    repoRoot,
	})
}

func NewWithAuditAndOptions(jobStore store.JobStore, backend backends.Backend, logger *log.Logger, auditLogger audit.Logger, runRoot, repoRoot string, opts Options) *Service {
	return NewWithParams(Params{
		JobStore:    jobStore,
		Backend:     backend,
		Logger:      logger,
		AuditLogger: auditLogger,
		RunRoot:     runRoot,
		RepoRoot:    repoRoot,
		Options:     opts,
	})
}

func NewWithAuditAndOptionsAndConfig(jobStore store.JobStore, backend backends.Backend, logger *log.Logger, auditLogger audit.Logger, runRoot, repoRoot string, opts Options, cfg *config.Config) *Service {
	return NewWithParams(Params{
		JobStore:    jobStore,
		Backend:     backend,
		Logger:      logger,
		AuditLogger: auditLogger,
		RunRoot:     runRoot,
		RepoRoot:    repoRoot,
		Options:     opts,
		Config:      cfg,
	})
}

func NewWithParams(params Params) *Service {
	if params.AuditLogger == nil {
		params.AuditLogger = audit.NewNopLogger()
	}
	models := defaultModelProfiles()
	runtimes := defaultRuntimeProfiles()
	gpuServices := gpuServiceSettings{}
	if params.Config != nil {
		models = modelProfiles{
			cpu:  strings.TrimSpace(params.Config.ModelProfileCPU),
			p40:  strings.TrimSpace(params.Config.ModelProfileP40),
			a100: strings.TrimSpace(params.Config.ModelProfileA100),
		}
		runtimes = runtimeProfiles{
			llamaCPP: runtimeConnection{
				BaseURL:        strings.TrimSpace(params.Config.RuntimeLlamaCPPBaseURL),
				TimeoutSeconds: params.Config.RuntimeLlamaCPPTimeoutSeconds,
			},
			vllm: runtimeConnection{
				BaseURL:        strings.TrimSpace(params.Config.RuntimeVLLMBaseURL),
				TimeoutSeconds: params.Config.RuntimeVLLMTimeoutSeconds,
			},
			sglang: runtimeConnection{
				BaseURL:        strings.TrimSpace(params.Config.RuntimeSGLangBaseURL),
				TimeoutSeconds: params.Config.RuntimeSGLangTimeoutSeconds,
			},
		}
		gpuServices = gpuServiceSettings{
			Enabled:               params.Config.GPUServiceEnabled,
			RegistryPath:          resolveGPUServiceRegistryPath(params.RepoRoot, params.Config.GPUServiceRegistryPath),
			ControlRequestPath:    resolveGPUServiceRegistryPath(params.RepoRoot, params.Config.GPUServiceControlRequestDir),
			ControlToken:          strings.TrimSpace(params.Config.GPUServiceControlToken),
			HealthIntervalSeconds: params.Config.GPUServiceHealthIntervalSeconds,
			StartupTimeoutSeconds: params.Config.GPUServiceStartupTimeoutSeconds,
		}
	}
	return &Service{
		store:       params.JobStore,
		backend:     params.Backend,
		logger:      params.Logger,
		auditLogger: params.AuditLogger,
		runRoot:     params.RunRoot,
		repoRoot:    params.RepoRoot,
		models:      models,
		runtimes:    runtimes,
		gpuServices: gpuServices,
		options:     normalizeOptions(params.Options),
	}
}

func resolveGPUServiceRegistryPath(repoRoot, registryPath string) string {
	registryPath = strings.TrimSpace(registryPath)
	if registryPath == "" || filepath.IsAbs(registryPath) {
		return registryPath
	}
	base := strings.TrimSpace(repoRoot)
	if base == "" {
		base = "."
	}
	resolved, err := filepath.Abs(filepath.Join(base, registryPath))
	if err != nil {
		return filepath.Clean(filepath.Join(base, registryPath))
	}
	return resolved
}

func durationMS(start time.Time) int64 {
	return time.Since(start).Milliseconds()
}

func (s *Service) GetJob(ctx context.Context, jobID string) (types.Job, error) {
	job, err := s.getJob(ctx, jobID)
	if err != nil {
		s.auditDeniedLookup(ctx, "job.get_status", jobID, err)
		return types.Job{}, err
	}
	s.audit(ctx, "job.get_status", "success", &job, nil)
	return job, nil
}

func (s *Service) getJob(ctx context.Context, jobID string) (types.Job, error) {
	job, err := s.store.GetJob(ctx, jobID)
	if err != nil {
		return types.Job{}, err
	}
	if err := authz.AuthorizeJobAccess(auth.PrincipalFromContext(ctx), job); err != nil {
		return types.Job{}, err
	}
	return s.reconcileJobView(ctx, job)
}

func (s *Service) resolveCacheHitJob(job types.Job) (types.Job, error) {
	sourceID := strings.TrimSpace(job.CacheSourceJobID)
	if job.CacheStatus != "hit" || sourceID == "" {
		return job, nil
	}
	source, err := s.store.GetJob(context.Background(), sourceID)
	if err != nil {
		return types.Job{}, err
	}
	if source.CacheStatus == "hit" && strings.TrimSpace(source.CacheSourceJobID) != "" && source.CacheSourceJobID != source.ID {
		source, err = s.resolveCacheHitJob(source)
		if err != nil {
			return types.Job{}, err
		}
	}
	if source.Result != nil {
		result := cloneResult(source.Result)
		if job.TaskType == "inspect_repo" {
			result = inspectRepoCacheHitResult(result)
		}
		job.Result = result
	}
	job.State = source.State
	job.BackendKind = "cache"
	if source.State == types.JobStateSucceeded {
		job.BackendState = "CACHE_HIT"
		job.CompletedAt = source.CompletedAt
		if source.StartedAt != nil {
			job.StartedAt = source.StartedAt
		}
	} else {
		job.BackendState = "CACHE_ALIAS"
		job.CompletedAt = nil
		if source.StartedAt != nil {
			job.StartedAt = source.StartedAt
		}
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
	if job.TaskType == "inspect_repo" && isTerminal(source.State) {
		timings := cloneMap(mapValue(mapValue(job.RuntimeDiagnostics)["broker_phase_timings_ms"]))
		if timings == nil {
			timings = map[string]any{}
		}
		if _, ok := timings["total_submit_ms"]; !ok {
			timings["total_submit_ms"] = int64(0)
		}
		attachInspectRepoBrokerTimings(&job, "cache_hit", timings)
	}
	if job.Result != nil && job.TaskType == "inspect_repo" {
		job.Result = inspectRepoReleasedResultWithBrokerRuntime(job, job.Result)
	}
	if isTerminal(source.State) {
		stored := job
		stored.Result = nil
		stored.UpdatedAt = time.Now().UTC()
		if err := s.store.UpdateJob(context.Background(), stored); err != nil {
			return types.Job{}, err
		}
	}
	return job, nil
}

func (s *Service) ListJobs(ctx context.Context) ([]types.Job, error) {
	jobs, err := s.listStoredJobs(ctx)
	if err != nil {
		return nil, err
	}
	principal := auth.PrincipalFromContext(ctx)
	filtered := filterAuthorizedJobs(principal, jobs)
	for index, job := range filtered {
		if isTerminal(job.State) && !(job.CacheStatus == "hit" && strings.TrimSpace(job.CacheSourceJobID) != "") {
			continue
		}
		refreshed, err := s.reconcileJobView(ctx, job)
		if err != nil {
			continue
		}
		filtered[index] = refreshed
	}
	s.audit(ctx, "job.list", "success", nil, map[string]any{
		"visible_count": len(filtered),
	})
	return filtered, nil
}

func (s *Service) reconcileJobView(ctx context.Context, job types.Job) (types.Job, error) {
	var err error
	job, err = s.refreshJobState(ctx, job)
	if err != nil {
		return types.Job{}, err
	}

	job, _ = s.maybeIngestJobOutputs(ctx, job)
	// Workers write a final completed heartbeat immediately before result.json.
	// Ingesting the result makes the job terminal, which used to skip the normal
	// progress refresh and lose that final 100% state. Do not ingest stale
	// running heartbeats after a result has already won the race.
	if job.Result != nil && job.Progress == nil && completedHeartbeatExists(s.runRoot, job.ID) {
		if updated, err := s.refreshProgress(ctx, job); err == nil {
			job = updated
		}
	}
	if job.Result == nil && !isTerminal(job.State) {
		if updated, err := s.refreshProgress(ctx, job); err == nil {
			job = updated
		}
	}
	job, err = s.resolveCacheHitJob(job)
	if err != nil {
		return types.Job{}, err
	}
	return job, nil
}

func (s *Service) loadRootJobsAuthorized(ctx context.Context, rootJobID string) ([]types.Job, error) {
	jobs, err := s.listStoredJobs(ctx)
	if err != nil {
		return nil, err
	}
	principal := auth.PrincipalFromContext(ctx)
	return ensureAuthorizedRootJobs(principal, rootJobID, jobs)
}

func (s *Service) authorizeForcedRootRelease(ctx context.Context, requestedBatches int) error {
	return s.authorizeForcedRootReleaseWithUsage(ctx, requestedBatches, 0)
}

func (s *Service) authorizeForcedRootReleaseWithUsage(ctx context.Context, requestedBatches, existingBatches int) error {
	return authorizeCumulativeNonAdminAction(
		ctx,
		requestedBatches,
		existingBatches,
		s.options.RootActionMaxAdditionalBatches,
		"max_additional_batches",
		"forced_release_batches",
	)
}

func (s *Service) authorizeFailedShardRetry(ctx context.Context, requestedShards int) error {
	return s.authorizeFailedShardRetryWithUsage(ctx, requestedShards, 0)
}

func (s *Service) authorizeFailedShardRetryWithUsage(ctx context.Context, requestedShards, existingShards int) error {
	return authorizeCumulativeNonAdminAction(
		ctx,
		requestedShards,
		existingShards,
		s.options.RootActionMaxRetriedShards,
		"retried_shards",
		"retried_shards",
	)
}

func (s *Service) GetRootJobStatus(ctx context.Context, rootJobID string) (types.RootJobStatus, error) {
	filtered, err := s.loadRootJobsAuthorized(ctx, rootJobID)
	if err != nil {
		return types.RootJobStatus{}, err
	}
	_, _ = s.releaseDispatchingRootChildren(ctx, rootJobID, 0, false)
	filtered, err = s.loadRootJobsAuthorized(ctx, rootJobID)
	if err != nil {
		return types.RootJobStatus{}, err
	}
	return rootJobStatusFromState(rootJobID, buildRootSummaryState(filtered, false)), nil
}

func (s *Service) RetryFailedRootShards(ctx context.Context, req types.RetryFailedRootShardsRequest) (types.RetryFailedRootShardsResponse, error) {
	rootJobID := strings.TrimSpace(req.RootJobID)
	if rootJobID == "" {
		return types.RetryFailedRootShardsResponse{}, errors.New("root_job_id is required")
	}

	filtered, err := s.loadRootJobsAuthorized(ctx, rootJobID)
	if err != nil {
		return types.RetryFailedRootShardsResponse{}, err
	}

	state := buildRootSummaryState(filtered, req.IncludeCancelled)
	effectiveChildren := state.effectiveChildren
	retryableCount := 0
	for _, job := range effectiveChildren {
		if shouldRetryShard(job, req.IncludeCancelled) {
			retryableCount++
		}
	}
	if err := s.authorizeFailedShardRetryWithUsage(ctx, retryableCount, state.usage.RetriedShardActions); err != nil {
		return types.RetryFailedRootShardsResponse{}, err
	}
	response := types.RetryFailedRootShardsResponse{
		RootJobID: rootJobID,
	}
	currentEffective := make([]types.Job, 0, len(effectiveChildren))
	for _, job := range effectiveChildren {
		if shouldRetryShard(job, req.IncludeCancelled) {
			retryReq := retrySubmitRequest(job)
			retryReq.TaskParams[taskParamRetryAction] = true
			resp, err := s.SubmitJob(ctx, retryReq)
			if err != nil {
				return types.RetryFailedRootShardsResponse{}, fmt.Errorf("retry shard %s: %w", job.ID, err)
			}
			retriedJob, err := s.getJob(ctx, resp.JobID)
			if err != nil {
				return types.RetryFailedRootShardsResponse{}, fmt.Errorf("lookup retried shard %s: %w", resp.JobID, err)
			}
			currentEffective = append(currentEffective, retriedJob)
			response.RetriedShards = append(response.RetriedShards, types.RetriedShardSubmission{
				PreviousJobID:  job.ID,
				JobID:          resp.JobID,
				State:          resp.State,
				Cache:          resp.Cache,
				StatusURL:      resp.StatusURL,
				ShardKey:       shardKeyOf(job),
				ShardIndex:     shardIndexOf(job),
				ShardCount:     shardCountOf(job),
				AggregationKey: aggregationKeyOf(job),
			})
			continue
		}
		currentEffective = append(currentEffective, job)
		response.SkippedShards = append(response.SkippedShards, types.SkippedShardRetry{
			JobID:      job.ID,
			ShardKey:   shardKeyOf(job),
			ShardIndex: shardIndexOf(job),
			ShardCount: shardCountOf(job),
			Reason:     retrySkipReason(job, req.IncludeCancelled),
		})
	}
	response.RetriedCount = len(response.RetriedShards)
	response.SkippedCount = len(response.SkippedShards)
	response.CumulativeRetriedShards = state.usage.RetriedShardActions + response.RetriedCount
	response.RemainingRetriedShardBudget = remainingNonAdminBudget(s.options.RootActionMaxRetriedShards, response.CumulativeRetriedShards, auth.PrincipalFromContext(ctx))

	if req.ResubmitReducer && len(response.RetriedShards) > 0 {
		if len(state.effectiveReducers) > 0 {
			resp, err := s.submitRetriedReducer(ctx, state.effectiveReducers[0], currentEffective)
			if err != nil {
				return types.RetryFailedRootShardsResponse{}, fmt.Errorf("resubmit reducer: %w", err)
			}
			response.ReducerJob = resp
		}
	}

	return response, nil
}

func (s *Service) ReleaseDeferredRootChunks(ctx context.Context, req types.ReleaseDeferredRootChunksRequest) (types.ReleaseDeferredRootChunksResponse, error) {
	rootJobID := strings.TrimSpace(req.RootJobID)
	if rootJobID == "" {
		return types.ReleaseDeferredRootChunksResponse{}, errors.New("root_job_id is required")
	}
	filtered, err := s.loadRootJobsAuthorized(ctx, rootJobID)
	if err != nil {
		return types.ReleaseDeferredRootChunksResponse{}, err
	}
	state := buildRootSummaryState(filtered, false)
	if err := s.authorizeForcedRootReleaseWithUsage(ctx, req.MaxAdditionalBatches, state.usage.ForcedReleasedChunks); err != nil {
		return types.ReleaseDeferredRootChunksResponse{}, err
	}
	release, err := s.releaseDispatchingRootChildren(ctx, rootJobID, req.MaxAdditionalBatches, req.MaxAdditionalBatches > 0)
	if err != nil {
		return types.ReleaseDeferredRootChunksResponse{}, err
	}
	status, err := s.GetRootJobStatus(ctx, rootJobID)
	if err != nil {
		return types.ReleaseDeferredRootChunksResponse{}, err
	}
	return types.ReleaseDeferredRootChunksResponse{
		RootJobID:                     rootJobID,
		ReleasedChunks:                release.ReleasedChunks,
		ReleasedChildren:              release.ReleasedChildren,
		ReducerReleased:               release.ReducerReleased,
		CumulativeForcedReleaseChunks: status.ForcedReleasedChunks,
		RemainingForcedReleaseBudget:  remainingNonAdminBudget(s.options.RootActionMaxAdditionalBatches, status.ForcedReleasedChunks, auth.PrincipalFromContext(ctx)),
		RootStatus:                    status,
	}, nil
}

func (s *Service) CancelJob(ctx context.Context, jobID string) (types.CancelJobResponse, error) {
	job, err := s.getJob(ctx, jobID)
	if err != nil {
		s.auditDeniedLookup(ctx, "job.cancel", jobID, err)
		return types.CancelJobResponse{}, err
	}

	if job.BackendRunID != "" {
		if err := s.backend.CancelRun(ctx, job.BackendRunID); err != nil {
			return types.CancelJobResponse{}, fmt.Errorf("cancel backend run: %w", err)
		}
	}

	now := time.Now().UTC()
	job.State = types.JobStateCancelled
	job.CompletedAt = &now
	job.UpdatedAt = now

	if err := s.store.UpdateJob(ctx, job); err != nil {
		return types.CancelJobResponse{}, fmt.Errorf("update job: %w", err)
	}

	s.logger.Printf("cancelled job=%s backend_run_id=%s", job.ID, job.BackendRunID)
	s.audit(ctx, "job.cancel", "success", &job, nil)

	return types.CancelJobResponse{
		JobID: job.ID,
		State: job.State,
	}, nil
}

func (s *Service) GetJobLogs(ctx context.Context, jobID, stream string, maxBytes int) (types.JobLogs, error) {
	job, err := s.getJob(ctx, jobID)
	if err != nil {
		s.auditDeniedLookup(ctx, "job.fetch_logs", jobID, err)
		return types.JobLogs{}, err
	}
	if err := policy.AuthorizeJobLogs(job); err != nil {
		s.audit(ctx, "job.fetch_logs", "policy_denied", &job, map[string]any{
			"stream": stream,
		})
		return types.JobLogs{}, err
	}

	if stream == "" {
		stream = "combined"
	}
	if maxBytes <= 0 {
		maxBytes = 16384
	}

	runDir := filepath.Join(s.runRoot, job.ID)
	stdoutPath := filepath.Join(runDir, "stdout.log")
	stderrPath := filepath.Join(runDir, "stderr.log")

	stdoutText, _ := readLogFile(stdoutPath)
	stderrText, _ := readLogFile(stderrPath)

	var content string
	var sourceRefs []string
	switch stream {
	case "stdout":
		content = stdoutText
		if stdoutText != "" {
			sourceRefs = append(sourceRefs, "stdout.log")
		}
	case "stderr":
		content = stderrText
		if stderrText != "" {
			sourceRefs = append(sourceRefs, "stderr.log")
		}
	case "combined":
		content, sourceRefs = combineLogs(stdoutText, stderrText)
	default:
		return types.JobLogs{}, fmt.Errorf("unsupported log stream: %s", stream)
	}

	content = redactLogContent(content)
	content, truncated := truncateLogContent(content, maxBytes)

	return types.JobLogs{
		JobID:      job.ID,
		State:      job.State,
		Stream:     stream,
		Content:    content,
		Truncated:  truncated,
		MaxBytes:   maxBytes,
		SourceRefs: sourceRefs,
	}, s.auditAndReturnLogs(ctx, job, stream, maxBytes)
}

func (s *Service) GetReleasedResult(ctx context.Context, jobID string) (types.JobResultRelease, error) {
	job, err := s.store.GetJob(ctx, jobID)
	if err != nil {
		s.auditDeniedLookup(ctx, "job.fetch_result", jobID, err)
		return types.JobResultRelease{}, err
	}
	if err := authz.AuthorizeJobAccess(auth.PrincipalFromContext(ctx), job); err != nil {
		s.auditDeniedLookup(ctx, "job.fetch_result", jobID, err)
		return types.JobResultRelease{}, err
	}
	if !skipInspectRepoResultProbe(ctx) {
		if releaseJob, ok, err := s.awaitDirectInspectRepoReleasedResult(job, localInspectRepoResultProbeWindow); err != nil {
			return types.JobResultRelease{}, err
		} else if ok {
			release, err := buildReleasedResult(releaseJob)
			if err != nil {
				return types.JobResultRelease{}, err
			}
			s.audit(ctx, "job.fetch_result", "success", &releaseJob, map[string]any{
				"artifact_count": len(release.Artifacts),
				"has_result":     release.Result != nil,
				"result_source":  "run_files_direct",
			})
			return release, nil
		}
	}
	if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
		return types.JobResultRelease{}, err
	} else if ok {
		release, err := buildReleasedResult(releaseJob)
		if err != nil {
			return types.JobResultRelease{}, err
		}
		s.audit(ctx, "job.fetch_result", "success", &releaseJob, map[string]any{
			"artifact_count": len(release.Artifacts),
			"has_result":     release.Result != nil,
			"result_source":  "run_files_direct",
		})
		return release, nil
	}
	job, err = s.getJob(ctx, jobID)
	if err != nil {
		s.auditDeniedLookup(ctx, "job.fetch_result", jobID, err)
		return types.JobResultRelease{}, err
	}
	release, err := buildReleasedResult(job)
	if err != nil {
		return types.JobResultRelease{}, err
	}
	s.audit(ctx, "job.fetch_result", "success", &job, map[string]any{
		"artifact_count": len(release.Artifacts),
		"has_result":     release.Result != nil,
	})
	return release, nil
}

func (s *Service) awaitInspectRepoRunResult(job types.Job, waitWindow time.Duration) {
	if !s.shouldProbeInspectRepoRunResult(job) || waitWindow <= 0 {
		return
	}
	deadline := time.Now().Add(waitWindow)
	for !time.Now().After(deadline) {
		if s.runResultExists(job) {
			return
		}
		time.Sleep(localInspectRepoResultProbeInterval)
	}
}

func (s *Service) awaitDirectInspectRepoReleasedResult(job types.Job, waitWindow time.Duration) (types.Job, bool, error) {
	if !s.shouldProbeInspectRepoRunResult(job) || waitWindow <= 0 {
		return job, false, nil
	}
	if waiter, ok := s.backend.(backends.LocalInspectRepoResultWaiter); ok {
		if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
			return job, false, err
		} else if ok {
			return releaseJob, true, nil
		}
		if waiter.AwaitLocalInspectRepoResult(context.Background(), job.BackendRunID, waitWindow) {
			if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
				return job, false, err
			} else if ok {
				return releaseJob, true, nil
			}
		}
		return job, false, nil
	}
	deadline := time.Now().Add(waitWindow)
	for {
		if releaseJob, ok, err := s.tryDirectInspectRepoReleasedResult(job); err != nil {
			return job, false, err
		} else if ok {
			return releaseJob, true, nil
		}
		if time.Now().After(deadline) {
			return job, false, nil
		}
		time.Sleep(localInspectRepoResultProbeInterval)
	}
}

func (s *Service) shouldProbeInspectRepoRunResult(job types.Job) bool {
	if job.TaskType != "inspect_repo" || job.Result != nil {
		return false
	}
	if job.CacheStatus == "hit" && strings.TrimSpace(job.CacheSourceJobID) != "" {
		return false
	}
	if job.BackendKind != "local" || strings.TrimSpace(job.BackendRunID) == "" {
		return false
	}
	return true
}

func (s *Service) tryDirectInspectRepoReleasedResult(job types.Job) (types.Job, bool, error) {
	if job.TaskType != "inspect_repo" || job.Result != nil {
		return job, false, nil
	}
	if job.CacheStatus == "hit" && strings.TrimSpace(job.CacheSourceJobID) != "" {
		return job, false, nil
	}
	result, err := s.readRunResult(job)
	if err != nil {
		return job, false, nil
	}
	transient := job
	if transient.State != types.JobStateSucceeded {
		now := time.Now().UTC()
		transient.State = types.JobStateSucceeded
		transient.BackendState = strings.TrimSpace(transient.BackendState)
		transient.UpdatedAt = now
		if transient.CompletedAt == nil {
			transient.CompletedAt = &now
		}
	}
	transient = s.applyIngestedOutputs(transient, result, s.readRunArtifacts(job))
	return transient, true, nil
}

func buildReleasedResult(job types.Job) (types.JobResultRelease, error) {
	result, artifacts, err := policy.FilterJobResult(job)
	if err != nil {
		return types.JobResultRelease{}, err
	}
	if result != nil {
		result = inspectRepoReleasedResultWithBrokerRuntime(job, result)
	}
	artifacts = filterJobArtifactsForRelease(job, artifacts)
	return types.JobResultRelease{
		JobID:                  job.ID,
		State:                  job.State,
		Result:                 result,
		RuntimeDiagnostics:     cloneMap(job.RuntimeDiagnostics),
		ExecutionQuality:       job.ExecutionQuality,
		DegradedLocalExecution: job.DegradedLocalExecution,
		RetryRecommended:       job.RetryRecommended,
		Artifacts:              artifacts,
	}, nil
}

func inspectRepoReleasedResultWithBrokerRuntime(job types.Job, result *types.Result) *types.Result {
	if result == nil || job.TaskType != "inspect_repo" || result.SchemaName != "repo_inspection_v2" {
		return result
	}
	brokerPhaseTimings := cloneMap(mapValue(mapValue(job.RuntimeDiagnostics)["broker_phase_timings_ms"]))
	brokerResultSource := stringValue(mapValue(job.RuntimeDiagnostics)["broker_result_source"])
	if len(brokerPhaseTimings) == 0 && brokerResultSource == "" {
		return result
	}
	cloned := cloneResult(result)
	payload := cloneMap(cloned.Payload)
	runtime := cloneMap(mapValue(payload["runtime"]))
	if runtime == nil {
		runtime = map[string]any{}
	}
	if len(brokerPhaseTimings) > 0 {
		runtime["broker_phase_timings_ms"] = brokerPhaseTimings
	}
	if brokerResultSource != "" {
		runtime["broker_result_source"] = brokerResultSource
	}
	payload["runtime"] = runtime
	cloned.Payload = payload
	return cloned
}

func attachInspectRepoBrokerRuntimeToRelease(release *types.JobResultRelease, brokerResultSource string, timings map[string]any) {
	if release == nil {
		return
	}
	job := types.Job{
		TaskType:           "inspect_repo",
		RuntimeDiagnostics: cloneMap(release.RuntimeDiagnostics),
	}
	attachInspectRepoBrokerTimings(&job, brokerResultSource, timings)
	release.RuntimeDiagnostics = cloneMap(job.RuntimeDiagnostics)
	release.Result = inspectRepoReleasedResultWithBrokerRuntime(job, release.Result)
}

func filterArtifactsByType(artifacts []types.Artifact, allowed ...string) []types.Artifact {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, artifactType := range allowed {
		allowedSet[artifactType] = struct{}{}
	}
	filtered := make([]types.Artifact, 0, len(artifacts))
	for _, artifact := range artifacts {
		if _, ok := allowedSet[artifact.ArtifactType]; ok {
			filtered = append(filtered, artifact)
		}
	}
	return filtered
}

func (s *Service) GetJobRetryRecommendation(ctx context.Context, jobID string) (types.JobRetryRecommendation, error) {
	job, err := s.getJob(ctx, jobID)
	if err != nil {
		return types.JobRetryRecommendation{}, err
	}
	if job.Result == nil {
		return types.JobRetryRecommendation{}, fmt.Errorf("job %q has no result", jobID)
	}
	rec, ok := retryRecommendationFromResult(job)
	if !ok {
		return types.JobRetryRecommendation{}, fmt.Errorf("job %q has no broker retry recommendation", jobID)
	}
	return rec, nil
}

func (s *Service) RetryJobWithRecommendation(ctx context.Context, jobID string) (types.SubmitJobResponse, error) {
	job, err := s.getJob(ctx, jobID)
	if err != nil {
		return types.SubmitJobResponse{}, err
	}
	if job.Result == nil {
		return types.SubmitJobResponse{}, fmt.Errorf("job %q has no result", jobID)
	}
	rec, ok := retryRecommendationFromResult(job)
	if !ok {
		return types.SubmitJobResponse{}, fmt.Errorf("job %q has no broker retry recommendation", jobID)
	}
	req := job.Request
	req.TaskParams = cloneTaskParams(job.Request.TaskParams)
	req.TaskParams[taskParamRetryOfJobID] = job.ID
	req.ExecutionProfile = rec.ExecutionProfile
	req.ExecutionProfile = mergePlacementHintIntoProfile(req.ExecutionProfile, rec.PlacementHint)
	req.ExecutionProfile = s.applyExecutionProfileDefaults(req.ExecutionProfile)
	req.TaskParams = mergePlacementHintIntoTaskParams(req.TaskParams, rec.PlacementHint)
	req.IdempotencyKey = ""
	return s.SubmitJob(ctx, req)
}

func (s *Service) GetArtifactMetadata(ctx context.Context, artifactID string, allowedTypes map[string]struct{}) (types.ArtifactMetadata, error) {
	principal := auth.PrincipalFromContext(ctx)
	jobs, err := s.store.ListJobs(ctx)
	if err != nil {
		return types.ArtifactMetadata{}, err
	}
	sort.SliceStable(jobs, func(i, j int) bool {
		return jobs[i].SubmittedAt.After(jobs[j].SubmittedAt)
	})
	for _, job := range jobs {
		if err := authz.AuthorizeJobAccess(principal, job); err != nil {
			continue
		}
		// Metadata contains no artifact path or content. Keep it available for
		// authorized jobs while still enforcing inspection's full-trace gate.
		eligible := filterJobArtifactsForRelease(job, job.Artifacts)
		for _, artifact := range eligible {
			if artifact.ArtifactID != artifactID {
				continue
			}
			if len(allowedTypes) > 0 {
				if _, ok := allowedTypes[artifact.ArtifactType]; !ok {
					continue
				}
			}
			schemaName := ""
			if job.Result != nil {
				schemaName = job.Result.SchemaName
			}
			return types.ArtifactMetadata{
				ArtifactID:     artifact.ArtifactID,
				ArtifactType:   artifact.ArtifactType,
				Classification: artifact.Classification,
				ContentHash:    artifact.ContentHash,
				SourceJobID:    job.ID,
				SourceTaskType: job.TaskType,
				SourceSchema:   schemaName,
				SubmittedBy:    job.SubmittedBy,
				CreatedAt:      job.CreatedAt.Format(time.RFC3339),
			}, nil
		}
	}
	return types.ArtifactMetadata{}, store.ErrNotFound
}

func (s *Service) LookupCache(ctx context.Context, req types.SubmitJobRequest) (types.CacheLookupResponse, error) {
	cacheLookup, err := s.lookupCompletedCacheJob(ctx, req)
	if err != nil {
		return types.CacheLookupResponse{}, err
	}
	resp := types.CacheLookupResponse{
		Status:     "uncacheable",
		TaskType:   req.TaskType,
		SchemaName: req.OutputSchema.Name,
		CacheKey:   cacheLookup.key,
	}
	if !cacheLookup.cacheable {
		return resp, nil
	}
	resp.Status = "miss"
	if cacheLookup.job == nil {
		return resp, nil
	}
	principal := auth.PrincipalFromContext(ctx)
	if err := authz.AuthorizeJobAccess(principal, *cacheLookup.job); err != nil {
		// Do not disclose inaccessible cache hits to non-admin callers.
		return resp, nil
	}
	resp.Status = "hit"
	resp.SourceJobID = cacheLookup.job.ID
	resp.ArtifactCount = len(cacheLookup.job.Artifacts)
	return resp, nil
}

func newJobID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("job_%d", time.Now().UnixNano())
	}
	return "job_" + hex.EncodeToString(buf)
}

func newRootJobID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("root_%d", time.Now().UnixNano())
	}
	return "root_" + hex.EncodeToString(buf)
}

func isTerminal(state types.JobState) bool {
	switch state {
	case types.JobStateSucceeded, types.JobStateFailed, types.JobStateCancelled, types.JobStatePreempted, types.JobStateTimedOut:
		return true
	default:
		return false
	}
}

func applyBrokerResultPolicies(job *types.Job, result *types.Result) {
	if job == nil || result == nil {
		return
	}
	if !isRAGLikeTask(job.TaskType) {
		return
	}
	// repo_inspection_v2 owns its quality states and escalation history. Legacy
	// policy promotion/guidance must not turn lexical evidence into an answer or
	// append fields outside the v2 contract.
	if job.TaskType == "inspect_repo" && result.SchemaName == "repo_inspection_v2" {
		quality, _ := result.Payload["quality"].(map[string]any)
		if stringValue(quality["result"]) == "failed" {
			job.State = types.JobStateFailed
			job.ResultError = "repo_inspection_gpu_tiers_exhausted"
		}
		return
	}
	payload := result.Payload
	policySignals, ok := payload["policy_signals"].(map[string]any)
	if !ok {
		return
	}

	warnings := collectStringSlice(policySignals["warnings"])
	existing := collectStringSlice(payload["warnings"])
	for _, warning := range warnings {
		switch warning {
		case "LOCAL_RETRIEVAL_DEGRADED":
			existing = appendUniqueString(existing, "broker_local_retrieval_degraded")
		case "NO_REAL_RETRIEVAL_BACKEND":
			existing = appendUniqueString(existing, "broker_no_real_retrieval_backend")
			payload["broker_retry_recommendation"] = brokerRetryRecommendation(*job)
			if job.ResultError == "" {
				job.ResultError = "broker_policy_no_real_retrieval_backend"
			}
		case "IGNORED_PATH_RETRIEVAL_CONTAMINATION":
			existing = appendUniqueString(existing, "broker_ignored_path_retrieval_contamination")
		}
	}
	if requiresStrictRetrievalQuality(*job) && hasAnyString(warnings, []string{
		"LOCAL_RETRIEVAL_DEGRADED",
		"NO_REAL_RETRIEVAL_BACKEND",
		"IGNORED_PATH_RETRIEVAL_CONTAMINATION",
	}) {
		existing = appendUniqueString(existing, "broker_retrieval_quality_gate_failed")
		job.State = types.JobStateFailed
		job.ResultError = "broker_policy_retrieval_quality_insufficient"
	}
	if len(existing) > 0 {
		payload["warnings"] = stringSliceToAny(existing)
	}
	applyAgentFacingGuidance(job, payload, warnings)
}

func applyAgentFacingGuidance(job *types.Job, payload map[string]any, warnings []string) {
	if payload == nil || job == nil {
		return
	}
	needsVerification := hasAnyString(warnings, []string{
		"LOCAL_RETRIEVAL_DEGRADED",
		"NO_REAL_RETRIEVAL_BACKEND",
		"IGNORED_PATH_RETRIEVAL_CONTAMINATION",
	})
	mode := "broker_evidence_ready"
	recommendation := defaultAgentNextAction(job.TaskType)
	confidence := "high"
	if hasAnyString(warnings, []string{"NO_REAL_RETRIEVAL_BACKEND"}) {
		mode = "lead_generation_only"
		recommendation = "Treat this result as a lead only. Retry with the recommended profile or inspect the cited files directly before making claims."
		confidence = "low"
	} else if needsVerification {
		mode = "verify_before_claiming"
		recommendation = "Use the cited evidence as a pointer, but verify the referenced files directly before making strong claims."
		confidence = "medium"
	}
	payload["usage_guidance"] = mergePayloadMap(mapValue(payload["usage_guidance"]), map[string]any{
		"mode":                      mode,
		"needs_direct_verification": needsVerification,
		"recommended_action":        recommendation,
	})
	if stringValue(payload["recommended_next_action"]) == "" {
		payload["recommended_next_action"] = recommendation
	}
	if stringValue(payload["confidence"]) == "" {
		payload["confidence"] = confidence
	}
	if _, ok := payload["must_cite_evidence"].(bool); !ok {
		payload["must_cite_evidence"] = true
	}
}

func mergePayloadMap(base, extra map[string]any) map[string]any {
	if len(base) == 0 && len(extra) == 0 {
		return nil
	}
	out := make(map[string]any, len(base)+len(extra))
	for key, value := range base {
		out[key] = value
	}
	for key, value := range extra {
		out[key] = value
	}
	return out
}

func defaultAgentNextAction(taskType string) string {
	switch taskType {
	case "inspect_repo":
		return "Answer from the cited files and line ranges, and call out the most relevant subsystem or symbol first."
	case "debug_with_local_context":
		return "Lead with the highest-confidence root cause candidate, then verify the cited code and logs before proposing a fix."
	case "summarize_logs":
		return "Summarize the dominant failure cluster first, then use the cited log evidence to guide the next debugging step."
	case "propose_patch":
		return "Keep the fix scoped to the cited files and validate it against the referenced evidence before changing code."
	default:
		return "Use the cited evidence directly in the answer and avoid claims that are not supported by the released refs."
	}
}

func mergeMetadata(base, extra map[string]any) map[string]any {
	if len(base) == 0 && len(extra) == 0 {
		return nil
	}
	out := make(map[string]any, len(base)+len(extra))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range extra {
		out[k] = v
	}
	return out
}

func resultSchemaName(result *types.Result) string {
	if result == nil {
		return ""
	}
	return result.SchemaName
}

func writeJSONFile(path string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func cloneTaskParams(in map[string]any) map[string]any {
	out := make(map[string]any, len(in)+1)
	for k, v := range in {
		out[k] = v
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func cloneArtifacts(in []types.Artifact) []types.Artifact {
	if len(in) == 0 {
		return nil
	}
	out := make([]types.Artifact, len(in))
	copy(out, in)
	return out
}

func compactNonEmptyStrings(in []string) []string {
	out := make([]string, 0, len(in))
	for _, item := range in {
		if strings.TrimSpace(item) != "" {
			out = append(out, item)
		}
	}
	return out
}

func normalizeOptions(opts Options) Options {
	if opts.ParallelMaxBatchSize <= 0 {
		opts.ParallelMaxBatchSize = 64
	}
	if opts.ParallelMaxActiveBatches < 0 {
		opts.ParallelMaxActiveBatches = 0
	}
	if opts.RootActionMaxAdditionalBatches <= 0 {
		opts.RootActionMaxAdditionalBatches = 1
	}
	if opts.RootActionMaxRetriedShards <= 0 {
		opts.RootActionMaxRetriedShards = 4
	}
	return opts
}

func retrySubmitRequest(job types.Job) types.SubmitJobRequest {
	req := job.Request
	req.TaskParams = cloneTaskParams(job.Request.TaskParams)
	req.Orchestration = types.OrchestrationRequest{
		ParentJobID:     job.ParentJobID,
		RootJobID:       job.RootJobID,
		Strategy:        orchestrationStrategyOf(job),
		ShardKey:        shardKeyOf(job),
		ShardIndex:      shardIndexOf(job),
		ShardCount:      shardCountOf(job),
		AggregationKey:  aggregationKeyOf(job),
		DependsOnJobIDs: dependsOnJobIDsOf(job),
	}
	return req
}

func remainingNonAdminBudget(limit, used int, principal auth.Principal) int {
	if auth.IsAdmin(principal) {
		return 0
	}
	remaining := limit - used
	if remaining < 0 {
		return 0
	}
	return remaining
}

func intFromAny(value any) int {
	switch typed := value.(type) {
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

func floatFromAny(value any) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	default:
		return 0
	}
}
