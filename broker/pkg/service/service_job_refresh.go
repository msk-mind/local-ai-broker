package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/schemas"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func (s *Service) refreshJobState(ctx context.Context, job types.Job) (types.Job, error) {
	if job.RootJobID != "" {
		job = s.refreshJobAfterRootRelease(ctx, job)
	}
	if job.CacheStatus == "hit" && strings.TrimSpace(job.CacheSourceJobID) != "" {
		return job, nil
	}
	if job.BackendRunID == "" || isTerminal(job.State) {
		return job, nil
	}
	if s.runResultExists(job) {
		return job, nil
	}

	runStatus, err := s.backend.GetRun(ctx, job.BackendRunID)
	if err != nil {
		return job, nil
	}
	if runStatus.State == "" || runStatus.State == job.State {
		return s.persistBackendMetadata(job, runStatus), nil
	}
	return s.applyBackendRunStatus(ctx, job, runStatus)
}

func (s *Service) refreshJobAfterRootRelease(ctx context.Context, job types.Job) types.Job {
	_, _ = s.releaseDispatchingRootChildren(ctx, job.RootJobID, 0, false)
	if refreshed, err := s.store.GetJob(ctx, job.ID); err == nil {
		return refreshed
	}
	return job
}

func (s *Service) persistBackendMetadata(job types.Job, runStatus backends.RunStatus) types.Job {
	previousDiagnostics := len(job.RuntimeDiagnostics)
	job = mergeBackendRunDiagnostics(job, runStatus)
	if (job.BackendState == "" && runStatus.RawState != "") || len(job.RuntimeDiagnostics) != previousDiagnostics {
		job.BackendState = runStatus.RawState
		job.BackendExitCode = runStatus.ExitCode
		_ = s.store.UpdateJob(context.Background(), job)
	}
	return job
}

func (s *Service) applyBackendRunStatus(ctx context.Context, job types.Job, runStatus backends.RunStatus) (types.Job, error) {
	now := time.Now().UTC()
	job.State = runStatus.State
	job.BackendState = runStatus.RawState
	job.BackendExitCode = runStatus.ExitCode
	job = mergeBackendRunDiagnostics(job, runStatus)
	if runStatus.State == types.JobStateFailed && strings.TrimSpace(job.ResultError) == "" {
		job.ResultError = "worker_failed_before_result"
	}
	job.UpdatedAt = now
	if runStatus.State == types.JobStateRunning && job.StartedAt == nil {
		job.StartedAt = &now
	}
	if isTerminal(runStatus.State) {
		job.CompletedAt = &now
	}
	if err := s.store.UpdateJob(ctx, job); err != nil {
		return types.Job{}, fmt.Errorf("update refreshed job: %w", err)
	}
	return job, nil
}

func mergeBackendRunDiagnostics(job types.Job, runStatus backends.RunStatus) types.Job {
	if len(runStatus.Diagnostics) == 0 {
		return job
	}
	job.RuntimeDiagnostics = mergeRuntimeDiagnostics(job.RuntimeDiagnostics, runStatus.Diagnostics)
	return job
}

func (s *Service) maybeIngestJobOutputs(ctx context.Context, job types.Job) (types.Job, error) {
	if job.CacheStatus == "hit" && strings.TrimSpace(job.CacheSourceJobID) != "" {
		return job, nil
	}
	if job.Result != nil {
		return job, nil
	}
	if job.State != types.JobStateSucceeded && !s.runResultExists(job) {
		return job, nil
	}
	updated, err := s.ingestRunOutputs(ctx, job)
	if err == nil {
		return updated, nil
	}

	now := time.Now().UTC()
	job.State = types.JobStateFailed
	job.ResultError = err.Error()
	job.UpdatedAt = now
	job.CompletedAt = &now
	_ = s.store.UpdateJob(ctx, job)
	return job, nil
}

func (s *Service) runResultExists(job types.Job) bool {
	resultPath := filepath.Join(s.runRoot, job.ID, "result.json")
	info, err := os.Stat(resultPath)
	return err == nil && !info.IsDir()
}

func completedHeartbeatExists(runRoot, jobID string) bool {
	path := filepath.Join(runRoot, jobID, "heartbeat.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var heartbeat struct {
		State string `json:"state"`
	}
	if json.Unmarshal(data, &heartbeat) != nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(heartbeat.State), "completed")
}

func (s *Service) readRunResult(job types.Job) (types.Result, error) {
	resultPath := filepath.Join(s.runRoot, job.ID, "result.json")
	resultBytes, err := os.ReadFile(resultPath)
	if err != nil {
		return types.Result{}, err
	}

	var result types.Result
	if err := json.Unmarshal(resultBytes, &result); err != nil {
		return types.Result{}, err
	}
	if err := schemas.ValidateResult(job.TaskType, job.Request.OutputSchema.Name, result); err != nil {
		return types.Result{}, err
	}
	if err := validateInspectionRequestEcho(job, result); err != nil {
		return types.Result{}, err
	}
	return result, nil
}

func validateInspectionRequestEcho(job types.Job, result types.Result) error {
	if job.TaskType != "inspect_repo" || result.SchemaName != "repo_inspection_v2" {
		return nil
	}
	wantQuery := strings.TrimSpace(stringValue(job.Request.TaskParams["query"]))
	gotQuery := strings.TrimSpace(stringValue(result.Payload["query"]))
	if wantQuery == "" || gotQuery != wantQuery {
		return fmt.Errorf("repo_inspection_v2 payload.query does not match the submitted query")
	}
	wantMode := strings.ToLower(strings.TrimSpace(stringValue(job.Request.TaskParams["mode"])))
	if wantMode == "" {
		wantMode = "auto"
	}
	gotMode := strings.ToLower(strings.TrimSpace(stringValue(result.Payload["mode"])))
	if gotMode != wantMode {
		return fmt.Errorf("repo_inspection_v2 payload.mode does not match the submitted mode")
	}
	return nil
}

func (s *Service) readRunArtifacts(job types.Job) []types.Artifact {
	artifactsPath := filepath.Join(s.runRoot, job.ID, "artifacts.json")
	artifactBytes, err := os.ReadFile(artifactsPath)
	if err != nil || len(artifactBytes) == 0 {
		return nil
	}
	var artifacts []types.Artifact
	if err := json.Unmarshal(artifactBytes, &artifacts); err != nil {
		return nil
	}
	return artifacts
}

func (s *Service) applyIngestedOutputs(job types.Job, result types.Result, artifacts []types.Artifact) types.Job {
	applyBrokerResultPolicies(&job, &result)
	job.Result = &result
	if len(artifacts) > 0 {
		job.Artifacts = artifacts
	}
	job.RuntimeDiagnostics = s.extractRuntimeDiagnostics(job, result)
	job.DegradedLocalExecution = isDegradedResult(job.TaskType, result, job.RuntimeDiagnostics)
	job.RetryRecommended = hasRetryRecommendation(job)
	job.ExecutionQuality = deriveResultExecutionQuality(job.TaskType, result, job.RuntimeDiagnostics, job.RetryRecommended)
	job.UpdatedAt = time.Now().UTC()
	return job
}

func (s *Service) refreshProgress(ctx context.Context, job types.Job) (types.Job, error) {
	heartbeatPath := filepath.Join(s.runRoot, job.ID, "heartbeat.json")
	heartbeatBytes, err := os.ReadFile(heartbeatPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return job, nil
		}
		return job, err
	}

	var heartbeat struct {
		JobID     string         `json:"job_id"`
		State     string         `json:"state"`
		Phase     string         `json:"phase"`
		Percent   int            `json:"percent"`
		Message   string         `json:"message"`
		Timestamp string         `json:"timestamp"`
		Metrics   map[string]any `json:"metrics"`
	}
	if err := json.Unmarshal(heartbeatBytes, &heartbeat); err != nil {
		return job, err
	}

	progress := &types.ProgressInfo{
		State:   heartbeat.State,
		Phase:   heartbeat.Phase,
		Percent: heartbeat.Percent,
		Message: heartbeat.Message,
		Metrics: heartbeat.Metrics,
	}
	now := time.Now().UTC()
	progress.LastUpdated = &now
	if heartbeat.Timestamp != "" {
		if ts, err := time.Parse(time.RFC3339, heartbeat.Timestamp); err == nil {
			progress.Timestamp = &ts
			progress.LastUpdated = &ts
		}
	}

	if progressEquals(job.Progress, progress) {
		return job, nil
	}

	job.Progress = progress
	job.UpdatedAt = now
	if err := s.store.UpdateJob(ctx, job); err != nil {
		return types.Job{}, fmt.Errorf("persist progress: %w", err)
	}
	return job, nil
}

func (s *Service) ingestRunOutputs(ctx context.Context, job types.Job) (types.Job, error) {
	result, err := s.readRunResult(job)
	if err != nil {
		return job, err
	}
	if job.State != types.JobStateSucceeded {
		now := time.Now().UTC()
		job.State = types.JobStateSucceeded
		job.BackendState = strings.TrimSpace(job.BackendState)
		job.UpdatedAt = now
		if job.CompletedAt == nil {
			job.CompletedAt = &now
		}
	}
	job = s.applyIngestedOutputs(job, result, s.readRunArtifacts(job))
	if err := s.store.UpdateJob(ctx, job); err != nil {
		return types.Job{}, fmt.Errorf("persist ingested outputs: %w", err)
	}
	return job, nil
}
