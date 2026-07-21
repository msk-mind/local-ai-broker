package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	localbackend "github.com/msk-mind/local-ai-broker/broker/pkg/backends/local"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type fakeBackend struct {
	status backends.RunStatus
}

func (f fakeBackend) Name() string { return "fake" }

func (f fakeBackend) SubmitRun(context.Context, types.Job) (backends.SubmitResponse, error) {
	return backends.SubmitResponse{
		BackendKind:  "fake",
		BackendRunID: "run-1",
		InitialState: types.JobStateQueued,
	}, nil
}

func (f fakeBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return f.status, nil
}

func (f fakeBackend) CancelRun(context.Context, string) error { return nil }

type mutableFakeBackend struct {
	status backends.RunStatus
}

func (f *mutableFakeBackend) Name() string { return "fake" }

func (f *mutableFakeBackend) SubmitRun(context.Context, types.Job) (backends.SubmitResponse, error) {
	return backends.SubmitResponse{
		BackendKind:  "fake",
		BackendRunID: "run-1",
		InitialState: types.JobStateQueued,
	}, nil
}

func (f *mutableFakeBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return f.status, nil
}

func (f *mutableFakeBackend) CancelRun(context.Context, string) error { return nil }

type countingFakeBackend struct {
	status      backends.RunStatus
	getRunCalls int
	submitCalls int
}

func (f *countingFakeBackend) Name() string { return "counting-fake" }

func (f *countingFakeBackend) SubmitRun(context.Context, types.Job) (backends.SubmitResponse, error) {
	f.submitCalls++
	return backends.SubmitResponse{
		BackendKind:  "counting-fake",
		BackendRunID: "run-1",
		InitialState: types.JobStateQueued,
	}, nil
}

func (f *countingFakeBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	f.getRunCalls++
	return f.status, nil
}

func (f *countingFakeBackend) CancelRun(context.Context, string) error { return nil }

type countingJobStore struct {
	*store.MemoryJobStore
	updateCalls int
}

func newCountingJobStore() *countingJobStore {
	return &countingJobStore{MemoryJobStore: store.NewMemoryJobStore()}
}

func (s *countingJobStore) UpdateJob(ctx context.Context, job types.Job) error {
	s.updateCalls++
	return s.MemoryJobStore.UpdateJob(ctx, job)
}

type cacheKeyLookupOnlyStore struct {
	*store.MemoryJobStore
	listJobsCalls int
}

func (s *cacheKeyLookupOnlyStore) ListJobs(context.Context) ([]types.Job, error) {
	s.listJobsCalls++
	return nil, errors.New("ListJobs should not be called")
}

type immediateLocalInspectRepoBackend struct {
	runRoot string
}

type delayedLocalInspectRepoCompletionBackend struct {
	runRoot string
	delay   time.Duration
}

type delayedLocalInspectRepoResultBackend struct {
	runRoot string
	delay   time.Duration
}

type delayedWarmQueuedLocalInspectRepoResultBackend struct {
	runRoot string
	delay   time.Duration
}

type countingDelayedLocalInspectRepoResultBackend struct {
	runRoot     string
	delay       time.Duration
	getRunCalls atomic.Int32
	submitCalls atomic.Int32
}

type signaledLocalInspectRepoResultBackend struct {
	runRoot     string
	delay       time.Duration
	getRunCalls atomic.Int32
	waiters     sync.Map
}

func (b *immediateLocalInspectRepoBackend) Name() string { return "local" }

func (b *immediateLocalInspectRepoBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	jobDir := filepath.Join(b.runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		return backends.SubmitResponse{}, err
	}
	result := types.Result{
		SchemaName:    "repo_inspection_v2",
		SchemaVersion: "2.0.0",
		Payload: map[string]any{
			"mode":     "evidence",
			"query":    stringValue(job.Request.TaskParams["query"]),
			"findings": []any{},
			"evidence": []any{
				map[string]any{"id": "ev_1", "path": "README.md", "source_refs": []any{map[string]any{"path": "README.md", "line_start": 1, "line_end": 1}}},
			},
			"quality": map[string]any{
				"result":       "evidence_only",
				"retrieval":    "lexical_degraded",
				"reranking":    "unavailable",
				"synthesis":    "not_requested",
				"answer_ready": false,
			},
			"warnings": []any{},
			"provenance": map[string]any{
				"index_fingerprint": "sha256:test",
			},
			"runtime": map[string]any{
				"attempts": []any{},
			},
			"retrieval": map[string]any{
				"lexical_candidates":  1,
				"semantic_candidates": 0,
				"reranked_candidates": 0,
			},
		},
	}
	resultBytes, err := json.Marshal(result)
	if err != nil {
		return backends.SubmitResponse{}, err
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), resultBytes, 0o644); err != nil {
		return backends.SubmitResponse{}, err
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		return backends.SubmitResponse{}, err
	}
	return backends.SubmitResponse{
		BackendKind:  "local",
		BackendRunID: job.ID,
		InitialState: types.JobStateDispatching,
	}, nil
}

func (b *immediateLocalInspectRepoBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
	}, nil
}

func (b *immediateLocalInspectRepoBackend) CancelRun(context.Context, string) error { return nil }

func (b *delayedLocalInspectRepoCompletionBackend) Name() string { return "local" }

func (b *delayedLocalInspectRepoCompletionBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	return backends.SubmitResponse{
		BackendKind:  "local",
		BackendRunID: job.ID,
		InitialState: types.JobStateRunning,
	}, nil
}

func (b *delayedLocalInspectRepoCompletionBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateRunning,
		RawState:     "RUNNING",
	}, nil
}

func (b *delayedLocalInspectRepoCompletionBackend) CancelRun(context.Context, string) error {
	return nil
}

func (b *delayedLocalInspectRepoResultBackend) Name() string { return "local" }

func (b *delayedLocalInspectRepoResultBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	go func() {
		time.Sleep(b.delay)
		_ = writeInspectRepoResultForTest(b.runRoot, job.ID, stringValue(job.Request.TaskParams["query"]))
	}()
	return backends.SubmitResponse{
		BackendKind:  "local",
		BackendRunID: job.ID,
		InitialState: types.JobStateRunning,
	}, nil
}

func (b *delayedLocalInspectRepoResultBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateRunning,
		RawState:     "RUNNING",
	}, nil
}

func (b *delayedLocalInspectRepoResultBackend) CancelRun(context.Context, string) error {
	return nil
}

func (b *delayedWarmQueuedLocalInspectRepoResultBackend) Name() string { return "local" }

func (b *delayedWarmQueuedLocalInspectRepoResultBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	jobDir := filepath.Join(b.runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		return backends.SubmitResponse{}, err
	}
	if err := os.WriteFile(filepath.Join(jobDir, "warm-request.marker"), []byte(job.ID+".json"), 0o644); err != nil {
		return backends.SubmitResponse{}, err
	}
	go func() {
		time.Sleep(b.delay)
		_ = writeInspectRepoResultForTest(b.runRoot, job.ID, stringValue(job.Request.TaskParams["query"]))
	}()
	return backends.SubmitResponse{
		BackendKind:  "local",
		BackendRunID: job.ID,
		InitialState: types.JobStateDispatching,
	}, nil
}

func (b *delayedWarmQueuedLocalInspectRepoResultBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateRunning,
		RawState:     "RUNNING",
	}, nil
}

func (b *delayedWarmQueuedLocalInspectRepoResultBackend) CancelRun(context.Context, string) error {
	return nil
}

func (b *countingDelayedLocalInspectRepoResultBackend) Name() string { return "local" }

func (b *countingDelayedLocalInspectRepoResultBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	b.submitCalls.Add(1)
	go func() {
		time.Sleep(b.delay)
		_ = writeInspectRepoResultForTest(b.runRoot, job.ID, stringValue(job.Request.TaskParams["query"]))
	}()
	return backends.SubmitResponse{
		BackendKind:  "local",
		BackendRunID: job.ID,
		InitialState: types.JobStateRunning,
	}, nil
}

func (b *countingDelayedLocalInspectRepoResultBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	b.getRunCalls.Add(1)
	return backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateRunning,
		RawState:     "RUNNING",
	}, nil
}

func (b *countingDelayedLocalInspectRepoResultBackend) CancelRun(context.Context, string) error {
	return nil
}

func (b *signaledLocalInspectRepoResultBackend) Name() string { return "local" }

func (b *signaledLocalInspectRepoResultBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	waiterAny, _ := b.waiters.LoadOrStore(job.ID, make(chan struct{}))
	waiter := waiterAny.(chan struct{})
	go func() {
		time.Sleep(b.delay)
		_ = writeInspectRepoResultForTest(b.runRoot, job.ID, stringValue(job.Request.TaskParams["query"]))
		close(waiter)
	}()
	return backends.SubmitResponse{
		BackendKind:  "local",
		BackendRunID: job.ID,
		InitialState: types.JobStateRunning,
	}, nil
}

func (b *signaledLocalInspectRepoResultBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	b.getRunCalls.Add(1)
	return backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateRunning,
		RawState:     "RUNNING",
	}, nil
}

func (b *signaledLocalInspectRepoResultBackend) CancelRun(context.Context, string) error {
	return nil
}

func (b *signaledLocalInspectRepoResultBackend) AwaitLocalInspectRepoResult(_ context.Context, backendRunID string, waitWindow time.Duration) bool {
	waiterAny, ok := b.waiters.Load(strings.TrimSpace(backendRunID))
	if !ok {
		return false
	}
	waiter := waiterAny.(chan struct{})
	timer := time.NewTimer(waitWindow)
	defer timer.Stop()
	select {
	case <-waiter:
		return true
	case <-timer.C:
		return false
	}
}

func writeInspectRepoResultForTest(runRoot, jobID, query string) error {
	jobDir := filepath.Join(runRoot, jobID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		return err
	}
	result := types.Result{
		SchemaName:    "repo_inspection_v2",
		SchemaVersion: "2.0.0",
		Payload: map[string]any{
			"mode":     "evidence",
			"query":    query,
			"findings": []any{},
			"evidence": []any{
				map[string]any{"id": "ev_1", "path": "README.md", "source_refs": []any{map[string]any{"path": "README.md", "line_start": 1, "line_end": 1}}},
			},
			"quality": map[string]any{
				"result":       "evidence_only",
				"retrieval":    "lexical_degraded",
				"reranking":    "unavailable",
				"synthesis":    "not_requested",
				"answer_ready": false,
			},
			"warnings": []any{},
			"provenance": map[string]any{
				"index_fingerprint": "sha256:test",
			},
			"runtime": map[string]any{
				"attempts": []any{},
			},
			"retrieval": map[string]any{
				"lexical_candidates":  1,
				"semantic_candidates": 0,
				"reranked_candidates": 0,
			},
		},
	}
	resultBytes, err := json.Marshal(result)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), resultBytes, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644)
}

type resolvingFakeBackend struct {
	status          backends.RunStatus
	resolvedProfile types.ExecutionProfile
}

func (f resolvingFakeBackend) Name() string { return "resolving-fake" }

func (f resolvingFakeBackend) ResolveExecutionProfile(context.Context, types.SubmitJobRequest) (types.ExecutionProfile, error) {
	return f.resolvedProfile, nil
}

func (f resolvingFakeBackend) SubmitRun(context.Context, types.Job) (backends.SubmitResponse, error) {
	return backends.SubmitResponse{
		BackendKind:  "resolving-fake",
		BackendRunID: "run-1",
		InitialState: types.JobStateQueued,
	}, nil
}

func (f resolvingFakeBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return f.status, nil
}

func (f resolvingFakeBackend) CancelRun(context.Context, string) error { return nil }

type fakeBatchBackend struct {
	status        backends.RunStatus
	batchCalls    int
	batchSizes    []int
	submittedJobs []types.Job
}

func (f *fakeBatchBackend) Name() string { return "fake-batch" }

func (f *fakeBatchBackend) SubmitRun(context.Context, types.Job) (backends.SubmitResponse, error) {
	return backends.SubmitResponse{
		BackendKind:  "fake-batch",
		BackendRunID: "single-run-1",
		InitialState: types.JobStateQueued,
	}, nil
}

func (f *fakeBatchBackend) SubmitRunBatch(_ context.Context, jobs []types.Job) ([]backends.SubmitResponse, error) {
	f.batchCalls++
	f.batchSizes = append(f.batchSizes, len(jobs))
	f.submittedJobs = append(f.submittedJobs, jobs...)
	responses := make([]backends.SubmitResponse, 0, len(jobs))
	for i := range jobs {
		responses = append(responses, backends.SubmitResponse{
			BackendKind:  "fake-batch",
			BackendRunID: "batch-run-" + string(rune('0'+i)),
			InitialState: types.JobStateQueued,
		})
	}
	return responses, nil
}

func (f *fakeBatchBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return f.status, nil
}

func (f *fakeBatchBackend) CancelRun(context.Context, string) error { return nil }

func newServiceThrottledBatchFixture(t *testing.T, opts Options) (*Service, *fakeBatchBackend) {
	t.Helper()
	backend := &fakeBatchBackend{
		status: backends.RunStatus{State: types.JobStateQueued, RawState: "PENDING"},
	}
	svc := NewWithAuditAndOptions(
		store.NewMemoryJobStore(),
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		t.TempDir(),
		".",
		opts,
	)
	return svc, backend
}

func newServiceRetryBudgetFixture(t *testing.T, retryBudget int) (*Service, *store.MemoryJobStore) {
	t.Helper()
	jobStore := store.NewMemoryJobStore()
	svc := NewWithAuditAndOptions(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		t.TempDir(),
		".",
		Options{RootActionMaxRetriedShards: retryBudget},
	)
	return svc, jobStore
}

func ctxAs(actor, role string) context.Context {
	return auth.WithPrincipal(context.Background(), auth.Principal{Actor: actor, Role: role})
}

func aliceUserCtx() context.Context {
	return ctxAs("alice", "user")
}

func bobUserCtx() context.Context {
	return ctxAs("bob", "user")
}

func adminCtx() context.Context {
	return ctxAs("admin", "admin")
}

func rootAdminCtx() context.Context {
	return ctxAs("root", "admin")
}

func seedFailedRetryRootJobs(t *testing.T, jobStore *store.MemoryJobStore, rootJobID, submittedBy string, count int) {
	t.Helper()
	now := time.Now().UTC()
	for i := 0; i < count; i++ {
		job := types.Job{
			ID:          "job_failed_" + string(rune('1'+i)),
			TaskType:    "document_summary",
			State:       types.JobStateFailed,
			RootJobID:   rootJobID,
			SubmittedBy: submittedBy,
			Request:     types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}},
			Orchestration: &types.OrchestrationInfo{
				RootJobID: rootJobID, Strategy: "fanout_child", ShardIndex: i, ShardCount: count,
			},
			CreatedAt:   now.Add(time.Duration(i) * time.Second),
			UpdatedAt:   now.Add(time.Duration(i) * time.Second),
			SubmittedAt: now.Add(time.Duration(i) * time.Second),
		}
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job: %v", err)
		}
	}
}

func serviceDocChildRequests(count int) []types.ParallelChildRequest {
	children := make([]types.ParallelChildRequest, 0, count)
	for i := 0; i < count; i++ {
		children = append(children, types.ParallelChildRequest{
			InputRefs:  []types.InputRef{{Type: "file", URI: "file:///tmp/doc-" + string(rune('a'+i)) + ".txt"}},
			ShardIndex: i,
			ShardCount: count,
		})
	}
	return children
}

func serviceFileChildRequests(count int) []types.ParallelChildRequest {
	children := make([]types.ParallelChildRequest, 0, count)
	for i := 0; i < count; i++ {
		children = append(children, types.ParallelChildRequest{
			InputRefs:  []types.InputRef{{Type: "file", URI: "file:///tmp/" + string(rune('a'+i)) + ".txt"}},
			ShardIndex: i,
			ShardCount: count,
		})
	}
	return children
}

func serviceRepoChildRequests(repoURI string) []types.ParallelChildRequest {
	return []types.ParallelChildRequest{
		{
			InputRefs:  []types.InputRef{{Type: "repo", URI: repoURI}},
			ShardKey:   "repo:src",
			ShardIndex: 0,
			ShardCount: 2,
		},
		{
			InputRefs:  []types.InputRef{{Type: "repo", URI: repoURI}},
			ShardKey:   "repo:tests",
			ShardIndex: 1,
			ShardCount: 2,
		},
	}
}

func loadJSONFileForTest(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read json file %s: %v", path, err)
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("decode json file %s: %v", path, err)
	}
	return payload
}

func anyStrings(values []any) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		text, _ := value.(string)
		if text != "" {
			out = append(out, text)
		}
	}
	return out
}

func inspectionEvidenceCorpus(payload map[string]any) string {
	evidence, _ := payload["evidence"].([]any)
	encoded, _ := json.Marshal(evidence)
	return string(encoded)
}

func submitAndRunInspectRepoJobForTest(t *testing.T, runRoot, repoRoot, inputRepo, query string) types.Job {
	t.Helper()

	backend := &mutableFakeBackend{}
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + inputRepo, ContentHash: "sha256:test", Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": query,
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      16000,
			PerChunkCompressionBudget: 192,
			FinalPackTokenBudget:      2048,
			RemoteModelContextBudget:  4000,
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	cmd := exec.Command(
		"python3",
		filepath.Join(repoRoot, "workers", "rag-compression", "main.py"),
		"--job-spec", filepath.Join(jobDir, "job_spec.json"),
		"--input-manifest", filepath.Join(jobDir, "input_manifest.json"),
		"--output-dir", jobDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run inspect_repo worker: %v: %s", err, string(output))
	}

	backend.status = backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
		ExitCode:     "0:0",
	}

	got, err := svc.GetJob(context.Background(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	return got
}

func makeFanoutJob(id, taskType, rootJobID string, state types.JobState, submittedAt time.Time, shardKey string, shardIndex, shardCount int) types.Job {
	return types.Job{
		ID:        id,
		TaskType:  taskType,
		State:     state,
		RootJobID: rootJobID,
		Orchestration: &types.OrchestrationInfo{
			RootJobID:  rootJobID,
			Strategy:   "fanout_child",
			ShardKey:   shardKey,
			ShardIndex: shardIndex,
			ShardCount: shardCount,
		},
		CreatedAt:   submittedAt,
		UpdatedAt:   submittedAt,
		SubmittedAt: submittedAt,
	}
}

func makeAggregatorJob(id, taskType, rootJobID, aggregationKey string, state types.JobState, submittedAt time.Time) types.Job {
	return types.Job{
		ID:        id,
		TaskType:  taskType,
		State:     state,
		RootJobID: rootJobID,
		Orchestration: &types.OrchestrationInfo{
			RootJobID:      rootJobID,
			Strategy:       "aggregator",
			AggregationKey: aggregationKey,
		},
		CreatedAt:   submittedAt,
		UpdatedAt:   submittedAt,
		SubmittedAt: submittedAt,
	}
}

func TestGetJobIngestsResultOnSuccess(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{
			status: backends.RunStatus{
				BackendRunID: "run-1",
				State:        types.JobStateSucceeded,
				RawState:     "COMPLETED",
				ExitCode:     "0:0",
			},
		},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_test",
		TaskType:     "document_summary",
		State:        types.JobStateQueued,
		BackendKind:  "fake",
		BackendRunID: "run-1",
		Request: types.SubmitJobRequest{
			TaskType:     "document_summary",
			OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "document_summary_v1",
  "schema_version": "1.0.0",
  "payload": {
    "summary": "placeholder summary"
  }
}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[
  {
    "artifact_id": "artifact_1",
    "artifact_type": "result_blob",
    "path": "result.json"
  }
]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", got.State)
	}
	if got.Result == nil || got.Result.SchemaName != "document_summary_v1" {
		t.Fatalf("expected ingested result, got %#v", got.Result)
	}
	if len(got.Artifacts) != 1 {
		t.Fatalf("expected 1 artifact, got %d", len(got.Artifacts))
	}
}

func TestSubmitJobStoresSubmittingActor(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	ctx := aliceUserCtx()
	resp, err := svc.SubmitJob(ctx, types.SubmitJobRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job, err := jobStore.GetJob(context.Background(), resp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if job.SubmittedBy != "alice" {
		t.Fatalf("expected submitted_by alice, got %q", job.SubmittedBy)
	}
}

func TestListJobsFiltersByActor(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		t.TempDir(),
		".",
	)

	now := time.Now().UTC()
	for _, job := range []types.Job{
		{ID: "job_alice", TaskType: "document_summary", State: types.JobStateQueued, SubmittedBy: "alice", CreatedAt: now, UpdatedAt: now, SubmittedAt: now},
		{ID: "job_bob", TaskType: "document_summary", State: types.JobStateQueued, SubmittedBy: "bob", CreatedAt: now, UpdatedAt: now, SubmittedAt: now.Add(time.Second)},
	} {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job: %v", err)
		}
	}

	jobs, err := svc.ListJobs(aliceUserCtx())
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if len(jobs) != 1 || jobs[0].ID != "job_alice" {
		t.Fatalf("unexpected filtered jobs: %#v", jobs)
	}

	allJobs, err := svc.ListJobs(rootAdminCtx())
	if err != nil {
		t.Fatalf("list admin jobs: %v", err)
	}
	if len(allJobs) != 2 {
		t.Fatalf("expected 2 jobs for admin, got %d", len(allJobs))
	}
}

func TestGetJobForbiddenForDifferentActor(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		t.TempDir(),
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_owned",
		TaskType:    "document_summary",
		State:       types.JobStateQueued,
		SubmittedBy: "alice",
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	if _, err := svc.GetJob(bobUserCtx(), job.ID); err == nil {
		t.Fatal("expected forbidden error")
	}
}

func TestRootScopedOperationsRequireAccessToEntireRoot(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	svc := New(jobStore, fakeBackend{}, log.New(io.Discard, "", 0), t.TempDir(), ".")

	now := time.Now().UTC()
	for _, job := range []types.Job{
		{
			ID: "job_alice_root", TaskType: "document_summary", State: types.JobStateQueued, RootJobID: "root_mixed",
			SubmittedBy: "alice",
			Request:     types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}},
			Orchestration: &types.OrchestrationInfo{
				RootJobID: "root_mixed", Strategy: "fanout_child", ShardIndex: 0, ShardCount: 2,
			},
			CreatedAt: now, UpdatedAt: now, SubmittedAt: now,
		},
		{
			ID: "job_bob_root", TaskType: "document_summary", State: types.JobStateDispatching, RootJobID: "root_mixed",
			SubmittedBy: "bob",
			Request:     types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}},
			Orchestration: &types.OrchestrationInfo{
				RootJobID: "root_mixed", Strategy: "fanout_child", ShardIndex: 1, ShardCount: 2,
			},
			CreatedAt: now.Add(time.Second), UpdatedAt: now.Add(time.Second), SubmittedAt: now.Add(time.Second),
		},
	} {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job: %v", err)
		}
	}

	if _, err := svc.GetRootJobStatus(aliceUserCtx(), "root_mixed"); err == nil {
		t.Fatal("expected forbidden root status")
	}
	if _, err := svc.RetryFailedRootShards(aliceUserCtx(), types.RetryFailedRootShardsRequest{RootJobID: "root_mixed"}); err == nil {
		t.Fatal("expected forbidden root retry")
	}
	if _, err := svc.ReleaseDeferredRootChunks(aliceUserCtx(), types.ReleaseDeferredRootChunksRequest{RootJobID: "root_mixed"}); err == nil {
		t.Fatal("expected forbidden root release")
	}
}

func TestSubmitJobWritesAuditEvent(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	auditLogger := audit.NewMemoryLogger()
	svc := NewWithAudit(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		auditLogger,
		runRoot,
		".",
	)

	ctx := aliceUserCtx()
	if _, err := svc.SubmitJob(ctx, types.SubmitJobRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	}); err != nil {
		t.Fatalf("submit job: %v", err)
	}

	if len(auditLogger.Events) == 0 {
		t.Fatal("expected audit events")
	}
	event := auditLogger.Events[len(auditLogger.Events)-1]
	if event.Action != "job.submit" || event.Actor != "alice" || event.Outcome != "success" {
		t.Fatalf("unexpected audit event: %#v", event)
	}
}

func TestSubmitJobNormalizesOrchestrationMetadata(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(context.Background()), types.SubmitJobRequest{
		TaskType: "document_summary",
		Orchestration: types.OrchestrationRequest{
			ParentJobID:    "job_parent_01",
			ShardKey:       "repo:src",
			ShardIndex:     2,
			ShardCount:     8,
			AggregationKey: "repo-pass-1",
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job, err := jobStore.GetJob(context.Background(), resp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if job.ParentJobID != "job_parent_01" {
		t.Fatalf("expected parent job id, got %q", job.ParentJobID)
	}
	if job.RootJobID != "job_parent_01" {
		t.Fatalf("expected root job id to default to parent, got %q", job.RootJobID)
	}
	if job.Orchestration == nil {
		t.Fatal("expected orchestration metadata")
	}
	if job.Orchestration.Strategy != "fanout_child" {
		t.Fatalf("expected fanout_child, got %q", job.Orchestration.Strategy)
	}
	if job.Orchestration.ShardIndex != 2 || job.Orchestration.ShardCount != 8 {
		t.Fatalf("unexpected shard metadata: %#v", job.Orchestration)
	}
}

func TestSubmitJobReturnsOpportunisticInlineReleaseFromDirectRunResultBeforeTerminalState(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	svc := New(store.NewMemoryJobStore(), &delayedLocalInspectRepoResultBackend{runRoot: runRoot, delay: 10 * time.Millisecond}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult == nil || resp.ReleasedResult.Result == nil {
		t.Fatalf("expected opportunistic inline released result from direct run files, got %#v", resp.ReleasedResult)
	}
	if resp.ReleasedResult.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded inline release, got %q", resp.ReleasedResult.State)
	}
}

func TestSubmitJobDoesNotAwaitWarmDaemonQueuedInspectRepoByDefault(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	if err := os.MkdirAll(filepath.Join(runRoot, "job_queued"), 0o755); err != nil {
		t.Fatalf("mkdir queued run dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runRoot, "job_queued", "warm-request.marker"), []byte("job_queued.json"), 0o644); err != nil {
		t.Fatalf("write warm marker: %v", err)
	}
	svc := New(store.NewMemoryJobStore(), fakeBackend{}, log.New(io.Discard, "", 0), runRoot, repoRoot)
	job := types.Job{
		ID:           "job_queued",
		TaskType:     "inspect_repo",
		BackendKind:  "local",
		BackendRunID: "job_queued",
	}
	if svc.shouldOpportunisticallyAwaitDirectWorkerInspectRepoRelease(job) {
		t.Fatal("expected warm-daemon queued inspect_repo to skip opportunistic direct-worker await")
	}
}

func TestSubmitJobReturnsOpportunisticInlineReleaseForWarmQueuedInspectRepoCompletionByDefault(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	svc := New(store.NewMemoryJobStore(), &delayedWarmQueuedLocalInspectRepoResultBackend{runRoot: runRoot, delay: 10 * time.Millisecond}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult == nil || resp.ReleasedResult.Result == nil {
		t.Fatalf("expected opportunistic inline released result for warm queued run, got %#v", resp.ReleasedResult)
	}
	if resp.ReleasedResult.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded inline release, got %q", resp.ReleasedResult.State)
	}
}

func TestSubmitJobWarmQueuedInspectRepoSlowCompletionStillReturnsPendingResponse(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	svc := New(store.NewMemoryJobStore(), &delayedWarmQueuedLocalInspectRepoResultBackend{runRoot: runRoot, delay: 250 * time.Millisecond}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	started := time.Now()
	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult != nil {
		t.Fatalf("expected pending submit response for slow warm queued completion, got %#v", resp.ReleasedResult)
	}
	if elapsed := time.Since(started); elapsed > 150*time.Millisecond {
		t.Fatalf("expected slow warm queued pending response to return within 150ms, got %v", elapsed)
	}
}

func TestSubmitJobDirectWorkerSlowInspectRepoReturnsPendingQuicklyByDefault(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	svc := New(store.NewMemoryJobStore(), &delayedLocalInspectRepoResultBackend{runRoot: runRoot, delay: 250 * time.Millisecond}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	started := time.Now()
	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult != nil {
		t.Fatalf("expected pending submit response for slow direct-worker completion, got %#v", resp.ReleasedResult)
	}
	if elapsed := time.Since(started); elapsed > 150*time.Millisecond {
		t.Fatalf("expected slow direct-worker pending response to return within 150ms, got %v", elapsed)
	}
}

func TestSubmitParallelJobsCreatesSharedRootChildren(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	resp, err := svc.SubmitParallelJobs(context.Background(), types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children: []types.ParallelChildRequest{
			{InputRefs: []types.InputRef{{Type: "file", URI: "file:///tmp/a.txt"}}, ShardKey: "repo:a", ShardIndex: 0, ShardCount: 2},
			{InputRefs: []types.InputRef{{Type: "file", URI: "file:///tmp/b.txt"}}, ShardKey: "repo:b", ShardIndex: 1, ShardCount: 2},
		},
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}
	if resp.RootJobID == "" {
		t.Fatal("expected root job id")
	}
	if resp.ChildCount != 2 || len(resp.Children) != 2 {
		t.Fatalf("unexpected child count: %#v", resp)
	}

	for _, child := range resp.Children {
		job, err := jobStore.GetJob(context.Background(), child.JobID)
		if err != nil {
			t.Fatalf("get child job: %v", err)
		}
		if job.RootJobID != resp.RootJobID {
			t.Fatalf("expected shared root job id %q, got %q", resp.RootJobID, job.RootJobID)
		}
		if job.Orchestration == nil || job.Orchestration.Strategy != "fanout_child" {
			t.Fatalf("expected fanout_child orchestration, got %#v", job.Orchestration)
		}
	}
}

func TestSubmitParallelJobsUsesBatchBackendForUncachedChildren(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	backend := &fakeBatchBackend{}
	svc := New(jobStore, backend, log.New(io.Discard, "", 0), runRoot, ".")

	resp, err := svc.SubmitParallelJobs(context.Background(), types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     serviceFileChildRequests(2),
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}
	if backend.batchCalls != 1 {
		t.Fatalf("expected 1 batch call, got %d", backend.batchCalls)
	}
	if len(backend.submittedJobs) != 2 {
		t.Fatalf("expected 2 submitted jobs, got %d", len(backend.submittedJobs))
	}
	for i, child := range resp.Children {
		job, err := jobStore.GetJob(context.Background(), child.JobID)
		if err != nil {
			t.Fatalf("get child job: %v", err)
		}
		if job.BackendRunID != "batch-run-0" && job.BackendRunID != "batch-run-1" {
			t.Fatalf("expected chunk-local batch run id, got %q for child %d", job.BackendRunID, i)
		}
	}
}

func TestSubmitParallelJobsChunksLargeBatchSubmission(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	backend := &fakeBatchBackend{}
	svc := NewWithAuditAndOptions(jobStore, backend, log.New(io.Discard, "", 0), audit.NewNopLogger(), runRoot, ".", Options{
		ParallelMaxBatchSize: 2,
	})

	children := serviceDocChildRequests(5)

	resp, err := svc.SubmitParallelJobs(context.Background(), types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     children,
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}
	if backend.batchCalls != 2 {
		t.Fatalf("expected 2 batch calls for chunked submission, got %d", backend.batchCalls)
	}
	if len(backend.batchSizes) != 2 || backend.batchSizes[0] != 2 || backend.batchSizes[1] != 2 {
		t.Fatalf("expected batch sizes [2 2], got %#v", backend.batchSizes)
	}
	if len(backend.submittedJobs) != 4 {
		t.Fatalf("expected 4 batched jobs, got %d", len(backend.submittedJobs))
	}
	if len(resp.Children) != 5 {
		t.Fatalf("expected 5 children, got %#v", resp)
	}
	lastJob, err := jobStore.GetJob(context.Background(), resp.Children[4].JobID)
	if err != nil {
		t.Fatalf("get last child job: %v", err)
	}
	if lastJob.BackendRunID != "single-run-1" {
		t.Fatalf("expected single-run fallback for final singleton chunk, got %q", lastJob.BackendRunID)
	}
}

func TestSubmitParallelJobsThrottlesActiveRootBatchesAndDeferredReducer(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	backend := &fakeBatchBackend{
		status: backends.RunStatus{State: types.JobStateQueued, RawState: "PENDING"},
	}
	svc := NewWithAuditAndOptions(jobStore, backend, log.New(io.Discard, "", 0), audit.NewNopLogger(), runRoot, ".", Options{
		ParallelMaxBatchSize:     2,
		ParallelMaxActiveBatches: 1,
	})

	children := make([]types.ParallelChildRequest, 0, 5)
	for i := 0; i < 5; i++ {
		children = append(children, types.ParallelChildRequest{
			InputRefs:  []types.InputRef{{Type: "file", URI: "file:///tmp/doc-" + string(rune('a'+i)) + ".txt"}},
			ShardIndex: i,
			ShardCount: 5,
		})
	}

	resp, err := svc.SubmitParallelJobs(context.Background(), types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     children,
		Reducer: &types.ParallelReducerRequest{
			TaskType:     "document_summary",
			OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		},
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}
	if backend.batchCalls != 1 || len(backend.batchSizes) != 1 || backend.batchSizes[0] != 2 {
		t.Fatalf("expected only first chunk to submit immediately, got calls=%d sizes=%#v", backend.batchCalls, backend.batchSizes)
	}
	if resp.ReducerJob == nil || resp.ReducerJob.State != types.JobStateDispatching {
		t.Fatalf("expected deferred reducer placeholder, got %#v", resp.ReducerJob)
	}
	if resp.Children[0].State != types.JobStateQueued || resp.Children[1].State != types.JobStateQueued {
		t.Fatalf("expected first chunk queued, got %#v", resp.Children[:2])
	}
	if resp.Children[2].State != types.JobStateDispatching || resp.Children[4].State != types.JobStateDispatching {
		t.Fatalf("expected later chunks dispatching, got %#v", resp.Children)
	}
	root0, err := svc.GetRootJobStatus(context.Background(), resp.RootJobID)
	if err != nil {
		t.Fatalf("get initial root status: %v", err)
	}
	if root0.DispatchingChildren != 3 || root0.PendingChildren != 3 || root0.ActiveChunks != 1 || root0.PendingChunks != 2 || !root0.ReducerDeferred {
		t.Fatalf("unexpected initial throttling metrics: %#v", root0)
	}

	backend.status = backends.RunStatus{State: types.JobStateSucceeded, RawState: "COMPLETED", ExitCode: "0:0"}
	if _, err := svc.GetRootJobStatus(context.Background(), resp.RootJobID); err != nil {
		t.Fatalf("get root status release 2nd chunk: %v", err)
	}
	if backend.batchCalls != 2 || len(backend.batchSizes) != 2 || backend.batchSizes[1] != 2 {
		t.Fatalf("expected second chunk release, got calls=%d sizes=%#v", backend.batchCalls, backend.batchSizes)
	}

	if _, err := svc.GetRootJobStatus(context.Background(), resp.RootJobID); err != nil {
		t.Fatalf("get root status release singleton chunk: %v", err)
	}
	lastJob, err := jobStore.GetJob(context.Background(), resp.Children[4].JobID)
	if err != nil {
		t.Fatalf("get last child job: %v", err)
	}
	if lastJob.BackendRunID != "single-run-1" || lastJob.State != types.JobStateQueued {
		t.Fatalf("expected singleton chunk dispatch after slots freed, got %#v", lastJob)
	}
	root2, err := svc.GetRootJobStatus(context.Background(), resp.RootJobID)
	if err != nil {
		t.Fatalf("get root status after singleton dispatch: %v", err)
	}
	if root2.PendingChildren != 0 || root2.ActiveChunks != 0 || root2.PendingChunks != 0 || root2.ReducerDeferred {
		t.Fatalf("unexpected post-dispatch throttling metrics: %#v", root2)
	}

	if _, err := svc.GetRootJobStatus(context.Background(), resp.RootJobID); err != nil {
		t.Fatalf("get root status release reducer: %v", err)
	}
	reducerJob, err := jobStore.GetJob(context.Background(), resp.ReducerJob.JobID)
	if err != nil {
		t.Fatalf("get reducer job: %v", err)
	}
	if reducerJob.BackendRunID == "" || reducerJob.State == types.JobStateDispatching {
		t.Fatalf("expected deferred reducer placeholder to submit in place, got %#v", reducerJob)
	}
	root3, err := svc.GetRootJobStatus(context.Background(), resp.RootJobID)
	if err != nil {
		t.Fatalf("get root status after reducer submit: %v", err)
	}
	if root3.ReducerDeferred {
		t.Fatalf("expected reducer_deferred=false after reducer submission, got %#v", root3)
	}
}

func TestReleaseDeferredRootChunksForcesAdditionalChunkRelease(t *testing.T) {
	svc, backend := newServiceThrottledBatchFixture(t, Options{
		ParallelMaxBatchSize:     2,
		ParallelMaxActiveBatches: 1,
	})

	children := make([]types.ParallelChildRequest, 0, 5)
	for i := 0; i < 5; i++ {
		children = append(children, types.ParallelChildRequest{
			InputRefs:  []types.InputRef{{Type: "file", URI: "file:///tmp/doc-" + string(rune('a'+i)) + ".txt"}},
			ShardIndex: i,
			ShardCount: 5,
		})
	}

	resp, err := svc.SubmitParallelJobs(context.Background(), types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     children,
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}
	if backend.batchCalls != 1 {
		t.Fatalf("expected initial throttled submission, got %d", backend.batchCalls)
	}

	release, err := svc.ReleaseDeferredRootChunks(context.Background(), types.ReleaseDeferredRootChunksRequest{
		RootJobID:            resp.RootJobID,
		MaxAdditionalBatches: 1,
	})
	if err != nil {
		t.Fatalf("release deferred root chunks: %v", err)
	}
	if release.ReleasedChunks != 1 || release.ReleasedChildren != 2 {
		t.Fatalf("expected one extra chunk release, got %#v", release)
	}
	if release.CumulativeForcedReleaseChunks != 1 || release.RemainingForcedReleaseBudget != 0 {
		t.Fatalf("expected direct forced-release counters, got %#v", release)
	}
	if backend.batchCalls != 2 || len(backend.batchSizes) != 2 || backend.batchSizes[1] != 2 {
		t.Fatalf("expected second batch submission via release action, got calls=%d sizes=%#v", backend.batchCalls, backend.batchSizes)
	}
	if release.RootStatus.PendingChunks != 1 || release.RootStatus.ActiveChunks != 2 {
		t.Fatalf("expected updated root throttling metrics after forced release, got %#v", release.RootStatus)
	}
}

func TestSubmitParallelJobsAttachesInspectRepoFingerprintHintToEligibleChildren(t *testing.T) {
	svc, backend := newServiceThrottledBatchFixture(t, Options{})
	repoA := t.TempDir()
	repoB := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoA, "README.md"), []byte("# repo a\n"), 0o644); err != nil {
		t.Fatalf("write repo a input: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoB, "README.md"), []byte("# repo b\n"), 0o644); err != nil {
		t.Fatalf("write repo b input: %v", err)
	}

	resp, err := svc.SubmitParallelJobs(aliceUserCtx(), types.SubmitParallelJobsRequest{
		TaskType:     "inspect_repo",
		TaskParams:   map[string]any{"query": "trace routing", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		Children: []types.ParallelChildRequest{
			{
				InputRefs:  []types.InputRef{{Type: "repo", URI: "file://" + repoA, Classification: "internal"}},
				ShardIndex: 0,
				ShardCount: 2,
			},
			{
				InputRefs:  []types.InputRef{{Type: "repo", URI: "file://" + repoB, Classification: "internal"}},
				ShardIndex: 1,
				ShardCount: 2,
			},
		},
	})
	if err != nil {
		t.Fatalf("submit parallel inspect_repo: %v", err)
	}
	if len(resp.Children) != 2 {
		t.Fatalf("expected 2 child submissions, got %#v", resp)
	}
	if backend.batchCalls != 1 {
		t.Fatalf("expected one batch backend submission, got %d", backend.batchCalls)
	}
	if len(backend.submittedJobs) != 2 {
		t.Fatalf("expected 2 submitted child jobs, got %d", len(backend.submittedJobs))
	}
	for _, job := range backend.submittedJobs {
		if got := stringValue(job.Request.TaskParams["_broker_repository_state_fingerprint"]); got != "" {
			t.Fatalf("expected child inspect_repo request to omit broker repository fingerprint hint: %#v", job.Request.TaskParams)
		}
	}
}

func TestSubmitParallelJobsOmitsInspectRepoFingerprintHintForExcludedChildren(t *testing.T) {
	svc, backend := newServiceThrottledBatchFixture(t, Options{})
	repoA := t.TempDir()
	repoB := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoA, "README.md"), []byte("# repo a\n"), 0o644); err != nil {
		t.Fatalf("write repo a input: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoB, "README.md"), []byte("# repo b\n"), 0o644); err != nil {
		t.Fatalf("write repo b input: %v", err)
	}

	_, err := svc.SubmitParallelJobs(aliceUserCtx(), types.SubmitParallelJobsRequest{
		TaskType:     "inspect_repo",
		TaskParams:   map[string]any{"query": "trace routing", "mode": "evidence", "excluded_dir_names": []string{"vendor"}},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		Children: []types.ParallelChildRequest{
			{
				InputRefs:  []types.InputRef{{Type: "repo", URI: "file://" + repoA, Classification: "internal"}},
				ShardIndex: 0,
				ShardCount: 2,
			},
			{
				InputRefs:  []types.InputRef{{Type: "repo", URI: "file://" + repoB, Classification: "internal"}},
				ShardIndex: 1,
				ShardCount: 2,
			},
		},
	})
	if err != nil {
		t.Fatalf("submit parallel inspect_repo with exclusions: %v", err)
	}
	if backend.batchCalls != 1 {
		t.Fatalf("expected one batch backend submission, got %d", backend.batchCalls)
	}
	if len(backend.submittedJobs) != 2 {
		t.Fatalf("expected 2 submitted child jobs, got %d", len(backend.submittedJobs))
	}
	for _, job := range backend.submittedJobs {
		if got := stringValue(job.Request.TaskParams["_broker_repository_state_fingerprint"]); got != "" {
			t.Fatalf("expected excluded child inspect_repo request to omit broker repository fingerprint hint: %#v", job.Request.TaskParams)
		}
	}
}

func TestReleaseDeferredRootChunksRejectsNonAdminRequestAboveCap(t *testing.T) {
	svc, _ := newServiceThrottledBatchFixture(t, Options{
		ParallelMaxBatchSize:           2,
		ParallelMaxActiveBatches:       1,
		RootActionMaxAdditionalBatches: 1,
	})

	resp, err := svc.SubmitParallelJobs(aliceUserCtx(), types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     serviceFileChildRequests(4),
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}

	_, err = svc.ReleaseDeferredRootChunks(aliceUserCtx(), types.ReleaseDeferredRootChunksRequest{
		RootJobID:            resp.RootJobID,
		MaxAdditionalBatches: 2,
	})
	if err == nil {
		t.Fatal("expected forbidden release over cap")
	}
}

func TestReleaseDeferredRootChunksAllowsAdminRequestAboveCap(t *testing.T) {
	svc, _ := newServiceThrottledBatchFixture(t, Options{
		ParallelMaxBatchSize:           2,
		ParallelMaxActiveBatches:       1,
		RootActionMaxAdditionalBatches: 1,
	})

	resp, err := svc.SubmitParallelJobs(adminCtx(), types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     serviceFileChildRequests(4),
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}

	release, err := svc.ReleaseDeferredRootChunks(adminCtx(), types.ReleaseDeferredRootChunksRequest{
		RootJobID:            resp.RootJobID,
		MaxAdditionalBatches: 2,
	})
	if err != nil {
		t.Fatalf("admin release deferred root chunks: %v", err)
	}
	if release.ReleasedChunks < 1 {
		t.Fatalf("expected admin release to succeed, got %#v", release)
	}
}

func TestReleaseDeferredRootChunksRejectsCumulativeNonAdminEscalation(t *testing.T) {
	svc, _ := newServiceThrottledBatchFixture(t, Options{
		ParallelMaxBatchSize:           2,
		ParallelMaxActiveBatches:       1,
		RootActionMaxAdditionalBatches: 2,
	})

	resp, err := svc.SubmitParallelJobs(aliceUserCtx(), types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     serviceFileChildRequests(8),
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}

	for range 2 {
		if _, err := svc.ReleaseDeferredRootChunks(aliceUserCtx(), types.ReleaseDeferredRootChunksRequest{
			RootJobID:            resp.RootJobID,
			MaxAdditionalBatches: 1,
		}); err != nil {
			t.Fatalf("expected cumulative release within cap: %v", err)
		}
	}
	if _, err := svc.ReleaseDeferredRootChunks(aliceUserCtx(), types.ReleaseDeferredRootChunksRequest{
		RootJobID:            resp.RootJobID,
		MaxAdditionalBatches: 1,
	}); err == nil {
		t.Fatal("expected cumulative forced release over cap to be forbidden")
	}
}

func TestRetryFailedRootShardsRejectsNonAdminRequestAboveCap(t *testing.T) {
	svc, jobStore := newServiceRetryBudgetFixture(t, 1)
	seedFailedRetryRootJobs(t, jobStore, "root_retry_cap", "alice", 2)

	_, err := svc.RetryFailedRootShards(aliceUserCtx(), types.RetryFailedRootShardsRequest{
		RootJobID: "root_retry_cap",
	})
	if err == nil {
		t.Fatal("expected forbidden retry over cap")
	}
}

func TestRetryFailedRootShardsAllowsAdminRequestAboveCap(t *testing.T) {
	svc, jobStore := newServiceRetryBudgetFixture(t, 1)
	seedFailedRetryRootJobs(t, jobStore, "root_retry_admin", "admin", 2)

	resp, err := svc.RetryFailedRootShards(adminCtx(), types.RetryFailedRootShardsRequest{
		RootJobID: "root_retry_admin",
	})
	if err != nil {
		t.Fatalf("admin retry failed root shards: %v", err)
	}
	if resp.RetriedCount != 2 {
		t.Fatalf("expected admin retry to succeed for both shards, got %#v", resp)
	}
}

func TestRetryFailedRootShardsRejectsCumulativeNonAdminEscalation(t *testing.T) {
	svc, jobStore := newServiceRetryBudgetFixture(t, 1)
	seedFailedRetryRootJobs(t, jobStore, "root_retry_cumulative", "alice", 1)

	userCtx := aliceUserCtx()
	first, err := svc.RetryFailedRootShards(userCtx, types.RetryFailedRootShardsRequest{RootJobID: "root_retry_cumulative"})
	if err != nil {
		t.Fatalf("first retry failed root shards: %v", err)
	}
	if first.RetriedCount != 1 {
		t.Fatalf("expected one retried shard, got %#v", first)
	}
	if first.CumulativeRetriedShards != 1 || first.RemainingRetriedShardBudget != 0 {
		t.Fatalf("expected direct retry counters, got %#v", first)
	}

	retriedJob, err := jobStore.GetJob(context.Background(), first.RetriedShards[0].JobID)
	if err != nil {
		t.Fatalf("get retried job: %v", err)
	}
	retriedJob.State = types.JobStateFailed
	retriedJob.BackendRunID = ""
	if err := jobStore.UpdateJob(context.Background(), retriedJob); err != nil {
		t.Fatalf("mark retried job failed again: %v", err)
	}

	if _, err := svc.RetryFailedRootShards(userCtx, types.RetryFailedRootShardsRequest{RootJobID: "root_retry_cumulative"}); err == nil {
		t.Fatal("expected cumulative retry over cap to be forbidden")
	}
}

func TestSubmitParallelJobsWithReducerCompletesLocally(t *testing.T) {
	runRoot, err := os.MkdirTemp("", "broker-runroot-*")
	if err != nil {
		t.Fatalf("mkdir run root: %v", err)
	}
	t.Cleanup(func() { removeAllRetry(runRoot) })
	repoRoot, err := os.MkdirTemp("", "broker-repo-*")
	if err != nil {
		t.Fatalf("mkdir repo root: %v", err)
	}
	t.Cleanup(func() { removeAllRetry(repoRoot) })
	if err := os.MkdirAll(filepath.Join(repoRoot, "src"), 0o755); err != nil {
		t.Fatalf("mkdir src: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(repoRoot, "tests"), 0o755); err != nil {
		t.Fatalf("mkdir tests: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "src", "main.py"), []byte("print('hello')\n"), 0o644); err != nil {
		t.Fatalf("write src file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "tests", "test_main.py"), []byte("def test_ok():\n    assert True\n"), 0o644); err != nil {
		t.Fatalf("write test file: %v", err)
	}

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("get wd: %v", err)
	}
	projectRoot := filepath.Clean(filepath.Join(wd, "..", "..", ".."))
	jobStore := store.NewMemoryJobStore()
	backend := localbackend.NewBackend(config.Config{
		LocalMode:       "command",
		LocalScriptPath: filepath.Join(projectRoot, "deploy", "local", "broker_worker.sh"),
		RunRootPath:     runRoot,
		RepoRootPath:    projectRoot,
	})
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		projectRoot,
	)

	resp, err := svc.SubmitParallelJobs(context.Background(), types.SubmitParallelJobsRequest{
		TaskType:     "repo_summary",
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
		Children:     serviceRepoChildRequests("file://" + repoRoot),
		Reducer: &types.ParallelReducerRequest{
			TaskType:     "repo_summary",
			OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
			TaskParams: map[string]any{
				"aggregate_wait_seconds": 15,
			},
		},
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}
	if resp.ReducerJob == nil {
		t.Fatal("expected reducer job")
	}

	deadline := time.Now().Add(20 * time.Second)
	for {
		job, err := svc.GetJob(context.Background(), resp.ReducerJob.JobID)
		if err != nil {
			t.Fatalf("get reducer job: %v", err)
		}
		if job.State == types.JobStateSucceeded {
			if job.Result == nil {
				t.Fatal("expected reducer result")
			}
			payload := job.Result.Payload
			metrics, _ := payload["aggregate_metrics"].(map[string]any)
			if metrics == nil {
				t.Fatalf("expected aggregate_metrics, got %#v", payload)
			}
			return
		}
		if job.State == types.JobStateFailed || job.State == types.JobStateCancelled || job.State == types.JobStateTimedOut {
			t.Fatalf("reducer ended unexpectedly: state=%s error=%s", job.State, job.ResultError)
		}
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for reducer completion")
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func TestGetRootJobStatusAggregatesChildrenAndReducer(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		t.TempDir(),
		".",
	)

	now := time.Now().UTC()
	for _, job := range []types.Job{
		makeFanoutJob("job_child_1", "repo_summary", "root_1", types.JobStateSucceeded, now, "", 0, 2),
		makeFanoutJob("job_child_2", "repo_summary", "root_1", types.JobStateSucceeded, now, "", 1, 2),
		func() types.Job {
			job := makeAggregatorJob("job_reduce", "repo_summary", "root_1", "repo-pass-1", types.JobStateRunning, now)
			job.Result = &types.Result{
				SchemaName:    "repo_summary_v1",
				SchemaVersion: "1.0.0",
				Payload: map[string]any{
					"summary": "agg",
					"aggregate_metrics": map[string]any{
						"children_total":     2,
						"children_succeeded": 2,
						"children_failed":    0,
						"coverage_fraction":  1.0,
					},
				},
			}
			return job
		}(),
	} {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job: %v", err)
		}
	}

	status, err := svc.GetRootJobStatus(context.Background(), "root_1")
	if err != nil {
		t.Fatalf("get root status: %v", err)
	}
	if status.TotalJobs != 3 || status.ReducerJobID != "job_reduce" {
		t.Fatalf("unexpected root status: %#v", status)
	}
	if status.State != types.JobStateRunning {
		t.Fatalf("expected running root state, got %q", status.State)
	}
	if len(status.ChildJobIDs) != 2 {
		t.Fatalf("expected 2 child job ids, got %#v", status.ChildJobIDs)
	}
	if status.ChildrenTotal != 2 || status.ChildrenSucceeded != 2 || status.CoverageFraction != 1.0 {
		t.Fatalf("expected reducer metrics on root status, got %#v", status)
	}
	if status.DispatchingChildren != 0 || status.PendingChildren != 0 || status.ActiveChunks != 0 || status.PendingChunks != 0 || status.ReducerDeferred {
		t.Fatalf("expected zero dispatch throttling metrics, got %#v", status)
	}
}

func TestGetRootJobStatusUsesEffectiveLatestShardAttempts(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	svc := New(jobStore, fakeBackend{}, log.New(io.Discard, "", 0), t.TempDir(), ".")

	now := time.Now().UTC()
	jobs := []types.Job{
		makeFanoutJob("job_child_failed", "repo_summary", "root_retry", types.JobStateFailed, now.Add(-2*time.Minute), "repo:src", 0, 2),
		makeFanoutJob("job_child_retry", "repo_summary", "root_retry", types.JobStateSucceeded, now.Add(-1*time.Minute), "repo:src", 0, 2),
		makeFanoutJob("job_child_2", "repo_summary", "root_retry", types.JobStateSucceeded, now, "repo:test", 1, 2),
		func() types.Job {
			job := makeAggregatorJob("job_reduce", "repo_summary", "root_retry", "repo-pass-1", types.JobStateSucceeded, now.Add(30*time.Second))
			job.Result = &types.Result{
				SchemaName: "repo_summary_v1", SchemaVersion: "1.0.0",
				Payload: map[string]any{"aggregate_metrics": map[string]any{
					"children_total": 2, "children_succeeded": 2, "children_failed": 0, "coverage_fraction": 1.0,
				}},
			}
			return job
		}(),
	}
	for _, job := range jobs {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job: %v", err)
		}
	}

	status, err := svc.GetRootJobStatus(context.Background(), "root_retry")
	if err != nil {
		t.Fatalf("get root status: %v", err)
	}
	if status.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded root state, got %#v", status)
	}
	if status.FailedJobs != 0 || status.SucceededJobs != 3 {
		t.Fatalf("expected only effective succeeded jobs to count, got %#v", status)
	}
	if len(status.ChildJobIDs) != 2 {
		t.Fatalf("expected 2 effective child jobs, got %#v", status.ChildJobIDs)
	}
}

func TestRetryFailedRootShardsRetriesOnlyFailedLatestChildrenAndResubmitsReducer(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(jobStore, fakeBackend{}, log.New(io.Discard, "", 0), runRoot, ".")

	now := time.Now().UTC()
	jobs := []types.Job{
		func() types.Job {
			job := makeFanoutJob("job_child_ok", "document_summary", "root_retry_2", types.JobStateSucceeded, now.Add(-2*time.Minute), "doc:a", 0, 2)
			job.Request = types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}}
			return job
		}(),
		func() types.Job {
			job := makeFanoutJob("job_child_failed", "document_summary", "root_retry_2", types.JobStateFailed, now.Add(-1*time.Minute), "doc:b", 1, 2)
			job.Request = types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}}
			return job
		}(),
		func() types.Job {
			job := makeAggregatorJob("job_reduce_failed", "document_summary", "root_retry_2", "aggregate", types.JobStateFailed, now)
			job.Request = types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}}
			return job
		}(),
	}
	for _, job := range jobs {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job: %v", err)
		}
	}

	resp, err := svc.RetryFailedRootShards(context.Background(), types.RetryFailedRootShardsRequest{
		RootJobID:       "root_retry_2",
		ResubmitReducer: true,
	})
	if err != nil {
		t.Fatalf("retry failed root shards: %v", err)
	}
	if resp.RetriedCount != 1 || len(resp.RetriedShards) != 1 {
		t.Fatalf("expected one retried shard, got %#v", resp)
	}
	if resp.RetriedShards[0].PreviousJobID != "job_child_failed" {
		t.Fatalf("expected failed shard to be retried, got %#v", resp.RetriedShards)
	}
	if resp.ReducerJob == nil {
		t.Fatalf("expected reducer resubmission, got %#v", resp)
	}
	if resp.SkippedCount != 1 || len(resp.SkippedShards) != 1 || resp.SkippedShards[0].Reason != "already_succeeded" {
		t.Fatalf("expected succeeded shard skip, got %#v", resp)
	}

	root, err := svc.GetRootJobStatus(context.Background(), "root_retry_2")
	if err != nil {
		t.Fatalf("get root status: %v", err)
	}
	if root.State != types.JobStateQueued {
		t.Fatalf("expected queued root state after retry/resubmission, got %#v", root)
	}
	if root.FailedJobs != 0 {
		t.Fatalf("expected stale failed attempts to be excluded from effective root state, got %#v", root)
	}
}

func TestSubmitParallelLogAnalysisWithReducerCompletesLocally(t *testing.T) {
	runRoot, err := os.MkdirTemp("", "broker-log-runroot-*")
	if err != nil {
		t.Fatalf("mkdir run root: %v", err)
	}
	t.Cleanup(func() { removeAllRetry(runRoot) })
	repoRoot, err := os.MkdirTemp("", "broker-log-repo-*")
	if err != nil {
		t.Fatalf("mkdir repo root: %v", err)
	}
	t.Cleanup(func() { removeAllRetry(repoRoot) })

	logA := filepath.Join(repoRoot, "a.log")
	if err := os.WriteFile(logA, []byte("fatal error: generated header missing\n"), 0o644); err != nil {
		t.Fatalf("write logA: %v", err)
	}
	logBMissing := filepath.Join(repoRoot, "missing.log")

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("get wd: %v", err)
	}
	projectRoot := filepath.Clean(filepath.Join(wd, "..", "..", ".."))
	jobStore := store.NewMemoryJobStore()
	backend := localbackend.NewBackend(config.Config{
		LocalMode:       "command",
		LocalScriptPath: filepath.Join(projectRoot, "deploy", "local", "broker_worker.sh"),
		RunRootPath:     runRoot,
		RepoRootPath:    projectRoot,
	})
	svc := New(jobStore, backend, log.New(io.Discard, "", 0), runRoot, projectRoot)

	resp, err := svc.SubmitParallelJobs(context.Background(), types.SubmitParallelJobsRequest{
		TaskType:     "log_analysis",
		OutputSchema: types.OutputSchemaRef{Name: "log_analysis_v1"},
		Children: []types.ParallelChildRequest{
			{InputRefs: []types.InputRef{{Type: "file", URI: "file://" + logA}}, ShardIndex: 0, ShardCount: 2},
			{InputRefs: []types.InputRef{{Type: "file", URI: "file://" + logBMissing}}, ShardIndex: 1, ShardCount: 2},
		},
		Reducer: &types.ParallelReducerRequest{
			TaskType:     "log_analysis",
			OutputSchema: types.OutputSchemaRef{Name: "log_analysis_v1"},
			TaskParams: map[string]any{
				"aggregate_wait_seconds": 15,
			},
		},
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}
	if resp.ReducerJob == nil {
		t.Fatal("expected reducer job")
	}

	deadline := time.Now().Add(20 * time.Second)
	for {
		job, err := svc.GetJob(context.Background(), resp.ReducerJob.JobID)
		if err != nil {
			t.Fatalf("get reducer job: %v", err)
		}
		if job.State == types.JobStateSucceeded {
			if job.Result == nil {
				t.Fatal("expected reducer result")
			}
			findings, _ := job.Result.Payload["top_findings"].([]any)
			if len(findings) == 0 {
				t.Fatalf("expected merged findings, got %#v", job.Result.Payload)
			}
			root, err := svc.GetRootJobStatus(context.Background(), resp.RootJobID)
			if err != nil {
				t.Fatalf("get root status: %v", err)
			}
			if root.CoverageFraction >= 1.0 || root.ChildrenSucceeded != 1 || root.ChildrenFailed != 1 {
				t.Fatalf("unexpected root metrics: %#v", root)
			}
			warnings, _ := job.Result.Payload["warnings"].([]any)
			if len(warnings) == 0 {
				t.Fatalf("expected partial-reduce warning, got %#v", job.Result.Payload)
			}
			return
		}
		if job.State == types.JobStateFailed || job.State == types.JobStateCancelled || job.State == types.JobStateTimedOut {
			t.Fatalf("reducer ended unexpectedly: state=%s error=%s", job.State, job.ResultError)
		}
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for reducer completion")
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func removeAllRetry(path string) {
	for range 20 {
		if err := os.RemoveAll(path); err == nil || errors.Is(err, os.ErrNotExist) {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	_ = os.RemoveAll(path)
}

func TestSubmitJobRunsToCompletionWithLocalBackend(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("get wd: %v", err)
	}
	repoRoot := filepath.Clean(filepath.Join(wd, "..", "..", ".."))
	backend := localbackend.NewBackend(config.Config{
		LocalMode:       "command",
		LocalScriptPath: filepath.Join(repoRoot, "deploy", "local", "broker_worker.sh"),
		RunRootPath:     runRoot,
		RepoRootPath:    repoRoot,
	})
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	inputPath := filepath.Join(t.TempDir(), "doc.txt")
	if err := os.WriteFile(inputPath, []byte("Local backend validation document.\nThis should complete through the real worker.\n"), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(context.Background()), types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + inputPath},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	deadline := time.Now().Add(15 * time.Second)
	for {
		job, err := svc.GetJob(context.Background(), resp.JobID)
		if err != nil {
			t.Fatalf("get job: %v", err)
		}
		if job.State == types.JobStateSucceeded {
			if job.Result == nil || job.Result.SchemaName != "document_summary_v1" {
				t.Fatalf("expected document_summary_v1 result, got %#v", job.Result)
			}
			return
		}
		if job.State == types.JobStateFailed || job.State == types.JobStateCancelled || job.State == types.JobStateTimedOut {
			t.Fatalf("job ended unexpectedly: state=%s error=%s", job.State, job.ResultError)
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for local backend job completion")
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func TestForbiddenGetJobWritesAuditEvent(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	auditLogger := audit.NewMemoryLogger()
	svc := NewWithAudit(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		auditLogger,
		t.TempDir(),
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_owned",
		TaskType:    "document_summary",
		State:       types.JobStateQueued,
		SubmittedBy: "alice",
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	if _, err := svc.GetJob(bobUserCtx(), job.ID); err == nil {
		t.Fatal("expected forbidden error")
	}

	if len(auditLogger.Events) == 0 {
		t.Fatal("expected audit events")
	}
	event := auditLogger.Events[len(auditLogger.Events)-1]
	if event.Action != "job.get_status" || event.Outcome != "forbidden" || event.JobID != job.ID {
		t.Fatalf("unexpected audit event: %#v", event)
	}
}

func TestSubmitJobStagesExecutionBundle(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		"/repo/root",
	)

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(context.Background()), types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file:///tmp/example.txt"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, resp.JobID)
	if _, err := os.Stat(filepath.Join(jobDir, "job_spec.json")); err != nil {
		t.Fatalf("expected job spec: %v", err)
	}
	if _, err := os.Stat(filepath.Join(jobDir, "execution_plan.json")); err != nil {
		t.Fatalf("expected execution plan: %v", err)
	}
	if _, err := os.Stat(filepath.Join(jobDir, "input_manifest.json")); err != nil {
		t.Fatalf("expected input manifest: %v", err)
	}
}

func TestSubmitJobAppliesTierModelDefaultsToExecutionPlan(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	cfg := config.Config{
		ModelProfileP40:               "gpt-oss-20b.p40",
		ModelProfileA100:              "qwen3-coder-30b.a100",
		RuntimeLlamaCPPBaseURL:        "http://127.0.0.1:8088",
		RuntimeLlamaCPPTimeoutSeconds: 17,
	}
	svc := NewWithAuditAndOptionsAndConfig(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		"/repo/root",
		Options{},
		&cfg,
	)

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(context.Background()), types.SubmitJobRequest{
		TaskType: "rag_compress",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file:///tmp/example-repo"},
		},
		ExecutionProfile: types.ExecutionProfile{
			Tier: "p40-rag-compression",
		},
		OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, resp.JobID)
	plan := loadJSONFileForTest(t, filepath.Join(jobDir, "execution_plan.json"))
	if plan["selected_model"] != "gpt-oss-20b.p40" {
		t.Fatalf("expected selected_model gpt-oss-20b.p40, got %#v", plan["selected_model"])
	}
	if plan["runtime_backend"] != "llama.cpp" {
		t.Fatalf("expected runtime_backend llama.cpp, got %#v", plan["runtime_backend"])
	}
	runtimeConnection, _ := plan["runtime_connection"].(map[string]any)
	if runtimeConnection["base_url"] != "http://127.0.0.1:8088" {
		t.Fatalf("expected runtime_connection.base_url, got %#v", runtimeConnection)
	}
	if runtimeConnection["timeout_seconds"] != float64(17) {
		t.Fatalf("expected runtime_connection.timeout_seconds 17, got %#v", runtimeConnection)
	}
}

func TestSubmitJobAppliesResolvedTierBeforeStagingExecutionPlan(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	cfg := config.Config{
		ModelProfileP40:  "gpt-oss-20b.p40",
		ModelProfileA100: "qwen3-coder-30b.a100",
	}
	svc := NewWithAuditAndOptionsAndConfig(
		jobStore,
		resolvingFakeBackend{
			resolvedProfile: types.ExecutionProfile{Tier: "a100-reasoning"},
		},
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		"/repo/root",
		Options{},
		&cfg,
	)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "rag_compress",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file:///tmp/example-repo"},
		},
		ExecutionProfile: types.ExecutionProfile{
			Tier: "p40-rag-compression",
		},
		OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, resp.JobID)
	plan := loadJSONFileForTest(t, filepath.Join(jobDir, "execution_plan.json"))
	if plan["resource_tier"] != "a100-reasoning" {
		t.Fatalf("expected resolved a100 resource_tier, got %#v", plan["resource_tier"])
	}
	if plan["selected_model"] != "qwen3-coder-30b.a100" {
		t.Fatalf("expected resolved selected_model qwen3-coder-30b.a100, got %#v", plan["selected_model"])
	}
}

func TestSubmitJobNormalizesAbsolutePathInputRefToFileURI(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		"/repo/root",
	)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		TaskParams: map[string]any{
			"query": "find the repository entrypoint",
		},
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "/tmp/example-repo"},
		},
		ExecutionProfile: types.ExecutionProfile{
			Tier: "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job, err := jobStore.GetJob(context.Background(), resp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got := job.Request.InputRefs[0].URI; got != "file:///tmp/example-repo" {
		t.Fatalf("expected normalized file URI, got %q", got)
	}

	manifest := loadJSONFileForTest(t, filepath.Join(runRoot, resp.JobID, "input_manifest.json"))
	inputRefs, _ := manifest["input_refs"].([]any)
	ref, _ := inputRefs[0].(map[string]any)
	if ref["uri"] != "file:///tmp/example-repo" {
		t.Fatalf("expected staged normalized file URI, got %#v", ref["uri"])
	}
}

func TestInspectRepoRequestWorkerNeverReceivesModelGPU(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(jobStore, fakeBackend{}, log.New(io.Discard, "", 0), runRoot, ".")

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType:   "inspect_repo",
		TaskParams: map[string]any{"query": "trace request routing", "mode": "auto"},
		InputRefs:  []types.InputRef{{Type: "repo", URI: "/tmp/example-repo"}},
		ExecutionProfile: types.ExecutionProfile{
			Tier: "v100-reasoning", Runtime: "vllm", Model: "/models/reasoning", Accelerator: "v100", GPUCount: 4,
			NodeList: "v100[1-4]", Constraint: "v100",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	job, err := jobStore.GetJob(context.Background(), resp.JobID)
	if err != nil {
		t.Fatalf("get submitted job: %v", err)
	}
	profile := job.Request.ExecutionProfile
	if profile.Tier != "cpu-rag-indexing" || profile.Runtime != "deterministic" || profile.GPUCount != 0 || profile.Model != "" || profile.Accelerator != "" {
		t.Fatalf("inspection request worker must remain CPU-only: %#v", profile)
	}
	if profile.NodeList != "" || profile.Constraint != "" {
		t.Fatalf("GPU placement leaked into inspection worker: %#v", profile)
	}
}

func TestInspectRepoExecutionPlanPointsToSharedGPUControlPlane(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	cfg := config.Config{
		GPUServiceEnabled:               true,
		GPUServiceRegistryPath:          ".broker/gpu-services.json",
		GPUServiceControlRequestDir:     ".broker/gpu-services.json.requests",
		GPUServiceControlToken:          "internal-control-token",
		GPUServiceHealthIntervalSeconds: 15,
		GPUServiceStartupTimeoutSeconds: 600,
	}
	svc := NewWithParams(Params{
		JobStore: jobStore, Backend: fakeBackend{}, Logger: log.New(io.Discard, "", 0),
		RunRoot: runRoot, RepoRoot: repoRoot, Config: &cfg,
	})

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType:     "inspect_repo",
		TaskParams:   map[string]any{"query": "trace request routing"},
		InputRefs:    []types.InputRef{{Type: "repo", URI: repoRoot}},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	plan := loadJSONFileForTest(t, filepath.Join(runRoot, resp.JobID, "execution_plan.json"))
	wantRegistry := filepath.Join(repoRoot, ".broker", "gpu-services.json")
	if plan["gpu_service_registry_path"] != wantRegistry {
		t.Fatalf("expected registry path %q, got %#v", wantRegistry, plan)
	}
	if plan["gpu_service_request_path"] != wantRegistry+".requests" {
		t.Fatalf("expected control request directory, got %#v", plan)
	}
	encoded, _ := json.Marshal(plan)
	if bytes.Contains(bytes.ToLower(encoded), []byte("bearer_token")) {
		t.Fatalf("execution plan must not copy endpoint credentials: %s", encoded)
	}
	if plan["gpu_service_control_token"] != "internal-control-token" {
		t.Fatalf("expected internal control token in protected execution plan: %#v", plan)
	}
	if plan["repo_inspection_cache_path"] != filepath.Join(repoRoot, ".broker", "repo-inspection-cache") {
		t.Fatalf("expected broker-owned inspection cache path, got %#v", plan)
	}
	if plan["repo_inspection_shared_cache_path"] != filepath.Join(repoRoot, ".broker", "repo-inspection-shared-cache") {
		t.Fatalf("expected broker-owned inspection shared cache path, got %#v", plan)
	}
	jobDir := filepath.Join(runRoot, resp.JobID)
	if info, err := os.Stat(jobDir); err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("expected protected job directory mode 0700, info=%#v err=%v", info, err)
	}
	planPath := filepath.Join(jobDir, "execution_plan.json")
	if info, err := os.Stat(planPath); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("expected protected execution plan mode 0600, info=%#v err=%v", info, err)
	}
}

func TestSubmitJobNormalizesRepoSchemeInputRefToFileURI(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		"/repo/root",
	)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		TaskParams: map[string]any{
			"query": "find the repository entrypoint",
		},
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "repo:///tmp/example-repo"},
		},
		ExecutionProfile: types.ExecutionProfile{
			Tier: "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job, err := jobStore.GetJob(context.Background(), resp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got := job.Request.InputRefs[0].URI; got != "file:///tmp/example-repo" {
		t.Fatalf("expected normalized file URI, got %q", got)
	}

	manifest := loadJSONFileForTest(t, filepath.Join(runRoot, resp.JobID, "input_manifest.json"))
	inputRefs, _ := manifest["input_refs"].([]any)
	ref, _ := inputRefs[0].(map[string]any)
	if ref["uri"] != "file:///tmp/example-repo" {
		t.Fatalf("expected staged normalized file URI, got %#v", ref["uri"])
	}
}

func TestSubmitJobUsesConfiguredInspectRepoSharedCachePathInExecutionPlan(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	sharedCacheRoot := filepath.Join(t.TempDir(), "shared-cache")
	jobStore := store.NewMemoryJobStore()
	cfg := config.Config{
		GPUServiceRegistryPath: filepath.Join(repoRoot, ".broker", "gpu-services.json"),
		GPUServiceControlToken: "internal-control-token",
		GPUServiceEnabled:      true,
	}
	svc := NewWithAuditAndOptionsAndConfig(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		repoRoot,
		Options{},
		&cfg,
	)

	withEnv := func() (restore func()) {
		previous, had := os.LookupEnv("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR")
		_ = os.Setenv("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", sharedCacheRoot)
		return func() {
			if had {
				_ = os.Setenv("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR", previous)
			} else {
				_ = os.Unsetenv("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR")
			}
		}
	}
	restore := withEnv()
	defer restore()

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType:     "inspect_repo",
		TaskParams:   map[string]any{"query": "trace request routing"},
		InputRefs:    []types.InputRef{{Type: "repo", URI: repoRoot}},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	plan := loadJSONFileForTest(t, filepath.Join(runRoot, resp.JobID, "execution_plan.json"))
	if plan["repo_inspection_shared_cache_path"] != sharedCacheRoot {
		t.Fatalf("expected configured inspect_repo shared cache path %q, got %#v", sharedCacheRoot, plan)
	}
}

func TestGetJobFailsOnInvalidResultSchema(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{
			status: backends.RunStatus{
				BackendRunID: "run-1",
				State:        types.JobStateSucceeded,
				RawState:     "COMPLETED",
				ExitCode:     "0:0",
			},
		},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_invalid",
		TaskType:     "document_summary",
		State:        types.JobStateQueued,
		BackendKind:  "fake",
		BackendRunID: "run-1",
		Request: types.SubmitJobRequest{
			TaskType:     "document_summary",
			OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "placeholder_v1",
  "schema_version": "1.0.0",
  "payload": {
    "summary": "placeholder summary"
  }
}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateFailed {
		t.Fatalf("expected failed state, got %q", got.State)
	}
	if got.ResultError == "" {
		t.Fatal("expected result error")
	}
}

func TestGetJobRefreshesProgressFromHeartbeat(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{
			status: backends.RunStatus{
				BackendRunID: "run-1",
				State:        types.JobStateRunning,
				RawState:     "RUNNING",
				ExitCode:     "0:0",
			},
		},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_progress",
		TaskType:     "document_summary",
		State:        types.JobStateQueued,
		BackendKind:  "fake",
		BackendRunID: "run-1",
		Request: types.SubmitJobRequest{
			TaskType:     "document_summary",
			OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	heartbeat := `{
  "job_id": "job_progress",
  "state": "running",
  "phase": "preprocessing",
  "percent": 35,
  "message": "Loading source document",
  "timestamp": "2026-06-26T12:00:00Z",
  "metrics": {
    "documents_processed": 1
  }
}`
	if err := os.WriteFile(filepath.Join(jobDir, "heartbeat.json"), []byte(heartbeat), 0o644); err != nil {
		t.Fatalf("write heartbeat: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateRunning {
		t.Fatalf("expected running, got %q", got.State)
	}
	if got.Progress == nil {
		t.Fatal("expected progress info")
	}
	if got.Progress.Phase != "preprocessing" || got.Progress.Percent != 35 {
		t.Fatalf("unexpected progress: %#v", got.Progress)
	}
	if got.Progress.Metrics["documents_processed"] != float64(1) {
		t.Fatalf("unexpected metrics: %#v", got.Progress.Metrics)
	}
}

func TestGetJobIngestsCompletedRunOutputsEvenWhenBackendStateLags(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{
			status: backends.RunStatus{
				BackendRunID: "run-1",
				State:        types.JobStateRunning,
				RawState:     "RUNNING",
				ExitCode:     "0:0",
			},
		},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_result_wins",
		TaskType:     "inspect_repo",
		State:        types.JobStateRunning,
		BackendKind:  "fake",
		BackendRunID: "run-1",
		BackendState: "RUNNING",
		Request: types.SubmitJobRequest{
			TaskType:     "inspect_repo",
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
			TaskParams:   map[string]any{"query": "where is gpu control plane", "mode": "answer"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
		StartedAt:   &now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "answer",
    "query": "where is gpu control plane",
    "answer": "The broker starts the GPU service control plane in StartGPUServiceControlPlane.",
    "findings": [{"summary":"It starts in broker/cmd/common/gpu_services.go.","evidence_refs":["ev_001"]}],
    "evidence": [{"id":"ev_001","path":"broker/cmd/common/gpu_services.go","source_refs":[{"path":"broker/cmd/common/gpu_services.go","line_start":18,"line_end":47}]}],
    "quality": {"result":"answer_ready","retrieval":"gpu","reranking":"gpu","synthesis":"gpu","answer_ready":true},
    "warnings": [],
    "provenance": {"index_fingerprint":"sha256:test"},
    "retrieval": {},
    "runtime": {"attempts":[
      {"operation":"semantic_retrieval","tier":"p40-retrieval","slurm_job_id":"123","gpu_count":1,"model_profile":"p40-retrieval-test","status":"succeeded","failure_category":"","escalation_reason":"primary_retrieval"},
      {"operation":"rerank","tier":"p40-retrieval","slurm_job_id":"123","gpu_count":1,"model_profile":"p40-retrieval-test","status":"succeeded","failure_category":"","escalation_reason":"semantic_candidates_ready"},
      {"operation":"synthesis","tier":"p40-synthesis","slurm_job_id":"124","gpu_count":1,"model_profile":"p40-synthesis-test","status":"succeeded","failure_category":"","escalation_reason":"primary_synthesis"}
    ]}
  }
}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded state, got %q job=%#v", got.State, got)
	}
	if got.Result == nil || got.Result.Payload["answer"] == nil {
		t.Fatalf("expected ingested result, got %#v", got.Result)
	}
	if got.CompletedAt == nil {
		t.Fatalf("expected completed timestamp, got %#v", got)
	}
}

func TestGetJobSkipsBackendStatusWhenResultAlreadyExists(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	backend := &countingFakeBackend{
		status: backends.RunStatus{
			BackendRunID: "run-1",
			State:        types.JobStateRunning,
			RawState:     "RUNNING",
			ExitCode:     "0:0",
		},
	}
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_result_short_circuit",
		TaskType:     "inspect_repo",
		State:        types.JobStateRunning,
		BackendKind:  "counting-fake",
		BackendRunID: "run-1",
		BackendState: "RUNNING",
		Request: types.SubmitJobRequest{
			TaskType:     "inspect_repo",
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
			TaskParams:   map[string]any{"query": "where is gpu control plane", "mode": "evidence"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
		StartedAt:   &now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "evidence",
    "query": "where is gpu control plane",
    "findings": [],
    "evidence": [{"id":"ev_001","path":"broker/cmd/common/gpu_services.go","source_refs":[{"path":"broker/cmd/common/gpu_services.go","line_start":18,"line_end":47}]}],
    "quality": {"result":"evidence_only","retrieval":"lexical_degraded","reranking":"unavailable","synthesis":"not_requested","answer_ready":false},
    "warnings": [],
    "provenance": {"index_fingerprint":"sha256:test"},
    "retrieval": {},
    "runtime": {"attempts":[]}
  }
}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded state, got %q", got.State)
	}
	if got.Result == nil {
		t.Fatalf("expected ingested result, got %#v", got.Result)
	}
	if backend.getRunCalls != 0 {
		t.Fatalf("expected no backend GetRun call when result already exists, got %d", backend.getRunCalls)
	}
}

func TestGetJobIgnoresStaleHeartbeatWhenResultAlreadyExists(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{
			status: backends.RunStatus{
				BackendRunID: "run-1",
				State:        types.JobStateRunning,
				RawState:     "RUNNING",
				ExitCode:     "0:0",
			},
		},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_result_beats_heartbeat",
		TaskType:     "inspect_repo",
		State:        types.JobStateRunning,
		BackendKind:  "fake",
		BackendRunID: "run-1",
		BackendState: "RUNNING",
		Request: types.SubmitJobRequest{
			TaskType:     "inspect_repo",
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
			TaskParams:   map[string]any{"query": "where is gpu control plane", "mode": "evidence"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
		StartedAt:   &now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "evidence",
    "query": "where is gpu control plane",
    "findings": [],
    "evidence": [{"id":"ev_001","path":"broker/cmd/common/gpu_services.go","source_refs":[{"path":"broker/cmd/common/gpu_services.go","line_start":18,"line_end":47}]}],
    "quality": {"result":"evidence_only","retrieval":"lexical_degraded","reranking":"unavailable","synthesis":"not_requested","answer_ready":false},
    "warnings": [],
    "provenance": {"index_fingerprint":"sha256:test"},
    "retrieval": {},
    "runtime": {"attempts":[]}
  }
}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "heartbeat.json"), []byte(`{
  "job_id": "job_result_beats_heartbeat",
  "state": "running",
  "phase": "gpu_first_retrieval",
  "percent": 35,
  "message": "stale heartbeat"
}`), 0o644); err != nil {
		t.Fatalf("write heartbeat: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded state, got %q", got.State)
	}
	if got.Progress != nil {
		t.Fatalf("expected stale heartbeat to be ignored after result ingestion, got %#v", got.Progress)
	}
}

func TestGetJobIngestsCompletedRunOutputsWithSingleStoreUpdate(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := newCountingJobStore()
	svc := New(
		jobStore,
		fakeBackend{
			status: backends.RunStatus{
				BackendRunID: "run-1",
				State:        types.JobStateRunning,
				RawState:     "RUNNING",
				ExitCode:     "0:0",
			},
		},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_single_update_ingest",
		TaskType:     "inspect_repo",
		State:        types.JobStateRunning,
		BackendKind:  "fake",
		BackendRunID: "run-1",
		BackendState: "RUNNING",
		Request: types.SubmitJobRequest{
			TaskType:     "inspect_repo",
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
			TaskParams:   map[string]any{"query": "where is gpu control plane", "mode": "evidence"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
		StartedAt:   &now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "evidence",
    "query": "where is gpu control plane",
    "findings": [],
    "evidence": [{"id":"ev_001","path":"broker/cmd/common/gpu_services.go","source_refs":[{"path":"broker/cmd/common/gpu_services.go","line_start":18,"line_end":47}]}],
    "quality": {"result":"evidence_only","retrieval":"lexical_degraded","reranking":"unavailable","synthesis":"not_requested","answer_ready":false},
    "warnings": [],
    "provenance": {"index_fingerprint":"sha256:test"},
    "retrieval": {},
    "runtime": {"attempts":[]}
  }
}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded state, got %q", got.State)
	}
	if got.Result == nil {
		t.Fatalf("expected ingested result, got %#v", got.Result)
	}
	if jobStore.updateCalls != 1 {
		t.Fatalf("expected single UpdateJob during completion ingest, got %d", jobStore.updateCalls)
	}
}

func TestGetJobAppliesBrokerRetrievalPolicyWarnings(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{
			status: backends.RunStatus{
				BackendRunID: "run-1",
				State:        types.JobStateSucceeded,
				RawState:     "COMPLETED",
				ExitCode:     "0:0",
			},
		},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_policy_result",
		TaskType:     "rag_compress",
		State:        types.JobStateQueued,
		BackendKind:  "fake",
		BackendRunID: "run-1",
		Request: types.SubmitJobRequest{
			TaskType:     "rag_compress",
			OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
			ExecutionProfile: types.ExecutionProfile{
				Backend: "slurm",
				Tier:    "p40-rag-compression",
				Runtime: "llama.cpp",
			},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "rag_evidence_pack_v1",
  "schema_version": "1.0.0",
  "payload": {
    "query": "why did the build fail?",
    "retrieval": {
      "strategies": ["ripgrep"],
      "chunks_considered": 1,
      "chunks_indexed": 1,
      "chunks_retrieved": 1,
      "chunks_reranked": 1,
      "chunks_deduplicated": 0,
      "chunks_compressed": 1,
      "requested_strategies": ["ripgrep"],
      "skipped_strategies": [],
      "strategy_hits": {},
      "strategy_stats": [{"strategy":"ripgrep","backend_mode":"fallback"}]
    },
    "retrieval_plan": {
      "requested_strategies": ["ripgrep"],
      "effective_strategies": ["ripgrep"]
    },
    "retrieval_trace": {
      "strategy_executions": [
        {"strategy":"ripgrep","backend_mode":"fallback","backend_detail":"deterministic_path_scan"}
      ]
    },
    "policy_signals": {
      "mode_counts": {"fallback": 1},
      "degraded_strategies": [{"strategy":"ripgrep","backend_mode":"fallback"}],
      "real_backend_required_recommended": true,
      "warnings": ["LOCAL_RETRIEVAL_DEGRADED", "NO_REAL_RETRIEVAL_BACKEND"]
    },
    "evidence": [{"id":"ev_001"}],
    "budget": {"retrieved_chunk_tokens": 10}
  }
}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.Result == nil {
		t.Fatal("expected result")
	}
	warnings, _ := got.Result.Payload["warnings"].([]any)
	if !containsAnyString(warnings, []string{"broker_local_retrieval_degraded", "broker_no_real_retrieval_backend"}) {
		t.Fatalf("expected broker retrieval policy warnings, got %#v", got.Result.Payload)
	}
	usageGuidance, _ := got.Result.Payload["usage_guidance"].(map[string]any)
	if usageGuidance["mode"] != "lead_generation_only" || usageGuidance["needs_direct_verification"] != true {
		t.Fatalf("expected lead-generation usage guidance, got %#v", got.Result.Payload)
	}
	if got.Result.Payload["confidence"] != "low" {
		t.Fatalf("expected low confidence for no-real-backend result, got %#v", got.Result.Payload)
	}
	if got.ResultError != "broker_policy_no_real_retrieval_backend" {
		t.Fatalf("expected broker policy result error, got %#v", got.ResultError)
	}
	retryRecommendation, _ := got.Result.Payload["broker_retry_recommendation"].(map[string]any)
	if retryRecommendation["recommended"] != true {
		t.Fatalf("expected broker retry recommendation, got %#v", got.Result.Payload)
	}
	executionProfile, _ := retryRecommendation["execution_profile"].(map[string]any)
	if executionProfile["tier"] != "a100-reasoning" {
		t.Fatalf("expected escalated retry tier, got %#v", retryRecommendation)
	}
	placementHint, _ := retryRecommendation["placement_hint"].(map[string]any)
	if placementHint["tier_preference"] != "a100-reasoning" || placementHint["preemptible"] != true {
		t.Fatalf("expected placement hint in retry recommendation, got %#v", retryRecommendation)
	}
}

func TestGetJobAllowsInspectRepoLexicalEvidenceWithoutPromotingItToAnswer(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{
			status: backends.RunStatus{
				BackendRunID: "run-1",
				State:        types.JobStateSucceeded,
				RawState:     "COMPLETED",
				ExitCode:     "0:0",
			},
		},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:           "job_inspect_repo_policy",
		TaskType:     "inspect_repo",
		State:        types.JobStateQueued,
		BackendKind:  "fake",
		BackendRunID: "run-1",
		Request: types.SubmitJobRequest{
			TaskType:     "inspect_repo",
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
			TaskParams:   map[string]any{"query": "audit this repo", "mode": "auto"},
			ExecutionProfile: types.ExecutionProfile{
				Backend: "slurm",
				Tier:    "p40-retrieval",
				Runtime: "llama.cpp",
			},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "auto",
    "query": "audit this repo",
    "findings": [],
    "evidence": [{"id":"ev_001","source_refs":[{"path":"src/main.py","line_start":1,"line_end":2}]}],
    "quality": {"result":"evidence_only","retrieval":"lexical_degraded","reranking":"unavailable","synthesis":"failed","answer_ready":false},
	    "warnings": ["gpu_retrieval_unavailable"],
	    "provenance": {"index_fingerprint":"sha256:test"},
	    "retrieval": {},
	    "runtime": {"attempts":[{"operation":"semantic_retrieval","tier":"p40-retrieval","gpu_count":1,"status":"failed","failure_category":"service_failure"}]}
  }
}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}

	got, err := svc.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected lexical evidence fallback to succeed, got state=%q job=%#v", got.State, got)
	}
	if got.ResultError != "" {
		t.Fatalf("expected no result error, got %#v", got.ResultError)
	}
	if got.ExecutionQuality != "evidence_only" || !got.DegradedLocalExecution {
		t.Fatalf("expected degraded evidence-only quality, got %#v", got)
	}
	if _, exists := got.Result.Payload["answer"]; exists {
		t.Fatalf("evidence-only result must not be promoted to an answer: %#v", got.Result.Payload)
	}
	if got.RuntimeDiagnostics["retrieval"] != "lexical_degraded" {
		t.Fatalf("expected stage quality in diagnostics, got %#v", got.RuntimeDiagnostics)
	}
}

func TestRepoInspectionAnswerFailureRetainsResultAndMarksJobFailed(t *testing.T) {
	job := types.Job{TaskType: "inspect_repo", State: types.JobStateSucceeded}
	result := types.Result{
		SchemaName: "repo_inspection_v2",
		Payload: map[string]any{
			"quality": map[string]any{"result": "failed"},
			"runtime": map[string]any{"attempts": []any{
				map[string]any{"tier": "p40-synthesis"},
				map[string]any{"tier": "v100-reasoning"},
				map[string]any{"tier": "a100-single"},
			}},
		},
	}
	applyBrokerResultPolicies(&job, &result)
	if job.State != types.JobStateFailed || job.ResultError != "repo_inspection_gpu_tiers_exhausted" {
		t.Fatalf("expected answer exhaustion failure, got %#v", job)
	}
	if _, ok := result.Payload["runtime"]; !ok {
		t.Fatalf("expected retry history to remain in result: %#v", result.Payload)
	}
}

func TestGetJobLogsRedactsAndTruncates(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:       "job_logs",
		TaskType: "log_analysis",
		State:    types.JobStateRunning,
		Request: types.SubmitJobRequest{
			TaskType:     "log_analysis",
			OutputSchema: types.OutputSchemaRef{Name: "log_analysis_v1"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "stdout.log"), []byte("token=abc123\nhello stdout\n"), 0o644); err != nil {
		t.Fatalf("write stdout: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "stderr.log"), []byte("Bearer secret-token-value\nline two\n"), 0o644); err != nil {
		t.Fatalf("write stderr: %v", err)
	}

	logs, err := svc.GetJobLogs(context.Background(), job.ID, "combined", 40)
	if err != nil {
		t.Fatalf("get job logs: %v", err)
	}
	if !logs.Truncated {
		t.Fatal("expected truncated logs")
	}
	if logs.Stream != "combined" {
		t.Fatalf("unexpected stream %q", logs.Stream)
	}
	if len(logs.SourceRefs) != 2 {
		t.Fatalf("expected 2 source refs, got %d", len(logs.SourceRefs))
	}
	if containsAny(logs.Content, []string{"abc123", "secret-token-value"}) {
		t.Fatalf("expected redacted secrets, got %q", logs.Content)
	}
}

func TestGetJobLogsDeniedForSensitiveClassification(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:       "job_sensitive_logs",
		TaskType: "log_analysis",
		State:    types.JobStateRunning,
		Request: types.SubmitJobRequest{
			TaskType: "log_analysis",
			InputRefs: []types.InputRef{
				{Type: "file", URI: "file:///tmp/secret.log", Classification: "phi"},
			},
			OutputSchema: types.OutputSchemaRef{Name: "log_analysis_v1"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	_, err := svc.GetJobLogs(context.Background(), job.ID, "combined", 1024)
	if err == nil {
		t.Fatal("expected policy denial")
	}
	if !strings.Contains(err.Error(), "policy denied") {
		t.Fatalf("expected policy denial error, got %v", err)
	}
}

func TestGetJobLogsOverrideAllowsSensitiveClassification(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:       "job_sensitive_override",
		TaskType: "log_analysis",
		State:    types.JobStateRunning,
		Request: types.SubmitJobRequest{
			TaskType: "log_analysis",
			InputRefs: []types.InputRef{
				{Type: "file", URI: "file:///tmp/secret.log", Classification: "restricted"},
			},
			TaskParams:   map[string]any{"allow_log_release": true},
			OutputSchema: types.OutputSchemaRef{Name: "log_analysis_v1"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "stdout.log"), []byte("hello\n"), 0o644); err != nil {
		t.Fatalf("write stdout: %v", err)
	}

	logs, err := svc.GetJobLogs(context.Background(), job.ID, "stdout", 1024)
	if err != nil {
		t.Fatalf("expected override to allow logs: %v", err)
	}
	if logs.Content == "" {
		t.Fatal("expected log content")
	}
}

func TestGetReleasedResultRedactsSensitiveFieldsAndWithholdsArtifacts(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:       "job_sensitive_result",
		TaskType: "repo_summary",
		State:    types.JobStateSucceeded,
		Request: types.SubmitJobRequest{
			TaskType: "repo_summary",
			InputRefs: []types.InputRef{
				{Type: "directory", URI: "file:///tmp/repo", Classification: "restricted"},
			},
			OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
		},
		Result: &types.Result{
			SchemaName:    "repo_summary_v1",
			SchemaVersion: "1.0.0",
			Payload: map[string]any{
				"summary": "summary",
				"entrypoints": []any{
					map[string]any{"path": "broker/cmd/main.go", "kind": "service_entrypoint"},
				},
				"warnings": []any{"worker_warning"},
			},
		},
		Artifacts: []types.Artifact{
			{ArtifactID: "artifact_1", ArtifactType: "chunk_manifest", Path: "/tmp/manifest.json"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	release, err := svc.GetReleasedResult(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get released result: %v", err)
	}
	if release.Result == nil {
		t.Fatal("expected result")
	}
	entrypoints := release.Result.Payload["entrypoints"].([]any)
	first := entrypoints[0].(map[string]any)
	if first["path"] != "[REDACTED]" {
		t.Fatalf("expected redacted path, got %#v", first["path"])
	}
	if len(release.Artifacts) != 0 {
		t.Fatalf("expected artifacts withheld, got %#v", release.Artifacts)
	}
	warnings := release.Result.Payload["warnings"].([]any)
	if !containsAnyString(warnings, []string{"broker_redacted_sensitive_fields", "broker_withheld_artifacts"}) {
		t.Fatalf("expected broker warnings, got %#v", warnings)
	}
}

func TestGetReleasedResultAllowsArtifactsWithOverrideButStripsPaths(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:       "job_sensitive_artifacts",
		TaskType: "log_analysis",
		State:    types.JobStateSucceeded,
		Request: types.SubmitJobRequest{
			TaskType: "log_analysis",
			InputRefs: []types.InputRef{
				{Type: "file", URI: "file:///tmp/build.log", Classification: "phi"},
			},
			TaskParams:   map[string]any{"allow_artifact_release": true},
			OutputSchema: types.OutputSchemaRef{Name: "log_analysis_v1"},
		},
		Result: &types.Result{
			SchemaName:    "log_analysis_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"summary": "summary"},
		},
		Artifacts: []types.Artifact{
			{ArtifactID: "artifact_1", ArtifactType: "redacted_excerpt", Path: "/tmp/excerpt.txt"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	release, err := svc.GetReleasedResult(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get released result: %v", err)
	}
	if len(release.Artifacts) != 1 {
		t.Fatalf("expected released artifacts, got %#v", release.Artifacts)
	}
	if release.Artifacts[0].Path != "" {
		t.Fatalf("expected stripped artifact path, got %#v", release.Artifacts[0].Path)
	}
}

func TestGetReleasedResultForInspectRepoUsesDirectRunFileFastPath(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	jobStore := newCountingJobStore()
	svc := New(jobStore, fakeBackend{}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	job := types.Job{
		ID:          "job_direct_release",
		TaskType:    "inspect_repo",
		State:       types.JobStateRunning,
		SubmittedBy: "alice",
		Request: types.SubmitJobRequest{
			TaskType: "inspect_repo",
			InputRefs: []types.InputRef{
				{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
			},
			TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		},
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	resultBytes := []byte(`{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "evidence",
    "query": "trace retry_job",
    "findings": [],
    "evidence": [{"id":"ev_001","path":"broker/pkg/service/service.go","source_refs":[{"path":"broker/pkg/service/service.go","line_start":1,"line_end":10}]}],
    "quality": {"result":"evidence_only","retrieval":"lexical_degraded","reranking":"unavailable","synthesis":"not_requested","answer_ready":false},
    "warnings": [],
    "provenance": {"index_fingerprint":"sha256:test"},
    "retrieval": {},
	    "runtime": {"attempts":[]}
  }
}`)
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), resultBytes, 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}

	release, err := svc.GetReleasedResult(aliceUserCtx(), job.ID)
	if err != nil {
		t.Fatalf("get released result: %v", err)
	}
	if release.Result == nil || release.Result.Payload["query"] != "trace retry_job" {
		t.Fatalf("unexpected release: %#v", release)
	}
	if jobStore.updateCalls != 0 {
		t.Fatalf("expected direct release path to avoid store update, got %d updates", jobStore.updateCalls)
	}
	storedJob, err := jobStore.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get stored job: %v", err)
	}
	if storedJob.Result != nil {
		t.Fatalf("expected stored job result to remain unset on fast path, got %#v", storedJob.Result)
	}
}

func TestGetReleasedResultForTerminalFailedInspectRepoUsesDirectRunFileFastPath(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	jobStore := newCountingJobStore()
	svc := New(jobStore, fakeBackend{}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	job := types.Job{
		ID:           "job_failed_direct_release",
		TaskType:     "inspect_repo",
		State:        types.JobStateFailed,
		BackendKind:  "local",
		BackendRunID: "job_failed_direct_release",
		SubmittedBy:  "alice",
		Request: types.SubmitJobRequest{
			TaskType: "inspect_repo",
			InputRefs: []types.InputRef{
				{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
			},
			TaskParams:   map[string]any{"query": "trace retry_job", "mode": "answer"},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		},
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	jobDir := filepath.Join(runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		t.Fatalf("mkdir job dir: %v", err)
	}
	resultBytes := []byte(`{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "answer",
    "query": "trace retry_job",
    "findings": [],
    "evidence": [{"id":"ev_001","path":"broker/pkg/service/service.go","source_refs":[{"path":"broker/pkg/service/service.go","line_start":1,"line_end":10}]}],
    "quality": {"result":"failed","retrieval":"lexical_degraded","reranking":"unavailable","synthesis":"failed","answer_ready":false},
    "warnings": ["ANSWER_REQUIRES_GPU_RETRIEVAL_AND_RERANK"],
    "provenance": {"index_fingerprint":"sha256:test"},
    "retrieval": {},
	    "runtime": {"attempts":[
	      {"operation":"semantic_retrieval","tier":"p40-retrieval","gpu_count":1,"status":"failed","failure_category":"service_unavailable"},
	      {"operation":"rerank","tier":"p40-retrieval","gpu_count":1,"status":"failed","failure_category":"service_unavailable"},
	      {"operation":"synthesis","tier":"p40-synthesis","gpu_count":1,"status":"failed","failure_category":"service_unavailable"},
	      {"operation":"synthesis","tier":"v100-reasoning","gpu_count":4,"status":"failed","failure_category":"service_unavailable"},
	      {"operation":"synthesis","tier":"a100-single","gpu_count":1,"status":"failed","failure_category":"service_unavailable"}
	    ]}
  }
}`)
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), resultBytes, 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}

	release, err := svc.GetReleasedResult(aliceUserCtx(), job.ID)
	if err != nil {
		t.Fatalf("get released result: %v", err)
	}
	if release.State != types.JobStateFailed {
		t.Fatalf("expected failed release state, got %#v", release)
	}
	if release.Result == nil || release.Result.Payload["query"] != "trace retry_job" {
		t.Fatalf("expected structured failed inspect result, got %#v", release)
	}
}

func TestGetReleasedResultForInspectRepoWaitsForDirectRunFileWithoutBackendRefresh(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	backend := &countingDelayedLocalInspectRepoResultBackend{runRoot: runRoot, delay: 250 * time.Millisecond}
	svc := New(jobStore, backend, log.New(io.Discard, "", 0), runRoot, repoRoot)

	job := types.Job{
		ID:           "job_direct_release_wait",
		TaskType:     "inspect_repo",
		State:        types.JobStateRunning,
		BackendKind:  "local",
		BackendRunID: "job_direct_release_wait",
		SubmittedBy:  "alice",
		Request: types.SubmitJobRequest{
			TaskType: "inspect_repo",
			InputRefs: []types.InputRef{
				{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
			},
			TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		},
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	go func() {
		time.Sleep(10 * time.Millisecond)
		_ = writeInspectRepoResultForTest(runRoot, job.ID, "trace retry_job")
	}()

	release, err := svc.GetReleasedResult(aliceUserCtx(), job.ID)
	if err != nil {
		t.Fatalf("get released result: %v", err)
	}
	if release.Result == nil || release.Result.Payload["query"] != "trace retry_job" {
		t.Fatalf("unexpected release: %#v", release)
	}
	if got := backend.getRunCalls.Load(); got != 0 {
		t.Fatalf("expected no backend GetRun calls before quick direct run-file release, got %d", got)
	}
}

func TestSubmitAndIngestDocumentSummaryWorker(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	inputPath := filepath.Join(runRoot, "source.txt")
	if err := os.WriteFile(inputPath, []byte("Worker integration test.\n- Point one\n- Point two\n"), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	backend := &mutableFakeBackend{}
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + inputPath, ContentHash: "sha256:test"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	cmd := exec.Command(
		"python3",
		filepath.Join(repoRoot, "workers", "document-summary", "main.py"),
		"--job-spec", filepath.Join(jobDir, "job_spec.json"),
		"--input-manifest", filepath.Join(jobDir, "input_manifest.json"),
		"--output-dir", jobDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run document worker: %v: %s", err, string(output))
	}

	backend.status = backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
		ExitCode:     "0:0",
	}

	got, err := svc.GetJob(context.Background(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", got.State)
	}
	if got.Result == nil || got.Result.SchemaName != "document_summary_v1" {
		t.Fatalf("expected document_summary_v1 result, got %#v", got.Result)
	}
	if len(got.Artifacts) != 1 {
		t.Fatalf("expected one artifact, got %d", len(got.Artifacts))
	}
	if got.Progress == nil || got.Progress.State != "completed" || got.Progress.Percent != 100 {
		t.Fatalf("expected completed progress, got %#v", got.Progress)
	}
}

func TestSubmitAndIngestLogAnalysisWorker(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	inputPath := filepath.Join(runRoot, "build.log")
	logText := "2026-06-26T12:01:00Z build started\nfatal error: generated/config.h: No such file or directory\n"
	if err := os.WriteFile(inputPath, []byte(logText), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	backend := &mutableFakeBackend{}
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "log_analysis",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + inputPath, ContentHash: "sha256:test"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "log_analysis_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	cmd := exec.Command(
		"python3",
		filepath.Join(repoRoot, "workers", "log-analysis", "main.py"),
		"--job-spec", filepath.Join(jobDir, "job_spec.json"),
		"--input-manifest", filepath.Join(jobDir, "input_manifest.json"),
		"--output-dir", jobDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run log worker: %v: %s", err, string(output))
	}

	backend.status = backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
		ExitCode:     "0:0",
	}

	got, err := svc.GetJob(context.Background(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", got.State)
	}
	if got.Result == nil || got.Result.SchemaName != "log_analysis_v1" {
		t.Fatalf("expected log_analysis_v1 result, got %#v", got.Result)
	}
	if len(got.Artifacts) != 1 {
		t.Fatalf("expected one artifact, got %d", len(got.Artifacts))
	}
	if got.Progress == nil || got.Progress.State != "completed" || got.Progress.Percent != 100 {
		t.Fatalf("expected completed progress, got %#v", got.Progress)
	}
}

func TestSubmitAndIngestRepoSummaryWorker(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot := filepath.Join(runRoot, "repo")
	if err := os.MkdirAll(filepath.Join(repoRoot, "broker"), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(repoRoot, "deploy", "slurm"), 0o755); err != nil {
		t.Fatalf("mkdir deploy dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "go.mod"), []byte("module example.com/test\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "broker", "main.go"), []byte("package main\nfunc main(){}\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	actualRepoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	backend := &mutableFakeBackend{}
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		actualRepoRoot,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "repo_summary",
		InputRefs: []types.InputRef{
			{Type: "directory", URI: "file://" + repoRoot, ContentHash: "sha256:test"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	cmd := exec.Command(
		"python3",
		filepath.Join(actualRepoRoot, "workers", "repo-summary", "main.py"),
		"--job-spec", filepath.Join(jobDir, "job_spec.json"),
		"--input-manifest", filepath.Join(jobDir, "input_manifest.json"),
		"--output-dir", jobDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run repo worker: %v: %s", err, string(output))
	}

	backend.status = backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
		ExitCode:     "0:0",
	}

	got, err := svc.GetJob(context.Background(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", got.State)
	}
	if got.Result == nil || got.Result.SchemaName != "repo_summary_v1" {
		t.Fatalf("expected repo_summary_v1 result, got %#v", got.Result)
	}
	if len(got.Artifacts) != 1 {
		t.Fatalf("expected one artifact, got %d", len(got.Artifacts))
	}
	if got.Progress == nil || got.Progress.State != "completed" || got.Progress.Percent != 100 {
		t.Fatalf("expected completed progress, got %#v", got.Progress)
	}
}

func TestSubmitAndIngestRAGCompressionWorker(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	inputPath := filepath.Join(runRoot, "build.log")
	inputText := "2026-06-26T12:01:00Z build started\nfatal error: generated/config.h: No such file or directory\n"
	if err := os.WriteFile(inputPath, []byte(inputText), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	backend := &mutableFakeBackend{}
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "rag_compress",
		InputRefs: []types.InputRef{
			{Type: "log", URI: "file://" + inputPath, ContentHash: "sha256:test", Classification: "restricted"},
		},
		TaskParams: map[string]any{
			"query": "why did the build fail?",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      64000,
			PerChunkCompressionBudget: 384,
			FinalEvidencePackBudget:   4000,
			RemoteModelContextBudget:  12000,
		},
		OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	cmd := exec.Command(
		"python3",
		filepath.Join(repoRoot, "workers", "rag-compression", "main.py"),
		"--job-spec", filepath.Join(jobDir, "job_spec.json"),
		"--input-manifest", filepath.Join(jobDir, "input_manifest.json"),
		"--output-dir", jobDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run rag worker: %v: %s", err, string(output))
	}

	backend.status = backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
		ExitCode:     "0:0",
	}

	got, err := svc.GetJob(context.Background(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", got.State)
	}
	if got.Result == nil || got.Result.SchemaName != "rag_evidence_pack_v1" {
		t.Fatalf("expected rag_evidence_pack_v1 result, got %#v", got.Result)
	}
	if got.RuntimeDiagnostics == nil {
		t.Fatalf("expected runtime diagnostics, got %#v", got.RuntimeDiagnostics)
	}
	if got.RuntimeDiagnostics["backend_mode"] == nil {
		t.Fatalf("expected runtime diagnostics backend_mode, got %#v", got.RuntimeDiagnostics)
	}
	if !got.DegradedLocalExecution {
		t.Fatalf("expected degraded_local_execution, got %#v", got)
	}
	if got.RetryRecommended {
		t.Fatalf("expected retry_recommended=false for deterministic degraded run, got %#v", got)
	}
	if got.ExecutionQuality != "degraded_local" {
		t.Fatalf("expected execution_quality degraded_local, got %#v", got.ExecutionQuality)
	}
	if got.Result.Payload["query"] != "why did the build fail?" {
		t.Fatalf("unexpected query payload: %#v", got.Result.Payload)
	}
	evidence, _ := got.Result.Payload["evidence"].([]any)
	if len(evidence) == 0 {
		t.Fatalf("expected evidence, got %#v", got.Result.Payload)
	}
	if len(got.Artifacts) == 0 {
		t.Fatalf("expected artifacts, got %#v", got.Artifacts)
	}
	if !artifactTypesInclude(got.Artifacts, "retrieval_plan", "retrieval_trace", "chunk_manifest", "rerank_result", "evidence_pack", "retrieval_result", "validation_report") {
		t.Fatalf("expected staged rag artifacts, got %#v", got.Artifacts)
	}
	retrievalPlanPath := artifactPathForType(got.Artifacts, "retrieval_plan")
	if retrievalPlanPath == "" {
		t.Fatalf("expected retrieval plan artifact path, got %#v", got.Artifacts)
	}
	retrievalPlan := loadJSONFileForTest(t, retrievalPlanPath)
	effective, ok := retrievalPlan["effective_strategies"].([]any)
	if !ok || len(effective) == 0 {
		t.Fatalf("expected effective retrieval strategies, got %#v", retrievalPlan)
	}
	retrievalTracePath := artifactPathForType(got.Artifacts, "retrieval_trace")
	if retrievalTracePath == "" {
		t.Fatalf("expected retrieval trace artifact path, got %#v", got.Artifacts)
	}
	retrievalTrace := loadJSONFileForTest(t, retrievalTracePath)
	executions, ok := retrievalTrace["strategy_executions"].([]any)
	if !ok || len(executions) == 0 {
		t.Fatalf("expected strategy executions in retrieval trace, got %#v", retrievalTrace)
	}
	firstExecution, ok := executions[0].(map[string]any)
	if !ok || firstExecution["backend_mode"] == nil {
		t.Fatalf("expected backend mode in retrieval trace, got %#v", retrievalTrace)
	}
	policySignals, ok := retrievalTrace["policy_signals"].(map[string]any)
	if ok && len(policySignals) > 0 {
		t.Fatalf("did not expect retrieval trace to duplicate policy signals, got %#v", retrievalTrace)
	}
	validationPath := artifactPathForType(got.Artifacts, "validation_report")
	if validationPath == "" {
		t.Fatalf("expected validation report artifact path, got %#v", got.Artifacts)
	}
	validation := loadJSONFileForTest(t, validationPath)
	if validation["chunks_indexed"].(float64) < 1 {
		t.Fatalf("expected indexed chunks in validation report, got %#v", validation)
	}
	validationPolicy, ok := validation["policy_signals"].(map[string]any)
	if !ok || validationPolicy["mode_counts"] == nil {
		t.Fatalf("expected policy signals in validation report, got %#v", validation)
	}
	stages, ok := validation["pipeline_stages"].([]any)
	if !ok || len(stages) < 8 {
		t.Fatalf("expected pipeline stages in validation report, got %#v", validation)
	}
	if got.Progress == nil || got.Progress.State != "completed" || got.Progress.Percent != 100 {
		t.Fatalf("expected completed progress, got %#v", got.Progress)
	}
}

func TestSubmitAndIngestRAGCompressionWorkerTrimsToFinalBudget(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	inputPath := filepath.Join(runRoot, "build.log")
	inputText := strings.Repeat("fatal error: generated/config.h missing during build step\n", 120)
	if err := os.WriteFile(inputPath, []byte(inputText), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	backend := &mutableFakeBackend{}
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "rag_compress",
		InputRefs: []types.InputRef{
			{Type: "log", URI: "file://" + inputPath, ContentHash: "sha256:test", Classification: "restricted"},
		},
		TaskParams: map[string]any{
			"query": "why did the build fail?",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      64000,
			PerChunkCompressionBudget: 384,
			FinalEvidencePackBudget:   80,
			RemoteModelContextBudget:  12000,
		},
		OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	cmd := exec.Command(
		"python3",
		filepath.Join(repoRoot, "workers", "rag-compression", "main.py"),
		"--job-spec", filepath.Join(jobDir, "job_spec.json"),
		"--input-manifest", filepath.Join(jobDir, "input_manifest.json"),
		"--output-dir", jobDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run rag worker: %v: %s", err, string(output))
	}

	backend.status = backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
		ExitCode:     "0:0",
	}

	got, err := svc.GetJob(context.Background(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.Result == nil {
		t.Fatal("expected result")
	}
	warnings, _ := got.Result.Payload["warnings"].([]any)
	if !containsAnyString(warnings, []string{"FINAL_EVIDENCE_PACK_TRIMMED"}) {
		t.Fatalf("expected trim warning, got %#v", got.Result.Payload)
	}
	budget, _ := got.Result.Payload["budget"].(map[string]any)
	if budget["final_pack_tokens"].(float64) > 80 {
		t.Fatalf("expected trimmed final budget <= 80, got %#v", budget)
	}
	retrieval, _ := got.Result.Payload["retrieval"].(map[string]any)
	if retrieval["strategy_hits"] == nil {
		t.Fatalf("expected strategy hits in retrieval payload, got %#v", got.Result.Payload)
	}
	if retrieval["strategy_stats"] == nil {
		t.Fatalf("expected strategy stats in retrieval payload, got %#v", got.Result.Payload)
	}
	retrievalTrace, _ := got.Result.Payload["retrieval_trace"].(map[string]any)
	if retrievalTrace["strategy_executions"] == nil {
		t.Fatalf("expected retrieval trace payload, got %#v", got.Result.Payload)
	}
	policySignals, _ := got.Result.Payload["policy_signals"].(map[string]any)
	if policySignals["mode_counts"] == nil {
		t.Fatalf("expected policy signals payload, got %#v", got.Result.Payload)
	}
	retrievalStats, _ := retrieval["strategy_stats"].([]any)
	if len(retrievalStats) == 0 {
		t.Fatalf("expected strategy stats payload, got %#v", got.Result.Payload)
	}
	firstStat, ok := retrievalStats[0].(map[string]any)
	if !ok || firstStat["backend_mode"] == nil {
		t.Fatalf("expected backend mode in strategy stats, got %#v", got.Result.Payload)
	}
}

func TestSubmitAndIngestRAGCompressionWorkerSkipsDefaultExcludedDirs(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	inputRepo := filepath.Join(runRoot, "repo")
	if err := os.MkdirAll(filepath.Join(inputRepo, "src"), 0o755); err != nil {
		t.Fatalf("mkdir src: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(inputRepo, ".venv", "lib"), 0o755); err != nil {
		t.Fatalf("mkdir venv: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "src", "main.py"), []byte("def run_service():\n    raise RuntimeError('primary failure')\n"), 0o644); err != nil {
		t.Fatalf("write source file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, ".venv", "lib", "noise.py"), []byte("def bad_dependency():\n    raise RuntimeError('primary failure')\n"), 0o644); err != nil {
		t.Fatalf("write venv file: %v", err)
	}

	backend := &mutableFakeBackend{}
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "rag_compress",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + inputRepo, ContentHash: "sha256:test", Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "primary failure",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      64000,
			PerChunkCompressionBudget: 384,
			FinalEvidencePackBudget:   4000,
			RemoteModelContextBudget:  12000,
		},
		OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	cmd := exec.Command(
		"python3",
		filepath.Join(repoRoot, "workers", "rag-compression", "main.py"),
		"--job-spec", filepath.Join(jobDir, "job_spec.json"),
		"--input-manifest", filepath.Join(jobDir, "input_manifest.json"),
		"--output-dir", jobDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run rag worker: %v: %s", err, string(output))
	}

	backend.status = backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
		ExitCode:     "0:0",
	}

	got, err := svc.GetJob(context.Background(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.Result == nil {
		t.Fatal("expected result")
	}
	evidence, _ := got.Result.Payload["evidence"].([]any)
	if len(evidence) == 0 {
		t.Fatalf("expected evidence, got %#v", got.Result.Payload)
	}
	for _, raw := range evidence {
		ev, ok := raw.(map[string]any)
		if !ok {
			t.Fatalf("unexpected evidence shape: %#v", raw)
		}
		refs, _ := ev["source_refs"].([]any)
		for _, sourceRefRaw := range refs {
			sourceRef, ok := sourceRefRaw.(map[string]any)
			if !ok {
				t.Fatalf("unexpected source ref shape: %#v", sourceRefRaw)
			}
			path, _ := sourceRef["path"].(string)
			if strings.Contains(path, ".venv") {
				t.Fatalf("expected excluded .venv paths to be skipped, got %#v", got.Result.Payload)
			}
		}
	}
	validationPath := artifactPathForType(got.Artifacts, "validation_report")
	if validationPath == "" {
		t.Fatalf("expected validation report artifact path, got %#v", got.Artifacts)
	}
	validation := loadJSONFileForTest(t, validationPath)
	excluded, ok := validation["excluded_dir_names"].([]any)
	if !ok || !containsAnyString(excluded, []string{".venv"}) {
		t.Fatalf("expected .venv in excluded dir names, got %#v", validation)
	}
}

func TestSubmitAndIngestInspectRepoWorkerReportsCoverageGaps(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	inputRepo := filepath.Join(runRoot, "repo")
	if err := os.MkdirAll(filepath.Join(inputRepo, "workers", "rag-compression"), 0o755); err != nil {
		t.Fatalf("mkdir worker dir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(inputRepo, "broker", "pkg", "service"), 0o755); err != nil {
		t.Fatalf("mkdir service dir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(inputRepo, "tests", "unit"), 0o755); err != nil {
		t.Fatalf("mkdir tests dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "go.mod"), []byte("module example.com/test\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "workers", "rag-compression", "main.py"), []byte(strings.Join([]string{
		"def build_result(): pass",
		"def build_evidence(): pass",
		"def build_repo_inspection_payload(): pass",
		"def execute_retrieval_plan(): pass",
		"def build_artifacts(): pass",
		"def rerank_candidates(): pass",
		"def select_chunks(): pass",
		"def summarize_chunk(): pass",
		"def classify_chunk_kind(): pass",
		"def detect_symbol(): pass",
		"def build_validation_report(): pass",
		"def enforce_final_pack_budget(): pass",
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write worker file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "service.go"), []byte("package service\n"), 0o644); err != nil {
		t.Fatalf("write service.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "service_artifacts.go"), []byte("package service\n"), 0o644); err != nil {
		t.Fatalf("write service_artifacts.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "service_job_refresh.go"), []byte("package service\n"), 0o644); err != nil {
		t.Fatalf("write service_job_refresh.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "service_access.go"), []byte("package service\n"), 0o644); err != nil {
		t.Fatalf("write service_access.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "service_test.go"), []byte("package service\nfunc TestRoot(t *testing.T) {}\n"), 0o644); err != nil {
		t.Fatalf("write service_test.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "tests", "unit", "test_workers.py"), []byte(strings.Join([]string{
		`rag_compression = load_module("rag_compression_worker", "workers/rag-compression/main.py")`,
		`rag_compression.query_terms_for("x", "inspect_repo", {})`,
		`rag_compression.repo_structure_executor({}, {}, "inspect_repo", [], {})`,
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write test_workers.py: %v", err)
	}

	got := submitAndRunInspectRepoJobForTest(t, runRoot, repoRoot, inputRepo, "Audit this repository for test coverage gaps")
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", got.State)
	}
	if got.Result == nil || got.Result.SchemaName != "repo_inspection_v2" {
		t.Fatalf("expected repo_inspection_v2 result, got %#v", got.Result)
	}
	quality, _ := got.Result.Payload["quality"].(map[string]any)
	if quality["result"] != "evidence_only" || quality["answer_ready"] != false {
		t.Fatalf("CPU-only inspection must remain evidence-only: %#v", got.Result.Payload)
	}
	if _, exists := got.Result.Payload["answer"]; exists || len(got.Result.Payload["findings"].([]any)) != 0 {
		t.Fatalf("CPU-only inspection synthesized findings: %#v", got.Result.Payload)
	}
	joined := inspectionEvidenceCorpus(got.Result.Payload)
	if !strings.Contains(joined, "workers/rag-compression/main.py") || !strings.Contains(joined, "service_test.go") {
		t.Fatalf("expected coverage-related lexical evidence, got %s", joined)
	}
	if !got.DegradedLocalExecution || got.ExecutionQuality != "evidence_only" {
		t.Fatalf("expected degraded evidence-only execution, got %#v", got)
	}
}

func TestSubmitAndIngestInspectRepoWorkerFindsRetryEntryPoints(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	inputRepo := filepath.Join(runRoot, "repo")
	if err := os.MkdirAll(filepath.Join(inputRepo, "broker", "pkg", "service"), 0o755); err != nil {
		t.Fatalf("mkdir service dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "service.go"), []byte(strings.Join([]string{
		"package service",
		"",
		"func RetryJobWithRecommendation(jobID string) string {",
		`	return "retry recommendation" + jobID`,
		"}",
		"",
		"func retrySubmitRequest() string {",
		`	return "retry request"`,
		"}",
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write service.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "service_runtime.go"), []byte(strings.Join([]string{
		"package service",
		"",
		"func retryRecommendationFromResult() string {",
		`	return "retry recommendation"`,
		"}",
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write service_runtime.go: %v", err)
	}

	got := submitAndRunInspectRepoJobForTest(t, runRoot, repoRoot, inputRepo, "Find retry logic and related entrypoints")
	if got.State != types.JobStateSucceeded || got.Result == nil {
		t.Fatalf("expected succeeded inspect_repo result, got %#v", got)
	}
	joined := inspectionEvidenceCorpus(got.Result.Payload)
	if !strings.Contains(joined, "RetryJobWithRecommendation") {
		t.Fatalf("expected RetryJobWithRecommendation in retry evidence, got %s", joined)
	}
	if !strings.Contains(joined, "retryRecommendationFromResult") {
		t.Fatalf("expected retryRecommendationFromResult in retry evidence, got %s", joined)
	}
	if _, exists := got.Result.Payload["answer"]; exists {
		t.Fatalf("CPU retry evidence must not be promoted to an answer: %#v", got.Result.Payload)
	}
}

func TestSubmitAndIngestInspectRepoWorkerFindsArtifactAuthorizationReviewPoints(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	inputRepo := filepath.Join(runRoot, "repo")
	if err := os.MkdirAll(filepath.Join(inputRepo, "broker", "pkg", "service"), 0o755); err != nil {
		t.Fatalf("mkdir service dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "service_artifacts.go"), []byte(strings.Join([]string{
		"package service",
		"",
		"type Principal struct{ Actor string }",
		"",
		"func resolveArtifactRef(principal Principal, artifactID string) string {",
		`	return "artifact " + artifactID + " for " + principal.Actor`,
		"}",
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write service_artifacts.go: %v", err)
	}
	if err := os.WriteFile(filepath.Join(inputRepo, "broker", "pkg", "service", "artifact_access.go"), []byte(strings.Join([]string{
		"package service",
		"",
		"type Principal struct{ Actor string }",
		"",
		"func artifactJobAccessible(principal Principal, submittedBy string) bool {",
		"	// artifact access check",
		"	return principal.Actor == submittedBy",
		"}",
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write artifact_access.go: %v", err)
	}

	got := submitAndRunInspectRepoJobForTest(t, runRoot, repoRoot, inputRepo, "Identify artifact authorization risks and relevant symbols")
	if got.State != types.JobStateSucceeded || got.Result == nil {
		t.Fatalf("expected succeeded inspect_repo result, got %#v", got)
	}
	joined := inspectionEvidenceCorpus(got.Result.Payload)
	if !strings.Contains(joined, "resolveArtifactRef") {
		t.Fatalf("expected resolveArtifactRef in artifact evidence, got %s", joined)
	}
	if !strings.Contains(joined, "artifactJobAccessible") {
		t.Fatalf("expected artifactJobAccessible in artifact evidence, got %s", joined)
	}
	if !strings.Contains(joined, "service_artifacts.go") {
		t.Fatalf("expected service_artifacts.go in artifact evidence, got %s", joined)
	}
}

func TestStageExecutionBundleResolvesArtifactInputs(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	artifactDir := filepath.Join(runRoot, "job_source")
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	artifactPath := filepath.Join(artifactDir, "evidence_pack.json")
	if err := os.WriteFile(artifactPath, []byte(`{"evidence":[{"id":"ev_001","claim":"generated header missing"}]}`), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	now := time.Now().UTC()
	jobStore := store.NewMemoryJobStore()
	sourceJob := types.Job{
		ID:          "job_source",
		TaskType:    "rag_compress",
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		Request: types.SubmitJobRequest{
			TaskType:     "rag_compress",
			TaskParams:   map[string]any{"allow_artifact_release": true},
			OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
		},
		Result: &types.Result{
			SchemaName:    "rag_evidence_pack_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"query": "why did the build fail?", "evidence": []any{}},
		},
		Artifacts: []types.Artifact{
			{ArtifactID: "artifact_evidence_pack", ArtifactType: "evidence_pack", Path: artifactPath, Classification: "restricted"},
		},
		CreatedAt:   now.Add(-time.Minute),
		UpdatedAt:   now.Add(-time.Minute),
		SubmittedAt: now.Add(-time.Minute),
	}
	if err := jobStore.CreateJob(context.Background(), sourceJob); err != nil {
		t.Fatalf("create source job: %v", err)
	}

	backend := &mutableFakeBackend{}
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	submitResp, err := svc.SubmitJob(aliceUserCtx(), types.SubmitJobRequest{
		TaskType: "propose_patch",
		InputRefs: []types.InputRef{
			{Type: "artifact", URI: "artifact://artifact_evidence_pack"},
		},
		TaskParams: map[string]any{
			"problem":             "fix the generated header issue",
			"validation_commands": []any{"go test ./..."},
		},
		OutputSchema: types.OutputSchemaRef{Name: "patch_proposal_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	inputManifest := loadJSONFileForTest(t, filepath.Join(jobDir, "input_manifest.json"))
	inputRefs, ok := inputManifest["input_refs"].([]any)
	if !ok || len(inputRefs) != 1 {
		t.Fatalf("unexpected input manifest: %#v", inputManifest)
	}
	firstRef := inputRefs[0].(map[string]any)
	metadata, ok := firstRef["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("expected metadata in input manifest, got %#v", firstRef)
	}
	if metadata["resolved_path"] != artifactPath {
		t.Fatalf("expected resolved artifact path %q, got %#v", artifactPath, metadata)
	}
	if metadata["source_job_id"] != "job_source" {
		t.Fatalf("expected source_job_id=job_source, got %#v", metadata)
	}
	if firstRef["classification"] != "restricted" {
		t.Fatalf("expected source classification to propagate into the manifest, got %#v", firstRef)
	}
	stored, err := jobStore.GetJob(context.Background(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get consuming job: %v", err)
	}
	if got := stored.Request.InputRefs[0].Classification; got != "restricted" {
		t.Fatalf("expected consuming policy state to retain restricted classification, got %q", got)
	}
}

func TestArtifactResolutionHonorsInspectFullTraceRelease(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	sourceDir := filepath.Join(runRoot, "job_inspect_source")
	if err := os.MkdirAll(sourceDir, 0o700); err != nil {
		t.Fatalf("mkdir source dir: %v", err)
	}
	evidencePath := filepath.Join(sourceDir, "evidence_pack.json")
	retrievalPath := filepath.Join(sourceDir, "retrieval_result.json")
	if err := os.WriteFile(evidencePath, []byte(`{"evidence":[]}`), 0o600); err != nil {
		t.Fatalf("write evidence artifact: %v", err)
	}
	if err := os.WriteFile(retrievalPath, []byte(`{"private_trace":true}`), 0o600); err != nil {
		t.Fatalf("write retrieval artifact: %v", err)
	}

	now := time.Now().UTC()
	jobStore := store.NewMemoryJobStore()
	sourceJob := types.Job{
		ID: "job_inspect_source", TaskType: "inspect_repo", State: types.JobStateSucceeded, SubmittedBy: "alice",
		Request: types.SubmitJobRequest{
			TaskType: "inspect_repo", TaskParams: map[string]any{"include_full_trace": false},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		},
		Result: &types.Result{SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: map[string]any{}},
		Artifacts: []types.Artifact{
			{ArtifactID: "artifact_evidence_pack", ArtifactType: "evidence_pack", Path: evidencePath, Classification: "internal"},
			{ArtifactID: "artifact_retrieval_result", ArtifactType: "retrieval_result", Path: retrievalPath, Classification: "internal"},
		},
		CreatedAt: now, UpdatedAt: now, SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), sourceJob); err != nil {
		t.Fatalf("create source job: %v", err)
	}
	svc := New(jobStore, &mutableFakeBackend{}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	request := func(artifactID string) (types.SubmitJobResponse, error) {
		return svc.SubmitJob(aliceUserCtx(), types.SubmitJobRequest{
			TaskType:     "propose_patch",
			InputRefs:    []types.InputRef{{Type: "artifact", URI: "artifact://" + artifactID}},
			TaskParams:   map[string]any{"problem": "inspect the evidence"},
			OutputSchema: types.OutputSchemaRef{Name: "patch_proposal_pack_v1"},
		})
	}
	if _, err := request("artifact_retrieval_result"); err == nil {
		t.Fatal("expected hidden retrieval trace to be unavailable through artifact resolution")
	}
	if _, err := svc.GetArtifactMetadata(aliceUserCtx(), "artifact_retrieval_result", nil); err == nil {
		t.Fatal("expected hidden retrieval trace metadata to be unavailable")
	}
	if _, err := request("artifact_evidence_pack"); err != nil {
		t.Fatalf("expected released evidence pack to remain consumable: %v", err)
	}
}

func TestSubmitAndIngestProposePatchWorkerFromArtifactEvidence(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	artifactDir := filepath.Join(runRoot, "job_source")
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	artifactPath := filepath.Join(artifactDir, "evidence_pack.json")
	if err := os.WriteFile(artifactPath, []byte(`{"evidence":[{"id":"ev_001","claim":"generated header missing","source_refs":[{"path":"broker/pkg/service/service.go","line_start":12,"line_end":34}]}]}`), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	now := time.Now().UTC()
	jobStore := store.NewMemoryJobStore()
	sourceJob := types.Job{
		ID:          "job_source",
		TaskType:    "rag_compress",
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		Request: types.SubmitJobRequest{
			TaskType:     "rag_compress",
			TaskParams:   map[string]any{"allow_artifact_release": true},
			OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
		},
		Result: &types.Result{
			SchemaName:    "rag_evidence_pack_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"query": "why did the build fail?", "evidence": []any{}},
		},
		Artifacts: []types.Artifact{
			{ArtifactID: "artifact_evidence_pack", ArtifactType: "evidence_pack", Path: artifactPath, Classification: "restricted"},
		},
		CreatedAt:   now.Add(-time.Minute),
		UpdatedAt:   now.Add(-time.Minute),
		SubmittedAt: now.Add(-time.Minute),
	}
	if err := jobStore.CreateJob(context.Background(), sourceJob); err != nil {
		t.Fatalf("create source job: %v", err)
	}

	backend := &mutableFakeBackend{}
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	submitResp, err := svc.SubmitJob(aliceUserCtx(), types.SubmitJobRequest{
		TaskType: "propose_patch",
		InputRefs: []types.InputRef{
			{Type: "artifact", URI: "artifact://artifact_evidence_pack"},
		},
		TaskParams: map[string]any{
			"problem":             "fix the generated header issue",
			"validation_commands": []any{"go test ./..."},
			"allowed_paths":       []any{"broker/pkg/service"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "patch_proposal_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	jobDir := filepath.Join(runRoot, submitResp.JobID)
	cmd := exec.Command(
		"python3",
		filepath.Join(repoRoot, "workers", "rag-compression", "main.py"),
		"--job-spec", filepath.Join(jobDir, "job_spec.json"),
		"--input-manifest", filepath.Join(jobDir, "input_manifest.json"),
		"--output-dir", jobDir,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run propose_patch worker: %v: %s", err, string(output))
	}

	backend.status = backends.RunStatus{
		BackendRunID: "run-1",
		State:        types.JobStateSucceeded,
		RawState:     "COMPLETED",
		ExitCode:     "0:0",
	}

	got, err := svc.GetJob(aliceUserCtx(), submitResp.JobID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.Result == nil || got.Result.SchemaName != "patch_proposal_pack_v1" {
		t.Fatalf("expected patch_proposal_pack_v1 result, got %#v", got.Result)
	}
	patches, ok := got.Result.Payload["patches"].([]any)
	if !ok || len(patches) == 0 {
		t.Fatalf("expected patch proposals, got %#v", got.Result.Payload)
	}
	if !artifactTypesInclude(got.Artifacts, "retrieval_plan", "retrieval_trace", "chunk_manifest", "rerank_result", "evidence_pack", "retrieval_result", "patch_plan", "validation_report") {
		t.Fatalf("expected patch and validation artifacts, got %#v", got.Artifacts)
	}
}

func artifactPathForType(artifacts []types.Artifact, artifactType string) string {
	for _, artifact := range artifacts {
		if artifact.ArtifactType == artifactType {
			return artifact.Path
		}
	}
	return ""
}

func TestSubmitJobCacheHitForDocumentSummary(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	inputPath := filepath.Join(runRoot, "doc.txt")
	if err := os.WriteFile(inputPath, []byte("same content"), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	firstResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + inputPath},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit first job: %v", err)
	}

	now := time.Now().UTC()
	firstJob, err := jobStore.GetJob(context.Background(), firstResp.JobID)
	if err != nil {
		t.Fatalf("get first job: %v", err)
	}
	firstJob.State = types.JobStateSucceeded
	firstJob.Result = &types.Result{
		SchemaName:    "document_summary_v1",
		SchemaVersion: "1.0.0",
		Payload:       map[string]any{"summary": "cached"},
	}
	firstJob.Artifacts = []types.Artifact{{ArtifactID: "artifact_1", ArtifactType: "excerpt"}}
	firstJob.CompletedAt = &now
	firstJob.UpdatedAt = now
	if err := jobStore.UpdateJob(context.Background(), firstJob); err != nil {
		t.Fatalf("update first job: %v", err)
	}

	secondResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + inputPath},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit second job: %v", err)
	}
	if secondResp.Cache.Status != "hit" {
		t.Fatalf("expected cache hit, got %q", secondResp.Cache.Status)
	}
	if secondResp.ReleasedResult == nil {
		t.Fatal("expected inline released result on cache hit")
	}
	if secondResp.ReleasedResult.Result == nil || secondResp.ReleasedResult.Result.Payload["summary"] != "cached" {
		t.Fatalf("expected cached inline released result, got %#v", secondResp.ReleasedResult)
	}

	secondJob, err := svc.GetJob(context.Background(), secondResp.JobID)
	if err != nil {
		t.Fatalf("get second job: %v", err)
	}
	if secondJob.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded state, got %q", secondJob.State)
	}
	if secondJob.BackendKind != "cache" {
		t.Fatalf("expected cache backend, got %q", secondJob.BackendKind)
	}
	if secondJob.CacheSourceJobID != firstJob.ID {
		t.Fatalf("expected cache source job id %q, got %q", firstJob.ID, secondJob.CacheSourceJobID)
	}
	if secondJob.Result == nil || secondJob.Result.Payload["summary"] != "cached" {
		t.Fatalf("expected cached result, got %#v", secondJob.Result)
	}

	storedSecondJob, err := jobStore.GetJob(context.Background(), secondResp.JobID)
	if err != nil {
		t.Fatalf("get stored second job: %v", err)
	}
	if storedSecondJob.Result != nil {
		t.Fatalf("expected stored cache-hit alias to omit duplicate result, got %#v", storedSecondJob.Result)
	}
}

func TestSubmitJobCacheHitForRepoSummary(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := filepath.Join(runRoot, "repo")
	if err := os.MkdirAll(filepath.Join(repoRoot, "broker"), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "broker", "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write repo file: %v", err)
	}

	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	firstResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "repo_summary",
		InputRefs: []types.InputRef{
			{Type: "directory", URI: "file://" + repoRoot},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit first job: %v", err)
	}

	now := time.Now().UTC()
	firstJob, err := jobStore.GetJob(context.Background(), firstResp.JobID)
	if err != nil {
		t.Fatalf("get first job: %v", err)
	}
	firstJob.State = types.JobStateSucceeded
	firstJob.Result = &types.Result{
		SchemaName:    "repo_summary_v1",
		SchemaVersion: "1.0.0",
		Payload:       map[string]any{"summary": "cached repo summary"},
	}
	firstJob.CompletedAt = &now
	firstJob.UpdatedAt = now
	if err := jobStore.UpdateJob(context.Background(), firstJob); err != nil {
		t.Fatalf("update first job: %v", err)
	}

	secondResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "repo_summary",
		InputRefs: []types.InputRef{
			{Type: "directory", URI: "file://" + repoRoot},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit second job: %v", err)
	}
	if secondResp.Cache.Status != "hit" {
		t.Fatalf("expected cache hit, got %q", secondResp.Cache.Status)
	}
	if secondResp.ReleasedResult == nil {
		t.Fatal("expected inline released result on cache hit")
	}
	if secondResp.ReleasedResult.Result == nil || secondResp.ReleasedResult.Result.Payload["summary"] != "cached repo summary" {
		t.Fatalf("expected cached repo inline released result, got %#v", secondResp.ReleasedResult)
	}

	secondJob, err := svc.GetJob(context.Background(), secondResp.JobID)
	if err != nil {
		t.Fatalf("get second job: %v", err)
	}
	if secondJob.CacheSourceJobID != firstJob.ID {
		t.Fatalf("expected cache source job id %q, got %q", firstJob.ID, secondJob.CacheSourceJobID)
	}
	if secondJob.Result == nil || secondJob.Result.Payload["summary"] != "cached repo summary" {
		t.Fatalf("expected cached repo result, got %#v", secondJob.Result)
	}
}

func TestSubmitJobCacheHitForInspectRepoAnswerReady(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := filepath.Join(runRoot, "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("mkdir repo: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "README.md"), []byte("# repo\n"), 0o644); err != nil {
		t.Fatalf("write repo file: %v", err)
	}

	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	firstResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace routing", "mode": "answer"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit first job: %v", err)
	}

	now := time.Now().UTC()
	firstJob, err := jobStore.GetJob(context.Background(), firstResp.JobID)
	if err != nil {
		t.Fatalf("get first job: %v", err)
	}
	firstJob.State = types.JobStateSucceeded
	firstJob.Result = &types.Result{
		SchemaName:    "repo_inspection_v2",
		SchemaVersion: "2.0.0",
		Payload: map[string]any{
			"mode":   "answer",
			"query":  "trace routing",
			"answer": "done",
			"findings": []any{
				map[string]any{"summary": "done", "evidence_refs": []any{"ev_1"}},
			},
			"evidence": []any{
				map[string]any{"id": "ev_1", "path": "README.md", "source_refs": []any{map[string]any{"path": "README.md", "line_start": 1, "line_end": 1}}},
			},
			"quality": map[string]any{
				"result":       "answer_ready",
				"retrieval":    "gpu",
				"reranking":    "gpu",
				"synthesis":    "gpu",
				"answer_ready": true,
			},
			"runtime": map[string]any{
				"attempts": []any{
					map[string]any{"operation": "semantic_retrieval", "status": "succeeded"},
					map[string]any{"operation": "rerank", "status": "succeeded"},
					map[string]any{"operation": "synthesis", "status": "succeeded"},
				},
				"worker_phase_timings_ms": map[string]any{
					"run_inspection": 84.031,
					"total":          90.768,
					"cache_hit":      false,
				},
			},
			"retrieval": map[string]any{
				"lexical_candidates":  1,
				"semantic_candidates": 1,
				"reranked_candidates": 1,
				"chunk_build_substage_timings_ms": map[string]any{
					"discover_source_files_ms":  23.891,
					"file_chunk_bundle_load_ms": 2.583,
				},
				"setup_timings_ms": map[string]any{
					"query_stage_cache_probe_ms": 2.004,
				},
				"stage_timings_ms": map[string]any{
					"build_syntax_chunks_ms":  68.796,
					"ensure_lexical_index_ms": 6.264,
				},
				"tail_timings_ms": map[string]any{
					"artifact_payloads_ms": 0.009,
				},
			},
		},
	}
	firstJob.Artifacts = []types.Artifact{{ArtifactID: "artifact_1", ArtifactType: "evidence_pack"}}
	firstJob.CompletedAt = &now
	firstJob.UpdatedAt = now
	if err := jobStore.UpdateJob(context.Background(), firstJob); err != nil {
		t.Fatalf("update first job: %v", err)
	}

	secondResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace routing", "mode": "answer"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit second job: %v", err)
	}
	if secondResp.Cache.Status != "hit" {
		t.Fatalf("expected cache hit, got %q", secondResp.Cache.Status)
	}
	if secondResp.ReleasedResult == nil || secondResp.ReleasedResult.Result == nil {
		t.Fatalf("expected inline released result, got %#v", secondResp.ReleasedResult)
	}
	if secondResp.ReleasedResult.Result.Payload["answer"] != "done" {
		t.Fatalf("expected cached answer-ready result, got %#v", secondResp.ReleasedResult.Result)
	}

	secondJob, err := svc.GetJob(context.Background(), secondResp.JobID)
	if err != nil {
		t.Fatalf("get second job: %v", err)
	}
	if secondJob.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded state, got %q", secondJob.State)
	}
	if secondJob.BackendKind != "cache" {
		t.Fatalf("expected cache backend, got %q", secondJob.BackendKind)
	}
	if secondJob.CacheSourceJobID != firstJob.ID {
		t.Fatalf("expected cache source job id %q, got %q", firstJob.ID, secondJob.CacheSourceJobID)
	}
	if secondJob.Result == nil || secondJob.Result.Payload["answer"] != "done" {
		t.Fatalf("expected resolved cache-hit result, got %#v", secondJob.Result)
	}
	retrieval, _ := secondJob.Result.Payload["retrieval"].(map[string]any)
	if retrieval["query_stage_cache_hit"] != true {
		t.Fatalf("expected cache-hit result to mark query_stage_cache_hit, got %#v", retrieval)
	}
	if retrieval["chunk_cache_reused_files"] != 0 || retrieval["chunk_cache_rebuilt_files"] != 0 {
		t.Fatalf("expected cache-hit retrieval chunk counters to clear stale work, got %#v", retrieval)
	}
	chunkBuildTimings, _ := retrieval["chunk_build_substage_timings_ms"].(map[string]any)
	if chunkBuildTimings["discover_source_files_ms"] != 0.0 || chunkBuildTimings["file_chunk_bundle_load_ms"] != 0.0 {
		t.Fatalf("expected cache-hit retrieval chunk-build timings to clear stale work, got %#v", chunkBuildTimings)
	}
	stageTimings, _ := retrieval["stage_timings_ms"].(map[string]any)
	if stageTimings["build_syntax_chunks_ms"] != 0.0 || stageTimings["ensure_lexical_index_ms"] != 0.0 {
		t.Fatalf("expected cache-hit retrieval stage timings to clear stale work, got %#v", stageTimings)
	}
	runtime, _ := secondJob.Result.Payload["runtime"].(map[string]any)
	if runtime["result_source"] != "broker_cache_hit" {
		t.Fatalf("expected broker cache-hit runtime source, got %#v", runtime)
	}
	if got := stringValue(runtime["broker_result_source"]); got != "cache_hit" {
		t.Fatalf("expected broker cache-hit broker_result_source, got %#v", runtime)
	}
	brokerPhaseTimings, _ := runtime["broker_phase_timings_ms"].(map[string]any)
	if brokerPhaseTimings == nil {
		t.Fatalf("expected broker cache-hit broker_phase_timings_ms, got %#v", runtime)
	}
	if brokerPhaseTimings["cache_key_ms"] == nil || brokerPhaseTimings["total_submit_ms"] == nil {
		t.Fatalf("expected cache-hit broker timings to include cache_key_ms and total_submit_ms, got %#v", brokerPhaseTimings)
	}
	workerPhaseTimings, _ := runtime["worker_phase_timings_ms"].(map[string]any)
	if workerPhaseTimings["cache_hit"] != true {
		t.Fatalf("expected cache-hit worker timings marker, got %#v", workerPhaseTimings)
	}
	if workerPhaseTimings["run_inspection"] != 0.0 || workerPhaseTimings["total"] != 0.0 {
		t.Fatalf("expected cache-hit worker timings to clear stale work, got %#v", workerPhaseTimings)
	}

	storedSecondJob, err := jobStore.GetJob(context.Background(), secondResp.JobID)
	if err != nil {
		t.Fatalf("get stored second job: %v", err)
	}
	if storedSecondJob.Result != nil {
		t.Fatalf("expected stored cache-hit alias to omit duplicate result, got %#v", storedSecondJob.Result)
	}
}

func TestSubmitJobCoalescesInflightInspectRepoRequest(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := filepath.Join(runRoot, "repo")
	if err := os.MkdirAll(repoRoot, 0o755); err != nil {
		t.Fatalf("mkdir repo: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "README.md"), []byte("# repo\n"), 0o644); err != nil {
		t.Fatalf("write repo file: %v", err)
	}

	jobStore := store.NewMemoryJobStore()
	backend := &countingFakeBackend{status: backends.RunStatus{State: types.JobStateQueued, RawState: "PENDING"}}
	svc := New(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		runRoot,
		repoRoot,
	)

	firstResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace routing", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit first job: %v", err)
	}
	secondResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace routing", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit second job: %v", err)
	}
	if backend.submitCalls != 1 {
		t.Fatalf("expected one backend submit, got %d", backend.submitCalls)
	}
	if secondResp.Cache.Status != "hit" {
		t.Fatalf("expected second submit to alias inflight job, got cache status %q", secondResp.Cache.Status)
	}
	if secondResp.State != types.JobStateQueued {
		t.Fatalf("expected queued alias state, got %q", secondResp.State)
	}
	if secondResp.ReleasedResult != nil {
		t.Fatalf("did not expect inline result for inflight alias, got %#v", secondResp.ReleasedResult)
	}

	secondJob, err := svc.GetJob(context.Background(), secondResp.JobID)
	if err != nil {
		t.Fatalf("get second job: %v", err)
	}
	if secondJob.CacheSourceJobID != firstResp.JobID {
		t.Fatalf("expected cache source job id %q, got %q", firstResp.JobID, secondJob.CacheSourceJobID)
	}
	if secondJob.State != types.JobStateQueued {
		t.Fatalf("expected queued alias job while source queued, got %q", secondJob.State)
	}

	now := time.Now().UTC()
	firstJob, err := jobStore.GetJob(context.Background(), firstResp.JobID)
	if err != nil {
		t.Fatalf("get first job: %v", err)
	}
	firstJob.State = types.JobStateSucceeded
	firstJob.Result = &types.Result{
		SchemaName:    "repo_inspection_v2",
		SchemaVersion: "2.0.0",
		Payload: map[string]any{
			"mode":     "evidence",
			"query":    "trace routing",
			"evidence": []any{},
			"quality": map[string]any{
				"result":       "evidence_only",
				"retrieval":    "lexical_degraded",
				"reranking":    "unavailable",
				"synthesis":    "not_requested",
				"answer_ready": false,
			},
			"retrieval": map[string]any{
				"lexical_candidates":  2,
				"semantic_candidates": 0,
				"reranked_candidates": 0,
			},
		},
	}
	firstJob.CompletedAt = &now
	firstJob.UpdatedAt = now
	if err := jobStore.UpdateJob(context.Background(), firstJob); err != nil {
		t.Fatalf("update first job: %v", err)
	}

	resolvedSecondJob, err := svc.GetJob(context.Background(), secondResp.JobID)
	if err != nil {
		t.Fatalf("get resolved second job: %v", err)
	}
	if resolvedSecondJob.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded alias after source completion, got %q", resolvedSecondJob.State)
	}
	if resolvedSecondJob.Result == nil {
		t.Fatal("expected resolved alias result")
	}
	retrieval, _ := resolvedSecondJob.Result.Payload["retrieval"].(map[string]any)
	if retrieval["query_stage_cache_hit"] != true {
		t.Fatalf("expected alias result to mark query_stage_cache_hit, got %#v", retrieval)
	}
	if retrieval["chunk_cache_reused_files"] != 0 || retrieval["chunk_cache_rebuilt_files"] != 0 {
		t.Fatalf("expected alias cache-hit retrieval chunk counters to clear stale work, got %#v", retrieval)
	}

	storedResolvedSecondJob, err := jobStore.GetJob(context.Background(), secondResp.JobID)
	if err != nil {
		t.Fatalf("get stored resolved second job: %v", err)
	}
	if storedResolvedSecondJob.State != types.JobStateSucceeded {
		t.Fatalf("expected persisted alias to become succeeded, got %q", storedResolvedSecondJob.State)
	}
	if storedResolvedSecondJob.BackendState != "CACHE_HIT" {
		t.Fatalf("expected persisted alias backend state CACHE_HIT, got %q", storedResolvedSecondJob.BackendState)
	}
	if storedResolvedSecondJob.Result != nil {
		t.Fatalf("expected persisted alias to continue omitting duplicate result, got %#v", storedResolvedSecondJob.Result)
	}
}

func TestListJobsRefreshesVisibleNonTerminalCacheAliases(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	svc := New(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		t.TempDir(),
		t.TempDir(),
	)

	now := time.Now().UTC()
	source := types.Job{
		ID:          "job_source",
		TaskType:    "document_summary",
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		Request: types.SubmitJobRequest{
			TaskType:     "document_summary",
			OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		},
		Result: &types.Result{
			SchemaName:    "document_summary_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"summary": "done"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
		CompletedAt: &now,
	}
	alias := types.Job{
		ID:               "job_alias",
		TaskType:         "document_summary",
		State:            types.JobStateRunning,
		SubmittedBy:      "alice",
		Request:          source.Request,
		CreatedAt:        now,
		UpdatedAt:        now,
		SubmittedAt:      now,
		CacheStatus:      "hit",
		CacheSourceJobID: source.ID,
		BackendKind:      "cache",
		BackendState:     "CACHE_ALIAS",
	}
	if err := jobStore.CreateJob(context.Background(), source); err != nil {
		t.Fatalf("create source job: %v", err)
	}
	if err := jobStore.CreateJob(context.Background(), alias); err != nil {
		t.Fatalf("create alias job: %v", err)
	}

	jobs, err := svc.ListJobs(aliceUserCtx())
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	var listedAlias *types.Job
	for i := range jobs {
		if jobs[i].ID == alias.ID {
			listedAlias = &jobs[i]
			break
		}
	}
	if listedAlias == nil {
		t.Fatal("expected listed alias job")
	}
	if listedAlias.State != types.JobStateSucceeded {
		t.Fatalf("expected listed alias to refresh to succeeded, got %q", listedAlias.State)
	}
	if listedAlias.BackendState != "CACHE_HIT" {
		t.Fatalf("expected listed alias backend state CACHE_HIT, got %q", listedAlias.BackendState)
	}
	if listedAlias.Result == nil || listedAlias.Result.Payload["summary"] != "done" {
		t.Fatalf("expected listed alias result to resolve from source, got %#v", listedAlias.Result)
	}

	storedAlias, err := jobStore.GetJob(context.Background(), alias.ID)
	if err != nil {
		t.Fatalf("get stored alias job: %v", err)
	}
	if storedAlias.State != types.JobStateSucceeded {
		t.Fatalf("expected stored alias to refresh to succeeded, got %q", storedAlias.State)
	}
	if storedAlias.BackendState != "CACHE_HIT" {
		t.Fatalf("expected stored alias backend state CACHE_HIT, got %q", storedAlias.BackendState)
	}
	if storedAlias.Result != nil {
		t.Fatalf("expected stored alias to omit duplicate result, got %#v", storedAlias.Result)
	}
}

func TestLookupReusableInflightJobUsesCacheKeyStoreLookup(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	jobStore := &cacheKeyLookupOnlyStore{MemoryJobStore: store.NewMemoryJobStore()}
	svc := New(jobStore, fakeBackend{}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	now := time.Now().UTC()
	for _, job := range []types.Job{
		{
			ID:          "job_old",
			TaskType:    "inspect_repo",
			State:       types.JobStateQueued,
			CacheKey:    "sha256:test",
			CreatedAt:   now,
			UpdatedAt:   now,
			SubmittedAt: now,
		},
		{
			ID:          "job_new",
			TaskType:    "inspect_repo",
			State:       types.JobStateRunning,
			CacheKey:    "sha256:test",
			CreatedAt:   now.Add(time.Second),
			UpdatedAt:   now.Add(time.Second),
			SubmittedAt: now.Add(time.Second),
		},
	} {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job %s: %v", job.ID, err)
		}
	}

	got, err := svc.lookupReusableInflightJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
	}, "sha256:test")
	if err != nil {
		t.Fatalf("lookup reusable inflight job: %v", err)
	}
	if got == nil || got.ID != "job_new" {
		t.Fatalf("expected latest cache-key candidate, got %#v", got)
	}
	if jobStore.listJobsCalls != 0 {
		t.Fatalf("expected ListJobs to be bypassed, got %d calls", jobStore.listJobsCalls)
	}
}

func TestCloneResultDeepCopiesPayload(t *testing.T) {
	original := &types.Result{
		SchemaName:    "repo_inspection_v2",
		SchemaVersion: "2.0.0",
		Payload: map[string]any{
			"quality": map[string]any{
				"result": "answer_ready",
			},
			"findings": []any{
				map[string]any{
					"summary":       "done",
					"evidence_refs": []any{"ev_1"},
				},
			},
			"warnings": []string{"A"},
		},
	}

	cloned := cloneResult(original)
	if cloned == nil || cloned.Payload == nil {
		t.Fatalf("expected cloned payload, got %#v", cloned)
	}

	quality := cloned.Payload["quality"].(map[string]any)
	quality["result"] = "evidence_only"
	findings := cloned.Payload["findings"].([]any)
	findings[0].(map[string]any)["summary"] = "changed"
	warnings := cloned.Payload["warnings"].([]string)
	warnings[0] = "B"

	if original.Payload["quality"].(map[string]any)["result"] != "answer_ready" {
		t.Fatalf("expected original quality to remain unchanged, got %#v", original.Payload["quality"])
	}
	if original.Payload["findings"].([]any)[0].(map[string]any)["summary"] != "done" {
		t.Fatalf("expected original findings to remain unchanged, got %#v", original.Payload["findings"])
	}
	if original.Payload["warnings"].([]string)[0] != "A" {
		t.Fatalf("expected original warnings to remain unchanged, got %#v", original.Payload["warnings"])
	}
}

func TestStartInspectRepoPrewarmSubmitsInspectRepoJob(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	repoRoot := t.TempDir()
	svc := New(jobStore, fakeBackend{}, log.New(io.Discard, "", 0), t.TempDir(), repoRoot)
	targetRepo := t.TempDir()

	started := svc.StartInspectRepoPrewarm(context.Background(), log.New(io.Discard, "", 0), targetRepo, "warm repo cache")
	if !started {
		t.Fatal("expected inspect_repo prewarm to start")
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		jobs, err := jobStore.ListJobs(context.Background())
		if err != nil {
			t.Fatalf("list jobs: %v", err)
		}
		if len(jobs) == 0 {
			time.Sleep(10 * time.Millisecond)
			continue
		}
		job := jobs[0]
		if job.TaskType != "inspect_repo" {
			t.Fatalf("task_type = %q, want inspect_repo", job.TaskType)
		}
		if job.SubmittedBy != brokerInspectRepoPrewarmActor {
			t.Fatalf("submitted_by = %q, want %q", job.SubmittedBy, brokerInspectRepoPrewarmActor)
		}
		if job.Request.OutputSchema.Name != "repo_inspection_v2" {
			t.Fatalf("output_schema = %#v", job.Request.OutputSchema)
		}
		if got := stringValue(job.Request.TaskParams["query"]); got != "warm repo cache" {
			t.Fatalf("query = %q", got)
		}
		if got := stringValue(job.Request.TaskParams["mode"]); got != "evidence" {
			t.Fatalf("mode = %q", got)
		}
		if got, ok := job.Request.TaskParams["_broker_index_prewarm"].(bool); !ok || !got {
			t.Fatalf("expected _broker_index_prewarm flag, got %#v", job.Request.TaskParams)
		}
		if len(job.Request.InputRefs) != 1 || job.Request.InputRefs[0].Type != "repo" {
			t.Fatalf("input_refs = %#v", job.Request.InputRefs)
		}
		if got, want := job.Request.InputRefs[0].URI, "file://"+targetRepo; got != want {
			t.Fatalf("prewarm repo URI = %q, want %q", got, want)
		}
		if !strings.HasPrefix(job.Request.InputRefs[0].URI, "file://") {
			t.Fatalf("unexpected prewarm repo URI: %q", job.Request.InputRefs[0].URI)
		}
		return
	}

	t.Fatal("timed out waiting for inspect_repo prewarm job submission")
}

func TestStartInspectRepoPrewarmSkipsBlankQuery(t *testing.T) {
	svc := New(store.NewMemoryJobStore(), fakeBackend{}, log.New(io.Discard, "", 0), t.TempDir(), t.TempDir())
	if svc.StartInspectRepoPrewarm(context.Background(), log.New(io.Discard, "", 0), t.TempDir(), "   ") {
		t.Fatal("expected blank prewarm query to be ignored")
	}
}

func TestSubmitJobReturnsInlineReleaseForImmediateLocalInspectRepoCompletion(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	backend := &immediateLocalInspectRepoBackend{runRoot: runRoot}
	svc := New(store.NewMemoryJobStore(), backend, log.New(io.Discard, "", 0), runRoot, repoRoot)

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(context.Background()), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult == nil || resp.ReleasedResult.Result == nil {
		t.Fatalf("expected inline released result, got %#v", resp.ReleasedResult)
	}
	if got := resp.ReleasedResult.Result.Payload["query"]; got != "trace retry_job" {
		t.Fatalf("expected inline result query echo, got %#v", got)
	}
	if resp.ReleasedResult.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded inline release, got %q", resp.ReleasedResult.State)
	}
}

func TestSubmitJobReturnsOpportunisticInlineReleaseForImmediateLocalInspectRepoCompletionByDefault(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	backend := &immediateLocalInspectRepoBackend{runRoot: runRoot}
	svc := New(store.NewMemoryJobStore(), backend, log.New(io.Discard, "", 0), runRoot, repoRoot)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult == nil || resp.ReleasedResult.Result == nil {
		t.Fatalf("expected opportunistic inline released result by default, got %#v", resp.ReleasedResult)
	}
	if resp.ReleasedResult.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded opportunistic inline release, got %q", resp.ReleasedResult.State)
	}
	runtime, _ := resp.ReleasedResult.Result.Payload["runtime"].(map[string]any)
	if runtime == nil {
		t.Fatalf("expected runtime payload, got %#v", resp.ReleasedResult.Result.Payload)
	}
	brokerTimings, _ := runtime["broker_phase_timings_ms"].(map[string]any)
	if brokerTimings == nil {
		t.Fatalf("expected broker_phase_timings_ms in runtime payload, got %#v", runtime)
	}
	if brokerTimings["total_submit_ms"] == nil {
		t.Fatalf("expected total_submit_ms in broker timings, got %#v", brokerTimings)
	}
	if got := stringValue(runtime["broker_result_source"]); got == "" {
		t.Fatalf("expected broker_result_source in runtime payload, got %#v", runtime)
	}
}

func TestSubmitJobReturnsInlineReleaseForInflightLocalInspectRepoAlias(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	svc := New(store.NewMemoryJobStore(), &delayedLocalInspectRepoCompletionBackend{runRoot: runRoot}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	sourceResp, err := svc.SubmitJob(aliceUserCtx(), req)
	if err != nil {
		t.Fatalf("submit source inspect_repo: %v", err)
	}
	go func() {
		time.Sleep(10 * time.Millisecond)
		_ = writeInspectRepoResultForTest(runRoot, sourceResp.JobID, "trace retry_job")
	}()

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(aliceUserCtx()), req)
	if err != nil {
		t.Fatalf("submit alias inspect_repo: %v", err)
	}
	if resp.ReleasedResult == nil || resp.ReleasedResult.Result == nil {
		t.Fatalf("expected inflight alias inline release, got %#v", resp.ReleasedResult)
	}
	if resp.Cache.Status != "hit" {
		t.Fatalf("cache status = %q, want hit", resp.Cache.Status)
	}
	if got := resp.ReleasedResult.Result.Payload["query"]; got != "trace retry_job" {
		t.Fatalf("expected inline alias result query echo, got %#v", got)
	}
}

func TestSubmitJobInflightLocalInspectRepoAliasThrottlesBackendRefreshPolling(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	backend := &countingDelayedLocalInspectRepoResultBackend{runRoot: runRoot, delay: 250 * time.Millisecond}
	svc := New(store.NewMemoryJobStore(), backend, log.New(io.Discard, "", 0), runRoot, repoRoot)

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	firstResp, err := svc.SubmitJob(aliceUserCtx(), req)
	if err != nil {
		t.Fatalf("submit source inspect_repo: %v", err)
	}
	if firstResp.ReleasedResult != nil {
		t.Fatalf("expected slow source submit to remain pending, got %#v", firstResp.ReleasedResult)
	}

	secondResp, err := svc.SubmitJob(WithPreferInlineLocalRelease(aliceUserCtx()), req)
	if err != nil {
		t.Fatalf("submit alias inspect_repo: %v", err)
	}
	if secondResp.ReleasedResult != nil {
		t.Fatalf("expected slow inflight alias to remain pending, got %#v", secondResp.ReleasedResult)
	}
	if secondResp.Cache.Status != "hit" {
		t.Fatalf("expected hit cache status for alias, got %q", secondResp.Cache.Status)
	}
	if got := backend.getRunCalls.Load(); got > 12 {
		t.Fatalf("expected throttled backend refresh polling, got %d GetRun calls", got)
	}
	resultPath := filepath.Join(runRoot, firstResp.JobID, "result.json")
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(resultPath); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for delayed test result write at %s", resultPath)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestSubmitJobInflightLocalInspectRepoAliasSkipsInitialBackendRefreshWhenRunFilesArriveQuickly(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	backend := &countingDelayedLocalInspectRepoResultBackend{runRoot: runRoot, delay: 250 * time.Millisecond}
	svc := New(store.NewMemoryJobStore(), backend, log.New(io.Discard, "", 0), runRoot, repoRoot)

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	sourceResp, err := svc.SubmitJob(aliceUserCtx(), req)
	if err != nil {
		t.Fatalf("submit source inspect_repo: %v", err)
	}
	if sourceResp.ReleasedResult != nil {
		t.Fatalf("expected slow source submit to remain pending, got %#v", sourceResp.ReleasedResult)
	}

	go func() {
		time.Sleep(10 * time.Millisecond)
		_ = writeInspectRepoResultForTest(runRoot, sourceResp.JobID, "trace retry_job")
	}()

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(aliceUserCtx()), req)
	if err != nil {
		t.Fatalf("submit alias inspect_repo: %v", err)
	}
	if resp.ReleasedResult == nil || resp.ReleasedResult.Result == nil {
		t.Fatalf("expected inflight alias inline release, got %#v", resp.ReleasedResult)
	}
	if got := backend.getRunCalls.Load(); got != 0 {
		t.Fatalf("expected no backend GetRun call before quick run-result release, got %d", got)
	}
}

func TestSubmitJobPreferredInlineReleaseUsesBackendCompletionSignal(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	backend := &signaledLocalInspectRepoResultBackend{runRoot: runRoot, delay: 10 * time.Millisecond}
	svc := New(store.NewMemoryJobStore(), backend, log.New(io.Discard, "", 0), runRoot, repoRoot)

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(aliceUserCtx()), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult == nil || resp.ReleasedResult.Result == nil {
		t.Fatalf("expected signaled inline released result, got %#v", resp.ReleasedResult)
	}
	if got := backend.getRunCalls.Load(); got != 0 {
		t.Fatalf("expected completion signal path to avoid backend GetRun calls, got %d", got)
	}
}

func TestGetReleasedResultUsesBackendCompletionSignal(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	backend := &signaledLocalInspectRepoResultBackend{runRoot: runRoot, delay: 50 * time.Millisecond}
	svc := New(store.NewMemoryJobStore(), backend, log.New(io.Discard, "", 0), runRoot, repoRoot)

	resp, err := svc.SubmitJob(aliceUserCtx(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult != nil {
		t.Fatalf("expected initial submit to remain pending, got %#v", resp.ReleasedResult)
	}

	release, err := svc.GetReleasedResult(aliceUserCtx(), resp.JobID)
	if err != nil {
		t.Fatalf("get released result: %v", err)
	}
	if release.Result == nil || release.Result.Payload["query"] != "trace retry_job" {
		t.Fatalf("unexpected release: %#v", release)
	}
	if got := backend.getRunCalls.Load(); got != 0 {
		t.Fatalf("expected completion signal path to avoid backend GetRun calls, got %d", got)
	}
}

func TestSubmitJobReturnsInlineReleaseFromDirectRunResultBeforeTerminalState(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	svc := New(store.NewMemoryJobStore(), &delayedLocalInspectRepoResultBackend{runRoot: runRoot, delay: 10 * time.Millisecond}, log.New(io.Discard, "", 0), runRoot, repoRoot)

	resp, err := svc.SubmitJob(WithPreferInlineLocalRelease(context.Background()), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace retry_job", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.ReleasedResult == nil || resp.ReleasedResult.Result == nil {
		t.Fatalf("expected inline released result from direct run files, got %#v", resp.ReleasedResult)
	}
	if resp.ReleasedResult.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded inline release, got %q", resp.ReleasedResult.State)
	}
}

func containsAny(text string, needles []string) bool {
	for _, needle := range needles {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

func containsAnyString(items []any, needles []string) bool {
	for _, item := range items {
		text, ok := item.(string)
		if !ok {
			continue
		}
		for _, needle := range needles {
			if text == needle {
				return true
			}
		}
	}
	return false
}

func artifactTypesInclude(artifacts []types.Artifact, required ...string) bool {
	if len(required) == 0 {
		return true
	}
	seen := make(map[string]struct{}, len(artifacts))
	for _, artifact := range artifacts {
		seen[artifact.ArtifactType] = struct{}{}
	}
	for _, want := range required {
		if _, ok := seen[want]; !ok {
			return false
		}
	}
	return true
}
