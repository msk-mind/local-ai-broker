package service

import (
	"context"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type captureSubmitBackend struct {
	job *types.Job
}

func (b *captureSubmitBackend) Name() string { return "capture-submit" }

func (b *captureSubmitBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	copied := job
	b.job = &copied
	return backends.SubmitResponse{
		BackendKind:  "capture-submit",
		BackendRunID: "run-capture",
		InitialState: types.JobStateQueued,
	}, nil
}

func (b *captureSubmitBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return backends.RunStatus{}, nil
}

func (b *captureSubmitBackend) CancelRun(context.Context, string) error { return nil }

type captureInlineWarmSubmitBackend struct {
	job    *types.Job
	bundle *backends.InlineExecutionBundle
}

func (b *captureInlineWarmSubmitBackend) Name() string { return "capture-inline-warm" }

func (b *captureInlineWarmSubmitBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	copied := job
	b.job = &copied
	return backends.SubmitResponse{
		BackendKind:  "capture-inline-warm",
		BackendRunID: "fallback-run",
		InitialState: types.JobStateQueued,
	}, nil
}

func (b *captureInlineWarmSubmitBackend) SubmitWarmInspectRepoRun(_ context.Context, job types.Job, bundle backends.InlineExecutionBundle) (backends.SubmitResponse, bool, error) {
	copied := job
	b.job = &copied
	copiedBundle := bundle
	b.bundle = &copiedBundle
	return backends.SubmitResponse{
		BackendKind:  "local",
		BackendRunID: "job-inline",
		InitialState: types.JobStateDispatching,
	}, true, nil
}

func (b *captureInlineWarmSubmitBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return backends.RunStatus{}, nil
}

func (b *captureInlineWarmSubmitBackend) CancelRun(context.Context, string) error { return nil }

func testFloatValue(value any) float64 {
	switch typed := value.(type) {
	case float64:
		return typed
	case float32:
		return float64(typed)
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case int32:
		return float64(typed)
	default:
		return 0
	}
}

func TestMaybeAttachInspectRepoFingerprintHintDoesNotAttachRequestCacheHash(t *testing.T) {
	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file:///tmp/repo"},
		},
		TaskParams: map[string]any{
			"query": "trace broker timeout",
			"mode":  "answer",
		},
	}

	maybeAttachInspectRepoFingerprintHint(&req, "git:abc123", nil, nil)

	if got := stringValue(req.TaskParams["_broker_repository_state_fingerprint"]); got != "" {
		t.Fatalf("expected broker request-cache hash to be omitted as inspect_repo fingerprint hint, got %q", got)
	}
	if got := req.InputRefs[0].ContentHash; got != "git:abc123" {
		t.Fatalf("expected broker-computed content hash to be preserved on inspect_repo input ref, got %q", got)
	}
}

func TestMaybeAttachInspectRepoFingerprintHintPreservesDirtyPaths(t *testing.T) {
	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file:///tmp/repo"},
		},
		TaskParams: map[string]any{
			"query": "trace broker timeout",
			"mode":  "answer",
		},
	}

	maybeAttachInspectRepoFingerprintHint(&req, "git:abc123", []string{"service.py", "pkg/api.go"}, nil)

	got, ok := req.TaskParams["_broker_touched_paths"].([]string)
	if !ok {
		t.Fatalf("expected dirty paths hint slice, got %#v", req.TaskParams["_broker_touched_paths"])
	}
	if len(got) != 2 || got[0] != "service.py" || got[1] != "pkg/api.go" {
		t.Fatalf("unexpected dirty paths hint: %#v", got)
	}
}

func TestMaybeAttachInspectRepoFingerprintHintPreservesCleanWorktreeFiles(t *testing.T) {
	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file:///tmp/repo"},
		},
		TaskParams: map[string]any{
			"query": "trace broker timeout",
			"mode":  "answer",
		},
	}

	maybeAttachInspectRepoFingerprintHint(&req, "git:abc123", nil, []string{"helper.py", "service.py"})

	got, ok := req.TaskParams["_broker_clean_worktree_files"].([]string)
	if !ok {
		t.Fatalf("expected clean worktree files hint slice, got %#v", req.TaskParams["_broker_clean_worktree_files"])
	}
	if len(got) != 2 || got[0] != "helper.py" || got[1] != "service.py" {
		t.Fatalf("unexpected clean worktree files hint: %#v", got)
	}
}

func TestMaybeAttachInspectRepoFingerprintHintSkipsExcludedRequests(t *testing.T) {
	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file:///tmp/repo"},
		},
		TaskParams: map[string]any{
			"query":              "trace broker timeout",
			"excluded_dir_names": []string{"vendor"},
		},
	}

	maybeAttachInspectRepoFingerprintHint(&req, "git:abc123", []string{"service.py"}, nil)

	if got := stringValue(req.TaskParams["_broker_repository_state_fingerprint"]); got != "" {
		t.Fatalf("expected broker fingerprint hint to be omitted for excluded requests, got %q", got)
	}
	if got := req.InputRefs[0].ContentHash; got != "" {
		t.Fatalf("expected excluded inspect_repo request to omit preserved content hash, got %q", got)
	}
}

func TestSubmitJobOmitsInspectRepoFingerprintHintToBackendRequest(t *testing.T) {
	runRoot := t.TempDir()
	repoDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoDir, "README.md"), []byte("# demo\n"), 0o644); err != nil {
		t.Fatalf("write inspect input: %v", err)
	}
	backend := &captureSubmitBackend{}
	svc := NewWithAuditAndOptions(
		store.NewMemoryJobStore(),
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		".",
		Options{},
	)

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoDir, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "trace routing",
			"mode":  "evidence",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	if _, err := svc.SubmitJob(aliceUserCtx(), req); err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if backend.job == nil {
		t.Fatal("expected backend to receive submitted job")
	}
	if got := stringValue(backend.job.Request.TaskParams["_broker_repository_state_fingerprint"]); got != "" {
		t.Fatalf("expected submitted inspect_repo request to omit broker repository fingerprint hint, got %q", got)
	}
	if got := backend.job.Request.InputRefs[0].ContentHash; !strings.HasPrefix(got, "git:") && !strings.HasPrefix(got, "meta:") {
		t.Fatalf("expected submitted inspect_repo request to preserve computed repo content hash, got %q", got)
	}
}

func TestSubmitJobOmitsInspectRepoFingerprintHintWhenExclusionsPresent(t *testing.T) {
	runRoot := t.TempDir()
	repoDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoDir, "README.md"), []byte("# demo\n"), 0o644); err != nil {
		t.Fatalf("write inspect input: %v", err)
	}
	backend := &captureSubmitBackend{}
	svc := NewWithAuditAndOptions(
		store.NewMemoryJobStore(),
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		".",
		Options{},
	)

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoDir, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query":              "trace routing",
			"mode":               "evidence",
			"excluded_dir_names": []string{"vendor"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	if _, err := svc.SubmitJob(aliceUserCtx(), req); err != nil {
		t.Fatalf("submit inspect_repo with exclusions: %v", err)
	}
	if backend.job == nil {
		t.Fatal("expected backend to receive submitted job")
	}
	if got := stringValue(backend.job.Request.TaskParams["_broker_repository_state_fingerprint"]); got != "" {
		t.Fatalf("expected submitted inspect_repo request with exclusions to omit broker fingerprint hint, got %q", got)
	}
	if got := backend.job.Request.InputRefs[0].ContentHash; got != "" {
		t.Fatalf("expected submitted inspect_repo request with exclusions to omit preserved content hash, got %q", got)
	}
}

func TestSubmitJobInlineWarmInspectRepoSkipsStagedExecutionBundleFiles(t *testing.T) {
	runRoot := t.TempDir()
	repoDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoDir, "README.md"), []byte("# demo\n"), 0o644); err != nil {
		t.Fatalf("write inspect input: %v", err)
	}
	backend := &captureInlineWarmSubmitBackend{}
	svc := NewWithAuditAndOptions(
		store.NewMemoryJobStore(),
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		".",
		Options{},
	)

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoDir, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "trace routing",
			"mode":  "evidence",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	resp, err := svc.SubmitJob(aliceUserCtx(), req)
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	if resp.JobID == "" {
		t.Fatal("expected job id")
	}
	if backend.bundle == nil {
		t.Fatal("expected inline warm execution bundle")
	}
	if got := stringValue(backend.bundle.JobSpec["job_id"]); got != resp.JobID {
		t.Fatalf("job spec job_id = %q, want %q", got, resp.JobID)
	}
	if _, ok := backend.bundle.InputManifest["input_refs"]; !ok {
		t.Fatalf("expected inline input manifest to include input_refs, got %#v", backend.bundle.InputManifest)
	}
	jobDir := filepath.Join(runRoot, resp.JobID)
	if _, err := os.Stat(filepath.Join(jobDir, "job_spec.json")); !os.IsNotExist(err) {
		t.Fatalf("expected inline warm path to skip staged job_spec.json, stat err=%v", err)
	}
	if _, err := os.Stat(filepath.Join(jobDir, "execution_plan.json")); !os.IsNotExist(err) {
		t.Fatalf("expected inline warm path to skip staged execution_plan.json, stat err=%v", err)
	}
	if _, err := os.Stat(filepath.Join(jobDir, "input_manifest.json")); !os.IsNotExist(err) {
		t.Fatalf("expected inline warm path to skip staged input_manifest.json, stat err=%v", err)
	}
}

func TestSubmitJobInspectRepoOmitsBrokerHintAndUsesInputManifestFingerprintInWorker(t *testing.T) {
	runRoot := t.TempDir()
	repoDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoDir, "service.py"), []byte("def retry_job(job_id):\n    return submit_job(job_id)\n\ndef submit_job(job_id):\n    return job_id\n"), 0o644); err != nil {
		t.Fatalf("write inspect input: %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoDir, "mcp.go"), []byte("package mcp\n\nfunc InspectRepo(query string) string {\n\treturn query\n}\n"), 0o644); err != nil {
		t.Fatalf("write inspect input: %v", err)
	}
	svc := NewWithAuditAndOptions(
		store.NewMemoryJobStore(),
		fakeBackend{},
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		".",
		Options{},
	)

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repoDir, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "Trace the retry_job service call chain",
			"mode":  "evidence",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	resp, err := svc.SubmitJob(aliceUserCtx(), req)
	if err != nil {
		t.Fatalf("submit inspect_repo: %v", err)
	}
	submitted, err := svc.GetJob(aliceUserCtx(), resp.JobID)
	if err != nil {
		t.Fatalf("get submitted inspect_repo job: %v", err)
	}
	if got := stringValue(submitted.Request.TaskParams["_broker_repository_state_fingerprint"]); got != "" {
		t.Fatalf("expected submitted inspect_repo request to omit broker fingerprint hint, got %q", got)
	}
	if got := submitted.Request.InputRefs[0].ContentHash; !strings.HasPrefix(got, "git:") && !strings.HasPrefix(got, "meta:") {
		t.Fatalf("expected submitted inspect_repo job to preserve content hash in input manifest, got %q", got)
	}

	jobDir := filepath.Join(runRoot, resp.JobID)
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
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

	result := loadJSONFileForTest(t, filepath.Join(jobDir, "result.json"))
	payload, _ := result["payload"].(map[string]any)
	retrieval, _ := payload["retrieval"].(map[string]any)
	setupTimings, _ := retrieval["setup_timings_ms"].(map[string]any)
	if got := testFloatValue(setupTimings["repository_fingerprint_ms"]); got != 0 {
		t.Fatalf("expected manifest-hinted worker run to skip repository fingerprinting, got %v ms", got)
	}
	sources, _ := retrieval["fingerprint_sources"].([]any)
	if len(sources) != 1 || stringValue(sources[0]) != "input_manifest" {
		t.Fatalf("expected worker run to report metadata fingerprint source, got %#v", retrieval["fingerprint_sources"])
	}
}
