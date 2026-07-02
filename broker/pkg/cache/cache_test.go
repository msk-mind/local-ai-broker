package cache

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestKeyForRequestFileTasks(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "doc.txt")
	if err := os.WriteFile(path, []byte("hello world"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	key, cacheable, err := KeyForRequest(types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + path},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("key for request: %v", err)
	}
	if !cacheable {
		t.Fatal("expected cacheable request")
	}
	if key == "" {
		t.Fatal("expected non-empty key")
	}
}

func TestFindCompletedJobByCacheKey(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_1",
		State:       types.JobStateSucceeded,
		CacheKey:    "sha256:test",
		Result:      &types.Result{SchemaName: "document_summary_v1", SchemaVersion: "1.0.0", Payload: map[string]any{"summary": "ok"}},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}
	found, err := FindCompletedJobByCacheKey(context.Background(), jobStore, "sha256:test")
	if err != nil {
		t.Fatalf("find by cache key: %v", err)
	}
	if found == nil || found.ID != "job_1" {
		t.Fatalf("expected job_1, got %#v", found)
	}
}

func TestKeyForRequestDirectoryTasks(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "broker"), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "broker", "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	key, cacheable, err := KeyForRequest(types.SubmitJobRequest{
		TaskType: "repo_summary",
		InputRefs: []types.InputRef{
			{Type: "directory", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
	})
	if err != nil {
		t.Fatalf("key for request: %v", err)
	}
	if !cacheable {
		t.Fatal("expected repo_summary request to be cacheable")
	}
	if key == "" {
		t.Fatal("expected non-empty key")
	}
}

func TestKeyForRequestDirectoryTasksUsesMetadataFingerprintForNonGitDirs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "broker", "main.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.WriteFile(path, []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	req := types.SubmitJobRequest{
		TaskType: "repo_summary",
		InputRefs: []types.InputRef{
			{Type: "directory", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
	}

	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request A: %v", err)
	}
	if !cacheable {
		t.Fatal("expected repo_summary request to be cacheable")
	}
	if keyA == "" {
		t.Fatal("expected non-empty key")
	}

	time.Sleep(2 * time.Millisecond)
	if err := os.WriteFile(path, []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go: %v", err)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request B: %v", err)
	}
	if !cacheable {
		t.Fatal("expected repo_summary request to be cacheable")
	}
	if keyA == keyB {
		t.Fatalf("expected metadata fingerprint cache key to change after file update, got %q", keyA)
	}
}

func TestKeyForRequestDirectoryTasksBecomesUncacheableWhenFingerprintBudgetExceeded(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "broker", "main.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.WriteFile(path, []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	originalMaxEntries := metadataFingerprintMaxEntries
	metadataFingerprintMaxEntries = 0
	defer func() { metadataFingerprintMaxEntries = originalMaxEntries }()

	key, cacheable, err := KeyForRequest(types.SubmitJobRequest{
		TaskType: "repo_summary",
		InputRefs: []types.InputRef{
			{Type: "directory", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
	})
	if err != nil {
		t.Fatalf("key for request: %v", err)
	}
	if cacheable {
		t.Fatalf("expected repo_summary request to fall back to uncacheable, got key=%q", key)
	}
	if key != "" {
		t.Fatalf("expected empty key when fingerprint budget is exceeded, got %q", key)
	}
}

func TestKeyForRequestDirectoryTasksUsesGitFingerprintWhenAvailable(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	req := types.SubmitJobRequest{
		TaskType: "repo_summary",
		InputRefs: []types.InputRef{
			{Type: "directory", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
	}

	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request A: %v", err)
	}
	if !cacheable {
		t.Fatal("expected repo_summary request to be cacheable")
	}
	if keyA == "" {
		t.Fatal("expected non-empty key")
	}

	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go: %v", err)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request B: %v", err)
	}
	if !cacheable {
		t.Fatal("expected repo_summary request to be cacheable")
	}
	if keyA == keyB {
		t.Fatalf("expected git fingerprint cache key to change for dirty repo, got %q", keyA)
	}
}

func TestKeyForRequestChangesWhenExecutionProfileChanges(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "doc.txt")
	if err := os.WriteFile(path, []byte("hello world"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	baseReq := types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + path},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		ExecutionProfile: types.ExecutionProfile{
			Tier:    "p40-rag-compression",
			Model:   "gpt-oss-20b.p40",
			Runtime: "llama.cpp",
		},
	}
	keyA, cacheable, err := KeyForRequest(baseReq)
	if err != nil {
		t.Fatalf("key for request A: %v", err)
	}
	if !cacheable {
		t.Fatal("expected cacheable request")
	}

	baseReq.ExecutionProfile.Model = "qwen3-coder-30b.a100"
	keyB, cacheable, err := KeyForRequest(baseReq)
	if err != nil {
		t.Fatalf("key for request B: %v", err)
	}
	if !cacheable {
		t.Fatal("expected cacheable request")
	}
	if keyA == keyB {
		t.Fatalf("expected cache key to change when model changes, got %q", keyA)
	}
}

func TestKeyForRequestInspectRepoIsNotCacheable(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# demo\n"), 0o644); err != nil {
		t.Fatalf("write README.md: %v", err)
	}

	key, cacheable, err := KeyForRequest(types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_pack_v1"},
	})
	if err != nil {
		t.Fatalf("key for request: %v", err)
	}
	if cacheable {
		t.Fatalf("expected inspect_repo to be uncacheable, got key=%q", key)
	}
	if key != "" {
		t.Fatalf("expected empty key for uncacheable inspect_repo, got %q", key)
	}
}

func TestKeyForRequestDebugWithLocalContextIsNotCacheable(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# demo\n"), 0o644); err != nil {
		t.Fatalf("write README.md: %v", err)
	}

	key, cacheable, err := KeyForRequest(types.SubmitJobRequest{
		TaskType: "debug_with_local_context",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "debug_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("key for request: %v", err)
	}
	if cacheable {
		t.Fatalf("expected debug_with_local_context to be uncacheable, got key=%q", key)
	}
	if key != "" {
		t.Fatalf("expected empty key for uncacheable debug_with_local_context, got %q", key)
	}
}

func runGitForTest(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v failed: %v: %s", args, err, string(output))
	}
}
