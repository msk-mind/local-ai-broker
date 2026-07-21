package cache

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type directLookupStore struct {
	job             types.Job
	listJobsCalls   int
	directFindCalls int
}

func (s *directLookupStore) CreateJob(context.Context, types.Job) error { return nil }
func (s *directLookupStore) GetJob(context.Context, string) (types.Job, error) {
	return types.Job{}, store.ErrNotFound
}
func (s *directLookupStore) UpdateJob(context.Context, types.Job) error { return nil }
func (s *directLookupStore) ListJobs(context.Context) ([]types.Job, error) {
	s.listJobsCalls++
	return []types.Job{s.job}, nil
}
func (s *directLookupStore) FindCompletedJobByCacheKey(context.Context, string) (types.Job, error) {
	s.directFindCalls++
	return s.job, nil
}

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

func TestKeyDetailsForRequestUsesProvidedInputRefContentHashWithoutFilesystemProbe(t *testing.T) {
	details, err := KeyDetailsForRequest(types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{
				Type:        "repo",
				URI:         "file:///path/that/does/not/need/to/exist",
				ContentHash: "git:provided-fingerprint",
			},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("key details for request: %v", err)
	}
	if !details.Cacheable {
		t.Fatal("expected cacheable request")
	}
	if details.Key == "" {
		t.Fatal("expected non-empty key")
	}
	if details.ContentHash != "git:provided-fingerprint" {
		t.Fatalf("expected provided content hash to be reused, got %q", details.ContentHash)
	}
	if len(details.DirtyPaths) != 0 {
		t.Fatalf("expected provided content hash path hint to omit dirty paths, got %#v", details.DirtyPaths)
	}
}

func TestKeyDetailsForRequestCapturesDirtyPathsForGitRepo(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "service.py"), []byte("def retry_job(job_id):\n    return job_id\n"), 0o644); err != nil {
		t.Fatalf("write tracked file: %v", err)
	}
	if err := exec.Command("git", "init", "-q", dir).Run(); err != nil {
		t.Fatalf("git init: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "config", "user.email", "test@example.invalid").Run(); err != nil {
		t.Fatalf("git config email: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "config", "user.name", "Test").Run(); err != nil {
		t.Fatalf("git config name: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "add", ".").Run(); err != nil {
		t.Fatalf("git add: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "commit", "-qm", "initial").Run(); err != nil {
		t.Fatalf("git commit: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "service.py"), []byte("def retry_job(job_id):\n    value = 1\n    return job_id + value\n"), 0o644); err != nil {
		t.Fatalf("write dirty file: %v", err)
	}

	details, err := KeyDetailsForRequest(types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("key details for request: %v", err)
	}
	if !details.Cacheable {
		t.Fatal("expected cacheable request")
	}
	if len(details.DirtyPaths) != 1 || details.DirtyPaths[0] != "service.py" {
		t.Fatalf("unexpected dirty paths: %#v", details.DirtyPaths)
	}
	if !strings.HasPrefix(details.ContentHash, "git:") {
		t.Fatalf("expected git fingerprint content hash, got %q", details.ContentHash)
	}
}

func TestKeyDetailsForRequestCapturesCleanWorktreeFilesForCleanGitRepo(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "service.py"), []byte("def retry_job(job_id):\n    return job_id\n"), 0o644); err != nil {
		t.Fatalf("write tracked file: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "helper.py"), []byte("def helper():\n    return retry_job(1)\n"), 0o644); err != nil {
		t.Fatalf("write tracked file: %v", err)
	}
	if err := exec.Command("git", "init", "-q", dir).Run(); err != nil {
		t.Fatalf("git init: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "config", "user.email", "test@example.invalid").Run(); err != nil {
		t.Fatalf("git config email: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "config", "user.name", "Test").Run(); err != nil {
		t.Fatalf("git config name: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "add", ".").Run(); err != nil {
		t.Fatalf("git add: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "commit", "-qm", "initial").Run(); err != nil {
		t.Fatalf("git commit: %v", err)
	}

	details, err := KeyDetailsForRequest(types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("key details for request: %v", err)
	}
	if !details.Cacheable {
		t.Fatal("expected cacheable request")
	}
	if len(details.DirtyPaths) != 0 {
		t.Fatalf("expected clean repo to omit dirty paths, got %#v", details.DirtyPaths)
	}
	if got := details.CleanWorktreeFiles; len(got) != 2 || got[0] != "helper.py" || got[1] != "service.py" {
		t.Fatalf("unexpected clean worktree files: %#v", got)
	}
	if !strings.HasPrefix(details.ContentHash, "git:") {
		t.Fatalf("expected git fingerprint content hash, got %q", details.ContentHash)
	}
}

func TestKeyDetailsForRequestFindsGitScopeFromFilesystemBeforeRevParseShowToplevel(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "service.py"), []byte("def retry_job(job_id):\n    return job_id\n"), 0o644); err != nil {
		t.Fatalf("write tracked file: %v", err)
	}
	if err := exec.Command("git", "init", "-q", dir).Run(); err != nil {
		t.Fatalf("git init: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "config", "user.email", "test@example.invalid").Run(); err != nil {
		t.Fatalf("git config email: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "config", "user.name", "Test").Run(); err != nil {
		t.Fatalf("git config name: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "add", ".").Run(); err != nil {
		t.Fatalf("git add: %v", err)
	}
	if err := exec.Command("git", "-C", dir, "commit", "-qm", "initial").Run(); err != nil {
		t.Fatalf("git commit: %v", err)
	}

	originalRunGit := runGitFunc
	originalUserCacheDirFunc := userCacheDirFunc
	t.Cleanup(func() {
		runGitFunc = originalRunGit
		userCacheDirFunc = originalUserCacheDirFunc
	})
	cacheDir := t.TempDir()
	userCacheDirFunc = func() (string, error) { return cacheDir, nil }
	runGitFunc = func(gitPath, workDir string, args ...string) (string, error) {
		if len(args) == 2 && args[0] == "rev-parse" && args[1] == "--show-toplevel" {
			return "", fmt.Errorf("unexpected rev-parse --show-toplevel call")
		}
		return runGit(gitPath, workDir, args...)
	}

	details, err := KeyDetailsForRequest(types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("key details for request: %v", err)
	}
	if !details.Cacheable {
		t.Fatal("expected cacheable request")
	}
	if !strings.HasPrefix(details.ContentHash, "git:") {
		t.Fatalf("expected git fingerprint content hash, got %q", details.ContentHash)
	}
}

func TestKeyForRequestIgnoresNonSemanticTaskParams(t *testing.T) {
	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{
				Type:        "repo",
				URI:         "file:///tmp/repo",
				ContentHash: "git:provided-fingerprint",
			},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		TaskParams: map[string]any{
			"query": "trace retry_job",
			"mode":  "evidence",
		},
	}

	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request A: %v", err)
	}
	if !cacheable {
		t.Fatal("expected inspect_repo request to be cacheable")
	}

	req.TaskParams["client_nonce"] = "worker-warm"
	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request B: %v", err)
	}
	if !cacheable {
		t.Fatal("expected inspect_repo request with client_nonce to stay cacheable")
	}
	if keyA != keyB {
		t.Fatalf("expected non-semantic client_nonce to be ignored, got %q vs %q", keyA, keyB)
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

func TestFindCompletedJobByCacheKeyPrefersDirectStoreLookup(t *testing.T) {
	now := time.Now().UTC()
	jobStore := &directLookupStore{
		job: types.Job{
			ID:          "job_direct",
			State:       types.JobStateSucceeded,
			CacheKey:    "sha256:direct",
			Result:      &types.Result{SchemaName: "document_summary_v1", Payload: map[string]any{"summary": "ok"}},
			CreatedAt:   now,
			UpdatedAt:   now,
			SubmittedAt: now,
		},
	}

	found, err := FindCompletedJobByCacheKey(context.Background(), jobStore, "sha256:direct")
	if err != nil {
		t.Fatalf("find by cache key: %v", err)
	}
	if found == nil || found.ID != "job_direct" {
		t.Fatalf("expected direct lookup job, got %#v", found)
	}
	if jobStore.directFindCalls != 1 {
		t.Fatalf("expected one direct lookup call, got %d", jobStore.directFindCalls)
	}
	if jobStore.listJobsCalls != 0 {
		t.Fatalf("expected ListJobs to be skipped, got %d calls", jobStore.listJobsCalls)
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

func TestKeyForRequestDirtyRepoChangesAcrossRepeatedEditsWithSameStatusShape(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(1) }\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(2) }\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go iteration A: %v", err)
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request A: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable dirty request A, got cacheable=%v key=%q", cacheable, keyA)
	}

	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(3) }\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go iteration B: %v", err)
	}
	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request B: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable dirty request B, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA == keyB {
		t.Fatalf("expected dirty git fingerprint cache key to change across repeated edits with same status shape, got %q", keyA)
	}
}

func TestKeyForRequestRepeatedSameStagedStateReusesCachedStagedEntriesWithoutIndexLookup(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(1) }\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalUserCacheDir := userCacheDirFunc
	originalDirtyMemo := gitDirtyFastpathMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	originalFingerprintMemo := gitFingerprintMemo.entries
	defer func() {
		runGitFunc = originalRunGit
		userCacheDirFunc = originalUserCacheDir
		gitDirtyFastpathMemo.entries = originalDirtyMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
		gitFingerprintMemo.entries = originalFingerprintMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitDirtyFastpathMemo.entries = nil
	gitCleanFastpathMemo.entries = nil
	gitFingerprintMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	if _, _, err := KeyForRequest(req); err != nil {
		t.Fatalf("warmup key for request: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(2) }\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go staged: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for staged request A: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable staged request A, got cacheable=%v key=%q", cacheable, keyA)
	}

	lsFilesCalls := 0
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		if len(args) >= 4 && args[1] == "ls-files" && args[2] == "-s" && args[3] == "-z" {
			lsFilesCalls++
		}
		return originalRunGit(gitPath, workdir, args...)
	}
	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for staged request B: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable staged request B, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA != keyB {
		t.Fatalf("expected repeated same staged state to reuse fingerprint, got %q vs %q", keyA, keyB)
	}
	if lsFilesCalls != 0 {
		t.Fatalf("expected repeated same staged state to skip ls-files index lookup, got %d calls", lsFilesCalls)
	}
}

func TestKeyForRequestRepeatedStagedStateWithNewUnstagedContentSkipsIndexLookupAndChangesKey(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(1) }\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalUserCacheDir := userCacheDirFunc
	originalDirtyMemo := gitDirtyFastpathMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	originalFingerprintMemo := gitFingerprintMemo.entries
	defer func() {
		runGitFunc = originalRunGit
		userCacheDirFunc = originalUserCacheDir
		gitDirtyFastpathMemo.entries = originalDirtyMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
		gitFingerprintMemo.entries = originalFingerprintMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitDirtyFastpathMemo.entries = nil
	gitCleanFastpathMemo.entries = nil
	gitFingerprintMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	if _, _, err := KeyForRequest(req); err != nil {
		t.Fatalf("warmup key for request: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(2) }\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go staged: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for staged request A: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable staged request A, got cacheable=%v key=%q", cacheable, keyA)
	}

	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(3) }\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go unstaged overlay: %v", err)
	}
	lsFilesCalls := 0
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		if len(args) >= 4 && args[1] == "ls-files" && args[2] == "-s" && args[3] == "-z" {
			lsFilesCalls++
		}
		return originalRunGit(gitPath, workdir, args...)
	}
	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for staged request B: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable staged request B, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA == keyB {
		t.Fatalf("expected staged state with new unstaged content to change fingerprint, got %q", keyA)
	}
	if lsFilesCalls != 0 {
		t.Fatalf("expected repeated staged state with unstaged overlay to skip ls-files index lookup, got %d calls", lsFilesCalls)
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

func TestGitFingerprintManifestSkipsRepeatedTreeProbe(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	headCalls := 0
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		if len(args) >= 2 && args[0] == "rev-parse" && args[1] == "HEAD^{tree}" {
			headCalls++
		}
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = originalRunGitExitCode

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("first key for request: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable first key, got cacheable=%v key=%q", cacheable, keyA)
	}
	if headCalls != 0 {
		t.Fatalf("expected filesystem/gitdir head fastpath to skip first tree probe, got %d", headCalls)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("second key for request: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable second key, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA != keyB {
		t.Fatalf("expected stable key across repeated calls, got %q vs %q", keyA, keyB)
	}
	if headCalls != 0 {
		t.Fatalf("expected manifest reuse to keep skipping tree probe, got %d calls", headCalls)
	}

	manifestPath := gitFingerprintManifestPath(dir, ".")
	if !strings.Contains(manifestPath, filepath.Join("local-ai-broker", "git-fingerprint-cache")) {
		t.Fatalf("unexpected manifest path %q", manifestPath)
	}
	if _, err := os.Stat(manifestPath); err != nil {
		t.Fatalf("expected manifest to exist at %q: %v", manifestPath, err)
	}
}

func TestGitFingerprintMemoSkipsRepeatedTreeProbeWithoutManifestRead(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	originalTTL := gitFingerprintMemoTTL
	originalMemo := gitFingerprintMemo.entries
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
		gitFingerprintMemoTTL = originalTTL
		gitFingerprintMemo.entries = originalMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitFingerprintMemoTTL = time.Minute
	gitFingerprintMemo.entries = nil

	headCalls := 0
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		if len(args) >= 2 && args[0] == "rev-parse" && args[1] == "HEAD^{tree}" {
			headCalls++
		}
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = originalRunGitExitCode

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("first key for request: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable first key, got cacheable=%v key=%q", cacheable, keyA)
	}
	if headCalls != 0 {
		t.Fatalf("expected head fastpath to skip first tree probe, got %d", headCalls)
	}

	manifestPath := gitFingerprintManifestPath(dir, ".")
	if err := os.Remove(manifestPath); err != nil {
		t.Fatalf("remove manifest: %v", err)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("second key for request: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable second key, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA != keyB {
		t.Fatalf("expected stable key across repeated calls, got %q vs %q", keyA, keyB)
	}
	if headCalls != 0 {
		t.Fatalf("expected memo reuse to keep skipping tree probe, got %d calls", headCalls)
	}
}

func TestGitFingerprintCleanSmallRepoFastpathSkipsSecondStatusProbe(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		t.Fatalf("first key for request failed: cacheable=%v err=%v", cacheable, err)
	}

	statusCalls := 0
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		if len(args) >= 1 && args[0] == "status" {
			statusCalls++
		}
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = originalRunGitExitCode

	keyA, cacheable, err := KeyForRequest(req)
	if err != nil || !cacheable || keyA == "" {
		t.Fatalf("second key for request failed: cacheable=%v key=%q err=%v", cacheable, keyA, err)
	}
	if statusCalls != 0 {
		t.Fatalf("expected clean fastpath to skip second status probe, got %d calls", statusCalls)
	}
}

func TestGitFingerprintCleanFastpathMemoSkipsRepeatedGitProbes(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	originalMemoTTL := gitFingerprintMemoTTL
	originalFingerprintMemo := gitFingerprintMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
		gitFingerprintMemoTTL = originalMemoTTL
		gitFingerprintMemo.entries = originalFingerprintMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitFingerprintMemoTTL = time.Minute
	gitFingerprintMemo.entries = nil
	gitCleanFastpathMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		t.Fatalf("first key for request failed: cacheable=%v err=%v", cacheable, err)
	}

	gitCalls := 0
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		gitCalls++
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = originalRunGitExitCode

	keyA, cacheable, err := KeyForRequest(req)
	if err != nil || !cacheable || keyA == "" {
		t.Fatalf("second key for request failed: cacheable=%v key=%q err=%v", cacheable, keyA, err)
	}
	if gitCalls != 0 {
		t.Fatalf("expected clean fastpath memo to skip repeated git probes, got %d calls", gitCalls)
	}
}

func TestGitFingerprintMemoInvalidatesWhenStatusChanges(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalTTL := gitFingerprintMemoTTL
	originalMemo := gitFingerprintMemo.entries
	defer func() {
		gitFingerprintMemoTTL = originalTTL
		gitFingerprintMemo.entries = originalMemo
	}()
	gitFingerprintMemoTTL = time.Minute
	gitFingerprintMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("first key for request: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable first key, got cacheable=%v key=%q", cacheable, keyA)
	}

	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go: %v", err)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("second key for request: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable second key, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA == keyB {
		t.Fatalf("expected memoized git fingerprint key to change after status update, got %q", keyA)
	}
}

func TestGitFingerprintUsesSplitGitProbesInsteadOfStatus(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	var commands []string
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = func(gitPath, workdir string, args ...string) (int, string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGitExitCode(gitPath, workdir, args...)
	}

	_, cacheable, err := KeyForRequest(types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("key for request: %v", err)
	}
	if !cacheable {
		t.Fatal("expected inspect_repo request to be cacheable")
	}

	joined := strings.Join(commands, "\n")
	if !strings.Contains(joined, "status --porcelain=v1 -z --untracked-files=all --ignored=no") {
		t.Fatalf("expected single porcelain status probe, got commands:\n%s", joined)
	}
	if strings.Contains(joined, "diff-index --quiet --cached HEAD") || strings.Contains(joined, "diff-files --quiet") || strings.Contains(joined, "--name-status") {
		t.Fatalf("expected split quiet/name-status probes to be removed, got commands:\n%s", joined)
	}
	if !strings.Contains(joined, ":(exclude).broker-live-tests") || !strings.Contains(joined, ":(exclude)slurm-*.out") {
		t.Fatalf("expected broker ephemeral excludes in commands, got commands:\n%s", joined)
	}
}

func TestGitFingerprintUsesPorcelainStatusWhenDirty(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go: %v", err)
	}

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	var commands []string
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = func(gitPath, workdir string, args ...string) (int, string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGitExitCode(gitPath, workdir, args...)
	}

	_, cacheable, err := KeyForRequest(types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("key for request: %v", err)
	}
	if !cacheable {
		t.Fatal("expected inspect_repo request to be cacheable")
	}

	joined := strings.Join(commands, "\n")
	if !strings.Contains(joined, "status --porcelain=v1 -z --untracked-files=all --ignored=no") {
		t.Fatalf("expected porcelain status probe on dirty repo, got commands:\n%s", joined)
	}
	if strings.Contains(joined, "diff-index --quiet --cached HEAD") || strings.Contains(joined, "diff-files --quiet") || strings.Contains(joined, "--name-status") {
		t.Fatalf("expected split quiet/name-status probes to be removed, got commands:\n%s", joined)
	}
}

func TestKeyForRequestRepeatedSameDirtyStateSkipsPorcelainStatus(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go: %v", err)
	}

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	originalDirtyMemo := gitDirtyFastpathMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	originalFingerprintMemo := gitFingerprintMemo.entries
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
		gitDirtyFastpathMemo.entries = originalDirtyMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
		gitFingerprintMemo.entries = originalFingerprintMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitDirtyFastpathMemo.entries = nil
	gitCleanFastpathMemo.entries = nil
	gitFingerprintMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("first key for request: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable dirty request, got cacheable=%v key=%q", cacheable, keyA)
	}

	var commands []string
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = func(gitPath, workdir string, args ...string) (int, string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGitExitCode(gitPath, workdir, args...)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("second key for request: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable dirty request, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA != keyB {
		t.Fatalf("expected unchanged dirty state to keep the same key, got %q vs %q", keyA, keyB)
	}
	joined := strings.Join(commands, "\n")
	if strings.Contains(joined, "status --porcelain=v1 -z --untracked-files=all --ignored=no") {
		t.Fatalf("expected dirty fastpath to skip porcelain status, got commands:\n%s", joined)
	}
}

func TestKeyForRequestDirtyFastpathInvalidatesOnNewDirtyFile(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go: %v", err)
	}

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	originalDirtyMemo := gitDirtyFastpathMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	originalFingerprintMemo := gitFingerprintMemo.entries
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
		gitDirtyFastpathMemo.entries = originalDirtyMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
		gitFingerprintMemo.entries = originalFingerprintMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitDirtyFastpathMemo.entries = nil
	gitCleanFastpathMemo.entries = nil
	gitFingerprintMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("first key for request: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable dirty request, got cacheable=%v key=%q", cacheable, keyA)
	}

	if err := os.WriteFile(filepath.Join(dir, "scratch.txt"), []byte("new file\n"), 0o644); err != nil {
		t.Fatalf("write scratch.txt: %v", err)
	}

	var commands []string
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = func(gitPath, workdir string, args ...string) (int, string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGitExitCode(gitPath, workdir, args...)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("second key for request: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable dirty request, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA == keyB {
		t.Fatalf("expected dirty fastpath invalidation to change key after new dirty file, got %q", keyA)
	}
	joined := strings.Join(commands, "\n")
	if !strings.Contains(joined, "status --porcelain=v1 -z --untracked-files=all --ignored=no") {
		t.Fatalf("expected invalidated dirty fastpath to fall back to porcelain status, got commands:\n%s", joined)
	}
}

func TestKeyForRequestRepeatedDirtyEditsReuseDirtyMemoWithoutStatusProbe(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(1) }\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalRunGitExitCode := runGitExitCodeFunc
	originalUserCacheDir := userCacheDirFunc
	originalDirtyMemo := gitDirtyFastpathMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	originalFingerprintMemo := gitFingerprintMemo.entries
	defer func() {
		runGitFunc = originalRunGit
		runGitExitCodeFunc = originalRunGitExitCode
		userCacheDirFunc = originalUserCacheDir
		gitDirtyFastpathMemo.entries = originalDirtyMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
		gitFingerprintMemo.entries = originalFingerprintMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitDirtyFastpathMemo.entries = nil
	gitCleanFastpathMemo.entries = nil
	gitFingerprintMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}

	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(2) }\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go iteration A: %v", err)
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request A: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable dirty request A, got cacheable=%v key=%q", cacheable, keyA)
	}

	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(3) }\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go iteration B: %v", err)
	}

	var commands []string
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGit(gitPath, workdir, args...)
	}
	runGitExitCodeFunc = func(gitPath, workdir string, args ...string) (int, string, error) {
		commands = append(commands, strings.Join(args, " "))
		return originalRunGitExitCode(gitPath, workdir, args...)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request B: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable dirty request B, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA == keyB {
		t.Fatalf("expected dirty git fingerprint cache key to change across repeated edits, got %q", keyA)
	}

	joined := strings.Join(commands, "\n")
	if strings.Contains(joined, "status --porcelain=v1 -z --untracked-files=all --ignored=no") {
		t.Fatalf("expected repeated dirty edit to reuse memoized dirty state without porcelain status, got commands:\n%s", joined)
	}
}

func TestDirtyPathStateSignaturesMatchDetectsChangedDirtyPathWithoutScopeScan(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "service.py")
	if err := os.WriteFile(target, []byte("print('a')\n"), 0o644); err != nil {
		t.Fatalf("write target: %v", err)
	}
	cached := map[string]worktreeSignatureMemoEntry{
		"service.py": {
			StateSignature: fileStateSignature(target),
		},
	}
	if !dirtyPathStateSignaturesMatch(dir, []string{"service.py"}, cached) {
		t.Fatalf("expected unchanged dirty path signature to match")
	}
	if err := os.WriteFile(target, []byte("print('b changed')\n"), 0o644); err != nil {
		t.Fatalf("rewrite target: %v", err)
	}
	if dirtyPathStateSignaturesMatch(dir, []string{"service.py"}, cached) {
		t.Fatalf("expected changed dirty path signature to invalidate fastpath")
	}
}


func TestGitStatusPayloadFromPorcelainPreservesRenameAndUntracked(t *testing.T) {
	staged, unstaged, untracked, err := gitStatusPayloadFromPorcelain("R  old.go\x00new.go\x00 M dirty.go\x00?? scratch.txt\x00")
	if err != nil {
		t.Fatalf("gitStatusPayloadFromPorcelain: %v", err)
	}
	stagedEntries, _ := parseGitNameStatus(staged)
	if len(stagedEntries) != 1 || stagedEntries[0].code != "R" || stagedEntries[0].sourcePath != "old.go" || stagedEntries[0].path != "new.go" {
		t.Fatalf("expected staged rename entry, got %#v", stagedEntries)
	}
	unstagedEntries, _ := parseGitNameStatus(unstaged)
	if len(unstagedEntries) != 1 || unstagedEntries[0].code != "M" || unstagedEntries[0].path != "dirty.go" {
		t.Fatalf("expected unstaged modify entry, got %#v", unstagedEntries)
	}
	untrackedPaths := parseGitUntracked(untracked)
	if len(untrackedPaths) != 1 || untrackedPaths[0] != "scratch.txt" {
		t.Fatalf("expected untracked scratch.txt, got %#v", untrackedPaths)
	}
}

func TestKeyForRequestIgnoresBrokerEphemeralUntrackedArtifacts(t *testing.T) {
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
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request A: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable request, got cacheable=%v key=%q", cacheable, keyA)
	}

	if err := os.WriteFile(filepath.Join(dir, "slurm-123.out"), []byte("noise\n"), 0o644); err != nil {
		t.Fatalf("write slurm output: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(dir, ".broker-live-tests"), 0o755); err != nil {
		t.Fatalf("mkdir .broker-live-tests: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".broker-live-tests", "result.txt"), []byte("noise\n"), 0o644); err != nil {
		t.Fatalf("write broker live test artifact: %v", err)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request B: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable request, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA != keyB {
		t.Fatalf("expected broker ephemeral artifacts to be ignored, got %q vs %q", keyA, keyB)
	}
}

func TestKeyForRequestDirectoryMetadataFingerprintIgnoresBrokerEphemeralArtifacts(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "directory", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request A: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable request, got cacheable=%v key=%q", cacheable, keyA)
	}

	if err := os.WriteFile(filepath.Join(dir, "slurm-123.out"), []byte("noise\n"), 0o644); err != nil {
		t.Fatalf("write slurm output: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(dir, ".broker-live-tests"), 0o755); err != nil {
		t.Fatalf("mkdir .broker-live-tests: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".broker-live-tests", "result.txt"), []byte("noise\n"), 0o644); err != nil {
		t.Fatalf("write broker live test artifact: %v", err)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("key for request B: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable request, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA != keyB {
		t.Fatalf("expected broker metadata fingerprint to ignore ephemeral artifacts, got %q vs %q", keyA, keyB)
	}
}

func TestFingerprintDirectoryMetadataUsesRGWhenAvailable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "broker", "main.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.WriteFile(path, []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	originalRunRG := runRGFilesFunc
	defer func() { runRGFilesFunc = originalRunRG }()

	rgCalls := 0
	runRGFilesFunc = func(root string) ([]string, error) {
		rgCalls++
		return []string{path}, nil
	}

	fingerprint, err := fingerprintDirectoryMetadata(dir)
	if err != nil {
		t.Fatalf("fingerprint directory metadata: %v", err)
	}
	if !strings.HasPrefix(fingerprint, "meta:") {
		t.Fatalf("expected metadata fingerprint, got %q", fingerprint)
	}
	if rgCalls != 1 {
		t.Fatalf("expected one rg file-list call, got %d", rgCalls)
	}
}

func TestFingerprintDirectoryMetadataFallsBackWhenRGUnavailable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "broker", "main.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.WriteFile(path, []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	originalRunRG := runRGFilesFunc
	defer func() { runRGFilesFunc = originalRunRG }()
	runRGFilesFunc = func(root string) ([]string, error) {
		return nil, exec.ErrNotFound
	}

	fingerprint, err := fingerprintDirectoryMetadata(dir)
	if err != nil {
		t.Fatalf("fingerprint directory metadata fallback: %v", err)
	}
	if !strings.HasPrefix(fingerprint, "meta:") {
		t.Fatalf("expected metadata fingerprint, got %q", fingerprint)
	}
}

func TestMetadataFingerprintManifestSkipsRewriteWhenUnchanged(t *testing.T) {
	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	originalUserCacheDir := userCacheDirFunc
	defer func() { userCacheDirFunc = originalUserCacheDir }()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	path := filepath.Join(dir, "broker", "main.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.WriteFile(path, []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	first, err := fingerprintDirectoryMetadata(dir)
	if err != nil {
		t.Fatalf("first fingerprint: %v", err)
	}
	manifestPath := metadataFingerprintManifestPath(dir)
	info, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatalf("stat metadata manifest: %v", err)
	}
	mtime := info.ModTime()
	time.Sleep(20 * time.Millisecond)

	second, err := fingerprintDirectoryMetadata(dir)
	if err != nil {
		t.Fatalf("second fingerprint: %v", err)
	}
	if first != second {
		t.Fatalf("expected stable metadata fingerprint, got %q vs %q", first, second)
	}
	info, err = os.Stat(manifestPath)
	if err != nil {
		t.Fatalf("restat metadata manifest: %v", err)
	}
	if !info.ModTime().Equal(mtime) {
		t.Fatalf("expected unchanged metadata manifest mtime, got %v -> %v", mtime, info.ModTime())
	}
}

func TestMetadataFingerprintManifestChangesWhenFileMetadataChanges(t *testing.T) {
	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	originalUserCacheDir := userCacheDirFunc
	defer func() { userCacheDirFunc = originalUserCacheDir }()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	path := filepath.Join(dir, "broker", "main.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir broker dir: %v", err)
	}
	if err := os.WriteFile(path, []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}

	first, err := fingerprintDirectoryMetadata(dir)
	if err != nil {
		t.Fatalf("first fingerprint: %v", err)
	}
	time.Sleep(2 * time.Millisecond)
	if err := os.WriteFile(path, []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatalf("rewrite main.go: %v", err)
	}
	second, err := fingerprintDirectoryMetadata(dir)
	if err != nil {
		t.Fatalf("second fingerprint: %v", err)
	}
	if first == second {
		t.Fatalf("expected metadata fingerprint to change, got %q", first)
	}
}

func TestGitScopeManifestSkipsRepeatedTopLevelProbe(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalRunGit := runGitFunc
	originalUserCacheDir := userCacheDirFunc
	defer func() {
		runGitFunc = originalRunGit
		userCacheDirFunc = originalUserCacheDir
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	topLevelCalls := 0
	runGitFunc = func(gitPath, workdir string, args ...string) (string, error) {
		if len(args) >= 2 && args[0] == "rev-parse" && args[1] == "--show-toplevel" {
			topLevelCalls++
		}
		return originalRunGit(gitPath, workdir, args...)
	}

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	keyA, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("first key for request: %v", err)
	}
	if !cacheable || keyA == "" {
		t.Fatalf("expected cacheable first key, got cacheable=%v key=%q", cacheable, keyA)
	}
	if topLevelCalls != 0 {
		t.Fatalf("expected filesystem git-scope fast path to skip top-level probe on first request, got %d", topLevelCalls)
	}

	keyB, cacheable, err := KeyForRequest(req)
	if err != nil {
		t.Fatalf("second key for request: %v", err)
	}
	if !cacheable || keyB == "" {
		t.Fatalf("expected cacheable second key, got cacheable=%v key=%q", cacheable, keyB)
	}
	if keyA != keyB {
		t.Fatalf("expected stable key, got %q vs %q", keyA, keyB)
	}
	if topLevelCalls != 0 {
		t.Fatalf("expected scope manifest to keep skipping top-level probe, got %d calls", topLevelCalls)
	}
}

func TestGitFingerprintManifestSkipsRewriteWhenUnchanged(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalUserCacheDir := userCacheDirFunc
	defer func() { userCacheDirFunc = originalUserCacheDir }()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		t.Fatalf("first key for request failed: cacheable=%v err=%v", cacheable, err)
	}

	manifestPath := gitFingerprintManifestPath(dir, ".")
	info, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatalf("stat fingerprint manifest: %v", err)
	}
	mtime := info.ModTime()
	time.Sleep(20 * time.Millisecond)

	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		t.Fatalf("second key for request failed: cacheable=%v err=%v", cacheable, err)
	}
	info, err = os.Stat(manifestPath)
	if err != nil {
		t.Fatalf("restat fingerprint manifest: %v", err)
	}
	if !info.ModTime().Equal(mtime) {
		t.Fatalf("expected unchanged fingerprint manifest mtime, got %v -> %v", mtime, info.ModTime())
	}
}

func TestGitScopeManifestSkipsRewriteWhenUnchanged(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}

	dir := t.TempDir()
	cacheHome := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheHome)
	runGitForTest(t, dir, "init")
	runGitForTest(t, dir, "config", "user.email", "test@example.com")
	runGitForTest(t, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write main.go: %v", err)
	}
	runGitForTest(t, dir, "add", "main.go")
	runGitForTest(t, dir, "commit", "-m", "init")

	originalUserCacheDir := userCacheDirFunc
	defer func() { userCacheDirFunc = originalUserCacheDir }()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		t.Fatalf("first key for request failed: cacheable=%v err=%v", cacheable, err)
	}

	manifestPath := gitScopeManifestPath(dir)
	info, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatalf("stat scope manifest: %v", err)
	}
	mtime := info.ModTime()
	time.Sleep(20 * time.Millisecond)

	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		t.Fatalf("second key for request failed: cacheable=%v err=%v", cacheable, err)
	}
	info, err = os.Stat(manifestPath)
	if err != nil {
		t.Fatalf("restat scope manifest: %v", err)
	}
	if !info.ModTime().Equal(mtime) {
		t.Fatalf("expected unchanged scope manifest mtime, got %v -> %v", mtime, info.ModTime())
	}
}

func TestKeyForRequestInspectRepoIsCacheable(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# demo\n"), 0o644); err != nil {
		t.Fatalf("write README.md: %v", err)
	}

	key, cacheable, err := KeyForRequest(types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("key for request: %v", err)
	}
	if !cacheable {
		t.Fatal("expected inspect_repo request to be cacheable")
	}
	if key == "" {
		t.Fatal("expected non-empty cache key for inspect_repo")
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

func BenchmarkKeyDetailsForRequestCleanRepoWarm(b *testing.B) {
	if _, err := exec.LookPath("git"); err != nil {
		b.Skip("git not available")
	}

	dir := b.TempDir()
	cacheHome := b.TempDir()
	runGitForBench(b, dir, "init")
	runGitForBench(b, dir, "config", "user.email", "test@example.com")
	runGitForBench(b, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		b.Fatalf("write main.go: %v", err)
	}
	runGitForBench(b, dir, "add", "main.go")
	runGitForBench(b, dir, "commit", "-m", "init")

	originalUserCacheDir := userCacheDirFunc
	originalMemoTTL := gitFingerprintMemoTTL
	originalFingerprintMemo := gitFingerprintMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	defer func() {
		userCacheDirFunc = originalUserCacheDir
		gitFingerprintMemoTTL = originalMemoTTL
		gitFingerprintMemo.entries = originalFingerprintMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitFingerprintMemoTTL = time.Minute
	gitFingerprintMemo.entries = nil
	gitCleanFastpathMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		b.Fatalf("warmup key for request failed: cacheable=%v err=%v", cacheable, err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		details, err := KeyDetailsForRequest(req)
		if err != nil {
			b.Fatalf("key details for request: %v", err)
		}
		if !details.Cacheable || details.Key == "" {
			b.Fatalf("expected cacheable request, got cacheable=%v key=%q", details.Cacheable, details.Key)
		}
	}
}

func BenchmarkKeyDetailsForRequestDirtyRepoWarm(b *testing.B) {
	if _, err := exec.LookPath("git"); err != nil {
		b.Skip("git not available")
	}

	dir := b.TempDir()
	cacheHome := b.TempDir()
	runGitForBench(b, dir, "init")
	runGitForBench(b, dir, "config", "user.email", "test@example.com")
	runGitForBench(b, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		b.Fatalf("write main.go: %v", err)
	}
	runGitForBench(b, dir, "add", "main.go")
	runGitForBench(b, dir, "commit", "-m", "init")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		b.Fatalf("rewrite main.go: %v", err)
	}

	originalUserCacheDir := userCacheDirFunc
	originalMemoTTL := gitFingerprintMemoTTL
	originalFingerprintMemo := gitFingerprintMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	originalDirtyMemo := gitDirtyFastpathMemo.entries
	defer func() {
		userCacheDirFunc = originalUserCacheDir
		gitFingerprintMemoTTL = originalMemoTTL
		gitFingerprintMemo.entries = originalFingerprintMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
		gitDirtyFastpathMemo.entries = originalDirtyMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitFingerprintMemoTTL = time.Minute
	gitFingerprintMemo.entries = nil
	gitCleanFastpathMemo.entries = nil
	gitDirtyFastpathMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		b.Fatalf("warmup key for request failed: cacheable=%v err=%v", cacheable, err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		details, err := KeyDetailsForRequest(req)
		if err != nil {
			b.Fatalf("key details for request: %v", err)
		}
		if !details.Cacheable || details.Key == "" {
			b.Fatalf("expected cacheable request, got cacheable=%v key=%q", details.Cacheable, details.Key)
		}
	}
}

func BenchmarkKeyDetailsForRequestRepeatedDirtyEditWarm(b *testing.B) {
	if _, err := exec.LookPath("git"); err != nil {
		b.Skip("git not available")
	}

	dir := b.TempDir()
	cacheHome := b.TempDir()
	runGitForBench(b, dir, "init")
	runGitForBench(b, dir, "config", "user.email", "test@example.com")
	runGitForBench(b, dir, "config", "user.name", "Test User")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		b.Fatalf("write main.go: %v", err)
	}
	runGitForBench(b, dir, "add", "main.go")
	runGitForBench(b, dir, "commit", "-m", "init")
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() { println(1) }\n"), 0o644); err != nil {
		b.Fatalf("rewrite main.go: %v", err)
	}

	originalUserCacheDir := userCacheDirFunc
	originalMemoTTL := gitFingerprintMemoTTL
	originalFingerprintMemo := gitFingerprintMemo.entries
	originalCleanMemo := gitCleanFastpathMemo.entries
	originalDirtyMemo := gitDirtyFastpathMemo.entries
	defer func() {
		userCacheDirFunc = originalUserCacheDir
		gitFingerprintMemoTTL = originalMemoTTL
		gitFingerprintMemo.entries = originalFingerprintMemo
		gitCleanFastpathMemo.entries = originalCleanMemo
		gitDirtyFastpathMemo.entries = originalDirtyMemo
	}()
	userCacheDirFunc = func() (string, error) { return cacheHome, nil }
	gitFingerprintMemoTTL = time.Minute
	gitFingerprintMemo.entries = nil
	gitCleanFastpathMemo.entries = nil
	gitDirtyFastpathMemo.entries = nil

	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	if _, cacheable, err := KeyForRequest(req); err != nil || !cacheable {
		b.Fatalf("warmup key for request failed: cacheable=%v err=%v", cacheable, err)
	}

	target := filepath.Join(dir, "main.go")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		body := fmt.Sprintf("package main\n\nfunc main() { println(%d) }\n", i+2)
		if err := os.WriteFile(target, []byte(body), 0o644); err != nil {
			b.Fatalf("rewrite main.go iteration %d: %v", i, err)
		}
		b.StartTimer()
		details, err := KeyDetailsForRequest(req)
		if err != nil {
			b.Fatalf("key details for request: %v", err)
		}
		if !details.Cacheable || details.Key == "" {
			b.Fatalf("expected cacheable request, got cacheable=%v key=%q", details.Cacheable, details.Key)
		}
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

func runGitForBench(b *testing.B, dir string, args ...string) {
	b.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		b.Fatalf("git %v failed: %v: %s", args, err, string(output))
	}
}
