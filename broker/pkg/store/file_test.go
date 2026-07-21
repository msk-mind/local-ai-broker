package store

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestFileJobStorePersistsJobs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	store, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_test",
		TaskType:    "document_summary",
		State:       types.JobStateQueued,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := store.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	reloaded, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("reload file store: %v", err)
	}

	got, err := reloaded.GetJob(context.Background(), "job_test")
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.ID != job.ID {
		t.Fatalf("expected %q, got %q", job.ID, got.ID)
	}
}

func TestFileJobStoreCreateJobPersistsWithNonDurableWriteHint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	jobStore, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_nondurable_create",
		TaskType:    "inspect_repo",
		State:       types.JobStateDispatching,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(WithNonDurableWrite(context.Background()), job); err != nil {
		t.Fatalf("create job with non-durable hint: %v", err)
	}

	reloaded, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("reload file store: %v", err)
	}
	got, err := reloaded.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get reloaded job: %v", err)
	}
	if got.ID != job.ID || got.State != job.State {
		t.Fatalf("expected persisted job %#v, got %#v", job, got)
	}
}

func TestFileJobStoreUpdateJobPersistsWithNonDurableWriteHint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	jobStore, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_nondurable_update",
		TaskType:    "inspect_repo",
		State:       types.JobStateDispatching,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	job.State = types.JobStateRunning
	job.UpdatedAt = now.Add(time.Second)
	if err := jobStore.UpdateJob(WithNonDurableWrite(context.Background()), job); err != nil {
		t.Fatalf("update job with non-durable hint: %v", err)
	}

	reloaded, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("reload file store: %v", err)
	}
	got, err := reloaded.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get reloaded job: %v", err)
	}
	if got.State != types.JobStateRunning {
		t.Fatalf("expected running state after non-durable update, got %#v", got.State)
	}
}

func TestFileJobStoreMergesWritesAcrossInstances(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	storeA, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store A: %v", err)
	}
	storeB, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store B: %v", err)
	}

	now := time.Now().UTC()
	jobA := types.Job{
		ID:          "job_a",
		TaskType:    "document_summary",
		State:       types.JobStateQueued,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	jobB := types.Job{
		ID:          "job_b",
		TaskType:    "log_analysis",
		State:       types.JobStateQueued,
		CreatedAt:   now.Add(time.Second),
		UpdatedAt:   now.Add(time.Second),
		SubmittedAt: now.Add(time.Second),
	}

	if err := storeA.CreateJob(context.Background(), jobA); err != nil {
		t.Fatalf("create job A: %v", err)
	}
	if err := storeB.CreateJob(context.Background(), jobB); err != nil {
		t.Fatalf("create job B: %v", err)
	}

	reloaded, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("reload file store: %v", err)
	}
	jobs, err := reloaded.ListJobs(context.Background())
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if len(jobs) != 2 {
		t.Fatalf("expected 2 jobs, got %d", len(jobs))
	}
	if jobs[0].ID != "job_a" || jobs[1].ID != "job_b" {
		t.Fatalf("expected deterministic ordering, got %#v", jobs)
	}
}

func TestFileJobStoreMissingJobErrors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	store, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	if _, err := store.GetJob(context.Background(), "missing"); err != ErrNotFound {
		t.Fatalf("expected ErrNotFound for get, got %v", err)
	}

	now := time.Now().UTC()
	err = store.UpdateJob(context.Background(), types.Job{
		ID:          "missing",
		TaskType:    "document_summary",
		State:       types.JobStateQueued,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	})
	if err != ErrNotFound {
		t.Fatalf("expected ErrNotFound for update, got %v", err)
	}
}

func TestFileJobStoreRejectsInvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")
	if err := os.WriteFile(path, []byte("{invalid"), 0o644); err != nil {
		t.Fatalf("write invalid json: %v", err)
	}

	if _, err := NewFileJobStore(path); err == nil {
		t.Fatal("expected invalid JSON to fail store load")
	}
}

func TestFileJobStoreFindCompletedJobByCacheKey(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	jobStore, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	for _, job := range []types.Job{
		{
			ID:          "job_old",
			TaskType:    "inspect_repo",
			State:       types.JobStateSucceeded,
			CacheKey:    "sha256:test",
			Result:      &types.Result{SchemaName: "repo_inspection_v2", Payload: map[string]any{"quality": map[string]any{"result": "evidence_only"}}},
			CreatedAt:   now,
			UpdatedAt:   now,
			SubmittedAt: now,
		},
		{
			ID:          "job_new",
			TaskType:    "inspect_repo",
			State:       types.JobStateSucceeded,
			CacheKey:    "sha256:test",
			Result:      &types.Result{SchemaName: "repo_inspection_v2", Payload: map[string]any{"quality": map[string]any{"result": "evidence_only"}}},
			CreatedAt:   now.Add(time.Second),
			UpdatedAt:   now.Add(time.Second),
			SubmittedAt: now.Add(time.Second),
		},
	} {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job %s: %v", job.ID, err)
		}
	}

	got, err := jobStore.FindCompletedJobByCacheKey(context.Background(), "sha256:test")
	if err != nil {
		t.Fatalf("find completed by cache key: %v", err)
	}
	if got.ID != "job_new" {
		t.Fatalf("expected latest cached job, got %#v", got)
	}
}

func TestFileJobStoreFindCompletedJobByCacheKeyUsesSidecarWithoutParsingJobs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	jobStore, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_cached",
		TaskType:    "inspect_repo",
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		CacheKey:    "sha256:test",
		Result:      &types.Result{SchemaName: "repo_inspection_v2", Payload: map[string]any{"quality": map[string]any{"result": "evidence_only"}}},
		Artifacts:   []types.Artifact{{ArtifactID: "artifact_1", ArtifactType: "evidence_pack"}},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create cached job: %v", err)
	}

	if err := os.WriteFile(path, []byte("{broken"), 0o644); err != nil {
		t.Fatalf("corrupt jobs file: %v", err)
	}

	got, err := jobStore.FindCompletedJobByCacheKey(context.Background(), "sha256:test")
	if err != nil {
		t.Fatalf("find completed by cache key using sidecar: %v", err)
	}
	if got.ID != job.ID || got.SubmittedBy != "alice" {
		t.Fatalf("unexpected sidecar lookup result: %#v", got)
	}
	if got.Result == nil || got.Result.SchemaName != "repo_inspection_v2" {
		t.Fatalf("expected schema from sidecar lookup, got %#v", got.Result)
	}
	if len(got.Artifacts) != 1 {
		t.Fatalf("expected artifact count from sidecar lookup, got %#v", got.Artifacts)
	}
}

func TestFileJobStoreFindCompletedJobByCacheKeyReturnsResidentFullJobWhenAvailable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	jobStore, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_cached_full",
		TaskType:    "inspect_repo",
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		CacheKey:    "sha256:test",
		Result: &types.Result{
			SchemaName:    "repo_inspection_v2",
			SchemaVersion: "2.0.0",
			Payload: map[string]any{
				"answer":  "done",
				"quality": map[string]any{"result": "answer_ready"},
			},
		},
		Artifacts:   []types.Artifact{{ArtifactID: "artifact_1", ArtifactType: "evidence_pack"}},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create cached job: %v", err)
	}

	got, err := jobStore.FindCompletedJobByCacheKey(context.Background(), "sha256:test")
	if err != nil {
		t.Fatalf("find completed by cache key: %v", err)
	}
	if got.ID != job.ID {
		t.Fatalf("expected resident job %q, got %#v", job.ID, got)
	}
	if got.Result == nil || got.Result.Payload["answer"] != "done" {
		t.Fatalf("expected full resident result, got %#v", got.Result)
	}
	if len(got.Artifacts) != 1 || got.Artifacts[0].ArtifactType != "evidence_pack" {
		t.Fatalf("expected resident artifacts, got %#v", got.Artifacts)
	}
}

func TestFileJobStoreListJobsByCacheKey(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	jobStore, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	for _, job := range []types.Job{
		{
			ID:          "job_queued",
			TaskType:    "inspect_repo",
			State:       types.JobStateQueued,
			CacheKey:    "sha256:test",
			CreatedAt:   now,
			UpdatedAt:   now,
			SubmittedAt: now,
		},
		{
			ID:          "job_running",
			TaskType:    "inspect_repo",
			State:       types.JobStateRunning,
			CacheKey:    "sha256:test",
			CreatedAt:   now.Add(time.Second),
			UpdatedAt:   now.Add(time.Second),
			SubmittedAt: now.Add(time.Second),
		},
		{
			ID:          "job_other",
			TaskType:    "inspect_repo",
			State:       types.JobStateRunning,
			CacheKey:    "sha256:other",
			CreatedAt:   now.Add(2 * time.Second),
			UpdatedAt:   now.Add(2 * time.Second),
			SubmittedAt: now.Add(2 * time.Second),
		},
	} {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job %s: %v", job.ID, err)
		}
	}

	got, err := jobStore.ListJobsByCacheKey(context.Background(), "sha256:test")
	if err != nil {
		t.Fatalf("list jobs by cache key: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 jobs, got %#v", got)
	}
}

func TestFileJobStoreGetJobReusesCachedJobsWhenFileMTimeUnchanged(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	jobStore, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_cached_read",
		TaskType:    "inspect_repo",
		State:       types.JobStateSucceeded,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create cached job: %v", err)
	}

	got, err := jobStore.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("initial get job: %v", err)
	}
	if got.ID != job.ID {
		t.Fatalf("unexpected initial job: %#v", got)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat jobs file: %v", err)
	}
	mtime := info.ModTime()

	if err := os.WriteFile(path, []byte("{broken"), 0o644); err != nil {
		t.Fatalf("corrupt jobs file: %v", err)
	}
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatalf("restore jobs file mtime: %v", err)
	}

	got, err = jobStore.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("cached get job: %v", err)
	}
	if got.ID != job.ID {
		t.Fatalf("expected cached job after unchanged mtime, got %#v", got)
	}

	if err := os.WriteFile(path, []byte("{}"), 0o644); err != nil {
		t.Fatalf("rewrite jobs file: %v", err)
	}
	if _, err := jobStore.GetJob(context.Background(), job.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("expected reload after jobs file mtime changed, got %v", err)
	}
}

func TestFileJobStoreCreateJobSkipsCacheIndexRewriteForNonCacheEligibleAlias(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "jobs.json")

	jobStore, err := NewFileJobStore(path)
	if err != nil {
		t.Fatalf("new file store: %v", err)
	}

	now := time.Now().UTC()
	source := types.Job{
		ID:          "job_source",
		TaskType:    "inspect_repo",
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		CacheKey:    "sha256:test",
		Result:      &types.Result{SchemaName: "repo_inspection_v2", Payload: map[string]any{"quality": map[string]any{"result": "answer_ready"}}},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), source); err != nil {
		t.Fatalf("create source job: %v", err)
	}

	info, err := os.Stat(path + ".cache_index")
	if err != nil {
		t.Fatalf("stat cache index: %v", err)
	}
	mtime := info.ModTime()
	time.Sleep(20 * time.Millisecond)

	alias := types.Job{
		ID:               "job_alias",
		TaskType:         "inspect_repo",
		State:            types.JobStateSucceeded,
		SubmittedBy:      "alice",
		CacheKey:         "sha256:test",
		CacheStatus:      "hit",
		CacheSourceJobID: source.ID,
		CreatedAt:        now.Add(time.Second),
		UpdatedAt:        now.Add(time.Second),
		SubmittedAt:      now.Add(time.Second),
	}
	if err := jobStore.CreateJob(context.Background(), alias); err != nil {
		t.Fatalf("create alias job: %v", err)
	}

	info, err = os.Stat(path + ".cache_index")
	if err != nil {
		t.Fatalf("restat cache index: %v", err)
	}
	if !info.ModTime().Equal(mtime) {
		t.Fatalf("expected unchanged cache index mtime, got %v -> %v", mtime, info.ModTime())
	}
}
