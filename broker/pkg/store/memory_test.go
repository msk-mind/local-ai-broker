package store

import (
	"context"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestMemoryJobStoreCRUD(t *testing.T) {
	jobStore := NewMemoryJobStore()
	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_1",
		TaskType:    "document_summary",
		State:       types.JobStateQueued,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}

	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}
	if err := jobStore.CreateJob(context.Background(), job); err == nil {
		t.Fatal("expected duplicate create error")
	}

	got, err := jobStore.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if got.ID != job.ID || got.State != types.JobStateQueued {
		t.Fatalf("unexpected job: %#v", got)
	}

	updated := got
	updated.State = types.JobStateSucceeded
	if err := jobStore.UpdateJob(context.Background(), updated); err != nil {
		t.Fatalf("update job: %v", err)
	}

	got, err = jobStore.GetJob(context.Background(), job.ID)
	if err != nil {
		t.Fatalf("get updated job: %v", err)
	}
	if got.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded state, got %#v", got)
	}
	if !got.UpdatedAt.After(now) {
		t.Fatalf("expected UpdatedAt to advance, got %v <= %v", got.UpdatedAt, now)
	}

	jobs, err := jobStore.ListJobs(context.Background())
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if len(jobs) != 1 || jobs[0].ID != job.ID {
		t.Fatalf("unexpected jobs: %#v", jobs)
	}
}

func TestMemoryJobStoreNotFound(t *testing.T) {
	jobStore := NewMemoryJobStore()
	job := types.Job{ID: "missing"}

	if _, err := jobStore.GetJob(context.Background(), "missing"); err != ErrNotFound {
		t.Fatalf("expected ErrNotFound from GetJob, got %v", err)
	}
	if err := jobStore.UpdateJob(context.Background(), job); err != ErrNotFound {
		t.Fatalf("expected ErrNotFound from UpdateJob, got %v", err)
	}
}

func TestMemoryJobStoreFindCompletedJobByCacheKey(t *testing.T) {
	jobStore := NewMemoryJobStore()
	now := time.Now().UTC()
	queued := types.Job{
		ID:          "job_queued",
		TaskType:    "inspect_repo",
		State:       types.JobStateQueued,
		CacheKey:    "sha256:test",
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	succeeded := types.Job{
		ID:          "job_succeeded",
		TaskType:    "inspect_repo",
		State:       types.JobStateSucceeded,
		CacheKey:    "sha256:test",
		Result:      &types.Result{SchemaName: "repo_inspection_v2", Payload: map[string]any{"quality": map[string]any{"result": "evidence_only"}}},
		CreatedAt:   now.Add(time.Second),
		UpdatedAt:   now.Add(time.Second),
		SubmittedAt: now.Add(time.Second),
	}
	for _, job := range []types.Job{queued, succeeded} {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job %s: %v", job.ID, err)
		}
	}
	got, err := jobStore.FindCompletedJobByCacheKey(context.Background(), "sha256:test")
	if err != nil {
		t.Fatalf("find completed by cache key: %v", err)
	}
	if got.ID != succeeded.ID {
		t.Fatalf("expected %s, got %#v", succeeded.ID, got)
	}
}

func TestMemoryJobStoreListJobsByCacheKey(t *testing.T) {
	jobStore := NewMemoryJobStore()
	now := time.Now().UTC()
	jobs := []types.Job{
		{
			ID:          "job_1",
			TaskType:    "inspect_repo",
			State:       types.JobStateQueued,
			CacheKey:    "sha256:test",
			CreatedAt:   now,
			UpdatedAt:   now,
			SubmittedAt: now,
		},
		{
			ID:          "job_2",
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
	}
	for _, job := range jobs {
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
