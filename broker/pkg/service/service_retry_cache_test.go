package service

import (
	"context"
	"io"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/cache"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestGetJobRetryRecommendationAndRetryJobWithRecommendation(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	backend := &mutableFakeBackend{status: backends.RunStatus{State: types.JobStateQueued, RawState: "PENDING"}}
	svc := NewWithAuditAndOptions(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		t.TempDir(),
		".",
		Options{},
	)

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_retry_source",
		TaskType:    "rag_compress",
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		Request: types.SubmitJobRequest{
			TaskType:       "rag_compress",
			TaskParams:     map[string]any{"query": "why does it fail"},
			OutputSchema:   types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
			IdempotencyKey: "source-idempotency",
		},
		Result: &types.Result{
			SchemaName:    "rag_evidence_pack_v1",
			SchemaVersion: "1.0.0",
			Payload: map[string]any{
				"broker_retry_recommendation": map[string]any{
					"recommended": true,
					"reason":      "no_real_retrieval_backend",
					"task_type":   "rag_compress",
					"execution_profile": map[string]any{
						"backend": "slurm",
						"tier":    "p40-rag-compression",
						"runtime": "llama.cpp",
						"model":   "gpt-oss-20b.p40",
					},
					"placement_hint": map[string]any{
						"backend_preference": "local",
						"tier_preference":    "a100-reasoning",
						"qos":                "scavenger",
						"nodelist":           "node-x",
						"constraint":         "gpu",
						"preemptible":        true,
						"rationale":          "retry with real backend",
					},
				},
			},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create source job: %v", err)
	}

	rec, err := svc.GetJobRetryRecommendation(aliceUserCtx(), job.ID)
	if err != nil {
		t.Fatalf("get retry recommendation: %v", err)
	}
	if !rec.Recommended || rec.ExecutionProfile.Tier != "p40-rag-compression" {
		t.Fatalf("unexpected recommendation: %#v", rec)
	}

	resp, err := svc.RetryJobWithRecommendation(aliceUserCtx(), job.ID)
	if err != nil {
		t.Fatalf("retry with recommendation: %v", err)
	}
	if resp.JobID == "" || resp.JobID == job.ID {
		t.Fatalf("expected new retry job id, got %#v", resp)
	}

	retriedJob, err := jobStore.GetJob(context.Background(), resp.JobID)
	if err != nil {
		t.Fatalf("load retried job: %v", err)
	}
	if retriedJob.Request.IdempotencyKey != "" {
		t.Fatalf("expected retry to clear idempotency key, got %#v", retriedJob.Request)
	}
	if retriedJob.Request.ExecutionProfile.Backend != "local" || retriedJob.Request.ExecutionProfile.Tier != "a100-reasoning" {
		t.Fatalf("expected placement hint to override execution profile, got %#v", retriedJob.Request.ExecutionProfile)
	}
	if retriedJob.Request.TaskParams[taskParamRetryOfJobID] != job.ID {
		t.Fatalf("expected retry-of marker, got %#v", retriedJob.Request.TaskParams)
	}
	if retriedJob.Request.TaskParams[taskParamRetryRationale] != "retry with real backend" {
		t.Fatalf("expected retry rationale in task params, got %#v", retriedJob.Request.TaskParams)
	}
}

func TestLookupCacheReportsAccessibleHitAndUncacheableTask(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	svc := NewWithAuditAndOptions(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		t.TempDir(),
		".",
		Options{},
	)

	dir := t.TempDir()
	inputPath := filepath.Join(dir, "example.txt")
	if err := os.WriteFile(inputPath, []byte("example"), 0o644); err != nil {
		t.Fatalf("write cacheable input: %v", err)
	}
	req := types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + inputPath},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	}
	cacheKey, cacheable, err := cache.KeyForRequest(req)
	if err != nil {
		t.Fatalf("compute cache key: %v", err)
	}
	if !cacheable {
		t.Fatal("expected document_summary to be cacheable")
	}

	now := time.Now().UTC()
	cachedJob := types.Job{
		ID:          "job_cached",
		TaskType:    req.TaskType,
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		Request:     req,
		Result: &types.Result{
			SchemaName:    "document_summary_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"summary": "done"},
		},
		Artifacts:   []types.Artifact{{ArtifactID: "artifact_1", ArtifactType: "summary"}},
		CacheKey:    cacheKey,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), cachedJob); err != nil {
		t.Fatalf("create cached job: %v", err)
	}

	resp, err := svc.LookupCache(aliceUserCtx(), req)
	if err != nil {
		t.Fatalf("lookup cache: %v", err)
	}
	if resp.Status != "hit" || resp.SourceJobID != cachedJob.ID || resp.ArtifactCount != 1 {
		t.Fatalf("unexpected cache hit response: %#v", resp)
	}

	inspectReq := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + dir, Classification: "internal"},
		},
		TaskParams:   map[string]any{"query": "trace routing", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	inspectKey, cacheable, err := cache.KeyForRequest(inspectReq)
	if err != nil {
		t.Fatalf("compute inspect_repo cache key: %v", err)
	}
	if !cacheable {
		t.Fatal("expected inspect_repo evidence request to be cacheable")
	}
	inspectJob := types.Job{
		ID:          "job_inspect_cached",
		TaskType:    inspectReq.TaskType,
		State:       types.JobStateSucceeded,
		SubmittedBy: "alice",
		Request:     inspectReq,
		Result: &types.Result{
			SchemaName:    "repo_inspection_v2",
			SchemaVersion: "2.0.0",
			Payload: map[string]any{
				"mode":  "evidence",
				"query": "trace routing",
				"quality": map[string]any{
					"result":       "evidence_only",
					"retrieval":    "gpu",
					"reranking":    "gpu",
					"synthesis":    "not_requested",
					"answer_ready": false,
				},
				"evidence": []any{},
			},
		},
		CacheKey:    inspectKey,
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), inspectJob); err != nil {
		t.Fatalf("create inspect cached job: %v", err)
	}

	inspectHit, err := svc.LookupCache(aliceUserCtx(), inspectReq)
	if err != nil {
		t.Fatalf("lookup inspect_repo cache hit: %v", err)
	}
	if inspectHit.Status != "hit" || inspectHit.SourceJobID != inspectJob.ID {
		t.Fatalf("expected inspect_repo evidence cache hit, got %#v", inspectHit)
	}
}

func TestLookupCacheSkipsInspectRepoAutoModeAndAllowsAnswerReadyAnswerMode(t *testing.T) {
	jobStore := store.NewMemoryJobStore()
	svc := NewWithAuditAndOptions(
		jobStore,
		fakeBackend{},
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		t.TempDir(),
		".",
		Options{},
	)

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# demo\n"), 0o644); err != nil {
		t.Fatalf("write inspect input: %v", err)
	}

	now := time.Now().UTC()
	for name, tc := range map[string]struct {
		mode        string
		payload     map[string]any
		wantStatus  string
	}{
		"answer_ready_auto": {
			mode: "auto",
			payload: map[string]any{
			"mode":  "auto",
			"query": "trace routing",
			"quality": map[string]any{
				"result":       "answer_ready",
				"retrieval":    "gpu",
				"reranking":    "gpu",
				"synthesis":    "gpu",
				"answer_ready": true,
			},
			"answer": "done",
			"findings": []any{
				map[string]any{"summary": "done", "evidence_refs": []any{"evidence_1"}},
			},
			"evidence": []any{
				map[string]any{"evidence_id": "evidence_1", "path": "README.md", "line_start": 1, "line_end": 1, "excerpt": "# demo"},
			},
			},
			wantStatus: "miss",
		},
		"answer_ready_answer": {
			mode: "answer",
			payload: map[string]any{
				"mode":  "answer",
				"query": "trace routing",
				"quality": map[string]any{
					"result":       "answer_ready",
					"retrieval":    "gpu",
					"reranking":    "gpu",
					"synthesis":    "gpu",
					"answer_ready": true,
				},
				"answer": "done",
				"findings": []any{
					map[string]any{"summary": "done", "evidence_refs": []any{"evidence_1"}},
				},
				"evidence": []any{
					map[string]any{"evidence_id": "evidence_1", "path": "README.md", "line_start": 1, "line_end": 1, "excerpt": "# demo"},
				},
			},
			wantStatus: "hit",
		},
		"evidence_only_auto": {
			mode: "auto",
			payload: map[string]any{
				"mode":  "auto",
				"query": "trace routing",
				"quality": map[string]any{
					"result":       "evidence_only",
					"retrieval":    "lexical_degraded",
					"reranking":    "unavailable",
					"synthesis":    "not_requested",
					"answer_ready": false,
				},
				"evidence": []any{},
			},
			wantStatus: "miss",
		},
	} {
		req := types.SubmitJobRequest{
			TaskType: "inspect_repo",
			InputRefs: []types.InputRef{
				{Type: "repo", URI: "file://" + dir, Classification: "internal"},
			},
			TaskParams:   map[string]any{"query": "trace routing", "mode": tc.mode},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		}
		key, cacheable, err := cache.KeyForRequest(req)
		if err != nil {
			t.Fatalf("%s: compute cache key: %v", name, err)
		}
		if !cacheable {
			t.Fatalf("%s: expected inspect_repo to be cacheable", name)
		}
		job := types.Job{
			ID:          "job_" + name,
			TaskType:    req.TaskType,
			State:       types.JobStateSucceeded,
			SubmittedBy: "alice",
			Request:     req,
			Result: &types.Result{
				SchemaName:    "repo_inspection_v2",
				SchemaVersion: "2.0.0",
				Payload:       tc.payload,
			},
			CacheKey:    key,
			CreatedAt:   now,
			UpdatedAt:   now,
			SubmittedAt: now,
		}
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("%s: create cached job: %v", name, err)
		}
		resp, err := svc.LookupCache(aliceUserCtx(), req)
		if err != nil {
			t.Fatalf("%s: lookup cache: %v", name, err)
		}
		if resp.Status != tc.wantStatus {
			t.Fatalf("%s: expected cache %s, got %#v", name, tc.wantStatus, resp)
		}
		if tc.wantStatus == "hit" && resp.SourceJobID != job.ID {
			t.Fatalf("%s: expected source job id %q, got %#v", name, job.ID, resp)
		}
	}
}
