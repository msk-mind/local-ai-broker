package types

import (
	"encoding/json"
	"testing"
	"time"
)

func TestJobJSONRoundTrip(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	job := Job{
		ID:       "job_123",
		TaskType: "inspect_repo",
		State:    JobStateSucceeded,
		Request: SubmitJobRequest{
			TaskType: "inspect_repo",
			InputRefs: []InputRef{
				{Type: "repo", URI: "file:///tmp/repo", Classification: "internal"},
			},
			TaskParams: map[string]any{"query": "audit this repo"},
			Constraints: Constraints{
				RetrievedChunkBudget:      16000,
				FinalEvidencePackBudget:   1200,
				RemoteModelContextBudget:  4000,
				AllowRemoteEscalation:     true,
				PerChunkCompressionBudget: 192,
			},
			ExecutionProfile: ExecutionProfile{
				Backend: "local",
				Tier:    "cpu-rag-indexing",
				Runtime: "deterministic",
			},
			OutputSchema: OutputSchemaRef{Name: "repo_inspection_v2"},
		},
		Result: &Result{
			SchemaName:    "repo_inspection_v2",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"query": "audit this repo"},
		},
		RuntimeDiagnostics:     map[string]any{"backend_mode": "real"},
		ExecutionQuality:       "real_local",
		DegradedLocalExecution: false,
		RetryRecommended:       false,
		Artifacts: []Artifact{
			{ArtifactID: "artifact_1", ArtifactType: "retrieval_result", Path: "/tmp/result.json"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}

	data, err := json.Marshal(job)
	if err != nil {
		t.Fatalf("marshal job: %v", err)
	}

	var decoded Job
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal job: %v", err)
	}

	if decoded.ID != job.ID || decoded.TaskType != job.TaskType || decoded.Request.OutputSchema.Name != "repo_inspection_v2" {
		t.Fatalf("unexpected decoded job: %#v", decoded)
	}
	if decoded.Artifacts[0].ArtifactID != "artifact_1" {
		t.Fatalf("unexpected artifact decoding: %#v", decoded.Artifacts)
	}
}

func TestRetryRecommendationJSONRoundTrip(t *testing.T) {
	rec := JobRetryRecommendation{
		JobID:       "job_123",
		Recommended: true,
		Reason:      "no_real_retrieval_backend",
		TaskType:    "rag_compress",
		ExecutionProfile: ExecutionProfile{
			Backend: "slurm",
			Tier:    "a100-reasoning",
			Runtime: "llama.cpp",
		},
		PlacementHint: PlacementHint{
			BackendPreference: "slurm",
			TierPreference:    "a100-reasoning",
			Preemptible:       true,
			Rationale:         "escalate to a real retrieval backend",
		},
		SourceResultError: "broker_policy_no_real_retrieval_backend",
	}

	data, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal retry recommendation: %v", err)
	}

	var decoded JobRetryRecommendation
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal retry recommendation: %v", err)
	}

	if !decoded.Recommended || decoded.ExecutionProfile.Tier != "a100-reasoning" || !decoded.PlacementHint.Preemptible {
		t.Fatalf("unexpected decoded recommendation: %#v", decoded)
	}
}
