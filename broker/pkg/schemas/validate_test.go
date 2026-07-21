package schemas

import (
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestValidateResultDocumentSummary(t *testing.T) {
	err := ValidateResult("document_summary", "document_summary_v1", types.Result{
		SchemaName:    "document_summary_v1",
		SchemaVersion: "1.0.0",
		Payload: map[string]any{
			"summary": "summary",
			"key_points": []any{
				"point 1",
			},
			"source_metadata": map[string]any{
				"path": "/tmp/doc.txt",
			},
		},
	})
	if err != nil {
		t.Fatalf("expected valid result, got %v", err)
	}
}

func TestValidateResultRejectsSchemaMismatch(t *testing.T) {
	err := ValidateResult("document_summary", "document_summary_v1", types.Result{
		SchemaName:    "placeholder_v1",
		SchemaVersion: "1.0.0",
		Payload: map[string]any{
			"summary": "summary",
		},
	})
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestValidateResultLogAnalysis(t *testing.T) {
	err := ValidateResult("log_analysis", "log_analysis_v1", types.Result{
		SchemaName:    "log_analysis_v1",
		SchemaVersion: "1.0.0",
		Payload: map[string]any{
			"summary": "summary",
			"top_findings": []any{
				map[string]any{"code": "TEST_FAILURE"},
			},
			"timeline": []any{
				map[string]any{"phase": "failure"},
			},
			"suggested_next_steps": []any{"Inspect the first error line."},
		},
	})
	if err != nil {
		t.Fatalf("expected valid result, got %v", err)
	}
}

func TestValidateResultRepoSummary(t *testing.T) {
	err := ValidateResult("repo_summary", "repo_summary_v1", types.Result{
		SchemaName:    "repo_summary_v1",
		SchemaVersion: "1.0.0",
		Payload: map[string]any{
			"summary":      "summary",
			"subsystems":   []any{map[string]any{"name": "broker"}},
			"entrypoints":  []any{map[string]any{"path": "cmd/main.go"}},
			"dependencies": []any{map[string]any{"name": "Go"}},
			"risks":        []any{"example risk"},
		},
	})
	if err != nil {
		t.Fatalf("expected valid result, got %v", err)
	}
}

func TestValidateResultRAGEvidencePack(t *testing.T) {
	err := ValidateResult("rag_compress", "rag_evidence_pack_v1", types.Result{
		SchemaName:    "rag_evidence_pack_v1",
		SchemaVersion: "1.0.0",
		Payload: map[string]any{
			"query": "why did it fail",
			"retrieval": map[string]any{
				"strategies": []any{"ripgrep"},
			},
			"retrieval_plan": map[string]any{
				"requested_strategies": []any{"ripgrep"},
				"effective_strategies": []any{"ripgrep"},
			},
			"retrieval_trace": map[string]any{
				"strategy_executions": []any{
					map[string]any{"strategy": "ripgrep", "candidate_count": 1},
				},
			},
			"evidence": []any{
				map[string]any{"id": "ev_001"},
			},
			"budget": map[string]any{
				"retrieved_chunk_tokens": 100,
			},
		},
	})
	if err != nil {
		t.Fatalf("expected valid result, got %v", err)
	}
}

func TestValidateResultPatchProposalPack(t *testing.T) {
	err := ValidateResult("propose_patch", "patch_proposal_pack_v1", types.Result{
		SchemaName:    "patch_proposal_pack_v1",
		SchemaVersion: "1.0.0",
		Payload: map[string]any{
			"summary": "summary",
			"patches": []any{
				map[string]any{"patch_ref": "artifact_patch_plan"},
			},
			"validation_steps": []any{"go test ./..."},
		},
	})
	if err != nil {
		t.Fatalf("expected valid result, got %v", err)
	}
}

func TestValidateRepoInspectionV2EvidenceOnly(t *testing.T) {
	err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName:    "repo_inspection_v2",
		SchemaVersion: "2.0.0",
		Payload: map[string]any{
			"mode":     "auto",
			"query":    "where is request routing implemented?",
			"findings": []any{},
			"evidence": []any{map[string]any{
				"id":          "ev_001",
				"source_refs": []any{map[string]any{"path": "broker/pkg/mcp/server.go", "line_start": 1, "line_end": 2}},
			}},
			"quality": map[string]any{
				"result":       "evidence_only",
				"retrieval":    "lexical_degraded",
				"reranking":    "unavailable",
				"synthesis":    "failed",
				"answer_ready": false,
			},
			"warnings":   []any{"gpu_retrieval_unavailable"},
			"provenance": map[string]any{"index_fingerprint": "sha256:test"},
			"retrieval":  map[string]any{},
			"runtime": map[string]any{"attempts": []any{map[string]any{
				"operation": "semantic_retrieval", "tier": "p40-retrieval", "status": "failed", "gpu_count": 1,
			}}},
		},
	})
	if err != nil {
		t.Fatalf("expected valid evidence-only result, got %v", err)
	}
}

func TestValidateRepoInspectionV2RequiresGPUForAnswer(t *testing.T) {
	base := map[string]any{
		"mode":   "auto",
		"query":  "where is request routing implemented?",
		"answer": "Routing is implemented by the MCP server.",
		"findings": []any{map[string]any{
			"summary":       "The server dispatches tool calls.",
			"evidence_refs": []any{"ev_001"},
		}},
		"evidence": []any{map[string]any{
			"id":          "ev_001",
			"source_refs": []any{map[string]any{"path": "broker/pkg/mcp/server.go", "line_start": 260, "line_end": 300}},
		}},
		"quality": map[string]any{
			"result":       "answer_ready",
			"retrieval":    "gpu",
			"reranking":    "gpu",
			"synthesis":    "gpu",
			"answer_ready": true,
		},
		"warnings":   []any{},
		"provenance": map[string]any{"index_fingerprint": "sha256:test"},
		"retrieval":  map[string]any{},
		"runtime": map[string]any{"attempts": []any{
			map[string]any{"operation": "semantic_retrieval", "tier": "p40-retrieval", "status": "succeeded", "gpu_count": 1, "model_profile": "retrieval", "slurm_job_id": "job-r"},
			map[string]any{"operation": "rerank", "tier": "p40-retrieval", "status": "succeeded", "gpu_count": 1, "model_profile": "reranker", "slurm_job_id": "job-r"},
			map[string]any{"operation": "synthesis", "tier": "p40-synthesis", "status": "succeeded", "gpu_count": 1, "model_profile": "synthesis", "slurm_job_id": "job-s"},
		}},
	}
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: base,
	}); err != nil {
		t.Fatalf("expected valid answer-ready result, got %v", err)
	}

	runtime := base["runtime"]
	delete(base, "runtime")
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: base,
	}); err == nil {
		t.Fatal("expected answer-ready result without GPU attempt history to be rejected")
	}
	base["runtime"] = runtime

	base["quality"].(map[string]any)["retrieval"] = "lexical_degraded"
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: base,
	}); err == nil {
		t.Fatal("expected CPU retrieval answer promotion to be rejected")
	}
}

func TestValidateRepoInspectionV2RejectsA100BeforeP40AndV100(t *testing.T) {
	payload := map[string]any{
		"mode": "answer", "query": "where is routing?", "answer": "In the server.",
		"findings": []any{map[string]any{"summary": "Routing is in the server.", "evidence_refs": []any{"ev_001"}}},
		"evidence": []any{map[string]any{"id": "ev_001", "source_refs": []any{map[string]any{"path": "server.go"}}}},
		"quality": map[string]any{
			"result": "answer_ready", "retrieval": "gpu", "reranking": "gpu", "synthesis": "gpu", "answer_ready": true,
		},
		"warnings": []any{}, "provenance": map[string]any{}, "retrieval": map[string]any{},
		"runtime": map[string]any{"attempts": []any{
			map[string]any{"operation": "semantic_retrieval", "tier": "p40-retrieval", "status": "succeeded", "gpu_count": 1, "model_profile": "retrieval", "slurm_job_id": "job-r"},
			map[string]any{"operation": "rerank", "tier": "p40-retrieval", "status": "succeeded", "gpu_count": 1, "model_profile": "reranker", "slurm_job_id": "job-r"},
			map[string]any{"operation": "synthesis", "tier": "a100-multigpu", "status": "succeeded", "gpu_count": 4, "model_profile": "a100", "slurm_job_id": "job-a"},
		}},
	}
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: payload,
	}); err == nil {
		t.Fatal("expected premature A100 escalation to be rejected")
	}
}

func TestValidateRepoInspectionV2RejectsUnknownFindingCitation(t *testing.T) {
	payload := map[string]any{
		"mode":   "answer",
		"query":  "where is routing?",
		"answer": "In the server.",
		"findings": []any{map[string]any{
			"summary":       "Routing is in the server.",
			"evidence_refs": []any{"ev_missing"},
		}},
		"evidence": []any{map[string]any{
			"id":          "ev_001",
			"source_refs": []any{map[string]any{"path": "server.go"}},
		}},
		"quality": map[string]any{
			"result": "answer_ready", "retrieval": "gpu", "reranking": "gpu", "synthesis": "gpu", "answer_ready": true,
		},
		"warnings": []any{}, "provenance": map[string]any{},
	}
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: payload,
	}); err == nil {
		t.Fatal("expected unknown evidence citation to be rejected")
	}
}

func TestValidateRepoInspectionV2AnswerFailureRetainsAttempts(t *testing.T) {
	payload := map[string]any{
		"mode":     "answer",
		"query":    "where is routing?",
		"findings": []any{},
		"evidence": []any{map[string]any{
			"id":          "ev_001",
			"source_refs": []any{map[string]any{"path": "server.go"}},
		}},
		"quality": map[string]any{
			"result": "failed", "retrieval": "gpu", "reranking": "gpu", "synthesis": "failed", "answer_ready": false,
		},
		"warnings": []any{"gpu_tiers_exhausted"}, "provenance": map[string]any{},
		"retrieval": map[string]any{},
		"runtime": map[string]any{"attempts": []any{
			map[string]any{"operation": "synthesis", "tier": "p40-synthesis", "status": "failed", "gpu_count": 1, "failure_category": "service_failure"},
			map[string]any{"operation": "synthesis", "tier": "v100-reasoning", "status": "failed", "gpu_count": 4, "failure_category": "timeout"},
			map[string]any{"operation": "synthesis", "tier": "a100-single", "status": "failed", "gpu_count": 1, "failure_category": "availability"},
		}},
	}
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: payload,
	}); err != nil {
		t.Fatalf("expected answer failure with attempts to validate, got %v", err)
	}
	payload["runtime"] = map[string]any{"attempts": []any{}}
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: payload,
	}); err == nil {
		t.Fatal("expected GPU answer failure without attempts to be rejected")
	}
}

func TestValidateRepoInspectionV2NoGPUAnswerFallbackNeedsNoFakeAttempt(t *testing.T) {
	payload := map[string]any{
		"mode":     "answer",
		"query":    "where is routing?",
		"findings": []any{},
		"evidence": []any{map[string]any{
			"id":          "ev_001",
			"source_refs": []any{map[string]any{"path": "server.go"}},
		}},
		"quality": map[string]any{
			"result": "failed", "retrieval": "lexical_degraded", "reranking": "unavailable", "synthesis": "failed", "answer_ready": false,
		},
		"warnings":   []any{"ANSWER_REQUIRES_GPU_RETRIEVAL_AND_RERANK"},
		"provenance": map[string]any{}, "retrieval": map[string]any{},
		"runtime": map[string]any{"attempts": []any{}},
	}
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: payload,
	}); err != nil {
		t.Fatalf("no-GPU answer fallback should not invent a GPU attempt: %v", err)
	}
}

func TestValidateRepoInspectionV2EmptyRepositoryFailureNeedsNoFakeGPUAttempt(t *testing.T) {
	payload := map[string]any{
		"mode": "answer", "query": "where is the implementation?", "findings": []any{}, "evidence": []any{},
		"quality": map[string]any{
			"result": "failed", "retrieval": "failed", "reranking": "unavailable", "synthesis": "failed", "answer_ready": false,
		},
		"warnings":   []any{"NO_SUPPORTED_REPOSITORY_SOURCES", "NO_REPOSITORY_EVIDENCE"},
		"provenance": map[string]any{}, "retrieval": map[string]any{},
		"runtime": map[string]any{"attempts": []any{}},
	}
	if err := ValidateResult("inspect_repo", "repo_inspection_v2", types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0", Payload: payload,
	}); err != nil {
		t.Fatalf("empty repository failure should not invent a GPU attempt: %v", err)
	}
}

func TestValidateResultRejectsInvalidPackShapes(t *testing.T) {
	cases := []types.Result{
		{
			SchemaName:    "debug_evidence_pack_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"problem": 7},
		},
		{
			SchemaName:    "log_evidence_pack_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"summary": "ok", "timeline": "bad"},
		},
		{
			SchemaName:    "repo_inspection_v2",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"query": "inspect", "symbols": "bad"},
		},
		{
			SchemaName:    "rag_evidence_pack_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"query": "why", "retrieval": []any{}},
		},
	}
	taskTypes := []string{
		"debug_with_local_context",
		"summarize_logs",
		"inspect_repo",
		"rag_compress",
	}

	for i := range cases {
		if err := ValidateResult(taskTypes[i], cases[i].SchemaName, cases[i]); err == nil {
			t.Fatalf("case %d: expected validation error", i)
		}
	}
}
