package service

import (
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestRuntimeDiagnosticsFromPayloadWithoutRuntimeMap(t *testing.T) {
	payload := map[string]any{
		"retrieval":  map[string]any{"runtime_backend_mode": "lexical"},
		"provenance": map[string]any{"runtime_backend": "cpu"},
	}

	diagnostics := runtimeDiagnosticsFromPayload(payload)
	if diagnostics["runtime_backend"] != "cpu" {
		t.Fatalf("runtime backend = %#v, want cpu", diagnostics["runtime_backend"])
	}
	if diagnostics["backend_mode"] != "lexical" {
		t.Fatalf("backend mode = %#v, want lexical", diagnostics["backend_mode"])
	}
}

func TestValidateInspectionRequestEcho(t *testing.T) {
	job := types.Job{
		TaskType: "inspect_repo",
		Request: types.SubmitJobRequest{TaskParams: map[string]any{
			"query": "trace the request", "mode": "answer",
		}},
	}
	result := types.Result{SchemaName: "repo_inspection_v2", Payload: map[string]any{
		"query": "trace the request", "mode": "answer",
	}}
	if err := validateInspectionRequestEcho(job, result); err != nil {
		t.Fatalf("expected matching request echo, got %v", err)
	}
	result.Payload["query"] = "different request"
	if err := validateInspectionRequestEcho(job, result); err == nil {
		t.Fatal("expected mismatched query echo to fail")
	}
}

func TestMergePlacementHintIntoProfile(t *testing.T) {
	profile := types.ExecutionProfile{
		Backend:    "slurm",
		Tier:       "cpu-rag-indexing",
		QOS:        "normal",
		NodeList:   "node-a",
		Constraint: "old",
	}
	hint := types.PlacementHint{
		BackendPreference: "local",
		TierPreference:    "p40-rag-compression",
		QOS:               "scavenger",
		NodeList:          "node-b",
		Constraint:        "gpu",
	}

	got := mergePlacementHintIntoProfile(profile, hint)

	if got.Backend != "local" || got.Tier != "p40-rag-compression" {
		t.Fatalf("unexpected backend/tier merge: %#v", got)
	}
	if got.QOS != "scavenger" || got.NodeList != "node-b" || got.Constraint != "gpu" {
		t.Fatalf("unexpected placement merge: %#v", got)
	}
}

func TestMergePlacementHintIntoTaskParams(t *testing.T) {
	params := map[string]any{"query": "why"}
	hint := types.PlacementHint{
		BackendPreference: "local",
		TierPreference:    "a100-reasoning",
		QOS:               "scavenger",
		NodeList:          "node-c",
		Constraint:        "a100",
		Preemptible:       true,
		Rationale:         "retry on real backend",
	}

	got := mergePlacementHintIntoTaskParams(params, hint)

	if got[taskParamRetryBackendPreference] != "local" {
		t.Fatalf("missing backend preference: %#v", got)
	}
	if got[taskParamRetryTierPreference] != "a100-reasoning" {
		t.Fatalf("missing tier preference: %#v", got)
	}
	if got[taskParamRetryQOS] != "scavenger" || got[taskParamRetryNodeList] != "node-c" {
		t.Fatalf("missing qos/nodelist: %#v", got)
	}
	if got[taskParamRetryConstraint] != "a100" || got[taskParamRetryRationale] != "retry on real backend" {
		t.Fatalf("missing retry details: %#v", got)
	}
	if got[taskParamRetryPreemptible] != true {
		t.Fatalf("expected preemptible=true, got %#v", got[taskParamRetryPreemptible])
	}
}
