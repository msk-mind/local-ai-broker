package tasks

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestSpecsIncludesInspectRepoAndCacheableTasks(t *testing.T) {
	spec, ok := FindSpec("inspect_repo")
	if !ok {
		t.Fatal("expected inspect_repo spec to exist")
	}
	if spec.SchemaName != "repo_inspection_v2" {
		t.Fatalf("unexpected schema name: %#v", spec)
	}
	if !IsCacheableTask("inspect_repo") {
		t.Fatal("expected inspect_repo to participate in exact-request cache keying")
	}
	cacheable := CacheableTaskNames()
	if len(cacheable) == 0 {
		t.Fatal("expected at least one cacheable task")
	}
}

func TestNormalizeTaskParams(t *testing.T) {
	params := map[string]any{"existing": "keep"}
	payload := map[string]any{
		"problem":              "  triage this failure  ",
		"retrieval_strategies": []any{"bm25"},
	}
	got := NormalizeTaskParams(params, payload, "debug_with_local_context")
	if got["existing"] != "keep" {
		t.Fatalf("expected existing value to survive: %#v", got)
	}
	if got["problem"] != "  triage this failure  " {
		t.Fatalf("unexpected problem normalization: %#v", got)
	}
	if got["retrieval_strategies"] == nil {
		t.Fatalf("expected retrieval strategies to be preserved: %#v", got)
	}
}

func TestDecodeSubmitRequest(t *testing.T) {
	raw := json.RawMessage(`{"problem":"diagnose this","input_refs":[{"type":"repo","uri":"file:///tmp/repo"}]}`)
	req, err := DecodeSubmitRequest(raw, Spec{Name: "debug_with_local_context", SchemaName: "debug_evidence_pack_v1"})
	if err != nil {
		t.Fatalf("decode submit request: %v", err)
	}
	if req.TaskType != "debug_with_local_context" || req.OutputSchema.Name != "debug_evidence_pack_v1" {
		t.Fatalf("unexpected decoded request: %#v", req)
	}
	if req.TaskParams["problem"] != "diagnose this" {
		t.Fatalf("expected normalized problem, got %#v", req.TaskParams)
	}
}

func TestDecodeInspectRepoRequiresQueryAndDefaultsMode(t *testing.T) {
	spec, _ := FindSpec("inspect_repo")
	if _, err := DecodeSubmitRequest(json.RawMessage(`{"input_refs":[{"type":"repo","uri":"file:///tmp/repo"}]}`), spec); err == nil {
		t.Fatal("expected missing inspect_repo query to fail")
	}
	req, err := DecodeSubmitRequest(json.RawMessage(`{"query":"trace MCP routing","input_refs":[{"type":"repo","uri":"file:///tmp/repo"}]}`), spec)
	if err != nil {
		t.Fatalf("decode inspect_repo request: %v", err)
	}
	if req.TaskParams["mode"] != "auto" {
		t.Fatalf("expected auto mode, got %#v", req.TaskParams)
	}
}

func TestDecodeInspectRepoRejectsUnknownMode(t *testing.T) {
	spec, _ := FindSpec("inspect_repo")
	_, err := DecodeSubmitRequest(json.RawMessage(`{"query":"trace MCP routing","mode":"fast","input_refs":[{"type":"repo","uri":"file:///tmp/repo"}]}`), spec)
	if err == nil {
		t.Fatal("expected invalid inspect_repo mode to fail")
	}
}

func TestDecodeInspectRepoRejectsBrokerReservedRoutingParameters(t *testing.T) {
	spec, _ := FindSpec("inspect_repo")
	for _, key := range []string{
		"gpu_services",
		"gpu_service_registry_path",
		"gpu_service_health_interval_seconds",
		"service_endpoints",
		"index_cache_dir",
		"repo_inspection_cache_path",
	} {
		raw := json.RawMessage(`{"query":"trace MCP routing","input_refs":[{"type":"repo","uri":"file:///tmp/repo"}],"task_params":{"` + key + `":"caller-controlled"}}`)
		if _, err := DecodeSubmitRequest(raw, spec); err == nil {
			t.Fatalf("expected reserved task parameter %q to fail", key)
		}
	}
}

func TestValidateInspectRepoBoundsQueryAndFinalPack(t *testing.T) {
	base := types.SubmitJobRequest{
		TaskType: "inspect_repo", TaskParams: map[string]any{"query": "trace routing", "mode": "auto"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	tooLong := base
	tooLong.TaskParams = map[string]any{"query": strings.Repeat("x", MaxInspectRepoQueryBytes+1), "mode": "auto"}
	if err := ValidateSubmitRequest(tooLong); err == nil {
		t.Fatal("expected oversized query to fail")
	}
	tooSmall := base
	tooSmall.Constraints.FinalPackTokenBudget = MinInspectRepoFinalPackTokens - 1
	if err := ValidateSubmitRequest(tooSmall); err == nil {
		t.Fatal("expected undersized final pack budget to fail")
	}
	legacyTooSmall := base
	legacyTooSmall.Constraints.FinalEvidencePackBudget = MinInspectRepoFinalPackTokens - 1
	if err := ValidateSubmitRequest(legacyTooSmall); err == nil {
		t.Fatal("expected deprecated undersized final evidence pack alias to fail")
	}
	base.Constraints.FinalPackTokenBudget = MinInspectRepoFinalPackTokens
	if err := ValidateSubmitRequest(base); err != nil {
		t.Fatalf("expected practical minimum final pack budget to pass: %v", err)
	}
}
