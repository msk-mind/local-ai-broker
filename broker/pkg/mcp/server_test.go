package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends/slurm"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/service"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/tasks"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

const (
	mcpRetryBudgetExceededMessage   = "cumulative retried_shards=2 would exceed non-admin limit 1"
	mcpReleaseBudgetExceededMessage = "cumulative forced_release_batches=2 would exceed non-admin limit 1"
)

func mcpTestPrincipal() auth.Principal {
	return auth.Principal{Actor: "mcp:test", Role: "user"}
}

func TestToolsList(t *testing.T) {
	server := newTestServer()
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/list",
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	result := resp.Result.(map[string]any)
	tools := result["tools"].([]map[string]any)
	if len(tools) != 17 {
		t.Fatalf("expected 17 tools, got %d", len(tools))
	}
}

func TestSubmitToolCall(t *testing.T) {
	server := newTestServer()
	params := map[string]any{
		"name": "submit_local_job",
		"arguments": map[string]any{
			"task_type": "document_summary",
			"input_refs": []map[string]any{
				{"type": "file", "uri": "file:///tmp/does-not-exist.txt"},
			},
			"output_schema": map[string]any{"name": "document_summary_v1"},
		},
	}
	paramBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  paramBytes,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	result := resp.Result.(map[string]any)
	if _, ok := result["structuredContent"]; !ok {
		t.Fatal("expected structuredContent")
	}
}

func TestSubmitAndMaybeWaitUsesInlineReleasedResult(t *testing.T) {
	server := &Server{}
	args := mustRawJSON(t, `{"wait_for_result": true}`)
	expected := types.JobResultRelease{
		JobID:  "job_cached",
		State:  types.JobStateSucceeded,
		Result: &types.Result{SchemaName: "document_summary_v1", Payload: map[string]any{"summary": "cached"}},
	}

	result, err := server.submitAndMaybeWait(context.Background(), args, "submit_local_job", func(context.Context) (types.SubmitJobResponse, error) {
		return types.SubmitJobResponse{
			JobID:          "job_cached",
			State:          types.JobStateSucceeded,
			ReleasedResult: &expected,
		}, nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	release := result["structuredContent"].(types.JobResultRelease)
	if release.JobID != expected.JobID || release.Result == nil || release.Result.Payload["summary"] != "cached" {
		t.Fatalf("expected inline released result, got %#v", release)
	}
}

func TestSubmitToolCallWaitForResultReturnsReleasedResult(t *testing.T) {
	runRoot := t.TempDir()
	svc := service.New(
		store.NewMemoryJobStore(),
		&waitForResultBackend{runRoot: runRoot, delay: 20 * time.Millisecond},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)
	server := NewServer(svc, mcpTestPrincipal())

	params := map[string]any{
		"name": "submit_local_job",
		"arguments": map[string]any{
			"task_type": "document_summary",
			"input_refs": []map[string]any{
				{"type": "file", "uri": "file:///tmp/does-not-exist.txt"},
			},
			"output_schema":    map[string]any{"name": "document_summary_v1"},
			"wait_for_result":  true,
			"max_wait_seconds": 2,
			"poll_interval_ms": 10,
		},
	}
	paramBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  paramBytes,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	release := resp.Result.(map[string]any)["structuredContent"].(types.JobResultRelease)
	if release.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded release, got %#v", release)
	}
	if release.Result == nil || release.Result.Payload["summary"] != "waited summary" {
		t.Fatalf("expected released result payload, got %#v", release)
	}
}

func TestSubmitToolCallWaitForResultReturnsDirectReleasedResultBeforeTerminalState(t *testing.T) {
	runRoot := t.TempDir()
	svc := service.New(
		store.NewMemoryJobStore(),
		&localInspectRepoEarlyResultBackend{runRoot: runRoot, delay: 20 * time.Millisecond},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)
	server := NewServer(svc, mcpTestPrincipal())

	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: mustRawJSON(t, `{
		  "name": "inspect_repo",
		  "arguments": {
		    "query": "trace inspect_repo timeout handling",
		    "mode": "evidence",
		    "input_refs": [{"type": "repo", "uri": "file:///tmp/repo"}],
		    "wait_for_result": true,
		    "max_wait_seconds": 2,
		    "poll_interval_ms": 10
		  }
		}`),
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	release := resp.Result.(map[string]any)["structuredContent"].(types.JobResultRelease)
	if release.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded release, got %#v", release)
	}
	if release.Result == nil || release.Result.SchemaName != "repo_inspection_v2" {
		t.Fatalf("expected inspect_repo released result, got %#v", release)
	}
	payload := release.Result.Payload
	if payload["mode"] != "evidence" {
		t.Fatalf("expected evidence mode payload, got %#v", payload)
	}
}

func TestSubmitToolCallWaitForResultTimeout(t *testing.T) {
	runRoot := t.TempDir()
	svc := service.New(
		store.NewMemoryJobStore(),
		&queuedOnlyBackend{},
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)
	server := NewServer(svc, mcpTestPrincipal())

	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: mustRawJSON(t, `{
		  "name": "submit_local_job",
		  "arguments": {
		    "task_type": "document_summary",
		    "input_refs": [{"type": "file", "uri": "file:///tmp/does-not-exist.txt"}],
		    "output_schema": {"name": "document_summary_v1"},
		    "wait_for_result": true,
		    "max_wait_seconds": 1,
		    "poll_interval_ms": 10
		  }
		}`),
	})
	if resp.Error == nil {
		t.Fatal("expected timeout error")
	}
	if !strings.Contains(resp.Error.Message, "timed out waiting for job") {
		t.Fatalf("expected timeout message, got %#v", resp.Error)
	}
}

func TestServeStdioInitialize(t *testing.T) {
	server := newTestServer()
	in := bytes.NewBufferString(frameJSON(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"auth":{"actor":"alice","role":"user"}}}`))
	out := &bytes.Buffer{}
	if err := server.ServeStdio(context.Background(), in, out); err != nil {
		t.Fatalf("serve stdio: %v", err)
	}
	if !strings.Contains(out.String(), "Content-Length:") {
		t.Fatalf("expected framed response, got %q", out.String())
	}
}

func TestServeStdioInitializeNDJSON(t *testing.T) {
	server := newTestServer()
	in := bytes.NewBufferString("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"auth\":{\"actor\":\"alice\",\"role\":\"user\"}}}\n")
	out := &bytes.Buffer{}
	if err := server.ServeStdio(context.Background(), in, out); err != nil {
		t.Fatalf("serve stdio: %v", err)
	}
	if strings.Contains(out.String(), "Content-Length:") {
		t.Fatalf("expected ndjson response, got %q", out.String())
	}
	if !strings.HasSuffix(out.String(), "\n") {
		t.Fatalf("expected newline-delimited response, got %q", out.String())
	}
}

func TestServeStdioEndToEndToolFlow(t *testing.T) {
	runRoot := t.TempDir()
	inputPath := filepath.Join(runRoot, "doc.txt")
	if err := os.WriteFile(inputPath, []byte("MCP protocol test document.\n- point\n"), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}

	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)
	server := NewServer(svc, auth.Principal{})

	messages := strings.Join([]string{
		frameJSON(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"auth":{"actor":"alice","role":"user"}}}`),
		frameJSON(`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`),
		frameJSON(fmt.Sprintf(`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"submit_local_job","arguments":{"task_type":"document_summary","input_refs":[{"type":"file","uri":"file://%s"}],"output_schema":{"name":"document_summary_v1"}}}}`, inputPath)),
	}, "")

	in := bytes.NewBufferString(messages)
	out := &bytes.Buffer{}
	if err := server.ServeStdio(context.Background(), in, out); err != nil {
		t.Fatalf("serve stdio: %v", err)
	}

	responses := decodeFramedResponses(t, out.Bytes())
	if len(responses) != 3 {
		t.Fatalf("expected 3 responses, got %d", len(responses))
	}

	var submitResp response
	if err := json.Unmarshal(responses[2], &submitResp); err != nil {
		t.Fatalf("unmarshal submit response: %v", err)
	}
	if submitResp.Error != nil {
		t.Fatalf("unexpected submit error: %#v", submitResp.Error)
	}

	result := submitResp.Result.(map[string]any)
	structured := result["structuredContent"].(map[string]any)
	if structured["job_id"] == "" {
		t.Fatal("expected job_id in structured content")
	}
}

func TestListLocalCapabilitiesToolCall(t *testing.T) {
	server := newTestServer()
	initResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      0,
		Method:  "initialize",
		Params:  mustRawJSON(t, `{"auth":{"actor":"alice","role":"user"}}`),
	})
	if initResp.Error != nil {
		t.Fatalf("initialize error: %v", initResp.Error)
	}
	params := map[string]any{
		"name":      "list_local_capabilities",
		"arguments": map[string]any{},
	}
	paramBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  paramBytes,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	result := resp.Result.(map[string]any)
	structured := result["structuredContent"].(map[string]any)
	taskTypes := structured["task_types"].([]map[string]any)
	if len(taskTypes) != 8 {
		t.Fatalf("expected 8 task types, got %d", len(taskTypes))
	}
	orchestration := structured["orchestration"].(map[string]any)
	if orchestration["independent_parallel_jobs"] != true {
		t.Fatalf("expected independent_parallel_jobs=true, got %#v", orchestration)
	}
	gpuServices := structured["gpu_services"].(map[string]any)
	if gpuServices["enabled"] != false {
		t.Fatalf("expected disabled GPU service snapshot in unit server, got %#v", gpuServices)
	}
	tiers := gpuServices["tiers"].([]map[string]any)
	if len(tiers) != 5 {
		t.Fatalf("expected all five GPU service tiers, got %#v", tiers)
	}
}

func TestListLocalCapabilitiesUsesLiveGPUProvider(t *testing.T) {
	base := newTestServer()
	server := NewServerWithGPUCapabilities(base.service, mcpTestPrincipal(), func(context.Context) (any, error) {
		return map[string]any{
			"enabled": true,
			"healthy": true,
			"tiers": []map[string]any{{
				"tier": "p40-retrieval", "active_replicas": 1,
				"queue_state": map[string]int{"running": 1},
				"endpoints":   []map[string]any{{"id": "gpu-p40-1", "healthy": true}},
			}},
		}, nil
	})
	server.setSessionPrincipal(mcpTestPrincipal())
	result, err := server.callTool(context.Background(), toolsCallParams{
		Name: "list_local_capabilities", Arguments: mustRawJSON(t, `{}`),
	})
	if err != nil {
		t.Fatalf("list capabilities: %v", err)
	}
	structured := result["structuredContent"].(map[string]any)
	gpuServices := structured["gpu_services"].(map[string]any)
	if gpuServices["enabled"] != true || gpuServices["healthy"] != true {
		t.Fatalf("expected live GPU snapshot, got %#v", gpuServices)
	}
}

func TestInspectRepoToolPublishesV2Contract(t *testing.T) {
	var inspect map[string]any
	for _, definition := range toolDefinitions() {
		if definition["name"] == "inspect_repo" {
			inspect = definition
			break
		}
	}
	if inspect == nil {
		t.Fatal("inspect_repo tool definition not found")
	}
	schema := inspect["inputSchema"].(map[string]any)
	required := schema["required"].([]string)
	if !containsString(required, "query") || !containsString(required, "input_refs") {
		t.Fatalf("inspect_repo must require query and input_refs: %#v", required)
	}
	properties := schema["properties"].(map[string]any)
	query := properties["query"].(map[string]any)
	if query["maxLength"] != tasks.MaxInspectRepoQueryBytes {
		t.Fatalf("unexpected inspect_repo query limit: %#v", query)
	}
	mode := properties["mode"].(map[string]any)
	if mode["default"] != "auto" {
		t.Fatalf("expected auto default mode, got %#v", mode)
	}
	constraints := properties["constraints"].(map[string]any)["properties"].(map[string]any)
	for _, key := range []string{
		"retrieval_token_budget", "evidence_token_budget", "final_pack_token_budget", "synthesis_context_token_budget",
	} {
		if _, ok := constraints[key]; !ok {
			t.Fatalf("missing token-explicit constraint %q: %#v", key, constraints)
		}
	}
	if _, exists := constraints["retrieved_chunk_budget"]; exists {
		t.Fatalf("legacy ambiguous budget must not be advertised: %#v", constraints)
	}
	finalPack := constraints["final_pack_token_budget"].(map[string]any)
	if finalPack["minimum"] != tasks.MinInspectRepoFinalPackTokens {
		t.Fatalf("unexpected inspect_repo final pack minimum: %#v", finalPack)
	}
	profile := properties["execution_profile"].(map[string]any)["properties"].(map[string]any)
	tiers := profile["tier"].(map[string]any)["enum"].([]string)
	for _, tier := range []string{"p40-retrieval", "p40-synthesis", "v100-reasoning", "a100-single", "a100-multigpu"} {
		if !containsString(tiers, tier) {
			t.Fatalf("missing GPU tier %q: %#v", tier, tiers)
		}
	}
	for _, legacyTier := range []string{"p40-rag-compression", "a100-reasoning"} {
		if !containsString(tiers, legacyTier) {
			t.Fatalf("shared RAG profile schema lost compatibility tier %q: %#v", legacyTier, tiers)
		}
	}
}

func containsString(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}

func TestRAGCompressToolCall(t *testing.T) {
	server := newTestServer()
	initResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      0,
		Method:  "initialize",
		Params:  mustRawJSON(t, `{"auth":{"actor":"alice","role":"user"}}`),
	})
	if initResp.Error != nil {
		t.Fatalf("initialize error: %v", initResp.Error)
	}
	params := map[string]any{
		"name": "rag_compress",
		"arguments": map[string]any{
			"query": "why did the build fail?",
			"input_refs": []map[string]any{
				{"type": "log", "uri": "file:///tmp/build.log"},
			},
			"constraints": map[string]any{
				"retrieved_chunk_budget": 64000,
			},
		},
	}
	paramBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  paramBytes,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	result := resp.Result.(map[string]any)
	structured := result["structuredContent"].(types.SubmitJobResponse)
	if structured.JobID == "" {
		t.Fatalf("expected job_id, got %#v", structured)
	}
}

func TestRAGCompressToolCallPreservesQuery(t *testing.T) {
	server := newTestServer()
	initResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      0,
		Method:  "initialize",
		Params:  mustRawJSON(t, `{"auth":{"actor":"alice","role":"user"}}`),
	})
	if initResp.Error != nil {
		t.Fatalf("initialize error: %v", initResp.Error)
	}
	callResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: mustRawJSON(t, `{
		  "name": "rag_compress",
		  "arguments": {
		    "query": "why did the build fail?",
		    "retrieval_strategies": ["ripgrep", "bm25"],
		    "input_refs": [{"type": "log", "uri": "file:///tmp/build.log"}]
		  }
		}`),
	})
	if callResp.Error != nil {
		t.Fatalf("unexpected error: %v", callResp.Error)
	}
	submit := callResp.Result.(map[string]any)["structuredContent"].(types.SubmitJobResponse)

	statusResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: mustRawJSON(t, fmt.Sprintf(`{
		  "name": "get_job_status",
		  "arguments": {"job_id": %q}
		}`, submit.JobID)),
	})
	if statusResp.Error != nil {
		t.Fatalf("unexpected status error: %v", statusResp.Error)
	}
	job := statusResp.Result.(map[string]any)["structuredContent"].(types.Job)
	if job.Request.TaskParams["query"] != "why did the build fail?" {
		t.Fatalf("expected normalized query task param, got %#v", job.Request.TaskParams)
	}
	if strategies, ok := job.Request.TaskParams["retrieval_strategies"].([]any); ok && len(strategies) == 2 {
	} else {
		t.Fatalf("expected normalized retrieval strategies, got %#v", job.Request.TaskParams)
	}
}

func TestInspectRepoToolCallUsesV2OutputSchema(t *testing.T) {
	server := newTestServer()
	server.setSessionPrincipal(mcpTestPrincipal())

	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: mustRawJSON(t, `{
		  "name": "inspect_repo",
		  "arguments": {
		    "query": "trace inspect_repo timeout handling",
		    "mode": "answer",
		    "input_refs": [{"type": "repo", "uri": "file:///tmp/repo"}]
		  }
		}`),
	})
	if resp.Error != nil {
		t.Fatalf("unexpected inspect_repo error: %v", resp.Error)
	}
	submit := resp.Result.(map[string]any)["structuredContent"].(types.SubmitJobResponse)

	statusResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: mustRawJSON(t, fmt.Sprintf(`{
		  "name": "get_job_status",
		  "arguments": {"job_id": %q}
		}`, submit.JobID)),
	})
	if statusResp.Error != nil {
		t.Fatalf("unexpected status error: %v", statusResp.Error)
	}
	job := statusResp.Result.(map[string]any)["structuredContent"].(types.Job)
	if job.Request.OutputSchema.Name != "repo_inspection_v2" {
		t.Fatalf("expected inspect_repo output schema repo_inspection_v2, got %#v", job.Request.OutputSchema)
	}
	if job.Request.TaskParams["mode"] != "answer" || job.Request.TaskParams["query"] != "trace inspect_repo timeout handling" {
		t.Fatalf("expected normalized inspect_repo task params, got %#v", job.Request.TaskParams)
	}
}

func TestGetJobStatusIncludesRuntimeDiagnostics(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := service.New(
		jobStore,
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		runRoot,
		t.TempDir(),
	)
	server := NewServer(svc, mcpTestPrincipal())

	now := time.Now().UTC()
	job := types.Job{
		ID:       "job_runtime_status",
		TaskType: "rag_compress",
		State:    types.JobStateSucceeded,
		Request: types.SubmitJobRequest{
			TaskType:     "rag_compress",
			OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
		},
		RuntimeDiagnostics: map[string]any{
			"backend_name":        "llama.cpp",
			"backend_mode":        "unavailable",
			"selected_model":      "gpt-oss-20b.p40",
			"resource_tier":       "p40-rag-compression",
			"endpoint_configured": true,
			"last_error":          "connection refused",
		},
		ExecutionQuality:       "no_real_backend",
		DegradedLocalExecution: true,
		RetryRecommended:       true,
		CreatedAt:              now,
		UpdatedAt:              now,
		SubmittedAt:            now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: mustRawJSON(t, `{
		  "name": "get_job_status",
		  "arguments": {"job_id": "job_runtime_status"}
		}`),
	})
	if resp.Error != nil {
		t.Fatalf("unexpected status error: %v", resp.Error)
	}
	got := resp.Result.(map[string]any)["structuredContent"].(types.Job)
	if got.RuntimeDiagnostics["backend_mode"] != "unavailable" {
		t.Fatalf("expected runtime diagnostics in get_job_status, got %#v", got.RuntimeDiagnostics)
	}
	if !got.DegradedLocalExecution || !got.RetryRecommended || got.ExecutionQuality != "no_real_backend" {
		t.Fatalf("expected summary flags in get_job_status, got %#v", got)
	}
}

func TestFetchResultPreservesIngestedDegradationAcrossCacheAlias(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	if err := os.WriteFile(filepath.Join(repoRoot, "README.md"), []byte("fixture\n"), 0o644); err != nil {
		t.Fatalf("write repo fixture: %v", err)
	}
	jobStore := store.NewMemoryJobStore()
	svc := service.New(jobStore, &queuedOnlyBackend{}, log.New(io.Discard, "", 0), runRoot, ".")
	ctx := auth.WithPrincipal(context.Background(), mcpTestPrincipal())
	req := types.SubmitJobRequest{
		TaskType:     "inspect_repo",
		InputRefs:    []types.InputRef{{Type: "repo", URI: "file://" + repoRoot, Classification: "internal"}},
		TaskParams:   map[string]any{"query": "trace routing", "mode": "evidence"},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}
	first, err := svc.SubmitJob(ctx, req)
	if err != nil {
		t.Fatalf("submit source job: %v", err)
	}
	result := types.Result{
		SchemaName: "repo_inspection_v2", SchemaVersion: "2.0.0",
		Payload: map[string]any{
			"mode": "evidence", "query": "trace routing", "findings": []any{},
			"evidence": []any{map[string]any{"id": "ev_1", "path": "README.md", "source_refs": []any{map[string]any{"path": "README.md", "line_start": 1, "line_end": 1}}}},
			"quality":  map[string]any{"result": "evidence_only", "retrieval": "lexical_degraded", "reranking": "unavailable", "synthesis": "not_requested", "answer_ready": false},
			"warnings": []any{}, "provenance": map[string]any{"index_fingerprint": "sha256:test"},
			"runtime": map[string]any{"attempts": []any{}}, "retrieval": map[string]any{"lexical_candidates": 1},
		},
	}
	resultBytes, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	jobDir := filepath.Join(runRoot, first.JobID)
	if err := os.WriteFile(filepath.Join(jobDir, "result.json"), resultBytes, 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644); err != nil {
		t.Fatalf("write artifacts: %v", err)
	}
	source, err := svc.GetJob(ctx, first.JobID)
	if err != nil {
		t.Fatalf("ingest source result: %v", err)
	}
	if !source.DegradedLocalExecution || source.ExecutionQuality != "evidence_only" {
		t.Fatalf("expected ingested degradation flags, got %#v", source)
	}

	second, err := svc.SubmitJob(ctx, req)
	if err != nil {
		t.Fatalf("submit cache alias: %v", err)
	}
	if second.Cache.Status != "hit" || second.ReleasedResult == nil {
		t.Fatalf("expected released cache alias, got %#v", second)
	}
	if !second.ReleasedResult.DegradedLocalExecution || second.ReleasedResult.ExecutionQuality != "evidence_only" {
		t.Fatalf("cache alias lost degradation flags: %#v", second.ReleasedResult)
	}

	server := NewServer(svc, mcpTestPrincipal())
	params := mustRawJSON(t, fmt.Sprintf(`{"name":"fetch_result","arguments":{"job_id":%q}}`, second.JobID))
	resp := server.handleRequest(ctx, request{JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: params})
	if resp.Error != nil {
		t.Fatalf("fetch_result error: %v", resp.Error)
	}
	structured := resp.Result.(map[string]any)["structuredContent"].(types.JobResultRelease)
	if !structured.DegradedLocalExecution || structured.ExecutionQuality != "evidence_only" {
		t.Fatalf("MCP release lost degradation flags: %#v", structured)
	}
}

func TestSubmitParallelJobsToolCall(t *testing.T) {
	server := newTestServer()
	initResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      0,
		Method:  "initialize",
		Params:  mustRawJSON(t, `{"auth":{"actor":"alice","role":"user"}}`),
	})
	if initResp.Error != nil {
		t.Fatalf("initialize error: %v", initResp.Error)
	}
	params := map[string]any{
		"name": "submit_parallel_jobs",
		"arguments": map[string]any{
			"task_type":     "document_summary",
			"output_schema": map[string]any{"name": "document_summary_v1"},
			"children": []map[string]any{
				{
					"input_refs":  []map[string]any{{"type": "file", "uri": "file:///tmp/a.txt"}},
					"shard_key":   "repo:a",
					"shard_index": 0,
					"shard_count": 2,
				},
				{
					"input_refs":  []map[string]any{{"type": "file", "uri": "file:///tmp/b.txt"}},
					"shard_key":   "repo:b",
					"shard_index": 1,
					"shard_count": 2,
				},
			},
			"reducer": map[string]any{
				"task_type":     "document_summary",
				"output_schema": map[string]any{"name": "document_summary_v1"},
			},
		},
	}
	paramBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  paramBytes,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	result := resp.Result.(map[string]any)
	structured := result["structuredContent"].(types.SubmitParallelJobsResponse)
	if structured.RootJobID == "" {
		t.Fatalf("expected root_job_id, got %#v", structured)
	}
	if len(structured.Children) != 2 {
		t.Fatalf("expected 2 children, got %#v", structured)
	}
	if structured.ReducerJob == nil {
		t.Fatalf("expected reducer job, got %#v", structured)
	}
}

func TestGetRootJobStatusToolCall(t *testing.T) {
	server := newTestServer()
	initResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      0,
		Method:  "initialize",
		Params:  mustRawJSON(t, `{"auth":{"actor":"alice","role":"user"}}`),
	})
	if initResp.Error != nil {
		t.Fatalf("initialize error: %v", initResp.Error)
	}

	submitParams := mustRawJSON(t, `{
	  "name": "submit_parallel_jobs",
	  "arguments": {
	    "task_type": "document_summary",
	    "output_schema": {"name": "document_summary_v1"},
	    "children": [
	      {"input_refs": [{"type":"file","uri":"file:///tmp/a.txt"}], "shard_index": 0, "shard_count": 2},
	      {"input_refs": [{"type":"file","uri":"file:///tmp/b.txt"}], "shard_index": 1, "shard_count": 2}
	    ]
	  }
	}`)
	submitResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: submitParams,
	})
	if submitResp.Error != nil {
		t.Fatalf("submit error: %v", submitResp.Error)
	}
	submitStructured := submitResp.Result.(map[string]any)["structuredContent"].(types.SubmitParallelJobsResponse)

	call := map[string]any{
		"name": "get_root_job_status",
		"arguments": map[string]any{
			"root_job_id": submitStructured.RootJobID,
		},
	}
	paramBytes, err := json.Marshal(call)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0", ID: 2, Method: "tools/call", Params: paramBytes,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	structured := resp.Result.(map[string]any)["structuredContent"].(types.RootJobStatus)
	if structured.RootJobID != submitStructured.RootJobID {
		t.Fatalf("unexpected root status: %#v", structured)
	}
	if structured.TotalJobs != 2 {
		t.Fatalf("expected 2 total jobs, got %#v", structured)
	}
}

func TestRetryFailedRootShardsToolCall(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := service.NewWithAuditAndOptions(
		jobStore,
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		nil,
		runRoot,
		".",
		service.Options{RootActionMaxRetriedShards: 2},
	)
	now := time.Now().UTC()
	for _, job := range []types.Job{
		{
			ID: "job_ok", TaskType: "document_summary", State: types.JobStateSucceeded, RootJobID: "root_retry_tool",
			Request:       types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}},
			Orchestration: &types.OrchestrationInfo{RootJobID: "root_retry_tool", Strategy: "fanout_child", ShardKey: "doc:a", ShardIndex: 0, ShardCount: 2},
			CreatedAt:     now.Add(-2 * time.Minute), UpdatedAt: now.Add(-2 * time.Minute), SubmittedAt: now.Add(-2 * time.Minute),
		},
		{
			ID: "job_failed", TaskType: "document_summary", State: types.JobStateFailed, RootJobID: "root_retry_tool",
			Request:       types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}},
			Orchestration: &types.OrchestrationInfo{RootJobID: "root_retry_tool", Strategy: "fanout_child", ShardKey: "doc:b", ShardIndex: 1, ShardCount: 2},
			CreatedAt:     now.Add(-1 * time.Minute), UpdatedAt: now.Add(-1 * time.Minute), SubmittedAt: now.Add(-1 * time.Minute),
		},
	} {
		if err := jobStore.CreateJob(context.Background(), job); err != nil {
			t.Fatalf("create job: %v", err)
		}
	}
	server := NewServer(svc, mcpTestPrincipal())

	params := mustRawJSON(t, `{
	  "name": "retry_failed_root_shards",
	  "arguments": {
	    "root_job_id": "root_retry_tool"
	  }
	}`)
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: params,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	structured := resp.Result.(map[string]any)["structuredContent"].(types.RetryFailedRootShardsResponse)
	if structured.RetriedCount != 1 {
		t.Fatalf("expected one retried shard, got %#v", structured)
	}
	if structured.CumulativeRetriedShards != 1 || structured.RemainingRetriedShardBudget != 1 {
		t.Fatalf("expected direct retry budget counters, got %#v", structured)
	}
}

func TestReleaseDeferredRootChunksToolCall(t *testing.T) {
	svc := newMCPThrottledBatchService(t, 2)
	server := NewServer(svc, mcpTestPrincipal())

	submitResp, err := submitParallelJobsAsMCPUser(svc, types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     throttledFourChildRequests(),
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}

	params := mustRawJSON(t, fmt.Sprintf(`{
	  "name": "release_deferred_root_chunks",
	  "arguments": {
	    "root_job_id": %q,
	    "max_additional_batches": 1
	  }
	}`, submitResp.RootJobID))
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: params,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	structured := resp.Result.(map[string]any)["structuredContent"].(types.ReleaseDeferredRootChunksResponse)
	if structured.ReleasedChunks != 1 || structured.ReleasedChildren != 2 {
		t.Fatalf("expected one released chunk with two children, got %#v", structured)
	}
	if structured.CumulativeForcedReleaseChunks != 1 || structured.RemainingForcedReleaseBudget != 1 {
		t.Fatalf("expected direct forced-release budget counters, got %#v", structured)
	}
}

func TestRetryFailedRootShardsToolCallReturnsErrorWhenCumulativeBudgetExceeded(t *testing.T) {
	server, rootJobID := newMCPRetryBudgetExceededFixture(t)
	second := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0", ID: 2, Method: "tools/call", Params: mustRawJSON(t, fmt.Sprintf(`{
		  "name": "retry_failed_root_shards",
		  "arguments": {"root_job_id": %q}
		}`, rootJobID)),
	})
	if second.Error == nil {
		t.Fatal("expected cumulative retry budget error")
	}
	assertToolErrorMessage(t, second.Error, mcpRetryBudgetExceededMessage)
}

func TestReleaseDeferredRootChunksToolCallReturnsErrorWhenCumulativeBudgetExceeded(t *testing.T) {
	server, rootJobID := newMCPReleaseBudgetExceededFixture(t)
	second := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0", ID: 2, Method: "tools/call", Params: mustRawJSON(t, fmt.Sprintf(`{
		  "name": "release_deferred_root_chunks",
		  "arguments": {"root_job_id": %q, "max_additional_batches": 1}
		}`, rootJobID)),
	})
	if second.Error == nil {
		t.Fatal("expected cumulative forced-release budget error")
	}
	assertToolErrorMessage(t, second.Error, mcpReleaseBudgetExceededMessage)
}

func TestFetchJobLogsToolCall(t *testing.T) {
	runRoot := t.TempDir()
	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)
	server := NewServer(svc, mcpTestPrincipal())

	submitResp, err := submitJobAsMCPUser(svc, types.SubmitJobRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}
	jobDir := filepath.Join(runRoot, submitResp.JobID)
	if err := os.WriteFile(filepath.Join(jobDir, "stdout.log"), []byte("token=abc123\n"), 0o644); err != nil {
		t.Fatalf("write stdout: %v", err)
	}

	params := map[string]any{
		"name": "fetch_job_logs",
		"arguments": map[string]any{
			"job_id": submitResp.JobID,
			"stream": "stdout",
		},
	}
	paramBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  paramBytes,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error)
	}
	result := resp.Result.(map[string]any)
	structured := result["structuredContent"].(types.JobLogs)
	if strings.Contains(structured.Content, "abc123") {
		t.Fatalf("expected redacted log content, got %q", structured.Content)
	}
}

func TestFetchJobLogsToolCallPolicyDenied(t *testing.T) {
	runRoot := t.TempDir()
	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)
	server := NewServer(svc, mcpTestPrincipal())

	submitResp, err := submitJobAsMCPUser(svc, types.SubmitJobRequest{
		TaskType: "log_analysis",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file:///tmp/build.log", Classification: "phi"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "log_analysis_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	params := map[string]any{
		"name": "fetch_job_logs",
		"arguments": map[string]any{
			"job_id": submitResp.JobID,
		},
	}
	paramBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  paramBytes,
	})
	if resp.Error == nil {
		t.Fatal("expected policy denial error")
	}
	if !strings.Contains(resp.Error.Message, "policy denied") {
		t.Fatalf("unexpected error: %#v", resp.Error)
	}
}

func TestFetchResultToolCallAppliesReleasePolicy(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := service.New(
		jobStore,
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)
	server := NewServer(svc, mcpTestPrincipal())

	now := time.Now().UTC()
	job := types.Job{
		ID:       "job_mcp_result_policy",
		TaskType: "repo_summary",
		State:    types.JobStateSucceeded,
		Request: types.SubmitJobRequest{
			TaskType: "repo_summary",
			InputRefs: []types.InputRef{
				{Type: "directory", URI: "file:///tmp/repo", Classification: "restricted"},
			},
			OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
		},
		Result: &types.Result{
			SchemaName:    "repo_summary_v1",
			SchemaVersion: "1.0.0",
			Payload: map[string]any{
				"summary": "summary",
				"entrypoints": []any{
					map[string]any{"path": "broker/main.go", "kind": "service_entrypoint"},
				},
			},
		},
		RuntimeDiagnostics: map[string]any{
			"backend_name":        "llama.cpp",
			"backend_mode":        "unavailable",
			"endpoint_configured": true,
			"last_error":          "connection refused",
		},
		ExecutionQuality:       "no_real_backend",
		DegradedLocalExecution: true,
		RetryRecommended:       true,
		Artifacts: []types.Artifact{
			{ArtifactID: "artifact_1", ArtifactType: "chunk_manifest", Path: "/tmp/manifest.json"},
		},
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	params := map[string]any{
		"name": "fetch_result",
		"arguments": map[string]any{
			"job_id": job.ID,
		},
	}
	paramBytes, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  paramBytes,
	})
	if resp.Error != nil {
		t.Fatalf("unexpected error: %#v", resp.Error)
	}
	result := resp.Result.(map[string]any)
	structured := result["structuredContent"].(types.JobResultRelease)
	if structured.Result == nil {
		t.Fatal("expected released result")
	}
	entrypoints := structured.Result.Payload["entrypoints"].([]any)
	first := entrypoints[0].(map[string]any)
	if first["path"] != "[REDACTED]" {
		t.Fatalf("expected redacted path, got %#v", first["path"])
	}
	if len(structured.Artifacts) != 0 {
		t.Fatalf("expected withheld artifacts, got %#v", structured.Artifacts)
	}
	if structured.RuntimeDiagnostics["backend_mode"] != "unavailable" {
		t.Fatalf("expected runtime diagnostics in structured release, got %#v", structured.RuntimeDiagnostics)
	}
	if !structured.DegradedLocalExecution || !structured.RetryRecommended || structured.ExecutionQuality != "no_real_backend" {
		t.Fatalf("expected summary flags in structured release, got %#v", structured)
	}
}

func TestRetryRecommendationTools(t *testing.T) {
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := service.New(
		jobStore,
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		runRoot,
		".",
	)
	server := NewServer(svc, mcpTestPrincipal())

	now := time.Now().UTC()
	job := types.Job{
		ID:          "job_mcp_retry_rec",
		TaskType:    "rag_compress",
		State:       types.JobStateSucceeded,
		SubmittedBy: "mcp:test",
		Request: types.SubmitJobRequest{
			TaskType:     "rag_compress",
			OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
			ExecutionProfile: types.ExecutionProfile{
				Backend: "slurm",
				Tier:    "p40-rag-compression",
				Runtime: "llama.cpp",
			},
		},
		Result: &types.Result{
			SchemaName:    "rag_evidence_pack_v1",
			SchemaVersion: "1.0.0",
			Payload: map[string]any{
				"query": "why did the build fail?",
				"broker_retry_recommendation": map[string]any{
					"recommended": true,
					"reason":      "no_real_retrieval_backend",
					"task_type":   "rag_compress",
					"execution_profile": map[string]any{
						"backend": "slurm",
						"tier":    "a100-reasoning",
						"runtime": "llama.cpp",
					},
					"placement_hint": map[string]any{
						"backend_preference": "slurm",
						"tier_preference":    "a100-reasoning",
						"qos":                "scavenger",
						"preemptible":        true,
					},
				},
			},
		},
		ResultError: "broker_policy_no_real_retrieval_backend",
		CreatedAt:   now,
		UpdatedAt:   now,
		SubmittedAt: now,
		CompletedAt: &now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}

	recResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: mustRawJSON(t, `{
		  "name": "get_retry_recommendation",
		  "arguments": {"job_id": "job_mcp_retry_rec"}
		}`),
	})
	if recResp.Error != nil {
		t.Fatalf("unexpected recommendation error: %#v", recResp.Error)
	}
	rec := recResp.Result.(map[string]any)["structuredContent"].(types.JobRetryRecommendation)
	if rec.ExecutionProfile.Tier != "a100-reasoning" {
		t.Fatalf("expected a100 retry recommendation, got %#v", rec)
	}
	if rec.PlacementHint.TierPreference != "a100-reasoning" || !rec.PlacementHint.Preemptible {
		t.Fatalf("expected placement hint on recommendation, got %#v", rec)
	}

	retryResp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: mustRawJSON(t, `{
		  "name": "retry_with_recommended_profile",
		  "arguments": {"job_id": "job_mcp_retry_rec"}
		}`),
	})
	if retryResp.Error != nil {
		t.Fatalf("unexpected retry error: %#v", retryResp.Error)
	}
	submit := retryResp.Result.(map[string]any)["structuredContent"].(types.SubmitJobResponse)
	retriedJob, err := jobStore.GetJob(context.Background(), submit.JobID)
	if err != nil {
		t.Fatalf("get retried job: %v", err)
	}
	if retriedJob.Request.ExecutionProfile.Tier != "a100-reasoning" {
		t.Fatalf("expected retried job to use recommended tier, got %#v", retriedJob.Request.ExecutionProfile)
	}
	if retriedJob.Request.ExecutionProfile.QOS != "scavenger" {
		t.Fatalf("expected retried job to use recommended qos, got %#v", retriedJob.Request.ExecutionProfile)
	}
	if retriedJob.Request.TaskParams["_broker_retry_qos"] != "scavenger" || retriedJob.Request.TaskParams["_broker_retry_preemptible"] != true {
		t.Fatalf("expected placement hint merged into task params, got %#v", retriedJob.Request.TaskParams)
	}
}

func newTestServer() *Server {
	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		".broker/runs",
		".",
	)
	return NewServer(svc, mcpTestPrincipal())
}

func submitJobAsMCPUser(svc *service.Service, req types.SubmitJobRequest) (*types.SubmitJobResponse, error) {
	resp, err := svc.SubmitJob(auth.WithPrincipal(context.Background(), mcpTestPrincipal()), req)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func submitParallelJobsAsMCPUser(svc *service.Service, req types.SubmitParallelJobsRequest) (*types.SubmitParallelJobsResponse, error) {
	resp, err := svc.SubmitParallelJobs(auth.WithPrincipal(context.Background(), mcpTestPrincipal()), req)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func newMCPThrottledBatchService(t *testing.T, releaseBudget int) *service.Service {
	t.Helper()
	return service.NewWithAuditAndOptions(
		store.NewMemoryJobStore(),
		&mcpTestBatchBackend{status: backends.RunStatus{State: types.JobStateQueued, RawState: "PENDING"}},
		log.New(io.Discard, "", 0),
		nil,
		t.TempDir(),
		".",
		service.Options{
			ParallelMaxBatchSize:           2,
			ParallelMaxActiveBatches:       1,
			RootActionMaxAdditionalBatches: releaseBudget,
		},
	)
}

func assertToolErrorMessage(t *testing.T, err *respError, expectedMessage string) {
	t.Helper()
	if err == nil {
		t.Fatal("expected tool error")
	}
	if err.Code != -32000 || !strings.Contains(err.Message, expectedMessage) {
		t.Fatalf("unexpected error: %#v expected_substring=%q", err, expectedMessage)
	}
}

func newMCPRetryBudgetExceededFixture(t *testing.T) (*Server, string) {
	t.Helper()
	runRoot := t.TempDir()
	jobStore := store.NewMemoryJobStore()
	svc := service.NewWithAuditAndOptions(
		jobStore,
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		nil,
		runRoot,
		".",
		service.Options{RootActionMaxRetriedShards: 1},
	)
	now := time.Now().UTC()
	rootJobID := "root_retry_tool_cap"
	job := types.Job{
		ID: "job_failed_once", TaskType: "document_summary", State: types.JobStateFailed, RootJobID: rootJobID,
		SubmittedBy: "mcp:test",
		Request:     types.SubmitJobRequest{TaskType: "document_summary", OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"}},
		Orchestration: &types.OrchestrationInfo{
			RootJobID: rootJobID, Strategy: "fanout_child", ShardIndex: 0, ShardCount: 1,
		},
		CreatedAt: now, UpdatedAt: now, SubmittedAt: now,
	}
	if err := jobStore.CreateJob(context.Background(), job); err != nil {
		t.Fatalf("create job: %v", err)
	}
	server := NewServer(svc, mcpTestPrincipal())
	first := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: mustRawJSON(t, fmt.Sprintf(`{
		  "name": "retry_failed_root_shards",
		  "arguments": {"root_job_id": %q}
		}`, rootJobID)),
	})
	if first.Error != nil {
		t.Fatalf("unexpected first retry error: %#v", first.Error)
	}
	firstStructured := first.Result.(map[string]any)["structuredContent"].(types.RetryFailedRootShardsResponse)
	retriedJob, err := jobStore.GetJob(context.Background(), firstStructured.RetriedShards[0].JobID)
	if err != nil {
		t.Fatalf("get retried job: %v", err)
	}
	retriedJob.State = types.JobStateFailed
	retriedJob.BackendRunID = ""
	if err := jobStore.UpdateJob(context.Background(), retriedJob); err != nil {
		t.Fatalf("mark retried job failed again: %v", err)
	}
	return server, rootJobID
}

func newMCPReleaseBudgetExceededFixture(t *testing.T) (*Server, string) {
	t.Helper()
	svc := newMCPThrottledBatchService(t, 1)
	server := NewServer(svc, mcpTestPrincipal())
	submitResp, err := submitParallelJobsAsMCPUser(svc, types.SubmitParallelJobsRequest{
		TaskType:     "document_summary",
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
		Children:     throttledSixChildRequests(),
	})
	if err != nil {
		t.Fatalf("submit parallel jobs: %v", err)
	}
	first := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0", ID: 1, Method: "tools/call", Params: mustRawJSON(t, fmt.Sprintf(`{
		  "name": "release_deferred_root_chunks",
		  "arguments": {"root_job_id": %q, "max_additional_batches": 1}
		}`, submitResp.RootJobID)),
	})
	if first.Error != nil {
		t.Fatalf("unexpected first release error: %#v", first.Error)
	}
	return server, submitResp.RootJobID
}

func TestToolsCallRequiresInitializedIdentity(t *testing.T) {
	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		t.TempDir(),
		".",
	)
	server := NewServer(svc, auth.Principal{})
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/list",
	})
	if resp.Error == nil {
		t.Fatal("expected identity error")
	}
	if !strings.Contains(resp.Error.Message, "identity") {
		t.Fatalf("unexpected error: %#v", resp.Error)
	}
}

func TestInitializeSetsSessionPrincipal(t *testing.T) {
	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(config.Config{}),
		log.New(io.Discard, "", 0),
		t.TempDir(),
		".",
	)
	server := NewServer(svc, auth.Principal{})
	resp := server.handleRequest(context.Background(), request{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params:  mustRawJSON(t, `{"auth":{"actor":"alice","role":"admin"}}`),
	})
	if resp.Error != nil {
		t.Fatalf("unexpected initialize error: %#v", resp.Error)
	}
	server.mu.RLock()
	principal := server.sessionPrincipal
	server.mu.RUnlock()
	if principal.Actor != "alice" || principal.Role != "admin" {
		t.Fatalf("unexpected session principal: %#v", principal)
	}
}

func mustRawJSON(t *testing.T, text string) json.RawMessage {
	t.Helper()
	return json.RawMessage(text)
}

func throttledSixChildRequests() []types.ParallelChildRequest {
	children := make([]types.ParallelChildRequest, 0, 6)
	for i := 0; i < 6; i++ {
		children = append(children, types.ParallelChildRequest{
			InputRefs:  []types.InputRef{{Type: "file", URI: "file:///tmp/" + string(rune('a'+i)) + ".txt"}},
			ShardIndex: i,
			ShardCount: 6,
		})
	}
	return children
}

func throttledFourChildRequests() []types.ParallelChildRequest {
	children := make([]types.ParallelChildRequest, 0, 4)
	for i := 0; i < 4; i++ {
		children = append(children, types.ParallelChildRequest{
			InputRefs:  []types.InputRef{{Type: "file", URI: "file:///tmp/" + string(rune('a'+i)) + ".txt"}},
			ShardIndex: i,
			ShardCount: 4,
		})
	}
	return children
}

func frameJSON(payload string) string {
	return "Content-Length: " + fmt.Sprintf("%d", len(payload)) + "\r\n\r\n" + payload
}

func decodeFramedResponses(t *testing.T, payload []byte) [][]byte {
	t.Helper()
	reader := bytes.NewReader(payload)
	bufReader := io.Reader(reader)
	data, err := io.ReadAll(bufReader)
	if err != nil {
		t.Fatalf("read payload: %v", err)
	}

	remaining := string(data)
	var out [][]byte
	for len(strings.TrimSpace(remaining)) > 0 {
		parts := strings.SplitN(remaining, "\r\n\r\n", 2)
		if len(parts) != 2 {
			t.Fatalf("invalid framed payload: %q", remaining)
		}
		header := parts[0]
		bodyAndRest := parts[1]

		var contentLength int
		if _, err := fmt.Sscanf(header, "Content-Length: %d", &contentLength); err != nil {
			t.Fatalf("parse content length: %v", err)
		}
		if len(bodyAndRest) < contentLength {
			t.Fatalf("body shorter than content length")
		}
		body := bodyAndRest[:contentLength]
		out = append(out, []byte(body))
		remaining = bodyAndRest[contentLength:]
	}
	return out
}

type mcpTestBatchBackend struct {
	status backends.RunStatus
}

type waitForResultBackend struct {
	runRoot  string
	delay    time.Duration
	mu       sync.Mutex
	complete map[string]bool
}

func (b *waitForResultBackend) Name() string { return "waitable" }

func (b *waitForResultBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	b.mu.Lock()
	if b.complete == nil {
		b.complete = make(map[string]bool)
	}
	b.complete[job.ID] = false
	b.mu.Unlock()

	go func() {
		time.Sleep(b.delay)
		jobDir := filepath.Join(b.runRoot, job.ID)
		_ = os.MkdirAll(jobDir, 0o755)
		_ = os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "document_summary_v1",
  "schema_version": "1.0.0",
  "payload": {
    "summary": "waited summary"
  }
}`), 0o644)
		_ = os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644)
		b.mu.Lock()
		b.complete[job.ID] = true
		b.mu.Unlock()
	}()

	return backends.SubmitResponse{
		BackendKind:  "waitable",
		BackendRunID: job.ID,
		InitialState: types.JobStateQueued,
	}, nil
}

func (b *waitForResultBackend) GetRun(_ context.Context, backendRunID string) (backends.RunStatus, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.complete[backendRunID] {
		return backends.RunStatus{State: types.JobStateSucceeded, RawState: "COMPLETED", ExitCode: "0:0"}, nil
	}
	return backends.RunStatus{State: types.JobStateQueued, RawState: "PENDING"}, nil
}

func (b *waitForResultBackend) CancelRun(context.Context, string) error { return nil }

type localInspectRepoEarlyResultBackend struct {
	runRoot  string
	delay    time.Duration
	mu       sync.Mutex
	complete map[string]bool
}

func (b *localInspectRepoEarlyResultBackend) Name() string { return "local-inspect-repo-early-result" }

func (b *localInspectRepoEarlyResultBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	b.mu.Lock()
	if b.complete == nil {
		b.complete = make(map[string]bool)
	}
	b.complete[job.ID] = false
	b.mu.Unlock()

	go func() {
		time.Sleep(b.delay)
		jobDir := filepath.Join(b.runRoot, job.ID)
		_ = os.MkdirAll(jobDir, 0o755)
		_ = os.WriteFile(filepath.Join(jobDir, "result.json"), []byte(`{
  "schema_name": "repo_inspection_v2",
  "schema_version": "2.0.0",
  "payload": {
    "mode": "evidence",
    "query": "trace inspect_repo timeout handling",
    "findings": [],
    "evidence": [
      {
        "id": "ev1",
        "path": "broker/pkg/mcp/server.go",
        "source_refs": [
          {
            "path": "broker/pkg/mcp/server.go",
            "line_start": 435,
            "line_end": 465
          }
        ]
      }
    ],
    "quality": {
      "result": "evidence_only",
      "retrieval": "lexical_degraded",
      "reranking": "unavailable",
      "synthesis": "not_requested",
      "answer_ready": false
    },
    "warnings": [],
    "provenance": {"index_fingerprint": "sha256:test"},
    "retrieval": {},
    "runtime": {"attempts": []}
  }
}`), 0o644)
		_ = os.WriteFile(filepath.Join(jobDir, "artifacts.json"), []byte(`[]`), 0o644)
		time.Sleep(150 * time.Millisecond)
		b.mu.Lock()
		b.complete[job.ID] = true
		b.mu.Unlock()
	}()

	return backends.SubmitResponse{
		BackendKind:  "local",
		BackendRunID: job.ID,
		InitialState: types.JobStateQueued,
	}, nil
}

func (b *localInspectRepoEarlyResultBackend) GetRun(_ context.Context, backendRunID string) (backends.RunStatus, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.complete[backendRunID] {
		return backends.RunStatus{State: types.JobStateSucceeded, RawState: "COMPLETED", ExitCode: "0:0"}, nil
	}
	return backends.RunStatus{State: types.JobStateQueued, RawState: "PENDING"}, nil
}

func (b *localInspectRepoEarlyResultBackend) CancelRun(context.Context, string) error { return nil }

type queuedOnlyBackend struct{}

func (b *queuedOnlyBackend) Name() string { return "queued-only" }

func (b *queuedOnlyBackend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	return backends.SubmitResponse{
		BackendKind:  "queued-only",
		BackendRunID: job.ID,
		InitialState: types.JobStateQueued,
	}, nil
}

func (b *queuedOnlyBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return backends.RunStatus{State: types.JobStateQueued, RawState: "PENDING"}, nil
}

func (b *queuedOnlyBackend) CancelRun(context.Context, string) error { return nil }

func (f *mcpTestBatchBackend) Name() string { return "mcp-fake-batch" }

func (f *mcpTestBatchBackend) SubmitRun(context.Context, types.Job) (backends.SubmitResponse, error) {
	return backends.SubmitResponse{
		BackendKind:  "mcp-fake-batch",
		BackendRunID: "single-run-1",
		InitialState: types.JobStateQueued,
	}, nil
}

func (f *mcpTestBatchBackend) SubmitRunBatch(_ context.Context, jobs []types.Job) ([]backends.SubmitResponse, error) {
	responses := make([]backends.SubmitResponse, 0, len(jobs))
	for i := range jobs {
		responses = append(responses, backends.SubmitResponse{
			BackendKind:  "mcp-fake-batch",
			BackendRunID: "batch-run-" + string(rune('0'+i)),
			InitialState: types.JobStateQueued,
		})
	}
	return responses, nil
}

func (f *mcpTestBatchBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return f.status, nil
}

func (f *mcpTestBatchBackend) CancelRun(context.Context, string) error { return nil }
