package service

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type executionTestBackend struct {
	name string
}

func (b executionTestBackend) Name() string { return b.name }

func (executionTestBackend) SubmitRun(context.Context, types.Job) (backends.SubmitResponse, error) {
	return backends.SubmitResponse{}, nil
}

func (executionTestBackend) GetRun(context.Context, string) (backends.RunStatus, error) {
	return backends.RunStatus{}, nil
}

func (executionTestBackend) CancelRun(context.Context, string) error { return nil }

func TestStageExecutionBundleUsesStableInspectRepoNodeLocalCacheForLocalBackend(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	svc := &Service{
		runRoot:  runRoot,
		repoRoot: repoRoot,
		backend:  executionTestBackend{name: "local"},
	}
	job := types.Job{
		ID:       "job_123",
		TaskType: "inspect_repo",
		Request: types.SubmitJobRequest{
			TaskType: "inspect_repo",
			TaskParams: map[string]any{
				"query": "trace routing",
				"mode":  "evidence",
			},
			InputRefs: []types.InputRef{
				{Type: "repo", URI: repoRoot},
			},
		},
	}

	if err := svc.stageExecutionBundle(context.Background(), &job); err != nil {
		t.Fatalf("stageExecutionBundle: %v", err)
	}

	executionPlanPath := filepath.Join(runRoot, job.ID, "execution_plan.json")
	payload, err := os.ReadFile(executionPlanPath)
	if err != nil {
		t.Fatalf("read execution plan: %v", err)
	}
	var plan map[string]any
	if err := json.Unmarshal(payload, &plan); err != nil {
		t.Fatalf("decode execution plan: %v", err)
	}
	if enabled, ok := plan["repo_inspection_use_node_local_cache"].(bool); !ok || !enabled {
		t.Fatalf("expected local backend inspect_repo to request node-local cache, got %#v", plan["repo_inspection_use_node_local_cache"])
	}
	namespace, ok := plan["repo_inspection_node_local_cache_namespace"].(string)
	if !ok || strings.TrimSpace(namespace) == "" {
		t.Fatalf("expected local backend inspect_repo node-local cache namespace, got %#v", plan["repo_inspection_node_local_cache_namespace"])
	}
}

func TestStageExecutionBundleMarksInspectRepoForNodeLocalCacheOnNonLocalBackend(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	svc := &Service{
		runRoot:  runRoot,
		repoRoot: repoRoot,
		backend:  executionTestBackend{name: "slurm"},
	}
	job := types.Job{
		ID:       "job_123",
		TaskType: "inspect_repo",
		Request: types.SubmitJobRequest{
			TaskType: "inspect_repo",
			TaskParams: map[string]any{
				"query": "trace routing",
				"mode":  "evidence",
			},
			InputRefs: []types.InputRef{
				{Type: "repo", URI: repoRoot},
			},
			ExecutionProfile: types.ExecutionProfile{Backend: "slurm"},
		},
	}

	if err := svc.stageExecutionBundle(context.Background(), &job); err != nil {
		t.Fatalf("stageExecutionBundle: %v", err)
	}

	executionPlanPath := filepath.Join(runRoot, job.ID, "execution_plan.json")
	payload, err := os.ReadFile(executionPlanPath)
	if err != nil {
		t.Fatalf("read execution plan: %v", err)
	}
	var plan map[string]any
	if err := json.Unmarshal(payload, &plan); err != nil {
		t.Fatalf("decode execution plan: %v", err)
	}
	if enabled, ok := plan["repo_inspection_use_node_local_cache"].(bool); !ok || !enabled {
		t.Fatalf("expected non-local inspect_repo to request node-local cache, got %#v", plan["repo_inspection_use_node_local_cache"])
	}
}
