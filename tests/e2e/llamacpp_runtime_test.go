package e2e

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	localbackend "github.com/msk-mind/local-ai-broker/broker/pkg/backends/local"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/service"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestLocalBackendRAGLlamaCPPRuntimeSmoke(t *testing.T) {
	if os.Getenv("OLLAMA_SLURM_E2E_LOOPBACK") != "1" {
		t.Skip("set OLLAMA_SLURM_E2E_LOOPBACK=1 to run loopback-binding e2e runtime smoke")
	}
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	fakeLLMURL, countFile := startFakeOpenAIServer(t, repoDir)
	inputRepo := filepath.Join(t.TempDir(), "repo")
	writeTestFile(t, filepath.Join(inputRepo, "src", "main.py"), "def run_service():\n    raise RuntimeError(\"smoke failure\")\n")

	backend := localbackend.NewBackend(config.Config{
		LocalMode:       "command",
		LocalScriptPath: filepath.Join(repoDir, "deploy", "local", "broker_worker.sh"),
		RunRootPath:     runRoot,
		RepoRootPath:    repoDir,
	})
	svc := service.NewWithAuditAndOptionsAndConfig(
		store.NewMemoryJobStore(),
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		repoDir,
		service.Options{},
		&config.Config{
			ModelProfileP40:               "gpt-oss-20b.p40",
			RuntimeLlamaCPPBaseURL:        fakeLLMURL,
			RuntimeLlamaCPPTimeoutSeconds: 10,
		},
	)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "rag_compress",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + inputRepo, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "Why does the service fail?",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      16000,
			PerChunkCompressionBudget: 192,
			FinalEvidencePackBudget:   1200,
			RemoteModelContextBudget:  4000,
		},
		ExecutionProfile: types.ExecutionProfile{
			Backend: "local",
			Tier:    "p40-rag-compression",
		},
		OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job := waitForJob(t, svc, resp.JobID, 15*time.Second)

	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded rag result, got state=%q result=%#v", job.State, job.Result)
	}

	retrieval, _ := job.Result.Payload["retrieval"].(map[string]any)
	if retrieval["runtime_backend_mode"] != "real" {
		t.Fatalf("expected live runtime backend mode, got %#v", retrieval)
	}
	provenance, _ := job.Result.Payload["provenance"].(map[string]any)
	if provenance["runtime_backend"] != "llama.cpp" {
		t.Fatalf("expected llama.cpp provenance, got %#v", provenance)
	}
	if !hasArtifact(job.Artifacts, "artifact_runtime_context") {
		t.Fatalf("expected artifact_runtime_context, got %#v", job.Artifacts)
	}
	requestCount, err := os.ReadFile(countFile)
	if err != nil {
		t.Fatalf("read fake request count: %v", err)
	}
	if strings.TrimSpace(string(requestCount)) == "0" {
		t.Fatal("expected fake llama.cpp endpoint to receive at least one request")
	}
}

func TestLocalBackendRAGLlamaCPPRuntimeUnavailableFallsBack(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	inputRepo := filepath.Join(t.TempDir(), "repo")
	writeTestFile(t, filepath.Join(inputRepo, "src", "main.py"), "def run_service():\n    raise RuntimeError(\"smoke failure\")\n")
	writeTestFile(t, filepath.Join(inputRepo, "build.log"), "fatal error: generated header missing\ntraceback: service failed to start\n")

	backend := localbackend.NewBackend(config.Config{
		LocalMode:       "command",
		LocalScriptPath: filepath.Join(repoDir, "deploy", "local", "broker_worker.sh"),
		RunRootPath:     runRoot,
		RepoRootPath:    repoDir,
	})
	svc := service.NewWithAuditAndOptionsAndConfig(
		store.NewMemoryJobStore(),
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		repoDir,
		service.Options{},
		&config.Config{
			ModelProfileP40:               "gpt-oss-20b.p40",
			RuntimeLlamaCPPBaseURL:        "http://127.0.0.1:9",
			RuntimeLlamaCPPTimeoutSeconds: 1,
		},
	)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "rag_compress",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + inputRepo, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "Why does the service fail?",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      16000,
			PerChunkCompressionBudget: 192,
			FinalEvidencePackBudget:   1200,
			RemoteModelContextBudget:  4000,
		},
		ExecutionProfile: types.ExecutionProfile{
			Backend: "local",
			Tier:    "p40-rag-compression",
		},
		OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job := waitForJob(t, svc, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded rag result, got state=%q result=%#v", job.State, job.Result)
	}

	retrieval, _ := job.Result.Payload["retrieval"].(map[string]any)
	if retrieval["runtime_backend_mode"] != "unavailable" {
		t.Fatalf("expected unavailable runtime backend mode, got %#v", retrieval)
	}
	if retrieval["compression_backend"] != "llama.cpp" {
		t.Fatalf("expected llama.cpp compression backend label, got %#v", retrieval)
	}

	policySignals, _ := job.Result.Payload["policy_signals"].(map[string]any)
	if policySignals == nil {
		t.Fatalf("expected policy_signals in payload: %#v", job.Result.Payload)
	}
	if policySignals["real_backend_required_recommended"] != true {
		t.Fatalf("expected real_backend_required_recommended=true, got %#v", policySignals)
	}

	warnings, _ := job.Result.Payload["warnings"].([]any)
	if len(warnings) == 0 {
		t.Fatalf("expected warnings for degraded local runtime: %#v", job.Result.Payload)
	}
	if !hasArtifact(job.Artifacts, "artifact_runtime_diagnostics") {
		t.Fatalf("expected artifact_runtime_diagnostics, got %#v", job.Artifacts)
	}

	diagnostics := readArtifactJSON(t, job.Artifacts, "artifact_runtime_diagnostics")
	if diagnostics["backend_mode"] != "unavailable" {
		t.Fatalf("expected runtime diagnostics backend_mode=unavailable, got %#v", diagnostics)
	}
	lastError, _ := diagnostics["last_error"].(string)
	if lastError == "" {
		t.Fatalf("expected runtime diagnostics last_error, got %#v", diagnostics)
	}
}

func TestLocalBackendInspectRepoUsesStructureRetrievalWithoutLLM(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	inputRepo := filepath.Join(t.TempDir(), "repo")
	writeTestFile(t, filepath.Join(inputRepo, "cmd", "demo", "main.go"), "package main\n\nfunc main() {}\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service.go"), "package service\n\ntype Service struct{}\n")
	writeTestFile(t, filepath.Join(inputRepo, "README.md"), "# Demo repo\n")

	backend := localbackend.NewBackend(config.Config{
		LocalMode:       "command",
		LocalScriptPath: filepath.Join(repoDir, "deploy", "local", "broker_worker.sh"),
		RunRootPath:     runRoot,
		RepoRootPath:    repoDir,
	})
	svc := service.NewWithAuditAndOptionsAndConfig(
		store.NewMemoryJobStore(),
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		repoDir,
		service.Options{},
		&config.Config{},
	)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + inputRepo, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "audit this repo",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      16000,
			PerChunkCompressionBudget: 192,
			FinalEvidencePackBudget:   1200,
			RemoteModelContextBudget:  4000,
		},
		ExecutionProfile: types.ExecutionProfile{
			Backend: "local",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo job: %v", err)
	}

	job := waitForJob(t, svc, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded inspect_repo result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.DegradedLocalExecution {
		t.Fatalf("expected inspect_repo not to be degraded, got %#v", job)
	}
	if job.RetryRecommended {
		t.Fatalf("expected retry_recommended=false, got %#v", job)
	}
	if job.ExecutionQuality != "real_local" {
		t.Fatalf("expected execution_quality real_local, got %#v", job.ExecutionQuality)
	}

	if job.RuntimeDiagnostics == nil {
		t.Fatalf("expected runtime diagnostics, got %#v", job)
	}
	if job.RuntimeDiagnostics["backend_mode"] != "real" {
		t.Fatalf("expected runtime backend_mode=real, got %#v", job.RuntimeDiagnostics)
	}

	warnings, _ := job.Result.Payload["warnings"].([]any)
	for _, warning := range warnings {
		text, _ := warning.(string)
		if text == "broker_no_real_retrieval_backend" || text == "broker_retrieval_quality_gate_failed" {
			t.Fatalf("unexpected retrieval quality warning in %#v", job.Result.Payload)
		}
	}
}

func readArtifactJSON(t *testing.T, artifacts []types.Artifact, artifactID string) map[string]any {
	t.Helper()

	for _, artifact := range artifacts {
		if artifact.ArtifactID != artifactID {
			continue
		}
		raw, err := os.ReadFile(artifact.Path)
		if err != nil {
			t.Fatalf("read artifact %s: %v", artifactID, err)
		}
		var payload map[string]any
		if err := json.Unmarshal(raw, &payload); err != nil {
			t.Fatalf("decode artifact %s: %v", artifactID, err)
		}
		return payload
	}
	t.Fatalf("artifact %s not found", artifactID)
	return nil
}
