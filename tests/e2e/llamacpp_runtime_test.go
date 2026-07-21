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

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)

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

func TestLocalBackendRepoSummaryProducesUsefulStructure(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	inputRepo := filepath.Join(t.TempDir(), "repo")
	writeTestFile(t, filepath.Join(inputRepo, "go.mod"), "module example.com/test\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "main.go"), "package main\nfunc main(){}\n")
	writeTestFile(t, filepath.Join(inputRepo, "workers", "task", "main.py"), "def main():\n    return 0\n")
	writeTestFile(t, filepath.Join(inputRepo, "deploy", "slurm", "broker_worker.slurm"), "#!/bin/bash\necho run\n")
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
		TaskType: "repo_summary",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + inputRepo, Classification: "internal"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit repo_summary job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded repo_summary result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.Result.SchemaName != "repo_summary_v1" {
		t.Fatalf("expected repo_summary_v1, got %#v", job.Result)
	}
	summary, _ := job.Result.Payload["summary"].(string)
	if !strings.Contains(summary, "Top content categories:") {
		t.Fatalf("expected summary categories, got %#v", job.Result.Payload)
	}
	subsystems, ok := job.Result.Payload["subsystems"].([]any)
	if !ok || len(subsystems) == 0 {
		t.Fatalf("expected subsystems, got %#v", job.Result.Payload)
	}
	entrypoints, ok := job.Result.Payload["entrypoints"].([]any)
	if !ok || len(entrypoints) == 0 {
		t.Fatalf("expected entrypoints, got %#v", job.Result.Payload)
	}
	dependencies, ok := job.Result.Payload["dependencies"].([]any)
	if !ok || len(dependencies) == 0 {
		t.Fatalf("expected dependencies, got %#v", job.Result.Payload)
	}
	depText := strings.Join(anyStrings(flattenMapField(dependencies, "name")), "\n")
	if !strings.Contains(depText, "Go toolchain") || !strings.Contains(depText, "Python 3") || !strings.Contains(depText, "Slurm") {
		t.Fatalf("expected Go, Python, and Slurm dependencies, got %#v", dependencies)
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

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
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

func TestLocalBackendDebugWithLocalContextProducesActionableHypotheses(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	inputRepo := filepath.Join(t.TempDir(), "repo")
	inputLog := filepath.Join(t.TempDir(), "debug.log")
	writeTestFile(t, filepath.Join(inputRepo, "src", "service.py"), "def run_service():\n    raise RuntimeError(\"demo failure\")\n")
	writeTestFile(t, inputLog, strings.Join([]string{
		"2026-07-09T12:00:00Z FAILED demo_test",
		"Traceback (most recent call last):",
		"RuntimeError: demo failure",
	}, "\n")+"\n")

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
		TaskType: "debug_with_local_context",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + inputRepo, Classification: "internal"},
			{Type: "log", URI: "file://" + inputLog, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"problem":       "demo_test fails with traceback",
			"failing_tests": []any{"demo_test"},
			"suspect_paths": []any{"src/service.py"},
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
		OutputSchema: types.OutputSchemaRef{Name: "debug_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit debug_with_local_context job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded debug result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.Result.SchemaName != "debug_evidence_pack_v1" {
		t.Fatalf("expected debug_evidence_pack_v1, got %#v", job.Result)
	}

	topHypotheses, ok := job.Result.Payload["top_hypotheses"].([]any)
	if !ok || len(topHypotheses) == 0 {
		t.Fatalf("expected top_hypotheses, got %#v", job.Result.Payload)
	}
	firstHypothesis, ok := topHypotheses[0].(map[string]any)
	if !ok || firstHypothesis["claim"] == nil {
		t.Fatalf("expected first hypothesis claim, got %#v", topHypotheses)
	}
	claim, _ := firstHypothesis["claim"].(string)
	if !strings.Contains(strings.ToLower(claim), "runtime exception") && !strings.Contains(strings.ToLower(claim), "test failure") {
		t.Fatalf("expected actionable hypothesis about runtime exception or test failure, got %#v", firstHypothesis)
	}

	failureSignature, ok := job.Result.Payload["failure_signature"].(map[string]any)
	if !ok {
		t.Fatalf("expected failure_signature, got %#v", job.Result.Payload)
	}
	testsValue, ok := failureSignature["tests"].([]any)
	if !ok || len(testsValue) == 0 || testsValue[0] != "demo_test" {
		t.Fatalf("expected failing test in failure_signature, got %#v", failureSignature)
	}

	followups, ok := job.Result.Payload["suggested_local_followups"].([]any)
	if !ok || len(followups) == 0 {
		t.Fatalf("expected suggested_local_followups, got %#v", job.Result.Payload)
	}
	firstFollowup, ok := followups[0].(map[string]any)
	if !ok || firstFollowup["tool"] != "rag_compress" {
		t.Fatalf("expected rag_compress followup, got %#v", followups)
	}

	if job.Result.Payload["recommended_next_action"] != "Verify the cited files directly, then answer using only the confirmed evidence." {
		t.Fatalf("unexpected next action: %#v", job.Result.Payload)
	}
}

func TestLocalBackendSummarizeLogsProducesClustersAndTimeline(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	inputLog := filepath.Join(t.TempDir(), "service.log")
	writeTestFile(t, inputLog, strings.Join([]string{
		"2026-07-09T12:00:00Z build started",
		"2026-07-09T12:00:01Z fatal error: generated/config.h missing",
		"2026-07-09T12:00:02Z undefined reference to demo_symbol",
		"2026-07-09T12:00:03Z FAILED demo_test",
	}, "\n")+"\n")

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
		TaskType: "summarize_logs",
		InputRefs: []types.InputRef{
			{Type: "log", URI: "file://" + inputLog, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "Summarize the root failure and dominant warnings.",
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
		OutputSchema: types.OutputSchemaRef{Name: "log_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit summarize_logs job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded summarize_logs result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.Result.SchemaName != "log_evidence_pack_v1" {
		t.Fatalf("expected log_evidence_pack_v1, got %#v", job.Result)
	}
	summary, _ := job.Result.Payload["summary"].(string)
	if !strings.Contains(strings.ToLower(summary), "build_error") {
		t.Fatalf("expected summary to identify build_error cluster, got %#v", job.Result.Payload)
	}
	timeline, ok := job.Result.Payload["timeline"].([]any)
	if !ok || len(timeline) == 0 {
		t.Fatalf("expected timeline entries, got %#v", job.Result.Payload)
	}
	clusters, ok := job.Result.Payload["clusters"].([]any)
	if !ok || len(clusters) == 0 {
		t.Fatalf("expected clusters, got %#v", job.Result.Payload)
	}
	firstCluster, ok := clusters[0].(map[string]any)
	if !ok || firstCluster["kind"] == nil {
		t.Fatalf("expected cluster kind, got %#v", clusters)
	}
	clusterKinds := strings.Join(anyStrings([]any{
		firstCluster["kind"],
	}), "\n")
	if clusterKinds == "" {
		t.Fatalf("expected non-empty cluster kind, got %#v", clusters)
	}
	if job.Result.Payload["recommended_next_action"] != "Verify the cited files directly, then answer using only the confirmed evidence." &&
		job.Result.Payload["recommended_next_action"] != "Summarize the dominant failure cluster first, then use the cited log evidence to guide the next step." {
		t.Fatalf("unexpected next action: %#v", job.Result.Payload)
	}
}

func TestLocalBackendProposePatchProducesActionablePatchPlan(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	artifactDir := filepath.Join(runRoot, "job_source")
	if err := os.MkdirAll(artifactDir, 0o755); err != nil {
		t.Fatalf("mkdir artifact dir: %v", err)
	}
	artifactPath := filepath.Join(artifactDir, "evidence_pack.json")
	if err := os.WriteFile(artifactPath, []byte(`{"evidence":[{"id":"ev_001","claim":"generated header missing","source_refs":[{"path":"broker/pkg/service/service.go","line_start":12,"line_end":34}]}]}`), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}

	now := time.Now().UTC()
	jobStore := store.NewMemoryJobStore()
	sourceJob := types.Job{
		ID:       "job_source",
		TaskType: "rag_compress",
		State:    types.JobStateSucceeded,
		Request: types.SubmitJobRequest{
			TaskType:     "rag_compress",
			TaskParams:   map[string]any{"allow_artifact_release": true},
			OutputSchema: types.OutputSchemaRef{Name: "rag_evidence_pack_v1"},
		},
		Result: &types.Result{
			SchemaName:    "rag_evidence_pack_v1",
			SchemaVersion: "1.0.0",
			Payload:       map[string]any{"query": "why did the build fail?", "evidence": []any{}},
		},
		Artifacts: []types.Artifact{
			{ArtifactID: "artifact_evidence_pack", ArtifactType: "evidence_pack", Path: artifactPath, Classification: "restricted"},
		},
		CreatedAt:   now.Add(-time.Minute),
		UpdatedAt:   now.Add(-time.Minute),
		SubmittedAt: now.Add(-time.Minute),
	}
	if err := jobStore.CreateJob(context.Background(), sourceJob); err != nil {
		t.Fatalf("create source job: %v", err)
	}

	backend := localbackend.NewBackend(config.Config{
		LocalMode:       "command",
		LocalScriptPath: filepath.Join(repoDir, "deploy", "local", "broker_worker.sh"),
		RunRootPath:     runRoot,
		RepoRootPath:    repoDir,
	})
	svc := service.NewWithAuditAndOptionsAndConfig(
		jobStore,
		backend,
		log.New(io.Discard, "", 0),
		audit.NewNopLogger(),
		runRoot,
		repoDir,
		service.Options{},
		&config.Config{},
	)

	resp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "propose_patch",
		InputRefs: []types.InputRef{
			{Type: "artifact", URI: "artifact://artifact_evidence_pack"},
		},
		TaskParams: map[string]any{
			"problem":             "fix the generated header issue",
			"validation_commands": []any{"go test ./..."},
			"allowed_paths":       []any{"broker/pkg/service"},
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
		OutputSchema: types.OutputSchemaRef{Name: "patch_proposal_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit propose_patch job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded propose_patch result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.Result.SchemaName != "patch_proposal_pack_v1" {
		t.Fatalf("expected patch_proposal_pack_v1, got %#v", job.Result)
	}
	patches, ok := job.Result.Payload["patches"].([]any)
	if !ok || len(patches) == 0 {
		t.Fatalf("expected patch proposals, got %#v", job.Result.Payload)
	}
	firstPatch, ok := patches[0].(map[string]any)
	if !ok {
		t.Fatalf("unexpected patch payload: %#v", patches[0])
	}
	paths, ok := firstPatch["paths"].([]any)
	if !ok || len(paths) == 0 || paths[0] != "broker/pkg/service" {
		t.Fatalf("expected broker/pkg/service patch path, got %#v", firstPatch)
	}
	rationale, _ := firstPatch["rationale"].(string)
	if !strings.Contains(rationale, "generated header missing") {
		t.Fatalf("expected patch rationale to use artifact evidence, got %#v", firstPatch)
	}
	validationSteps, ok := job.Result.Payload["validation_steps"].([]any)
	if !ok || len(validationSteps) == 0 || validationSteps[0] != "go test ./..." {
		t.Fatalf("expected validation steps, got %#v", job.Result.Payload)
	}
	if !hasArtifact(job.Artifacts, "artifact_patch_plan") {
		t.Fatalf("expected artifact_patch_plan, got %#v", job.Artifacts)
	}
	patchPlan := readArtifactJSON(t, job.Artifacts, "artifact_patch_plan")
	if patchPlan["paths"] == nil {
		t.Fatalf("expected patch plan paths, got %#v", patchPlan)
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
			FinalPackTokenBudget:      2048,
			RemoteModelContextBudget:  4000,
		},
		ExecutionProfile: types.ExecutionProfile{
			Backend: "local",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded inspect_repo result, got state=%q result=%#v", job.State, job.Result)
	}
	if !job.DegradedLocalExecution {
		t.Fatalf("expected CPU-only inspect_repo to be evidence-only, got %#v", job)
	}
	if job.RetryRecommended {
		t.Fatalf("expected retry_recommended=false, got %#v", job)
	}
	if job.ExecutionQuality != "evidence_only" {
		t.Fatalf("expected execution_quality evidence_only, got %#v", job.ExecutionQuality)
	}

	if job.RuntimeDiagnostics == nil {
		t.Fatalf("expected runtime diagnostics, got %#v", job)
	}
	if job.RuntimeDiagnostics["retrieval"] != "lexical_degraded" {
		t.Fatalf("expected lexical fallback diagnostics, got %#v", job.RuntimeDiagnostics)
	}
	if _, exists := job.Result.Payload["answer"]; exists {
		t.Fatalf("CPU-only inspection must omit answer: %#v", job.Result.Payload)
	}
}

func TestLocalBackendInspectRepoFindsRetryEntryPoints(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	inputRepo := filepath.Join(t.TempDir(), "repo")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service.go"), strings.Join([]string{
		"package service",
		"",
		"func RetryJobWithRecommendation(jobID string) string {",
		`	return "retry recommendation" + jobID`,
		"}",
		"",
		"func retrySubmitRequest() string {",
		`	return "retry request"`,
		"}",
	}, "\n")+"\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service_runtime.go"), strings.Join([]string{
		"package service",
		"",
		"func retryRecommendationFromResult() string {",
		`	return "retry recommendation"`,
		"}",
	}, "\n")+"\n")

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
			"query": "Find retry logic and related entrypoints",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      16000,
			PerChunkCompressionBudget: 192,
			FinalPackTokenBudget:      2048,
			RemoteModelContextBudget:  4000,
		},
		ExecutionProfile: types.ExecutionProfile{
			Backend: "local",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded inspect_repo result, got state=%q result=%#v", job.State, job.Result)
	}
	joined := inspectionEvidenceCorpus(job.Result.Payload)
	if !strings.Contains(joined, "RetryJobWithRecommendation") {
		t.Fatalf("expected RetryJobWithRecommendation in retry evidence, got %s", joined)
	}
	if !strings.Contains(joined, "retryRecommendationFromResult") {
		t.Fatalf("expected retryRecommendationFromResult in retry evidence, got %s", joined)
	}
}

func TestLocalBackendInspectRepoFindsArtifactAuthorizationReviewPoints(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	inputRepo := filepath.Join(t.TempDir(), "repo")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service_artifacts.go"), strings.Join([]string{
		"package service",
		"",
		"type Principal struct{ Actor string }",
		"",
		"func resolveArtifactRef(principal Principal, artifactID string) string {",
		`	return "artifact " + artifactID + " for " + principal.Actor`,
		"}",
	}, "\n")+"\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "artifact_access.go"), strings.Join([]string{
		"package service",
		"",
		"type Principal struct{ Actor string }",
		"",
		"func artifactJobAccessible(principal Principal, submittedBy string) bool {",
		"	// artifact access check",
		"	return principal.Actor == submittedBy",
		"}",
	}, "\n")+"\n")

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
			"query": "Identify artifact authorization risks and relevant symbols",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      16000,
			PerChunkCompressionBudget: 192,
			FinalPackTokenBudget:      2048,
			RemoteModelContextBudget:  4000,
		},
		ExecutionProfile: types.ExecutionProfile{
			Backend: "local",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded inspect_repo result, got state=%q result=%#v", job.State, job.Result)
	}
	joined := inspectionEvidenceCorpus(job.Result.Payload)
	if !strings.Contains(joined, "resolveArtifactRef") {
		t.Fatalf("expected resolveArtifactRef in artifact evidence, got %s", joined)
	}
	if !strings.Contains(joined, "artifactJobAccessible") {
		t.Fatalf("expected artifactJobAccessible in artifact evidence, got %s", joined)
	}
	if !strings.Contains(joined, "service_artifacts.go") {
		t.Fatalf("expected service_artifacts.go in artifact evidence, got %s", joined)
	}
}

func TestLocalBackendInspectRepoFindsSimplicityAndDryIssues(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	runRoot := t.TempDir()
	repoDir := repoRoot(t)
	inputRepo := filepath.Join(t.TempDir(), "repo")
	writeTestFile(t, filepath.Join(inputRepo, "workers", "rag-compression", "main.py"), strings.Repeat("def load_json():\n    pass\n", 950)+
		"def write_json():\n    pass\n"+
		"def emit_heartbeat():\n    pass\n"+
		"def resolve_file_uri():\n    pass\n")
	writeTestFile(t, filepath.Join(inputRepo, "workers", "document-summary", "main.py"), "def load_json():\n    pass\ndef write_json():\n    pass\ndef emit_heartbeat():\n    pass\ndef resolve_file_uri():\n    pass\n")
	writeTestFile(t, filepath.Join(inputRepo, "workers", "log-analysis", "main.py"), "def load_json():\n    pass\ndef write_json():\n    pass\ndef emit_heartbeat():\n    pass\ndef resolve_file_uri():\n    pass\n")
	writeTestFile(t, filepath.Join(inputRepo, "workers", "repo-summary", "main.py"), "def load_json():\n    pass\ndef write_json():\n    pass\ndef emit_heartbeat():\n    pass\ndef resolve_file_uri():\n    pass\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service_execution.go"), "package service\n")
	writeTestFile(t, filepath.Join(inputRepo, "tests", "e2e", "llamacpp_runtime_test.go"), "func TestLocalBackendX() { SubmitJob(); waitForJob(); TaskType: }\n")
	writeTestFile(t, filepath.Join(inputRepo, "tests", "e2e", "command_mode_test.go"), "func TestCommandModeX() { SubmitJob(); waitForJob(); TaskType: }\n")

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
			"query": "Audit this repository for simplicity and DRY-ness. Identify duplicated logic and repeated orchestration.",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      16000,
			PerChunkCompressionBudget: 192,
			FinalPackTokenBudget:      2048,
			RemoteModelContextBudget:  4000,
		},
		ExecutionProfile: types.ExecutionProfile{
			Backend: "local",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit inspect_repo job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, resp.JobID, 15*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded inspect_repo result, got state=%q result=%#v", job.State, job.Result)
	}
	joined := inspectionEvidenceCorpus(job.Result.Payload)
	if !strings.Contains(joined, "workers/rag-compression/main.py") {
		t.Fatalf("expected worker module evidence, got %s", joined)
	}
	if !strings.Contains(joined, "load_json") {
		t.Fatalf("expected repeated helper evidence, got %s", joined)
	}
	if _, exists := job.Result.Payload["answer"]; exists {
		t.Fatalf("CPU-only DRY audit must not synthesize an answer: %#v", job.Result.Payload)
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
