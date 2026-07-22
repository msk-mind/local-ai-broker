package e2e

import (
	"context"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends/slurm"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/service"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestCommandModeDocumentSummarySmoke(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	baseDir := t.TempDir()
	runRoot := filepath.Join(baseDir, "runs")

	repoDir := repoRoot(t)
	setupFakeSlurmEnv(t, repoDir, baseDir)

	inputPath := filepath.Join(baseDir, "source.txt")
	writeTestFile(t, inputPath, "Smoke test document.\n- alpha\n- beta\n")

	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmStatusCmd:  "sacct",
		SlurmCancelCmd:  "scancel",
		SlurmScriptPath: filepath.Join(repoDir, "deploy", "slurm", "broker_worker.slurm"),
	}

	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(cfg),
		log.New(io.Discard, "", 0),
		runRoot,
		repoDir,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "document_summary",
		InputRefs: []types.InputRef{
			{Type: "file", URI: "file://" + inputPath, ContentHash: "sha256:test"},
		},
		OutputSchema: types.OutputSchemaRef{Name: "document_summary_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, submitResp.JobID, 5*time.Second)

	if job.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", job.State)
	}
	if job.Result == nil {
		t.Fatal("expected ingested result")
	}
	if !strings.Contains(job.Result.SchemaName, "document_summary_v1") {
		t.Fatalf("unexpected schema name %q", job.Result.SchemaName)
	}
	summary, _ := job.Result.Payload["summary"].(string)
	if !strings.Contains(summary, "Opening sentence: Smoke test document.") {
		t.Fatalf("expected opening sentence in summary, got %#v", job.Result.Payload)
	}
	keyPoints, ok := job.Result.Payload["key_points"].([]any)
	if !ok || len(keyPoints) < 2 {
		t.Fatalf("expected key points for bullet lines, got %#v", job.Result.Payload)
	}
	if keyPoints[0] != "- alpha" || keyPoints[1] != "- beta" {
		t.Fatalf("unexpected key points: %#v", keyPoints)
	}
}

func TestCommandModeInspectRepoProducesUsefulCoverageFindings(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	baseDir := t.TempDir()
	runRoot := filepath.Join(baseDir, "runs")

	repoDir := repoRoot(t)
	setupFakeSlurmEnv(t, repoDir, baseDir)

	inputRepo := filepath.Join(baseDir, "repo")
	writeTestFile(t, filepath.Join(inputRepo, "go.mod"), "module example.com/test\n")
	writeTestFile(t, filepath.Join(inputRepo, "workers", "rag-compression", "main.py"), strings.Join([]string{
		"def build_result(): pass",
		"def build_evidence(): pass",
		"def build_repo_inspection_payload(): pass",
		"def execute_retrieval_plan(): pass",
		"def build_artifacts(): pass",
		"def rerank_candidates(): pass",
		"def select_chunks(): pass",
		"def summarize_chunk(): pass",
		"def classify_chunk_kind(): pass",
		"def detect_symbol(): pass",
		"def build_validation_report(): pass",
		"def enforce_final_pack_budget(): pass",
	}, "\n")+"\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service.go"), "package service\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service_artifacts.go"), "package service\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service_job_refresh.go"), "package service\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service_access.go"), "package service\n")
	writeTestFile(t, filepath.Join(inputRepo, "broker", "pkg", "service", "service_test.go"), "package service\nfunc TestRoot(t *testing.T) {}\n")
	writeTestFile(t, filepath.Join(inputRepo, "tests", "unit", "test_workers.py"), strings.Join([]string{
		`rag_compression = load_module("rag_compression_worker", "workers/rag-compression/main.py")`,
		`rag_compression.query_terms_for("x", "inspect_repo", {})`,
		`rag_compression.repo_structure_executor({}, {}, "inspect_repo", [], {})`,
	}, "\n")+"\n")

	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmStatusCmd:  "sacct",
		SlurmCancelCmd:  "scancel",
		SlurmScriptPath: filepath.Join(repoDir, "deploy", "slurm", "broker_worker.slurm"),
	}

	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(cfg),
		log.New(io.Discard, "", 0),
		runRoot,
		repoDir,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + inputRepo, Classification: "internal"},
		},
		TaskParams: map[string]any{
			"query": "Audit this repository for test coverage gaps",
		},
		Constraints: types.Constraints{
			RetrievedChunkBudget:      16000,
			PerChunkCompressionBudget: 192,
			FinalPackTokenBudget:      2048,
			RemoteModelContextBudget:  4000,
		},
		ExecutionProfile: types.ExecutionProfile{
			Backend: "slurm",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, submitResp.JobID, 10*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded inspect_repo result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.Result.SchemaName != "repo_inspection_v2" {
		t.Fatalf("unexpected schema name %q", job.Result.SchemaName)
	}
	quality, _ := job.Result.Payload["quality"].(map[string]any)
	if quality["result"] != "evidence_only" || quality["answer_ready"] != false {
		t.Fatalf("CPU command-mode inspection must be evidence-only, got %#v", job.Result.Payload)
	}
	joined := inspectionEvidenceCorpus(job.Result.Payload)
	if !strings.Contains(joined, "workers/rag-compression/main.py") || !strings.Contains(joined, "service_test.go") {
		t.Fatalf("expected coverage-related lexical evidence, got %s", joined)
	}
}

func TestCommandModeSummarizeLogsProducesUsefulClusters(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	baseDir := t.TempDir()
	runRoot := filepath.Join(baseDir, "runs")

	repoDir := repoRoot(t)
	setupFakeSlurmEnv(t, repoDir, baseDir)

	inputLog := filepath.Join(baseDir, "service.log")
	writeTestFile(t, inputLog, strings.Join([]string{
		"2026-07-09T12:00:00Z build started",
		"2026-07-09T12:00:01Z fatal error: generated/config.h missing",
		"2026-07-09T12:00:02Z undefined reference to demo_symbol",
		"2026-07-09T12:00:03Z FAILED demo_test",
	}, "\n")+"\n")

	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmStatusCmd:  "sacct",
		SlurmCancelCmd:  "scancel",
		SlurmScriptPath: filepath.Join(repoDir, "deploy", "slurm", "broker_worker.slurm"),
	}

	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(cfg),
		log.New(io.Discard, "", 0),
		runRoot,
		repoDir,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
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
			Backend: "slurm",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "log_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, submitResp.JobID, 10*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded summarize_logs result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.Result.SchemaName != "log_evidence_pack_v1" {
		t.Fatalf("unexpected schema name %q", job.Result.SchemaName)
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
}

func TestCommandModeDebugWithLocalContextProducesActionableHypotheses(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	baseDir := t.TempDir()
	runRoot := filepath.Join(baseDir, "runs")

	repoDir := repoRoot(t)
	setupFakeSlurmEnv(t, repoDir, baseDir)

	inputRepo := filepath.Join(baseDir, "repo")
	inputLog := filepath.Join(baseDir, "debug.log")
	writeTestFile(t, filepath.Join(inputRepo, "src", "service.py"), "def run_service():\n    raise RuntimeError(\"demo failure\")\n")
	writeTestFile(t, inputLog, strings.Join([]string{
		"2026-07-09T12:00:00Z FAILED demo_test",
		"Traceback (most recent call last):",
		"RuntimeError: demo failure",
	}, "\n")+"\n")

	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmStatusCmd:  "sacct",
		SlurmCancelCmd:  "scancel",
		SlurmScriptPath: filepath.Join(repoDir, "deploy", "slurm", "broker_worker.slurm"),
	}

	svc := service.New(
		store.NewMemoryJobStore(),
		slurm.NewBackend(cfg),
		log.New(io.Discard, "", 0),
		runRoot,
		repoDir,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
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
			Backend: "slurm",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "debug_evidence_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, submitResp.JobID, 10*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded debug result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.Result.SchemaName != "debug_evidence_pack_v1" {
		t.Fatalf("unexpected schema name %q", job.Result.SchemaName)
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
}

func TestCommandModeProposePatchProducesActionablePatchPlan(t *testing.T) {
	if _, err := os.Stat("/usr/bin/bash"); err != nil {
		t.Skip("bash not available")
	}
	if _, err := os.Stat("/usr/bin/python3"); err != nil {
		t.Skip("python3 not available")
	}

	baseDir := t.TempDir()
	runRoot := filepath.Join(baseDir, "runs")

	repoDir := repoRoot(t)
	setupFakeSlurmEnv(t, repoDir, baseDir)

	artifactDir := filepath.Join(baseDir, "job_source")
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

	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmStatusCmd:  "sacct",
		SlurmCancelCmd:  "scancel",
		SlurmScriptPath: filepath.Join(repoDir, "deploy", "slurm", "broker_worker.slurm"),
	}

	svc := service.New(
		jobStore,
		slurm.NewBackend(cfg),
		log.New(io.Discard, "", 0),
		runRoot,
		repoDir,
	)

	submitResp, err := svc.SubmitJob(context.Background(), types.SubmitJobRequest{
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
			Backend: "slurm",
			Tier:    "cpu-rag-indexing",
		},
		OutputSchema: types.OutputSchemaRef{Name: "patch_proposal_pack_v1"},
	})
	if err != nil {
		t.Fatalf("submit job: %v", err)
	}

	job := waitForJob(t, svc, runRoot, submitResp.JobID, 10*time.Second)
	if job.State != types.JobStateSucceeded || job.Result == nil {
		t.Fatalf("expected succeeded propose_patch result, got state=%q result=%#v", job.State, job.Result)
	}
	if job.Result.SchemaName != "patch_proposal_pack_v1" {
		t.Fatalf("unexpected schema name %q", job.Result.SchemaName)
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
}
