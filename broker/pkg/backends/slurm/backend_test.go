package slurm

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/gpuservice"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type fakeRunner struct {
	outputs  map[string][]byte
	errors   map[string]error
	lastArgs []string
}

func (f fakeRunner) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	f.lastArgs = append([]string(nil), args...)
	if len(f.outputs) == 0 && len(f.errors) == 0 {
		return nil, nil
	}
	key := name
	if len(args) > 0 {
		if _, ok := f.outputs[args[0]]; ok {
			key = args[0]
		} else if _, ok := f.errors[args[0]]; ok {
			key = args[0]
		}
	}
	return f.outputs[key], f.errors[key]
}

func TestParseSlurmState(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want types.JobState
	}{
		{name: "pending", in: "PENDING\n", want: types.JobStateQueued},
		{name: "running", in: "RUNNING\n", want: types.JobStateRunning},
		{name: "completed", in: "COMPLETED\n", want: types.JobStateSucceeded},
		{name: "cancelled", in: "CANCELLED by 123\n", want: types.JobStateCancelled},
		{name: "timeout", in: "TIMEOUT\n", want: types.JobStateTimedOut},
		{name: "preempted", in: "PREEMPTED\n", want: types.JobStatePreempted},
		{name: "failed", in: "FAILED\n", want: types.JobStateFailed},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseSlurmState([]byte(tt.in)); got != tt.want {
				t.Fatalf("expected %q, got %q", tt.want, got)
			}
		})
	}
}

func TestFormatSlurmTimeLimit(t *testing.T) {
	for _, test := range []struct {
		seconds int
		want    string
	}{
		{seconds: 1, want: "00:00:01"},
		{seconds: 3661, want: "01:01:01"},
		{seconds: 86400, want: "24:00:00"},
	} {
		if got := formatSlurmTimeLimit(test.seconds); got != test.want {
			t.Fatalf("formatSlurmTimeLimit(%d) = %q, want %q", test.seconds, got, test.want)
		}
	}
}

func TestSubmitRunCommandMode(t *testing.T) {
	t.Setenv("PATH", "/usr/bin:/bin")
	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmScriptPath: "deploy/slurm/broker_worker.slurm",
	}
	backend := NewBackendWithRunner(cfg, fakeRunner{
		outputs: map[string][]byte{"--parsable": []byte("12345\n")},
	})

	resp, err := backend.SubmitRun(context.Background(), types.Job{TaskType: "log_analysis"})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if resp.BackendRunID != "12345" {
		t.Fatalf("expected backend run id 12345, got %q", resp.BackendRunID)
	}
}

func TestSubmitRunExportsExplicitEnvWithoutALL(t *testing.T) {
	t.Setenv("PATH", "/usr/bin:/bin")
	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmScriptPath: "deploy/slurm/broker_worker.slurm",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		ID:       "job_123",
		TaskType: "log_analysis",
		Request: types.SubmitJobRequest{
			TaskParams:   map[string]any{"_broker_run_root": "/runs", "_broker_repo_root": "/repo"},
			OutputSchema: types.OutputSchemaRef{Name: "log_summary_v1"},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}

	exportArg := argValueAfter(runner.args, "--export")
	if findPart(exportArg, "ALL") == "ALL" {
		t.Fatalf("expected explicit export without ALL, got %q", exportArg)
	}
	for _, want := range []string{
		"PATH=/usr/bin:/bin",
		"BROKER_JOB_ID=job_123",
		"BROKER_TASK_TYPE=log_analysis",
		"BROKER_REPO_ROOT=/repo",
		"BROKER_OUTPUT_DIR=/runs/job_123",
		"BROKER_OUTPUT_SCHEMA=log_summary_v1",
	} {
		if !strings.Contains(exportArg, want) {
			t.Fatalf("expected export to contain %q, got %q", want, exportArg)
		}
	}
}

func TestSubmitRunCommandModeError(t *testing.T) {
	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmScriptPath: "deploy/slurm/broker_worker.slurm",
	}
	backend := NewBackendWithRunner(cfg, fakeRunner{
		outputs: map[string][]byte{"--parsable": []byte("boom")},
		errors:  map[string]error{"--parsable": errors.New("exit 1")},
	})

	if _, err := backend.SubmitRun(context.Background(), types.Job{TaskType: "log_analysis"}); err == nil {
		t.Fatal("expected error")
	}
}

func TestGetRunCommandMode(t *testing.T) {
	cfg := config.Config{
		SlurmMode:      "command",
		SlurmStatusCmd: "sacct",
	}
	backend := NewBackendWithRunner(cfg, fakeRunner{
		outputs: map[string][]byte{"--jobs": []byte("RUNNING|0:0\n")},
	})

	status, err := backend.GetRun(context.Background(), "12345")
	if err != nil {
		t.Fatalf("get run: %v", err)
	}
	if status.State != types.JobStateRunning {
		t.Fatalf("expected running, got %q", status.State)
	}
	if status.RawState != "RUNNING" {
		t.Fatalf("expected raw state RUNNING, got %q", status.RawState)
	}
}

func TestGetRunCommandModeArrayChildMatchesExactTask(t *testing.T) {
	cfg := config.Config{
		SlurmMode:      "command",
		SlurmStatusCmd: "sacct",
	}
	backend := NewBackendWithRunner(cfg, fakeRunner{
		outputs: map[string][]byte{
			"--jobs": []byte("98765|COMPLETED|0:0\n98765_0|FAILED|1:0\n98765_1|RUNNING|0:0\n"),
		},
	})

	status, err := backend.GetRun(context.Background(), "98765_1")
	if err != nil {
		t.Fatalf("get run: %v", err)
	}
	if status.State != types.JobStateRunning || status.RawState != "RUNNING" || status.ExitCode != "0:0" {
		t.Fatalf("unexpected array child status: %#v", status)
	}
}

func TestSubmitRunAddsDependencyArgs(t *testing.T) {
	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmScriptPath: "deploy/slurm/broker_worker.slurm",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		TaskType: "repo_summary",
		Request: types.SubmitJobRequest{
			TaskParams: map[string]any{
				"_dependency_backend_run_ids": []string{"111", "222"},
			},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if len(runner.args) < 3 || runner.args[0] != "--parsable" || runner.args[1] != "--job-name" || runner.args[2] != "broker-repo_summary" {
		t.Fatalf("unexpected submit prefix: %#v", runner.args)
	}
	if argValueAfter(runner.args, "--dependency") != "afterany:111:222" {
		t.Fatalf("expected dependency arg, got %#v", runner.args)
	}
}

func TestSubmitRunAddsQOSArg(t *testing.T) {
	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmScriptPath: "deploy/slurm/broker_worker.slurm",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		TaskType: "rag_compress",
		Request: types.SubmitJobRequest{
			ExecutionProfile: types.ExecutionProfile{
				QOS: "scavenger",
			},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if argValueAfter(runner.args, "--qos") != "scavenger" {
		t.Fatalf("expected --qos scavenger, got %#v", runner.args)
	}
}

func TestSubmitRunAddsPartitionFromTier(t *testing.T) {
	cfg := config.Config{
		SlurmMode:         "command",
		SlurmSubmitCmd:    "sbatch",
		SlurmScriptPath:   "deploy/slurm/broker_worker.slurm",
		SlurmPartitionGPU: "hpc",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		TaskType: "rag_compress",
		Request: types.SubmitJobRequest{
			ExecutionProfile: types.ExecutionProfile{
				Tier: "a100-reasoning",
			},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if argValueAfter(runner.args, "--partition") != "hpc" {
		t.Fatalf("expected --partition hpc, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--gres") != "gpu:1" {
		t.Fatalf("expected generic GPU request, got %#v", runner.args)
	}
}

func TestSubmitRunAddsTypedGPURequestFromTierDefaults(t *testing.T) {
	cfg := config.Config{
		SlurmMode:           "command",
		SlurmSubmitCmd:      "sbatch",
		SlurmScriptPath:     "deploy/slurm/broker_worker.slurm",
		SlurmPartitionGPU:   "hpc",
		SlurmGPURequestMode: "gres",
		SlurmGPUTypeP40:     "p40",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		TaskType: "rag_compress",
		Request: types.SubmitJobRequest{
			ExecutionProfile: types.ExecutionProfile{
				Tier: "p40-rag-compression",
			},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if argValueAfter(runner.args, "--partition") != "hpc" {
		t.Fatalf("expected --partition hpc, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--gres") != "gpu:p40:1" {
		t.Fatalf("expected typed --gres request, got %#v", runner.args)
	}
}

func TestSubmitRunPrefersExplicitAcceleratorOverride(t *testing.T) {
	cfg := config.Config{
		SlurmMode:           "command",
		SlurmSubmitCmd:      "sbatch",
		SlurmScriptPath:     "deploy/slurm/broker_worker.slurm",
		SlurmPartitionGPU:   "hpc",
		SlurmGPURequestMode: "gres",
		SlurmGPUTypeP40:     "p40",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		TaskType: "rag_compress",
		Request: types.SubmitJobRequest{
			ExecutionProfile: types.ExecutionProfile{
				Tier:        "p40-rag-compression",
				Accelerator: "l40s",
			},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if argValueAfter(runner.args, "--gres") != "gpu:l40s:1" {
		t.Fatalf("expected explicit accelerator override, got %#v", runner.args)
	}
}

func TestSubmitRunSupportsGpusFlagMode(t *testing.T) {
	cfg := config.Config{
		SlurmMode:           "command",
		SlurmSubmitCmd:      "sbatch",
		SlurmScriptPath:     "deploy/slurm/broker_worker.slurm",
		SlurmPartitionGPU:   "hpc",
		SlurmGPURequestMode: "gpus",
		SlurmGPUTypeA100:    "a100",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		TaskType: "patch_generation",
		Request: types.SubmitJobRequest{
			ExecutionProfile: types.ExecutionProfile{
				Tier: "a100-reasoning",
			},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if argValueAfter(runner.args, "--gpus") != "a100:1" {
		t.Fatalf("expected --gpus a100:1, got %#v", runner.args)
	}
}

func TestSubmitRunAddsNodeListAndConstraintFromTierDefaults(t *testing.T) {
	cfg := config.Config{
		SlurmMode:          "command",
		SlurmSubmitCmd:     "sbatch",
		SlurmScriptPath:    "deploy/slurm/broker_worker.slurm",
		SlurmNodeListP40:   "pllimsksparky[1-4]",
		SlurmConstraintP40: "p40",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		TaskType: "rag_compress",
		Request: types.SubmitJobRequest{
			ExecutionProfile: types.ExecutionProfile{
				Tier: "p40-rag-compression",
			},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if argValueAfter(runner.args, "--nodelist") != "pllimsksparky[1-4]" {
		t.Fatalf("expected --nodelist pllimsksparky[1-4], got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--nodes") != "1" {
		t.Fatalf("expected --nodes 1, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--ntasks") != "1" {
		t.Fatalf("expected --ntasks 1, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--constraint") != "p40" {
		t.Fatalf("expected --constraint p40, got %#v", runner.args)
	}
}

func TestSubmitRunPrefersExplicitNodeListAndConstraintOverrides(t *testing.T) {
	cfg := config.Config{
		SlurmMode:          "command",
		SlurmSubmitCmd:     "sbatch",
		SlurmScriptPath:    "deploy/slurm/broker_worker.slurm",
		SlurmNodeListP40:   "pllimsksparky[1-4]",
		SlurmConstraintP40: "p40",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	_, err := backend.SubmitRun(context.Background(), types.Job{
		TaskType: "rag_compress",
		Request: types.SubmitJobRequest{
			ExecutionProfile: types.ExecutionProfile{
				Tier:       "p40-rag-compression",
				NodeList:   "pllimsksparky2",
				Constraint: "gpu24g",
			},
		},
	})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if argValueAfter(runner.args, "--nodelist") != "pllimsksparky2" {
		t.Fatalf("expected explicit --nodelist override, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--nodes") != "1" {
		t.Fatalf("expected --nodes 1, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--ntasks") != "1" {
		t.Fatalf("expected --ntasks 1, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--constraint") != "gpu24g" {
		t.Fatalf("expected explicit --constraint override, got %#v", runner.args)
	}
}

func TestResolveExecutionProfileKeepsP40WhenAvailable(t *testing.T) {
	cfg := config.Config{
		SlurmMode:                   "command",
		SlurmInfoCmd:                "sinfo",
		SlurmEnableDynamicPlacement: true,
		SlurmPartitionGPU:           "hpc",
		SlurmGPUTypeP40:             "p40",
		SlurmGPUTypeA100:            "a100",
		SlurmNodeListP40:            "pllimsksparky[1-4]",
	}
	backend := NewBackendWithRunner(cfg, fakeRunner{
		outputs: map[string][]byte{
			"sinfo": []byte("hpc|pllimsksparky2|(null)|gpu:p40:4|idle\nhpc|a100box|(null)|gpu:a100:4|idle\n"),
		},
	})

	profile, err := backend.ResolveExecutionProfile(context.Background(), types.SubmitJobRequest{
		ExecutionProfile: types.ExecutionProfile{Tier: "p40-rag-compression"},
	})
	if err != nil {
		t.Fatalf("resolve execution profile: %v", err)
	}
	if profile.Tier != "p40-rag-compression" {
		t.Fatalf("expected p40 tier to remain, got %#v", profile)
	}
}

func TestResolveExecutionProfilePromotesToA100WhenP40Unavailable(t *testing.T) {
	cfg := config.Config{
		SlurmMode:                   "command",
		SlurmInfoCmd:                "sinfo",
		SlurmEnableDynamicPlacement: true,
		SlurmPartitionGPU:           "hpc",
		SlurmGPUTypeP40:             "p40",
		SlurmGPUTypeA100:            "a100",
		SlurmNodeListP40:            "pllimsksparky[1-4]",
	}
	backend := NewBackendWithRunner(cfg, fakeRunner{
		outputs: map[string][]byte{
			"sinfo": []byte("hpc|pllimsksparky2|(null)|gpu:p40:4|alloc\nhpc|a100box|(null)|gpu:a100:4|idle\n"),
		},
	})

	profile, err := backend.ResolveExecutionProfile(context.Background(), types.SubmitJobRequest{
		ExecutionProfile: types.ExecutionProfile{Tier: "p40-rag-compression"},
	})
	if err != nil {
		t.Fatalf("resolve execution profile: %v", err)
	}
	if profile.Tier != "a100-reasoning" {
		t.Fatalf("expected a100 promotion, got %#v", profile)
	}
}

func TestResolveExecutionProfileSkipsDynamicSelectionForExplicitOverrides(t *testing.T) {
	cfg := config.Config{
		SlurmMode:                   "command",
		SlurmInfoCmd:                "sinfo",
		SlurmEnableDynamicPlacement: true,
		SlurmPartitionGPU:           "hpc",
		SlurmGPUTypeP40:             "p40",
		SlurmGPUTypeA100:            "a100",
	}
	backend := NewBackendWithRunner(cfg, fakeRunner{
		outputs: map[string][]byte{
			"sinfo": []byte("hpc|a100box|(null)|gpu:a100:4|idle\n"),
		},
	})

	profile, err := backend.ResolveExecutionProfile(context.Background(), types.SubmitJobRequest{
		ExecutionProfile: types.ExecutionProfile{
			Tier:       "p40-rag-compression",
			Constraint: "gpu24g",
		},
	})
	if err != nil {
		t.Fatalf("resolve execution profile: %v", err)
	}
	if profile.Tier != "p40-rag-compression" {
		t.Fatalf("expected explicit override to keep p40 tier, got %#v", profile)
	}
}

func TestResolveExecutionProfileUsesQueuePressureToRankEligibleTiers(t *testing.T) {
	cfg := config.Config{
		SlurmMode:                   "command",
		SlurmInfoCmd:                "sinfo",
		SlurmEnableDynamicPlacement: true,
		SlurmPartitionGPU:           "hpc",
		SlurmGPUTypeP40:             "p40",
		SlurmGPUTypeA100:            "a100",
		SlurmNodeListP40:            "pllimsksparky[1-4]",
	}
	backend := NewBackendWithRunner(cfg, fakeRunner{
		outputs: map[string][]byte{
			"sinfo":  []byte("hpc|pllimsksparky2|(null)|gpu:p40:4|idle\nhpc|a100box|(null)|gpu:a100:4|idle\n"),
			"squeue": []byte("hpc|PENDING|n/a\nhpc|PENDING|n/a\nhpc|PENDING|n/a\nhpc|RUNNING|pllimsksparky2\n"),
		},
	})

	profile, err := backend.ResolveExecutionProfile(context.Background(), types.SubmitJobRequest{
		ExecutionProfile: types.ExecutionProfile{Tier: "p40-rag-compression"},
	})
	if err != nil {
		t.Fatalf("resolve execution profile: %v", err)
	}
	if profile.Tier != "a100-reasoning" {
		t.Fatalf("expected queue-aware promotion to a100, got %#v", profile)
	}
}

func TestSubmitRunBatchCommandMode(t *testing.T) {
	t.Setenv("PATH", "/usr/bin:/bin")
	runRoot := t.TempDir()
	cfg := config.Config{
		SlurmMode:       "command",
		SlurmSubmitCmd:  "sbatch",
		SlurmScriptPath: "deploy/slurm/broker_worker.slurm",
	}
	runner := &recordingRunner{
		output: []byte("98765\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	jobs := []types.Job{
		{
			ID:        "job_a",
			TaskType:  "repo_summary",
			RootJobID: "root_batch_1",
			Request: types.SubmitJobRequest{
				TaskParams:   map[string]any{"_broker_run_root": runRoot, "_broker_repo_root": "/repo"},
				OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
			},
		},
		{
			ID:        "job_b",
			TaskType:  "repo_summary",
			RootJobID: "root_batch_1",
			Request: types.SubmitJobRequest{
				TaskParams:   map[string]any{"_broker_run_root": runRoot, "_broker_repo_root": "/repo"},
				OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
			},
		},
	}

	resp, err := backend.SubmitRunBatch(context.Background(), jobs)
	if err != nil {
		t.Fatalf("submit run batch: %v", err)
	}
	if len(resp) != 2 || resp[0].BackendRunID != "98765_0" || resp[1].BackendRunID != "98765_1" {
		t.Fatalf("unexpected batch responses: %#v", resp)
	}
	if !containsArg(runner.args, "--array") || !containsArg(runner.args, "0-1") {
		t.Fatalf("expected sbatch array args, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--nodes") != "1" {
		t.Fatalf("expected --nodes 1 for arrays, got %#v", runner.args)
	}
	if argValueAfter(runner.args, "--ntasks") != "1" {
		t.Fatalf("expected --ntasks 1 for arrays, got %#v", runner.args)
	}
	exportArg := argValueAfter(runner.args, "--export")
	if findPart(exportArg, "ALL") == "ALL" {
		t.Fatalf("expected explicit export without ALL, got %q", exportArg)
	}
	if !strings.Contains(exportArg, "BROKER_ARRAY_MANIFEST=") {
		t.Fatalf("expected manifest export, got %q", exportArg)
	}
	if !strings.Contains(exportArg, "PATH=/usr/bin:/bin") {
		t.Fatalf("expected PATH passthrough export, got %q", exportArg)
	}
	manifestPath := strings.TrimPrefix(findPart(exportArg, "BROKER_ARRAY_MANIFEST="), "BROKER_ARRAY_MANIFEST=")
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if !strings.Contains(string(manifestBytes), "\"broker_job_id\": \"job_a\"") {
		t.Fatalf("unexpected manifest: %s", string(manifestBytes))
	}
	if filepath.Dir(manifestPath) != filepath.Join(runRoot, "_slurm_arrays") {
		t.Fatalf("unexpected manifest dir: %s", manifestPath)
	}
}

func TestSubmitGPUServiceRequestsFourTypedV100GPUs(t *testing.T) {
	root := t.TempDir()
	t.Setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/opt/nvidia/lib64")
	t.Setenv("CUDA_HOME", "/usr/local/cuda")
	cfg := config.Config{
		SlurmMode:            "command",
		SlurmSubmitCmd:       "sbatch",
		SlurmGPURequestMode:  "gres",
		GPUServiceScriptPath: "deploy/slurm/gpu_service.slurm",
	}
	runner := &recordingRunner{output: []byte("77777\n")}
	backend := NewBackendWithRunner(cfg, runner)
	request := gpuservice.LaunchRequest{
		ServiceID:                "gpu-v100-reasoning-test",
		Tier:                     gpuservice.TierV100Reasoning,
		Role:                     gpuservice.RoleSynthesis,
		RegistryPath:             filepath.Join(root, "registry.json"),
		RegistrationToken:        "registration-secret",
		HeartbeatIntervalSeconds: 15,
		LeaseDurationSeconds:     4 * 60 * 60,
		Capabilities:             []string{gpuservice.OperationChatCompletions},
		Deployment: gpuservice.DeploymentProfile{
			Name:               "v100-strong",
			Model:              "/models/v100-strong",
			Quantization:       "bf16",
			ContextLimitTokens: 65536,
			Runtime:            "vllm",
			RuntimeArgs:        []string{"--tensor-parallel-size", "4"},
		},
		Placement: gpuservice.Placement{
			Partition:  "v100-partition",
			GPU:        gpuservice.GPU{Type: "v100", Count: 4},
			NodeList:   "v100node01",
			Constraint: "v100",
		},
	}
	jobID, err := backend.SubmitGPUService(context.Background(), request)
	if err != nil {
		t.Fatalf("submit GPU service: %v", err)
	}
	if jobID != "77777" {
		t.Fatalf("unexpected job id %q", jobID)
	}
	if got := argValueAfter(runner.args, "--gres"); got != "gpu:v100:4" {
		t.Fatalf("V100 placement requested %q, args=%#v", got, runner.args)
	}
	if got := argValueAfter(runner.args, "--partition"); got != "v100-partition" {
		t.Fatalf("unexpected partition %q", got)
	}
	if argValueAfter(runner.args, "--nodes") != "1" || argValueAfter(runner.args, "--ntasks") != "1" {
		t.Fatalf("tensor-parallel service must stay on one node: %#v", runner.args)
	}
	if runner.args[len(runner.args)-1] != "deploy/slurm/gpu_service.slurm" {
		t.Fatalf("inspection worker script was used: %#v", runner.args)
	}
	export := argValueAfter(runner.args, "--export")
	if !strings.Contains(export, "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/nvidia/lib64") {
		t.Fatalf("expected CUDA library export, got %q", export)
	}
	if !strings.Contains(export, "CUDA_HOME=/usr/local/cuda") {
		t.Fatalf("expected CUDA_HOME export, got %q", export)
	}
	specPath := strings.TrimPrefix(findPart(export, "BROKER_GPU_SERVICE_SPEC_PATH="), "BROKER_GPU_SERVICE_SPEC_PATH=")
	content, err := os.ReadFile(specPath)
	if err != nil {
		t.Fatalf("read launch spec: %v", err)
	}
	if !strings.Contains(string(content), `"registration_token": "registration-secret"`) ||
		!strings.Contains(string(content), `"count": 4`) {
		t.Fatalf("unexpected launch spec: %s", content)
	}
	info, err := os.Stat(specPath)
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("launch spec permissions info=%v err=%v", info, err)
	}
}

func TestGPUServiceStatusPreservesTerminalFailureCategory(t *testing.T) {
	for _, test := range []struct {
		name     string
		rawState string
		want     gpuservice.FailureCategory
	}{
		{name: "out of memory", rawState: "OUT_OF_MEMORY", want: gpuservice.FailureOOM},
		{name: "timeout", rawState: "TIMEOUT", want: gpuservice.FailureTimeout},
	} {
		t.Run(test.name, func(t *testing.T) {
			backend := NewBackendWithRunner(config.Config{SlurmMode: "command", SlurmStatusCmd: "sacct"}, fakeRunner{
				outputs: map[string][]byte{"--jobs": []byte("999|" + test.rawState + "|1:0\n")},
			})
			status, err := backend.GPUServiceStatus(context.Background(), "999")
			if err != nil {
				t.Fatalf("GPU service status: %v", err)
			}
			if status.State != gpuservice.JobStateFailed || status.RawState != test.rawState || status.FailureCategory != test.want {
				t.Fatalf("unexpected structured status: %#v", status)
			}
		})
	}
}

func TestSubmitRunBatchExportsCUDAEnvironment(t *testing.T) {
	runRoot := t.TempDir()
	t.Setenv("PATH", "/usr/bin:/bin")
	t.Setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
	t.Setenv("CUDA_PATH", "/usr/local/cuda")

	cfg := config.Config{
		SlurmMode:      "command",
		SlurmSubmitCmd: "sbatch",
	}
	runner := &recordingRunner{
		output: []byte("12345\n"),
	}
	backend := NewBackendWithRunner(cfg, runner)

	jobs := []types.Job{
		{
			ID:        "job_cuda",
			TaskType:  "repo_summary",
			RootJobID: "root_cuda_1",
			Request: types.SubmitJobRequest{
				TaskParams:   map[string]any{"_broker_run_root": runRoot, "_broker_repo_root": "/repo"},
				OutputSchema: types.OutputSchemaRef{Name: "repo_summary_v1"},
			},
		},
	}

	if _, err := backend.SubmitRunBatch(context.Background(), jobs); err != nil {
		t.Fatalf("submit run batch: %v", err)
	}

	exportArg := argValueAfter(runner.args, "--export")
	if !strings.Contains(exportArg, "LD_LIBRARY_PATH=/usr/local/cuda/lib64") {
		t.Fatalf("expected LD_LIBRARY_PATH passthrough export, got %q", exportArg)
	}
	if !strings.Contains(exportArg, "CUDA_PATH=/usr/local/cuda") {
		t.Fatalf("expected CUDA_PATH passthrough export, got %q", exportArg)
	}
}

func TestCancelRunUsesExactArrayChildID(t *testing.T) {
	cfg := config.Config{
		SlurmMode:      "command",
		SlurmCancelCmd: "scancel",
	}
	runner := &recordingRunner{}
	backend := NewBackendWithRunner(cfg, runner)

	if err := backend.CancelRun(context.Background(), "98765_3"); err != nil {
		t.Fatalf("cancel run: %v", err)
	}
	if len(runner.args) != 1 || runner.args[0] != "98765_3" {
		t.Fatalf("expected exact array child cancel target, got %#v", runner.args)
	}
}

func TestParseSqueueStateMatchesArrayChild(t *testing.T) {
	runRef := parseRunRef("98765_4")
	rawState := parseSqueueState([]byte("98765|RUNNING\n98765_4|FAILED\n"), runRef)
	if rawState != "FAILED" {
		t.Fatalf("expected FAILED, got %q", rawState)
	}
}

type recordingRunner struct {
	args   []string
	output []byte
	err    error
}

func (r *recordingRunner) Run(_ context.Context, _ string, args ...string) ([]byte, error) {
	r.args = append([]string(nil), args...)
	return r.output, r.err
}

func containsArg(args []string, want string) bool {
	for _, arg := range args {
		if arg == want {
			return true
		}
	}
	return false
}

func argValueAfter(args []string, key string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == key {
			return args[i+1]
		}
	}
	return ""
}

func findPart(value, prefix string) string {
	for _, part := range strings.Split(value, ",") {
		if strings.HasPrefix(part, prefix) {
			return part
		}
	}
	return ""
}
