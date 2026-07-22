package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadUsesEnvOverridesAndDefaults(t *testing.T) {
	t.Setenv("BROKER_LISTEN_ADDR", "127.0.0.1:9090")
	t.Setenv("BROKER_AUDIT_ROTATE_BYTES", "2048")
	t.Setenv("BROKER_AUDIT_KEEP_ARCHIVES", "7")
	t.Setenv("BROKER_SLURM_ENABLE_DYNAMIC_PLACEMENT", "true")
	t.Setenv("BROKER_RUNTIME_LLAMACPP_TIMEOUT_SECONDS", "33")
	t.Setenv("BROKER_GPU_SERVICE_REGISTRY_PATH", "/shared/gpu-services.json")
	t.Setenv("BROKER_GPU_SERVICE_CONTROL_TOKEN", "control-secret")
	t.Setenv("BROKER_GPU_SERVICE_V100_REASONING_PROFILE", "v100-strong")
	t.Setenv("BROKER_GPU_SERVICE_V100_REASONING_MODEL_PATH", "/models/v100")
	t.Setenv("BROKER_GPU_SERVICE_V100_REASONING_QUANTIZATION", "bf16")
	t.Setenv("BROKER_GPU_SERVICE_V100_REASONING_CONTEXT_LIMIT_TOKENS", "65536")
	t.Setenv("BROKER_GPU_SERVICE_V100_REASONING_RUNTIME", "vllm")
	t.Setenv("BROKER_GPU_SERVICE_V100_REASONING_RUNTIME_ARGS_JSON", `["--tensor-parallel-size","4"]`)
	t.Setenv("BROKER_INSPECT_REPO_PREWARM_ENABLED", "true")
	t.Setenv("BROKER_INSPECT_REPO_PREWARM_URI", "file:///workspace/target-repo")
	t.Setenv("BROKER_INSPECT_REPO_PREWARM_QUERY", "warm repo inspection cache")

	cfg := Load()
	if cfg.ListenAddr != "127.0.0.1:9090" {
		t.Fatalf("unexpected listen addr: %#v", cfg.ListenAddr)
	}
	if cfg.AuditRotateBytes != 2048 {
		t.Fatalf("unexpected audit rotate bytes: %#v", cfg.AuditRotateBytes)
	}
	if cfg.AuditKeepArchives != 7 {
		t.Fatalf("unexpected audit keep archives: %#v", cfg.AuditKeepArchives)
	}
	if !cfg.SlurmEnableDynamicPlacement {
		t.Fatal("expected dynamic placement to be enabled")
	}
	if cfg.RuntimeLlamaCPPTimeoutSeconds != 33 {
		t.Fatalf("unexpected llama.cpp timeout: %#v", cfg.RuntimeLlamaCPPTimeoutSeconds)
	}
	if cfg.ModelProfileP40 != DefaultModelProfileP40 || cfg.ModelProfileA100 != DefaultModelProfileA100 {
		t.Fatalf("unexpected model profile defaults: %#v", cfg)
	}
	if cfg.ModelProfileP40 != "" || cfg.ModelProfileA100 != "" {
		t.Fatalf("model artifacts must not have hardcoded defaults: %#v", cfg)
	}
	if cfg.GPUServiceRegistryPath != "/shared/gpu-services.json" || cfg.GPUServiceControlRequestDir != "/shared/gpu-services.json.requests" {
		t.Fatalf("unexpected GPU service paths: %#v", cfg)
	}
	if cfg.GPUServiceLeaseDurationSeconds != 4*60*60 || cfg.GPUServiceP40Retrieval.MinReplicas != 1 || cfg.GPUServiceP40Retrieval.MaxReplicas != 2 {
		t.Fatalf("unexpected warm service defaults: %#v", cfg)
	}
	if cfg.GPUServiceV100Reasoning.Profile != "v100-strong" || cfg.GPUServiceV100Reasoning.ContextLimitTokens != 65536 {
		t.Fatalf("unexpected V100 deployment: %#v", cfg.GPUServiceV100Reasoning)
	}
	args, err := cfg.GPUServiceV100Reasoning.RuntimeArgs()
	if err != nil || len(args) != 2 || args[1] != "4" {
		t.Fatalf("unexpected V100 runtime args %#v: %v", args, err)
	}
	if !cfg.InspectRepoPrewarmEnabled || cfg.InspectRepoPrewarmURI != "file:///workspace/target-repo" || cfg.InspectRepoPrewarmQuery != "warm repo inspection cache" {
		t.Fatalf("unexpected inspect_repo prewarm config: %#v", cfg)
	}
}

func TestLoadEnablesLocalInspectRepoWarmByDefaultForLocalCommandMode(t *testing.T) {
	t.Setenv("BROKER_BACKEND", "local")
	t.Setenv("BROKER_LOCAL_MODE", "command")
	if err := os.Unsetenv("BROKER_LOCAL_INSPECT_REPO_WARM_ENABLED"); err != nil {
		t.Fatalf("unset warm daemon env: %v", err)
	}

	cfg := Load()
	if !cfg.LocalInspectRepoWarmEnabled {
		t.Fatalf("expected local inspect_repo warm daemon to default on for local command mode: %#v", cfg)
	}
}

func TestLoadAllowsExplicitOptOutOfLocalInspectRepoWarmDaemon(t *testing.T) {
	t.Setenv("BROKER_BACKEND", "local")
	t.Setenv("BROKER_LOCAL_MODE", "command")
	t.Setenv("BROKER_LOCAL_INSPECT_REPO_WARM_ENABLED", "false")

	cfg := Load()
	if cfg.LocalInspectRepoWarmEnabled {
		t.Fatalf("expected explicit opt-out to disable local inspect_repo warm daemon: %#v", cfg)
	}
}

func TestGPUServiceRuntimeArgsRejectMalformedOrEmptyConfiguration(t *testing.T) {
	for _, raw := range []string{"", "not-json", "[]", `["--port",""]`} {
		if _, err := (GPUServiceDeploymentConfig{RuntimeArgsJSON: raw}).RuntimeArgs(); err == nil {
			t.Fatalf("expected %q to fail", raw)
		}
	}
}

func TestLoadResolvesGPUServicePathsAgainstRepositoryRoot(t *testing.T) {
	repoRoot := t.TempDir()
	t.Setenv("BROKER_REPO_ROOT_PATH", repoRoot)
	t.Setenv("BROKER_GPU_SERVICE_REGISTRY_PATH", ".state/registry.json")
	t.Setenv("BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR", ".state/control")
	t.Setenv("BROKER_GPU_SERVICE_SCRIPT_PATH", "deploy/slurm/gpu_service.slurm")

	cfg := Load()
	if got, want := cfg.GPUServiceRegistryPath, filepath.Join(repoRoot, ".state/registry.json"); got != want {
		t.Fatalf("registry path = %q, want %q", got, want)
	}
	if got, want := cfg.GPUServiceControlRequestDir, filepath.Join(repoRoot, ".state/control"); got != want {
		t.Fatalf("control path = %q, want %q", got, want)
	}
	if got, want := cfg.GPUServiceScriptPath, filepath.Join(repoRoot, "deploy/slurm/gpu_service.slurm"); got != want {
		t.Fatalf("script path = %q, want %q", got, want)
	}
}
