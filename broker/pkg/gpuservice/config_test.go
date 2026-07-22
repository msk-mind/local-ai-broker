package gpuservice

import (
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
)

func TestValidateConfigRequiresEveryDeploymentWithoutStartingServices(t *testing.T) {
	cfg := config.Config{GPUServiceEnabled: false}
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("disabled configuration should be valid: %v", err)
	}

	cfg = validGPUServiceConfig(t.TempDir())
	cfg.GPUServiceV100Reasoning.ModelPath = ""
	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("expected missing exact V100 model path to fail")
	}
	cfg = validGPUServiceConfig(t.TempDir())
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("validate complete configuration: %v", err)
	}
}

func TestProfilesFromConfigRepresentTypedAdaptivePlacements(t *testing.T) {
	cfg := validGPUServiceConfig(t.TempDir())
	profiles, err := ProfilesFromConfig(cfg)
	if err != nil {
		t.Fatalf("profiles from config: %v", err)
	}
	if len(profiles) != 5 {
		t.Fatalf("expected five deployment profiles, got %d", len(profiles))
	}
	byTier := map[Tier]Profile{}
	for _, profile := range profiles {
		byTier[profile.Tier] = profile
	}
	if got := byTier[TierV100Reasoning]; got.Placement.GPU != (GPU{Type: "v100", Count: 4}) ||
		got.Placement.Partition != "v100-partition" || got.Deployment.RuntimeArgs[1] != "{gpu_count}" {
		t.Fatalf("unexpected V100 profile: %#v", got)
	}
	if got := byTier[TierA100Single].Placement.GPU.Count; got != 1 {
		t.Fatalf("single A100 profile requested %d GPUs", got)
	}
	if got := byTier[TierA100Multigpu].Placement.GPU.Count; got != 4 {
		t.Fatalf("multi-A100 profile requested %d GPUs", got)
	}
}

func TestValidateConfigRequiresTensorParallelArgsAndStrongerV100Profile(t *testing.T) {
	cfg := validGPUServiceConfig(t.TempDir())
	cfg.GPUServiceV100Reasoning.RuntimeArgsJSON = `["--tensor-parallel-size","4"]`
	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("expected literal multigpu count without {gpu_count} to fail")
	}

	cfg = validGPUServiceConfig(t.TempDir())
	cfg.GPUServiceV100Reasoning.Profile = cfg.GPUServiceP40Synthesis.Profile
	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("expected V100 to reject the P40 synthesis profile")
	}

	cfg = validGPUServiceConfig(t.TempDir())
	cfg.GPUServiceV100Reasoning.ModelPath = cfg.GPUServiceP40Synthesis.ModelPath
	if err := ValidateConfig(cfg); err == nil {
		t.Fatal("expected V100 to reject the P40 synthesis model artifact")
	}
}

func TestNewManagerFromConfigDoesNotStartSchedulerWork(t *testing.T) {
	cfg := validGPUServiceConfig(t.TempDir())
	scheduler := newFakeServiceScheduler()
	manager, err := NewManagerFromConfig(cfg, scheduler)
	if err != nil {
		t.Fatalf("construct manager: %v", err)
	}
	if manager == nil || scheduler.launchCount() != 0 {
		t.Fatalf("constructor started scheduler work: manager=%v launches=%d", manager, scheduler.launchCount())
	}
}

func validGPUServiceConfig(root string) config.Config {
	deployment := func(name string, minReplicas, maxReplicas int) config.GPUServiceDeploymentConfig {
		return config.GPUServiceDeploymentConfig{
			Profile:            name,
			ModelPath:          "/models/" + name,
			Quantization:       "bf16",
			ContextLimitTokens: 32768,
			Runtime:            "vllm",
			RuntimeArgsJSON:    `["--served-model-name","` + name + `"]`,
			MinReplicas:        minReplicas,
			MaxReplicas:        maxReplicas,
		}
	}
	v100 := deployment("v100-reasoning", 0, 1)
	v100.RuntimeArgsJSON = `["--tensor-parallel-size","{gpu_count}"]`
	a100Multi := deployment("a100-multigpu", 0, 1)
	a100Multi.RuntimeArgsJSON = `["--tensor-parallel-size","{gpu_count}"]`
	return config.Config{
		GPUServiceEnabled:                 true,
		GPUServiceRegistryPath:            root + "/registry.json",
		GPUServiceControlToken:            "control-token",
		GPUServiceControlRequestDir:       root + "/requests",
		GPUServiceScriptPath:              "deploy/slurm/gpu_service.slurm",
		GPUServiceLeaseDurationSeconds:    4 * 60 * 60,
		GPUServiceHealthIntervalSeconds:   15,
		GPUServiceHeartbeatTimeoutSeconds: 45,
		GPUServiceStartupTimeoutSeconds:   600,
		GPUServiceP40Retrieval:            deployment("p40-retrieval", 1, 2),
		GPUServiceP40Synthesis:            deployment("p40-synthesis", 1, 2),
		GPUServiceV100Reasoning:           v100,
		GPUServiceA100Single:              deployment("a100-single", 0, 1),
		GPUServiceA100Multigpu:            a100Multi,
		SlurmPartitionP40:                 "p40-partition",
		SlurmPartitionV100:                "v100-partition",
		SlurmPartitionA100:                "a100-partition",
		SlurmGPUTypeP40:                   "p40",
		SlurmGPUTypeV100:                  "v100",
		SlurmGPUTypeA100:                  "a100",
		SlurmNodeListV100:                 "v100node[01-02]",
		SlurmConstraintV100:               "v100",
	}
}
