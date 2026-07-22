package gpuservice

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
)

var ErrControlPlaneDisabled = errors.New("GPU service control plane is disabled")

// ValidateConfig validates an enabled service control plane without starting
// scheduler jobs. Disabled configurations remain valid so existing CPU-only
// unit tests and deployments do not need placeholder model artifacts.
func ValidateConfig(cfg config.Config) error {
	if !cfg.GPUServiceEnabled {
		return nil
	}
	if strings.TrimSpace(cfg.GPUServiceRegistryPath) == "" {
		return errors.New("BROKER_GPU_SERVICE_REGISTRY_PATH is required")
	}
	if strings.TrimSpace(cfg.GPUServiceControlToken) == "" {
		return errors.New("BROKER_GPU_SERVICE_CONTROL_TOKEN is required")
	}
	if strings.TrimSpace(cfg.GPUServiceControlRequestDir) == "" {
		return errors.New("BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR is required")
	}
	if strings.TrimSpace(cfg.GPUServiceScriptPath) == "" {
		return errors.New("BROKER_GPU_SERVICE_SCRIPT_PATH is required")
	}
	if cfg.GPUServiceLeaseDurationSeconds <= 0 || cfg.GPUServiceHealthIntervalSeconds <= 0 ||
		cfg.GPUServiceHeartbeatTimeoutSeconds <= 0 || cfg.GPUServiceStartupTimeoutSeconds <= 0 {
		return errors.New("GPU service lease, health, heartbeat, and startup durations must be positive")
	}
	if cfg.GPUServiceHeartbeatTimeoutSeconds < cfg.GPUServiceHealthIntervalSeconds {
		return errors.New("GPU service heartbeat timeout must be at least the health interval")
	}
	_, err := ProfilesFromConfig(cfg)
	return err
}

// ProfilesFromConfig builds all five required deployments. It never supplies
// a model path, quantization, context limit, runtime, or runtime argument.
func ProfilesFromConfig(cfg config.Config) ([]Profile, error) {
	// Keep this table ordered: adaptive synthesis relies on the stable tier
	// order, while the small descriptor makes the shared placement rules clear.
	definitions := []struct {
		tier       Tier
		role       Role
		operations []string
		deployment config.GPUServiceDeploymentConfig
		partition  string
		gpuType    string
		gpuCount   int
		nodeList   string
		constraint string
	}{
		{
			tier:       TierP40Retrieval,
			role:       RoleRetrieval,
			operations: []string{OperationEmbeddings, OperationIndexStatus, OperationIndexUpsert, OperationVectorSearch, OperationRerank},
			deployment: cfg.GPUServiceP40Retrieval,
			partition:  firstNonEmpty(cfg.SlurmPartitionP40, cfg.SlurmPartitionGPU), gpuType: cfg.SlurmGPUTypeP40, gpuCount: 1, nodeList: cfg.SlurmNodeListP40, constraint: cfg.SlurmConstraintP40,
		},
		{
			tier:       TierP40Synthesis,
			role:       RoleSynthesis,
			operations: []string{OperationChatCompletions},
			deployment: cfg.GPUServiceP40Synthesis,
			partition:  firstNonEmpty(cfg.SlurmPartitionP40, cfg.SlurmPartitionGPU), gpuType: cfg.SlurmGPUTypeP40, gpuCount: 1, nodeList: cfg.SlurmNodeListP40, constraint: cfg.SlurmConstraintP40,
		},
		{
			tier:       TierV100Reasoning,
			role:       RoleSynthesis,
			operations: []string{OperationChatCompletions},
			deployment: cfg.GPUServiceV100Reasoning,
			partition:  firstNonEmpty(cfg.SlurmPartitionV100, cfg.SlurmPartitionGPU), gpuType: cfg.SlurmGPUTypeV100, gpuCount: 4, nodeList: cfg.SlurmNodeListV100, constraint: cfg.SlurmConstraintV100,
		},
		{
			tier:       TierA100Single,
			role:       RoleSynthesis,
			operations: []string{OperationChatCompletions},
			deployment: cfg.GPUServiceA100Single,
			partition:  firstNonEmpty(cfg.SlurmPartitionA100, cfg.SlurmPartitionGPU), gpuType: cfg.SlurmGPUTypeA100, gpuCount: 1, nodeList: cfg.SlurmNodeListA100, constraint: cfg.SlurmConstraintA100,
		},
		{
			tier:       TierA100Multigpu,
			role:       RoleSynthesis,
			operations: []string{OperationChatCompletions},
			deployment: cfg.GPUServiceA100Multigpu,
			partition:  firstNonEmpty(cfg.SlurmPartitionA100, cfg.SlurmPartitionGPU), gpuType: cfg.SlurmGPUTypeA100, gpuCount: 4, nodeList: cfg.SlurmNodeListA100, constraint: cfg.SlurmConstraintA100,
		},
	}

	profiles := make([]Profile, 0, len(definitions))
	for _, definition := range definitions {
		args, err := definition.deployment.RuntimeArgs()
		if err != nil {
			return nil, fmt.Errorf("tier %s: %w", definition.tier, err)
		}
		profile := Profile{
			Tier:                definition.tier,
			Role:                definition.role,
			SupportedOperations: append([]string(nil), definition.operations...),
			Deployment: DeploymentProfile{
				Name:               strings.TrimSpace(definition.deployment.Profile),
				Model:              strings.TrimSpace(definition.deployment.ModelPath),
				Quantization:       strings.TrimSpace(definition.deployment.Quantization),
				ContextLimitTokens: definition.deployment.ContextLimitTokens,
				Runtime:            strings.TrimSpace(definition.deployment.Runtime),
				RuntimeArgs:        args,
			},
			Placement:   Placement{Partition: definition.partition, GPU: GPU{Type: definition.gpuType, Count: definition.gpuCount}, NodeList: definition.nodeList, Constraint: definition.constraint},
			MinReplicas: definition.deployment.MinReplicas,
			MaxReplicas: definition.deployment.MaxReplicas,
		}
		if err := profile.Validate(); err != nil {
			return nil, err
		}
		if profile.Placement.GPU.Count > 1 && !containsRuntimePlaceholder(profile.Deployment.RuntimeArgs, "gpu_count") {
			return nil, fmt.Errorf("tier %s runtime args must use {gpu_count} for tensor-parallel placement", profile.Tier)
		}
		profiles = append(profiles, profile)
	}
	if strings.EqualFold(
		strings.TrimSpace(cfg.GPUServiceV100Reasoning.Profile),
		strings.TrimSpace(cfg.GPUServiceP40Synthesis.Profile),
	) || strings.TrimSpace(cfg.GPUServiceV100Reasoning.ModelPath) == strings.TrimSpace(cfg.GPUServiceP40Synthesis.ModelPath) {
		return nil, errors.New("v100-reasoning must use a model profile and artifact distinct from p40-synthesis")
	}
	return profiles, nil
}

func containsRuntimePlaceholder(args []string, name string) bool {
	placeholder := "{" + name + "}"
	for _, arg := range args {
		if strings.Contains(arg, placeholder) {
			return true
		}
	}
	return false
}

type Timing struct {
	LeaseDuration    time.Duration
	HealthInterval   time.Duration
	HeartbeatTimeout time.Duration
	StartupTimeout   time.Duration
}

func TimingFromConfig(cfg config.Config) Timing {
	return Timing{
		LeaseDuration:    time.Duration(cfg.GPUServiceLeaseDurationSeconds) * time.Second,
		HealthInterval:   time.Duration(cfg.GPUServiceHealthIntervalSeconds) * time.Second,
		HeartbeatTimeout: time.Duration(cfg.GPUServiceHeartbeatTimeoutSeconds) * time.Second,
		StartupTimeout:   time.Duration(cfg.GPUServiceStartupTimeoutSeconds) * time.Second,
	}
}

// NewManagerFromConfig is the production wiring point used by both broker
// transports. Construction validates configuration but does not submit jobs;
// callers start scheduler activity explicitly with Manager.Run/Reconcile.
func NewManagerFromConfig(cfg config.Config, scheduler Scheduler) (*Manager, error) {
	if !cfg.GPUServiceEnabled {
		return nil, ErrControlPlaneDisabled
	}
	if err := ValidateConfig(cfg); err != nil {
		return nil, err
	}
	profiles, err := ProfilesFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	timing := TimingFromConfig(cfg)
	registry, err := NewAuthenticatedFileRegistry(cfg.GPUServiceRegistryPath, timing.LeaseDuration, cfg.GPUServiceControlToken)
	if err != nil {
		return nil, err
	}
	spool, err := NewControlSpool(cfg.GPUServiceControlRequestDir, cfg.GPUServiceControlToken)
	if err != nil {
		return nil, err
	}
	healthTimeout := 10 * time.Second
	if timing.HealthInterval < healthTimeout {
		healthTimeout = timing.HealthInterval
	}
	return NewManager(registry, scheduler, NewHTTPHealthChecker(healthTimeout), ManagerOptions{
		Profiles:     profiles,
		Timing:       timing,
		ControlToken: cfg.GPUServiceControlToken,
		ControlSpool: spool,
	})
}
