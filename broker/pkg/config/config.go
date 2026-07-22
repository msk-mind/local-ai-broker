package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	// Legacy task-level profiles intentionally default to empty. Exact model
	// artifacts are operator-supplied deployment configuration.
	DefaultModelProfileP40                = ""
	DefaultModelProfileA100               = ""
	DefaultRuntimeConnectionTimeoutSecs   = 20
	DefaultGPUServiceLeaseDurationSecs    = 4 * 60 * 60
	DefaultGPUServiceHealthIntervalSecs   = 15
	DefaultGPUServiceHeartbeatTimeoutSecs = 45
	DefaultGPUServiceStartupTimeoutSecs   = 10 * 60
)

// GPUServiceDeploymentConfig describes one operator-supplied model service.
// Model artifacts and runtime arguments intentionally have no defaults: an
// enabled control plane must be given an exact deployment for every tier.
type GPUServiceDeploymentConfig struct {
	Profile            string
	ModelPath          string
	Quantization       string
	ContextLimitTokens int
	Runtime            string
	RuntimeArgsJSON    string
	MinReplicas        int
	MaxReplicas        int
}

// RuntimeArgs parses the configured JSON array without applying shell
// expansion. This keeps exact runtime arguments portable through Slurm.
func (p GPUServiceDeploymentConfig) RuntimeArgs() ([]string, error) {
	if strings.TrimSpace(p.RuntimeArgsJSON) == "" {
		return nil, fmt.Errorf("runtime args JSON is required")
	}
	var args []string
	if err := json.Unmarshal([]byte(p.RuntimeArgsJSON), &args); err != nil {
		return nil, fmt.Errorf("parse runtime args JSON: %w", err)
	}
	if len(args) == 0 {
		return nil, fmt.Errorf("runtime args JSON must contain at least one argument")
	}
	for i, arg := range args {
		if strings.TrimSpace(arg) == "" {
			return nil, fmt.Errorf("runtime args JSON contains an empty argument at index %d", i)
		}
	}
	return args, nil
}

type Config struct {
	ListenAddr                        string
	JobStorePath                      string
	RunRootPath                       string
	RepoRootPath                      string
	InspectRepoPrewarmEnabled         bool
	InspectRepoPrewarmURI             string
	InspectRepoPrewarmQuery           string
	AuditLogPath                      string
	AuditVerifyMode                   string
	AuditRotateBytes                  int64
	AuditKeepArchives                 int
	AuditMaintainIntervalSeconds      int
	AuthMode                          string
	StaticTokens                      string
	MCPActor                          string
	MCPRole                           string
	BackendKind                       string
	SlurmMode                         string
	SlurmSubmitCmd                    string
	SlurmStatusCmd                    string
	SlurmCancelCmd                    string
	SlurmInfoCmd                      string
	SlurmScriptPath                   string
	SlurmPartitionCPU                 string
	SlurmPartitionGPU                 string
	SlurmPartitionP40                 string
	SlurmPartitionV100                string
	SlurmPartitionA100                string
	SlurmGPURequestMode               string
	SlurmGPUTypeP40                   string
	SlurmGPUTypeV100                  string
	SlurmGPUTypeA100                  string
	SlurmNodeListCPU                  string
	SlurmNodeListP40                  string
	SlurmNodeListV100                 string
	SlurmNodeListA100                 string
	SlurmConstraintCPU                string
	SlurmConstraintP40                string
	SlurmConstraintV100               string
	SlurmConstraintA100               string
	SlurmEnableDynamicPlacement       bool
	GPUServiceEnabled                 bool
	GPUServiceRegistryPath            string
	GPUServiceControlToken            string
	GPUServiceControlRequestDir       string
	GPUServiceScriptPath              string
	GPUServiceLeaseDurationSeconds    int
	GPUServiceHealthIntervalSeconds   int
	GPUServiceHeartbeatTimeoutSeconds int
	GPUServiceStartupTimeoutSeconds   int
	GPUServiceP40Retrieval            GPUServiceDeploymentConfig
	GPUServiceP40Synthesis            GPUServiceDeploymentConfig
	GPUServiceV100Reasoning           GPUServiceDeploymentConfig
	GPUServiceA100Single              GPUServiceDeploymentConfig
	GPUServiceA100Multigpu            GPUServiceDeploymentConfig
	ModelProfileCPU                   string
	ModelProfileP40                   string
	ModelProfileA100                  string
	RuntimeLlamaCPPBaseURL            string
	RuntimeLlamaCPPTimeoutSeconds     int
	RuntimeVLLMBaseURL                string
	RuntimeVLLMTimeoutSeconds         int
	RuntimeSGLangBaseURL              string
	RuntimeSGLangTimeoutSeconds       int
	LocalMode                         string
	LocalScriptPath                   string
	LocalInspectRepoWarmEnabled       bool
	ParallelMaxBatchSize              int
	ParallelMaxActiveBatches          int
	RootActionMaxAdditionalBatches    int
	RootActionMaxRetriedShards        int
}

func Load() Config {
	localInspectRepoWarmEnabled := false
	if raw, ok := os.LookupEnv("BROKER_LOCAL_INSPECT_REPO_WARM_ENABLED"); ok {
		localInspectRepoWarmEnabled = parseBool(raw, false)
	} else {
		backendKind := envOrDefault("BROKER_BACKEND", "slurm")
		localMode := envOrDefault("BROKER_LOCAL_MODE", "command")
		localInspectRepoWarmEnabled = strings.EqualFold(strings.TrimSpace(backendKind), "local") &&
			strings.EqualFold(strings.TrimSpace(localMode), "command")
	}
	cfg := Config{
		ListenAddr:                        envOrDefault("BROKER_LISTEN_ADDR", ":8081"),
		JobStorePath:                      envOrDefault("BROKER_JOB_STORE_PATH", ".broker/jobs.json"),
		RunRootPath:                       envOrDefault("BROKER_RUN_ROOT_PATH", ".broker/runs"),
		RepoRootPath:                      envOrDefault("BROKER_REPO_ROOT_PATH", "."),
		InspectRepoPrewarmEnabled:         envOrDefaultBool("BROKER_INSPECT_REPO_PREWARM_ENABLED", false),
		InspectRepoPrewarmURI:             envOrDefault("BROKER_INSPECT_REPO_PREWARM_URI", ""),
		InspectRepoPrewarmQuery:           envOrDefault("BROKER_INSPECT_REPO_PREWARM_QUERY", "broker inspect_repo index prewarm"),
		AuditLogPath:                      envOrDefault("BROKER_AUDIT_LOG_PATH", ".broker/audit.jsonl"),
		AuditVerifyMode:                   envOrDefault("BROKER_AUDIT_VERIFY_MODE", "fail"),
		AuditRotateBytes:                  envOrDefaultInt64("BROKER_AUDIT_ROTATE_BYTES", 10*1024*1024),
		AuditKeepArchives:                 envOrDefaultInt("BROKER_AUDIT_KEEP_ARCHIVES", 10),
		AuditMaintainIntervalSeconds:      envOrDefaultInt("BROKER_AUDIT_MAINTAIN_INTERVAL_SECONDS", 300),
		AuthMode:                          envOrDefault("BROKER_AUTH_MODE", "header"),
		StaticTokens:                      envOrDefault("BROKER_STATIC_TOKENS", ""),
		MCPActor:                          envOrDefault("BROKER_MCP_ACTOR", ""),
		MCPRole:                           envOrDefault("BROKER_MCP_ROLE", "user"),
		BackendKind:                       envOrDefault("BROKER_BACKEND", "slurm"),
		SlurmMode:                         envOrDefault("BROKER_SLURM_MODE", "stub"),
		SlurmSubmitCmd:                    envOrDefault("BROKER_SLURM_SUBMIT_CMD", "sbatch"),
		SlurmStatusCmd:                    envOrDefault("BROKER_SLURM_STATUS_CMD", "sacct"),
		SlurmCancelCmd:                    envOrDefault("BROKER_SLURM_CANCEL_CMD", "scancel"),
		SlurmInfoCmd:                      envOrDefault("BROKER_SLURM_INFO_CMD", "sinfo"),
		SlurmScriptPath:                   envOrDefault("BROKER_SLURM_SCRIPT_PATH", "deploy/slurm/broker_worker.slurm"),
		SlurmPartitionCPU:                 envOrDefault("BROKER_SLURM_PARTITION_CPU", ""),
		SlurmPartitionGPU:                 envOrDefault("BROKER_SLURM_PARTITION_GPU", ""),
		SlurmPartitionP40:                 envOrDefault("BROKER_SLURM_PARTITION_P40", ""),
		SlurmPartitionV100:                envOrDefault("BROKER_SLURM_PARTITION_V100", ""),
		SlurmPartitionA100:                envOrDefault("BROKER_SLURM_PARTITION_A100", ""),
		SlurmGPURequestMode:               envOrDefault("BROKER_SLURM_GPU_REQUEST_MODE", "gres"),
		SlurmGPUTypeP40:                   envOrDefault("BROKER_SLURM_GPU_TYPE_P40", ""),
		SlurmGPUTypeV100:                  envOrDefault("BROKER_SLURM_GPU_TYPE_V100", ""),
		SlurmGPUTypeA100:                  envOrDefault("BROKER_SLURM_GPU_TYPE_A100", ""),
		SlurmNodeListCPU:                  envOrDefault("BROKER_SLURM_NODELIST_CPU", ""),
		SlurmNodeListP40:                  envOrDefault("BROKER_SLURM_NODELIST_P40", ""),
		SlurmNodeListV100:                 envOrDefault("BROKER_SLURM_NODELIST_V100", ""),
		SlurmNodeListA100:                 envOrDefault("BROKER_SLURM_NODELIST_A100", ""),
		SlurmConstraintCPU:                envOrDefault("BROKER_SLURM_CONSTRAINT_CPU", ""),
		SlurmConstraintP40:                envOrDefault("BROKER_SLURM_CONSTRAINT_P40", ""),
		SlurmConstraintV100:               envOrDefault("BROKER_SLURM_CONSTRAINT_V100", ""),
		SlurmConstraintA100:               envOrDefault("BROKER_SLURM_CONSTRAINT_A100", ""),
		SlurmEnableDynamicPlacement:       envOrDefaultBool("BROKER_SLURM_ENABLE_DYNAMIC_PLACEMENT", false),
		GPUServiceEnabled:                 envOrDefaultBool("BROKER_GPU_SERVICE_ENABLED", false),
		GPUServiceRegistryPath:            envOrDefault("BROKER_GPU_SERVICE_REGISTRY_PATH", ".broker/gpu-services.json"),
		GPUServiceControlToken:            envOrDefault("BROKER_GPU_SERVICE_CONTROL_TOKEN", ""),
		GPUServiceControlRequestDir:       envOrDefault("BROKER_GPU_SERVICE_CONTROL_REQUEST_DIR", envOrDefault("BROKER_GPU_SERVICE_REGISTRY_PATH", ".broker/gpu-services.json")+".requests"),
		GPUServiceScriptPath:              envOrDefault("BROKER_GPU_SERVICE_SCRIPT_PATH", ""),
		GPUServiceLeaseDurationSeconds:    envOrDefaultInt("BROKER_GPU_SERVICE_LEASE_DURATION_SECONDS", DefaultGPUServiceLeaseDurationSecs),
		GPUServiceHealthIntervalSeconds:   envOrDefaultInt("BROKER_GPU_SERVICE_HEALTH_INTERVAL_SECONDS", DefaultGPUServiceHealthIntervalSecs),
		GPUServiceHeartbeatTimeoutSeconds: envOrDefaultInt("BROKER_GPU_SERVICE_HEARTBEAT_TIMEOUT_SECONDS", DefaultGPUServiceHeartbeatTimeoutSecs),
		GPUServiceStartupTimeoutSeconds:   envOrDefaultInt("BROKER_GPU_SERVICE_STARTUP_TIMEOUT_SECONDS", DefaultGPUServiceStartupTimeoutSecs),
		GPUServiceP40Retrieval:            loadGPUServiceDeployment("P40_RETRIEVAL", 1, 2),
		GPUServiceP40Synthesis:            loadGPUServiceDeployment("P40_SYNTHESIS", 1, 2),
		GPUServiceV100Reasoning:           loadGPUServiceDeployment("V100_REASONING", 0, 1),
		GPUServiceA100Single:              loadGPUServiceDeployment("A100_SINGLE", 0, 1),
		GPUServiceA100Multigpu:            loadGPUServiceDeployment("A100_MULTIGPU", 0, 1),
		ModelProfileCPU:                   envOrDefault("BROKER_MODEL_PROFILE_CPU", ""),
		ModelProfileP40:                   envOrDefault("BROKER_MODEL_PROFILE_P40", DefaultModelProfileP40),
		ModelProfileA100:                  envOrDefault("BROKER_MODEL_PROFILE_A100", DefaultModelProfileA100),
		RuntimeLlamaCPPBaseURL:            envOrDefault("BROKER_RUNTIME_LLAMACPP_BASE_URL", ""),
		RuntimeLlamaCPPTimeoutSeconds:     envOrDefaultInt("BROKER_RUNTIME_LLAMACPP_TIMEOUT_SECONDS", DefaultRuntimeConnectionTimeoutSecs),
		RuntimeVLLMBaseURL:                envOrDefault("BROKER_RUNTIME_VLLM_BASE_URL", ""),
		RuntimeVLLMTimeoutSeconds:         envOrDefaultInt("BROKER_RUNTIME_VLLM_TIMEOUT_SECONDS", DefaultRuntimeConnectionTimeoutSecs),
		RuntimeSGLangBaseURL:              envOrDefault("BROKER_RUNTIME_SGLANG_BASE_URL", ""),
		RuntimeSGLangTimeoutSeconds:       envOrDefaultInt("BROKER_RUNTIME_SGLANG_TIMEOUT_SECONDS", DefaultRuntimeConnectionTimeoutSecs),
		LocalMode:                         envOrDefault("BROKER_LOCAL_MODE", "command"),
		LocalScriptPath:                   envOrDefault("BROKER_LOCAL_SCRIPT_PATH", "deploy/local/broker_worker.sh"),
		LocalInspectRepoWarmEnabled:       localInspectRepoWarmEnabled,
		ParallelMaxBatchSize:              envOrDefaultInt("BROKER_PARALLEL_MAX_BATCH_SIZE", 64),
		ParallelMaxActiveBatches:          envOrDefaultInt("BROKER_PARALLEL_MAX_ACTIVE_BATCHES", 0),
		RootActionMaxAdditionalBatches:    envOrDefaultInt("BROKER_ROOT_ACTION_MAX_ADDITIONAL_BATCHES", 1),
		RootActionMaxRetriedShards:        envOrDefaultInt("BROKER_ROOT_ACTION_MAX_RETRIED_SHARDS", 4),
	}
	return resolveGPUServicePaths(cfg)
}

// resolveGPUServicePaths gives every broker component the same absolute
// registry, control-spool, and launcher paths. In particular, the request
// service, scheduler backend, and reconciler must not interpret a relative
// path against three potentially different working directories.
func resolveGPUServicePaths(cfg Config) Config {
	cfg.GPUServiceRegistryPath = resolveAgainstRepoRoot(cfg.RepoRootPath, cfg.GPUServiceRegistryPath)
	cfg.GPUServiceControlRequestDir = resolveAgainstRepoRoot(cfg.RepoRootPath, cfg.GPUServiceControlRequestDir)
	cfg.GPUServiceScriptPath = resolveAgainstRepoRoot(cfg.RepoRootPath, cfg.GPUServiceScriptPath)
	return cfg
}

func resolveAgainstRepoRoot(repoRoot, value string) string {
	value = strings.TrimSpace(value)
	if value == "" || filepath.IsAbs(value) {
		return value
	}
	repoRoot = strings.TrimSpace(repoRoot)
	if repoRoot == "" {
		repoRoot = "."
	}
	absoluteRoot, err := filepath.Abs(repoRoot)
	if err != nil {
		return filepath.Clean(filepath.Join(repoRoot, value))
	}
	return filepath.Clean(filepath.Join(absoluteRoot, value))
}

func loadGPUServiceDeployment(name string, minReplicas, maxReplicas int) GPUServiceDeploymentConfig {
	prefix := "BROKER_GPU_SERVICE_" + name + "_"
	return GPUServiceDeploymentConfig{
		Profile:            envOrDefault(prefix+"PROFILE", ""),
		ModelPath:          envOrDefault(prefix+"MODEL_PATH", ""),
		Quantization:       envOrDefault(prefix+"QUANTIZATION", ""),
		ContextLimitTokens: envOrDefaultInt(prefix+"CONTEXT_LIMIT_TOKENS", 0),
		Runtime:            envOrDefault(prefix+"RUNTIME", ""),
		RuntimeArgsJSON:    envOrDefault(prefix+"RUNTIME_ARGS_JSON", ""),
		MinReplicas:        envOrDefaultInt(prefix+"MIN_REPLICAS", minReplicas),
		MaxReplicas:        envOrDefaultInt(prefix+"MAX_REPLICAS", maxReplicas),
	}
}

func DefaultModelProfiles() (cpu, p40, a100 string) {
	return "", DefaultModelProfileP40, DefaultModelProfileA100
}

func DefaultRuntimeTimeoutSeconds() int {
	return DefaultRuntimeConnectionTimeoutSecs
}

func envOrDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envOrDefaultInt(key string, fallback int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return fallback
}

func envOrDefaultInt64(key string, fallback int64) int64 {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseInt(value, 10, 64); err == nil {
			return parsed
		}
	}
	return fallback
}

func envOrDefaultBool(key string, fallback bool) bool {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return parseBool(value, fallback)
	}
	return fallback
}

func parseBool(raw string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}
