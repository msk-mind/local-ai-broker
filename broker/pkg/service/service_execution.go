package service

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func requiresStrictRetrievalQuality(job types.Job) bool {
	if job.Request.TaskParams == nil {
		return false
	}
	return boolValue(job.Request.TaskParams[taskParamStrictRetrievalQuality])
}

func cloneMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}
	output := make(map[string]any, len(input))
	for key, value := range input {
		output[key] = cloneAny(value)
	}
	return output
}

func cloneSlice(input []any) []any {
	if input == nil {
		return nil
	}
	output := make([]any, len(input))
	for i, value := range input {
		output[i] = cloneAny(value)
	}
	return output
}

func cloneAny(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		return cloneSlice(typed)
	case []string:
		return append([]string(nil), typed...)
	case []int:
		return append([]int(nil), typed...)
	case []int64:
		return append([]int64(nil), typed...)
	case []float64:
		return append([]float64(nil), typed...)
	case []bool:
		return append([]bool(nil), typed...)
	default:
		return value
	}
}

func stringValue(value any) string {
	if text, ok := value.(string); ok {
		return strings.TrimSpace(text)
	}
	return ""
}

func mapValue(value any) map[string]any {
	if out, ok := value.(map[string]any); ok {
		return out
	}
	return map[string]any{}
}

func boolValue(value any) bool {
	if out, ok := value.(bool); ok {
		return out
	}
	return false
}

func hasAnyString(items []string, expected []string) bool {
	lookup := make(map[string]struct{}, len(items))
	for _, item := range items {
		lookup[item] = struct{}{}
	}
	for _, item := range expected {
		if _, ok := lookup[item]; ok {
			return true
		}
	}
	return false
}

func collectStringSlice(value any) []string {
	switch typed := value.(type) {
	case []string:
		return append([]string(nil), typed...)
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok && strings.TrimSpace(text) != "" {
				out = append(out, text)
			}
		}
		return out
	default:
		return nil
	}
}

func appendUniqueString(values []string, value string) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

func stringSliceToAny(values []string) []any {
	out := make([]any, 0, len(values))
	for _, value := range values {
		out = append(out, value)
	}
	return out
}

func (s *Service) stageExecutionBundle(ctx context.Context, job *types.Job) error {
	jobDir := filepath.Join(s.runRoot, job.ID)
	if err := os.MkdirAll(jobDir, 0o700); err != nil {
		return err
	}

	jobSpecPath := filepath.Join(jobDir, "job_spec.json")
	executionPlanPath := filepath.Join(jobDir, "execution_plan.json")
	inputManifestPath := filepath.Join(jobDir, "input_manifest.json")

	bundle, err := s.executionBundle(ctx, job)
	if err != nil {
		return err
	}

	if err := writeJSONFile(jobSpecPath, bundle.JobSpec); err != nil {
		return err
	}
	if err := writeJSONFile(executionPlanPath, bundle.ExecutionPlan); err != nil {
		return err
	}
	if err := writeJSONFile(inputManifestPath, bundle.InputManifest); err != nil {
		return err
	}

	return nil
}

func (s *Service) executionBundle(ctx context.Context, job *types.Job) (backends.InlineExecutionBundle, error) {
	resolvedInputRefs := append([]types.InputRef(nil), job.Request.InputRefs...)
	if !inputRefsReadyForExecution(resolvedInputRefs) {
		var err error
		resolvedInputRefs, err = s.resolveInputRefs(ctx, *job)
		if err != nil {
			return backends.InlineExecutionBundle{}, err
		}
	}
	job.Request.InputRefs = resolvedInputRefs

	executionPlan := map[string]any{
		"job_id":             job.ID,
		"task_type":          job.TaskType,
		"execution_profile":  job.Request.ExecutionProfile,
		"selected_model":     job.Request.ExecutionProfile.Model,
		"runtime_backend":    job.Request.ExecutionProfile.Runtime,
		"resource_tier":      job.Request.ExecutionProfile.Tier,
		"runtime_connection": s.runtimeConnectionPlan(job.Request.ExecutionProfile),
	}
	if job.TaskType == "inspect_repo" && s.gpuServices.Enabled {
		executionPlan["gpu_service_registry_path"] = s.gpuServices.RegistryPath
		executionPlan["gpu_service_request_path"] = s.gpuServices.ControlRequestPath
		executionPlan["gpu_service_health_interval_seconds"] = s.gpuServices.HealthIntervalSeconds
		executionPlan["gpu_service_startup_timeout_seconds"] = s.gpuServices.StartupTimeoutSeconds
		executionPlan["gpu_service_control_token"] = s.gpuServices.ControlToken
	}
	if job.TaskType == "inspect_repo" {
		executionPlan["repo_inspection_cache_path"] = s.repoInspectionCachePath()
		executionPlan["repo_inspection_shared_cache_path"] = s.repoInspectionSharedCachePath()
		executionPlan["repo_inspection_use_node_local_cache"] = true
		if s.inspectRepoUsesLocalBackend(*job) {
			executionPlan["repo_inspection_node_local_cache_namespace"] = stableInspectRepoNodeLocalNamespace(
				s.repoInspectionCachePath(),
				s.repoInspectionSharedCachePath(),
			)
		}
	}

	return backends.InlineExecutionBundle{
		JobSpec: map[string]any{
			"job_id":        job.ID,
			"task_type":     job.TaskType,
			"task_params":   s.executionTaskParams(*job),
			"output_schema": job.Request.OutputSchema,
			"constraints":   job.Request.Constraints,
		},
		ExecutionPlan: executionPlan,
		InputManifest: map[string]any{
			"job_id":     job.ID,
			"input_refs": resolvedInputRefs,
		},
	}, nil
}

func (s *Service) inspectRepoUsesLocalBackend(job types.Job) bool {
	backendName := strings.TrimSpace(job.Request.ExecutionProfile.Backend)
	if backendName == "" && s.backend != nil {
		backendName = strings.TrimSpace(s.backend.Name())
	}
	return strings.EqualFold(backendName, "local")
}

func stableInspectRepoNodeLocalNamespace(cachePath, sharedCachePath string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(cachePath) + "\x00" + strings.TrimSpace(sharedCachePath)))
	return "inspect-repo-" + hex.EncodeToString(sum[:8])
}

func (s *Service) repoInspectionCachePath() string {
	repoRoot := strings.TrimSpace(s.repoRoot)
	if repoRoot != "" {
		return filepath.Join(repoRoot, ".broker", "repo-inspection-cache")
	}
	return filepath.Join(s.runRoot, "repo-inspection-cache")
}

func (s *Service) repoInspectionSharedCachePath() string {
	if configured := strings.TrimSpace(os.Getenv("BROKER_REPO_INSPECTION_SHARED_CACHE_DIR")); configured != "" {
		return configured
	}
	repoRoot := strings.TrimSpace(s.repoRoot)
	if repoRoot != "" {
		return filepath.Join(repoRoot, ".broker", "repo-inspection-shared-cache")
	}
	return filepath.Join(s.runRoot, "repo-inspection-shared-cache")
}

func inputRefsReadyForExecution(inputRefs []types.InputRef) bool {
	for _, input := range inputRefs {
		if !isArtifactInputRef(input) {
			continue
		}
		if input.Metadata == nil {
			return false
		}
		if strings.TrimSpace(stringValue(input.Metadata["resolved_path"])) == "" {
			return false
		}
		if strings.TrimSpace(stringValue(input.Metadata["source_job_id"])) == "" {
			return false
		}
	}
	return true
}

func (s *Service) executionTaskParams(job types.Job) map[string]any {
	taskParams := cloneTaskParams(job.Request.TaskParams)
	taskParams[taskParamExcludeDirs] = stringSliceToAny(defaultBrokerExcludedDirs())
	return taskParams
}

func defaultBrokerExcludedDirs() []string {
	return []string{
		".git",
		".broker",
		"__pycache__",
		".pytest_cache",
		".mypy_cache",
		".ruff_cache",
		".tox",
		".venv",
		"venv",
		"env",
		"node_modules",
		"site-packages",
		"build",
		"dist",
	}
}

func defaultModelProfiles() modelProfiles {
	cpu, p40, a100 := config.DefaultModelProfiles()
	return modelProfiles{
		cpu:  cpu,
		p40:  p40,
		a100: a100,
	}
}

func defaultRuntimeProfiles() runtimeProfiles {
	timeoutSeconds := config.DefaultRuntimeTimeoutSeconds()
	return runtimeProfiles{
		llamaCPP: runtimeConnection{TimeoutSeconds: timeoutSeconds},
		vllm:     runtimeConnection{TimeoutSeconds: timeoutSeconds},
		sglang:   runtimeConnection{TimeoutSeconds: timeoutSeconds},
	}
}

func (s *Service) applyExecutionProfileDefaults(profile types.ExecutionProfile) types.ExecutionProfile {
	switch strings.TrimSpace(profile.Tier) {
	case "cpu-rag-indexing":
		if strings.TrimSpace(profile.Runtime) == "" {
			profile.Runtime = "deterministic"
		}
		if strings.TrimSpace(profile.Model) == "" {
			profile.Model = s.models.cpu
		}
	case "p40-rag-compression":
		if strings.TrimSpace(profile.Runtime) == "" {
			profile.Runtime = "llama.cpp"
		}
		if strings.TrimSpace(profile.Model) == "" {
			profile.Model = s.models.p40
		}
	case "a100-reasoning":
		if strings.TrimSpace(profile.Runtime) == "" {
			profile.Runtime = "llama.cpp"
		}
		if strings.TrimSpace(profile.Model) == "" {
			profile.Model = s.models.a100
		}
	}
	return profile
}

func (s *Service) runtimeConnectionPlan(profile types.ExecutionProfile) map[string]any {
	runtimeName := strings.TrimSpace(profile.Runtime)
	connection := runtimeConnection{}
	switch runtimeName {
	case "llama.cpp":
		connection = s.runtimes.llamaCPP
	case "vllm":
		connection = s.runtimes.vllm
	case "sglang":
		connection = s.runtimes.sglang
	}
	return map[string]any{
		"base_url":        strings.TrimSpace(connection.BaseURL),
		"timeout_seconds": connection.TimeoutSeconds,
	}
}
