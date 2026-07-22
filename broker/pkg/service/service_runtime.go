package service

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func (s *Service) extractRuntimeDiagnostics(job types.Job, result types.Result) map[string]any {
	diagnostics := sanitizeRuntimeDiagnostics(runtimeDiagnosticsFromPayload(result.Payload))
	diagnostics = mergeRuntimeDiagnostics(diagnostics, sanitizeRuntimeDiagnostics(
		validationRuntimeDiagnostics(s.runRoot, job.ID, job.Artifacts),
	))
	diagnostics = mergeRuntimeDiagnostics(diagnostics, sanitizeRuntimeDiagnostics(
		artifactJSONForType(s.runRoot, job.ID, job.Artifacts, "runtime_diagnostics"),
	))
	return diagnostics
}

func mergeRuntimeDiagnostics(base, extra map[string]any) map[string]any {
	if len(base) == 0 && len(extra) == 0 {
		return nil
	}
	merged := make(map[string]any, len(base)+len(extra))
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range extra {
		merged[key] = value
	}
	return merged
}

func artifactJSONForType(runRoot, jobID string, artifacts []types.Artifact, artifactType string) map[string]any {
	for _, artifact := range artifacts {
		if artifact.ArtifactType != artifactType || strings.TrimSpace(artifact.Path) == "" {
			continue
		}
		var payload map[string]any
		if bytes, err := os.ReadFile(artifact.Path); err == nil && len(bytes) > 0 {
			if err := json.Unmarshal(bytes, &payload); err == nil {
				return payload
			}
		}
	}
	fallbackPath := filepath.Join(runRoot, jobID, artifactType+".json")
	var payload map[string]any
	if bytes, err := os.ReadFile(fallbackPath); err == nil && len(bytes) > 0 {
		if err := json.Unmarshal(bytes, &payload); err == nil {
			return payload
		}
	}
	return nil
}

func validationRuntimeDiagnostics(runRoot, jobID string, artifacts []types.Artifact) map[string]any {
	validation := artifactJSONForType(runRoot, jobID, artifacts, "validation_report")
	if len(validation) == 0 {
		return nil
	}
	diagnostics, _ := validation["runtime_diagnostics"].(map[string]any)
	return diagnostics
}

func runtimeDiagnosticsFromPayload(payload map[string]any) map[string]any {
	if payload == nil {
		return nil
	}
	retrieval, _ := payload["retrieval"].(map[string]any)
	provenance, _ := payload["provenance"].(map[string]any)
	runtime, _ := payload["runtime"].(map[string]any)
	if len(retrieval) == 0 && len(provenance) == 0 && len(runtime) == 0 {
		return nil
	}
	diagnostics := cloneMap(runtime)
	if diagnostics == nil {
		diagnostics = map[string]any{}
	}
	if value := strings.TrimSpace(stringValue(provenance["runtime_backend"])); value != "" {
		diagnostics["runtime_backend"] = value
		diagnostics["backend_name"] = value
	}
	if value := strings.TrimSpace(stringValue(provenance["model"])); value != "" {
		diagnostics["selected_model"] = value
		diagnostics["backend_detail"] = value
	}
	if value := strings.TrimSpace(stringValue(provenance["resource_tier"])); value != "" {
		diagnostics["resource_tier"] = value
	}
	if value := strings.TrimSpace(stringValue(retrieval["runtime_backend_mode"])); value != "" {
		diagnostics["backend_mode"] = value
	}
	if value := strings.TrimSpace(stringValue(retrieval["runtime_backend_detail"])); value != "" {
		diagnostics["backend_detail"] = value
	}
	if quality, ok := payload["quality"].(map[string]any); ok {
		for _, key := range []string{"result", "retrieval", "reranking", "synthesis", "answer_ready"} {
			if value, exists := quality[key]; exists {
				diagnostics[key] = value
			}
		}
	}
	return diagnostics
}

func sanitizeRuntimeDiagnostics(input map[string]any) map[string]any {
	if len(input) == 0 {
		return nil
	}
	output := map[string]any{}
	for _, key := range []string{
		"runtime_backend",
		"selected_model",
		"resource_tier",
		"backend_name",
		"backend_mode",
		"backend_detail",
		"llm_available",
		"endpoint_configured",
		"timeout_seconds",
		"last_error",
		"result",
		"retrieval",
		"reranking",
		"synthesis",
		"answer_ready",
		"index_fingerprint",
		"endpoint_health",
		"backend_failure_category",
		"worker_output_dir",
		"stdout_log",
		"stderr_log",
	} {
		value, ok := input[key]
		if !ok {
			continue
		}
		switch typed := value.(type) {
		case string:
			if strings.TrimSpace(typed) != "" {
				output[key] = typed
			}
		case bool:
			output[key] = typed
		case float64, int, int32, int64:
			output[key] = typed
		}
	}
	if attempts, ok := input["attempts"].([]any); ok {
		output["attempts"] = sanitizeRuntimeAttempts(attempts, 12)
	}
	if len(output) == 0 {
		return nil
	}
	return output
}

func sanitizeRuntimeAttempts(attempts []any, limit int) []any {
	if len(attempts) > limit {
		attempts = attempts[:limit]
	}
	out := make([]any, 0, len(attempts))
	for _, raw := range attempts {
		attempt, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		clean := map[string]any{}
		for _, key := range []string{
			"tier", "operation", "status", "slurm_job_id", "job_id",
			"gpu_count", "model_profile", "attempt", "failure_category", "escalation_reason",
		} {
			if value, exists := attempt[key]; exists {
				switch value.(type) {
				case string, bool, float64, int, int32, int64:
					clean[key] = value
				}
			}
		}
		if len(clean) > 0 {
			out = append(out, clean)
		}
	}
	return out
}

func isDegradedLocalExecution(diagnostics map[string]any) bool {
	mode := strings.ToLower(strings.TrimSpace(stringValue(diagnostics["backend_mode"])))
	if mode == "" || mode == "real" {
		return false
	}
	return true
}

func isDegradedResult(taskType string, result types.Result, diagnostics map[string]any) bool {
	if taskType == "inspect_repo" && result.SchemaName == "repo_inspection_v2" {
		quality, _ := result.Payload["quality"].(map[string]any)
		if stringValue(quality["result"]) == "failed" {
			return true
		}
		if hasDegradedRetrievalPolicy(result.Payload) {
			return true
		}
		// Evidence mode intentionally stops before synthesis. A GPU-backed
		// evidence pack is authoritative evidence, not degraded execution.
		if stringValue(result.Payload["mode"]) == "evidence" {
			return stringValue(quality["retrieval"]) != "gpu" || stringValue(quality["reranking"]) != "gpu"
		}
		return stringValue(quality["result"]) != "answer_ready"
	}
	if hasDegradedRetrievalPolicy(result.Payload) {
		return true
	}
	return isDegradedLocalExecution(diagnostics)
}

func hasDegradedRetrievalPolicy(payload map[string]any) bool {
	if payload == nil {
		return false
	}
	for _, raw := range []any{payload["warnings"], mapValue(payload["policy_signals"])["warnings"]} {
		for _, warning := range collectStringSlice(raw) {
			switch warning {
			case "LOCAL_RETRIEVAL_DEGRADED", "NO_REAL_RETRIEVAL_BACKEND", "IGNORED_PATH_RETRIEVAL_CONTAMINATION",
				"broker_local_retrieval_degraded", "broker_no_real_retrieval_backend", "broker_ignored_path_retrieval_contamination":
				return true
			}
		}
	}
	return false
}

func hasRetryRecommendation(job types.Job) bool {
	rec, ok := retryRecommendationFromResult(job)
	return ok && rec.Recommended
}

func deriveExecutionQuality(diagnostics map[string]any, retryRecommended bool) string {
	if retryRecommended {
		return "no_real_backend"
	}
	mode := strings.ToLower(strings.TrimSpace(stringValue(diagnostics["backend_mode"])))
	switch mode {
	case "real":
		return "real_local"
	case "heuristic", "fallback", "unavailable", "configured_local_llm":
		return "degraded_local"
	default:
		return ""
	}
}

func deriveResultExecutionQuality(taskType string, result types.Result, diagnostics map[string]any, retryRecommended bool) string {
	if taskType == "inspect_repo" && result.SchemaName == "repo_inspection_v2" {
		quality, _ := result.Payload["quality"].(map[string]any)
		switch stringValue(quality["result"]) {
		case "answer_ready":
			return "answer_ready"
		case "evidence_only":
			return "evidence_only"
		default:
			return "failed"
		}
	}
	if hasDegradedRetrievalPolicy(result.Payload) {
		if retryRecommended {
			return "no_real_backend"
		}
		return "degraded_local"
	}
	return deriveExecutionQuality(diagnostics, retryRecommended)
}

func retryRecommendationFromResult(job types.Job) (types.JobRetryRecommendation, bool) {
	if job.Result == nil {
		return types.JobRetryRecommendation{}, false
	}
	raw, ok := job.Result.Payload["broker_retry_recommendation"].(map[string]any)
	if !ok {
		return types.JobRetryRecommendation{}, false
	}
	profileMap, ok := raw["execution_profile"].(map[string]any)
	if !ok {
		return types.JobRetryRecommendation{}, false
	}
	return types.JobRetryRecommendation{
		JobID:             job.ID,
		Recommended:       raw["recommended"] == true,
		Reason:            stringValue(raw["reason"]),
		TaskType:          stringValue(raw["task_type"]),
		ExecutionProfile:  executionProfileFromMap(profileMap),
		PlacementHint:     placementHintFromMap(mapValue(raw["placement_hint"])),
		SourceResultError: job.ResultError,
	}, true
}

func brokerRetryRecommendation(job types.Job) map[string]any {
	profile := recommendedExecutionProfile(job.Request.ExecutionProfile)
	placement := recommendedPlacementHint(job, profile)
	return map[string]any{
		"recommended":       true,
		"reason":            "no_real_retrieval_backend",
		"task_type":         job.TaskType,
		"execution_profile": profile,
		"placement_hint":    placement,
	}
}

func executionProfileFromMap(value map[string]any) types.ExecutionProfile {
	return types.ExecutionProfile{
		Backend:    stringValue(value["backend"]),
		Tier:       stringValue(value["tier"]),
		Runtime:    stringValue(value["runtime"]),
		Model:      stringValue(value["model"]),
		QOS:        stringValue(value["qos"]),
		NodeList:   stringValue(value["nodelist"]),
		Constraint: stringValue(value["constraint"]),
		GPUCount:   intValue(value["gpu_count"]),
	}
}

func intValue(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
}

func placementHintFromMap(value map[string]any) types.PlacementHint {
	return types.PlacementHint{
		BackendPreference: stringValue(value["backend_preference"]),
		TierPreference:    stringValue(value["tier_preference"]),
		QOS:               stringValue(value["qos"]),
		NodeList:          stringValue(value["nodelist"]),
		Constraint:        stringValue(value["constraint"]),
		Preemptible:       boolValue(value["preemptible"]),
		Rationale:         stringValue(value["rationale"]),
	}
}

func recommendedPlacementHint(job types.Job, profile map[string]any) map[string]any {
	tier := stringValue(profile["tier"])
	hint := map[string]any{
		"backend_preference": stringValue(profile["backend"]),
		"tier_preference":    tier,
		"preemptible":        true,
	}
	if value := firstNonEmpty(stringValue(profile["nodelist"]), job.Request.ExecutionProfile.NodeList); value != "" {
		hint["nodelist"] = value
	}
	if value := firstNonEmpty(stringValue(profile["constraint"]), job.Request.ExecutionProfile.Constraint); value != "" {
		hint["constraint"] = value
	}
	switch tier {
	case "p40-rag-compression", "p40-retrieval", "p40-synthesis":
		hint["qos"] = firstNonEmpty(job.Request.ExecutionProfile.QOS, "scavenger")
		hint["rationale"] = "Prefer a warm P40 service before starting an on-demand reasoning service."
	case "v100-reasoning":
		hint["qos"] = firstNonEmpty(job.Request.ExecutionProfile.QOS, "scavenger")
		hint["rationale"] = "Use the four-GPU V100 reasoning profile after the warm P40 synthesis attempt failed."
	case "a100-reasoning", "a100-single", "a100-multigpu":
		hint["qos"] = firstNonEmpty(job.Request.ExecutionProfile.QOS, "scavenger")
		hint["rationale"] = "Use A100 only after recorded P40 and four-GPU V100 failures."
	default:
		hint["qos"] = firstNonEmpty(job.Request.ExecutionProfile.QOS, "normal")
		hint["rationale"] = "Use the broker-recommended local tier with non-blocking placement."
	}
	return hint
}

func mergePlacementHintIntoProfile(profile types.ExecutionProfile, hint types.PlacementHint) types.ExecutionProfile {
	if strings.TrimSpace(hint.BackendPreference) != "" {
		profile.Backend = hint.BackendPreference
	}
	if strings.TrimSpace(hint.TierPreference) != "" {
		profile.Tier = hint.TierPreference
	}
	if strings.TrimSpace(hint.QOS) != "" {
		profile.QOS = hint.QOS
	}
	if strings.TrimSpace(hint.NodeList) != "" {
		profile.NodeList = hint.NodeList
	}
	if strings.TrimSpace(hint.Constraint) != "" {
		profile.Constraint = hint.Constraint
	}
	return profile
}

func mergePlacementHintIntoTaskParams(taskParams map[string]any, hint types.PlacementHint) map[string]any {
	if taskParams == nil {
		taskParams = make(map[string]any)
	}
	if strings.TrimSpace(hint.BackendPreference) != "" {
		taskParams[taskParamRetryBackendPreference] = hint.BackendPreference
	}
	if strings.TrimSpace(hint.TierPreference) != "" {
		taskParams[taskParamRetryTierPreference] = hint.TierPreference
	}
	if strings.TrimSpace(hint.QOS) != "" {
		taskParams[taskParamRetryQOS] = hint.QOS
	}
	if strings.TrimSpace(hint.NodeList) != "" {
		taskParams[taskParamRetryNodeList] = hint.NodeList
	}
	if strings.TrimSpace(hint.Constraint) != "" {
		taskParams[taskParamRetryConstraint] = hint.Constraint
	}
	taskParams[taskParamRetryPreemptible] = hint.Preemptible
	if strings.TrimSpace(hint.Rationale) != "" {
		taskParams[taskParamRetryRationale] = hint.Rationale
	}
	return taskParams
}

func recommendedExecutionProfile(current types.ExecutionProfile) map[string]any {
	tier := strings.TrimSpace(current.Tier)
	nextTier := "p40-rag-compression"
	gpuCount := 1
	switch tier {
	case "cpu-rag-indexing":
		nextTier = "p40-rag-compression"
	case "p40-rag-compression":
		nextTier = "a100-reasoning"
	case "a100-reasoning":
		nextTier = "a100-reasoning"
	case "p40-retrieval":
		nextTier = "p40-synthesis"
	case "p40-synthesis":
		nextTier = "v100-reasoning"
		gpuCount = 4
	case "v100-reasoning":
		nextTier = "a100-single"
	case "a100-single":
		nextTier = "a100-multigpu"
		gpuCount = 4
	case "a100-multigpu":
		nextTier = "a100-multigpu"
		gpuCount = 4
	}
	backend := strings.TrimSpace(current.Backend)
	if backend == "" {
		backend = "slurm"
	}
	runtime := strings.TrimSpace(current.Runtime)
	if runtime == "" {
		runtime = "llama.cpp"
	}
	return map[string]any{
		"backend":    backend,
		"tier":       nextTier,
		"runtime":    runtime,
		"gpu_count":  gpuCount,
		"nodelist":   strings.TrimSpace(current.NodeList),
		"constraint": strings.TrimSpace(current.Constraint),
	}
}

func isRAGLikeTask(taskType string) bool {
	switch taskType {
	case "rag_compress", "debug_with_local_context", "summarize_logs", "inspect_repo", "propose_patch":
		return true
	default:
		return false
	}
}
