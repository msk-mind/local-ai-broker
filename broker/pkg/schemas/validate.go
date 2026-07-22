package schemas

import (
	"fmt"
	"strings"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func ValidateResult(taskType, expectedSchema string, result types.Result) error {
	if expectedSchema == "" {
		return fmt.Errorf("expected schema is required")
	}
	if result.SchemaName != expectedSchema {
		return fmt.Errorf("schema name mismatch: expected %q, got %q", expectedSchema, result.SchemaName)
	}
	if result.SchemaVersion == "" {
		return fmt.Errorf("schema version is required")
	}
	if result.Payload == nil {
		return fmt.Errorf("payload is required")
	}

	switch expectedSchema {
	case "document_summary_v1":
		return validateDocumentSummary(taskType, result.Payload)
	case "log_analysis_v1":
		return validateLogAnalysis(taskType, result.Payload)
	case "repo_summary_v1":
		return validateRepoSummary(taskType, result.Payload)
	case "rag_evidence_pack_v1":
		return validateRAGEvidencePack(taskType, result.Payload)
	case "debug_evidence_pack_v1":
		return validateDebugEvidencePack(taskType, result.Payload)
	case "log_evidence_pack_v1":
		return validateLogEvidencePack(taskType, result.Payload)
	case "repo_inspection_v2":
		if result.SchemaVersion != "2.0.0" {
			return fmt.Errorf("schema repo_inspection_v2 requires version 2.0.0")
		}
		return validateRepoInspectionV2(taskType, result.Payload)
	case "patch_proposal_pack_v1":
		return validatePatchProposalPack(taskType, result.Payload)
	default:
		if _, ok := result.Payload["summary"].(string); !ok {
			return fmt.Errorf("payload.summary must be a string")
		}
		return nil
	}
}

func validateDocumentSummary(taskType string, payload map[string]any) error {
	if taskType != "document_summary" {
		return fmt.Errorf("schema document_summary_v1 is incompatible with task type %q", taskType)
	}
	if _, ok := payload["summary"].(string); !ok {
		return fmt.Errorf("payload.summary must be a string")
	}
	if keyPoints, exists := payload["key_points"]; exists {
		if _, ok := keyPoints.([]any); !ok {
			return fmt.Errorf("payload.key_points must be an array")
		}
	}
	if sourceMetadata, exists := payload["source_metadata"]; exists {
		if _, ok := sourceMetadata.(map[string]any); !ok {
			return fmt.Errorf("payload.source_metadata must be an object")
		}
	}
	return nil
}

func validateLogAnalysis(taskType string, payload map[string]any) error {
	if taskType != "log_analysis" {
		return fmt.Errorf("schema log_analysis_v1 is incompatible with task type %q", taskType)
	}
	if _, ok := payload["summary"].(string); !ok {
		return fmt.Errorf("payload.summary must be a string")
	}
	if findings, exists := payload["top_findings"]; exists {
		if _, ok := findings.([]any); !ok {
			return fmt.Errorf("payload.top_findings must be an array")
		}
	}
	if timeline, exists := payload["timeline"]; exists {
		if _, ok := timeline.([]any); !ok {
			return fmt.Errorf("payload.timeline must be an array")
		}
	}
	if nextSteps, exists := payload["suggested_next_steps"]; exists {
		if _, ok := nextSteps.([]any); !ok {
			return fmt.Errorf("payload.suggested_next_steps must be an array")
		}
	}
	return nil
}

func validateRepoSummary(taskType string, payload map[string]any) error {
	if taskType != "repo_summary" {
		return fmt.Errorf("schema repo_summary_v1 is incompatible with task type %q", taskType)
	}
	if _, ok := payload["summary"].(string); !ok {
		return fmt.Errorf("payload.summary must be a string")
	}
	if subsystems, exists := payload["subsystems"]; exists {
		if _, ok := subsystems.([]any); !ok {
			return fmt.Errorf("payload.subsystems must be an array")
		}
	}
	if entrypoints, exists := payload["entrypoints"]; exists {
		if _, ok := entrypoints.([]any); !ok {
			return fmt.Errorf("payload.entrypoints must be an array")
		}
	}
	if dependencies, exists := payload["dependencies"]; exists {
		if _, ok := dependencies.([]any); !ok {
			return fmt.Errorf("payload.dependencies must be an array")
		}
	}
	if risks, exists := payload["risks"]; exists {
		if _, ok := risks.([]any); !ok {
			return fmt.Errorf("payload.risks must be an array")
		}
	}
	return nil
}

func validateRAGEvidencePack(taskType string, payload map[string]any) error {
	if taskType != "rag_compress" {
		return fmt.Errorf("schema rag_evidence_pack_v1 is incompatible with task type %q", taskType)
	}
	return validateEvidencePackShape(payload, true)
}

func validateDebugEvidencePack(taskType string, payload map[string]any) error {
	if taskType != "debug_with_local_context" {
		return fmt.Errorf("schema debug_evidence_pack_v1 is incompatible with task type %q", taskType)
	}
	if _, ok := payload["problem"].(string); !ok {
		return fmt.Errorf("payload.problem must be a string")
	}
	if hypotheses, exists := payload["top_hypotheses"]; exists {
		if _, ok := hypotheses.([]any); !ok {
			return fmt.Errorf("payload.top_hypotheses must be an array")
		}
	}
	if evidence, exists := payload["evidence"]; exists {
		if _, ok := evidence.([]any); !ok {
			return fmt.Errorf("payload.evidence must be an array")
		}
	}
	return nil
}

func validateLogEvidencePack(taskType string, payload map[string]any) error {
	if taskType != "summarize_logs" {
		return fmt.Errorf("schema log_evidence_pack_v1 is incompatible with task type %q", taskType)
	}
	if _, ok := payload["summary"].(string); !ok {
		return fmt.Errorf("payload.summary must be a string")
	}
	if timeline, exists := payload["timeline"]; exists {
		if _, ok := timeline.([]any); !ok {
			return fmt.Errorf("payload.timeline must be an array")
		}
	}
	if clusters, exists := payload["clusters"]; exists {
		if _, ok := clusters.([]any); !ok {
			return fmt.Errorf("payload.clusters must be an array")
		}
	}
	if evidence, exists := payload["evidence"]; exists {
		if _, ok := evidence.([]any); !ok {
			return fmt.Errorf("payload.evidence must be an array")
		}
	}
	return nil
}

func validateRepoInspectionV2(taskType string, payload map[string]any) error {
	if taskType != "inspect_repo" {
		return fmt.Errorf("schema repo_inspection_v2 is incompatible with task type %q", taskType)
	}
	query, ok := payload["query"].(string)
	if !ok || strings.TrimSpace(query) == "" {
		return fmt.Errorf("payload.query must be a non-empty string")
	}
	mode, ok := payload["mode"].(string)
	if !ok {
		return fmt.Errorf("payload.mode must be a string")
	}
	mode = strings.ToLower(strings.TrimSpace(mode))
	switch mode {
	case "auto", "evidence", "answer":
	default:
		return fmt.Errorf("payload.mode must be one of auto, evidence, or answer")
	}

	evidence, ok := payload["evidence"].([]any)
	if !ok {
		return fmt.Errorf("payload.evidence must be an array")
	}
	if len(evidence) > 12 {
		return fmt.Errorf("payload.evidence cannot contain more than 12 chunks")
	}
	evidenceIDs := make(map[string]struct{}, len(evidence))
	for index, raw := range evidence {
		item, ok := raw.(map[string]any)
		if !ok {
			return fmt.Errorf("payload.evidence[%d] must be an object", index)
		}
		id, ok := item["id"].(string)
		id = strings.TrimSpace(id)
		if !ok || id == "" {
			return fmt.Errorf("payload.evidence[%d].id must be a non-empty string", index)
		}
		if _, exists := evidenceIDs[id]; exists {
			return fmt.Errorf("payload.evidence[%d].id %q is duplicated", index, id)
		}
		evidenceIDs[id] = struct{}{}
		sourceRefs, ok := item["source_refs"].([]any)
		if !ok || len(sourceRefs) == 0 {
			return fmt.Errorf("payload.evidence[%d].source_refs must be a non-empty array", index)
		}
		for refIndex, rawRef := range sourceRefs {
			ref, ok := rawRef.(map[string]any)
			if !ok || strings.TrimSpace(stringField(ref, "path")) == "" {
				return fmt.Errorf("payload.evidence[%d].source_refs[%d].path must be a non-empty string", index, refIndex)
			}
		}
	}

	findings, ok := payload["findings"].([]any)
	if !ok {
		return fmt.Errorf("payload.findings must be an array")
	}
	for index, raw := range findings {
		finding, ok := raw.(map[string]any)
		if !ok {
			return fmt.Errorf("payload.findings[%d] must be an object", index)
		}
		if strings.TrimSpace(stringField(finding, "summary")) == "" {
			return fmt.Errorf("payload.findings[%d].summary must be a non-empty string", index)
		}
		refs, ok := finding["evidence_refs"].([]any)
		if !ok || len(refs) == 0 {
			return fmt.Errorf("payload.findings[%d].evidence_refs must be a non-empty array", index)
		}
		for _, rawRef := range refs {
			ref, ok := rawRef.(string)
			if !ok || strings.TrimSpace(ref) == "" {
				return fmt.Errorf("payload.findings[%d].evidence_refs must contain strings", index)
			}
			if _, exists := evidenceIDs[strings.TrimSpace(ref)]; !exists {
				return fmt.Errorf("payload.findings[%d] references unknown evidence %q", index, ref)
			}
		}
	}

	quality, ok := payload["quality"].(map[string]any)
	if !ok {
		return fmt.Errorf("payload.quality must be an object")
	}
	resultState := strings.TrimSpace(stringField(quality, "result"))
	retrievalState := strings.TrimSpace(stringField(quality, "retrieval"))
	rerankingState := strings.TrimSpace(stringField(quality, "reranking"))
	synthesisState := strings.TrimSpace(stringField(quality, "synthesis"))
	answerReady, ok := quality["answer_ready"].(bool)
	if !ok {
		return fmt.Errorf("payload.quality.answer_ready must be a boolean")
	}
	if !oneOf(retrievalState, "gpu", "lexical_degraded", "failed") {
		return fmt.Errorf("payload.quality.retrieval has an invalid state")
	}
	if !oneOf(rerankingState, "gpu", "unavailable", "failed") {
		return fmt.Errorf("payload.quality.reranking has an invalid state")
	}
	if !oneOf(synthesisState, "gpu", "not_requested", "failed") {
		return fmt.Errorf("payload.quality.synthesis has an invalid state")
	}
	retrievalDiagnostics, ok := payload["retrieval"].(map[string]any)
	if !ok {
		return fmt.Errorf("payload.retrieval must be an object")
	}
	_ = retrievalDiagnostics
	runtime, ok := payload["runtime"].(map[string]any)
	if !ok {
		return fmt.Errorf("payload.runtime must be an object")
	}
	rawAttempts, ok := runtime["attempts"].([]any)
	if !ok {
		return fmt.Errorf("payload.runtime.attempts must be an array")
	}
	attempts, err := validateGPUAttempts(rawAttempts)
	if err != nil {
		return err
	}

	answer, answerPresent := payload["answer"]
	switch resultState {
	case "evidence_only":
		if answerReady {
			return fmt.Errorf("evidence-only result cannot be answer-ready")
		}
		if answerPresent {
			return fmt.Errorf("evidence-only result must omit payload.answer")
		}
		if len(findings) != 0 {
			return fmt.Errorf("evidence-only result cannot contain synthesized findings")
		}
	case "answer_ready":
		if !answerReady {
			return fmt.Errorf("answer-ready result must set quality.answer_ready")
		}
		answerText, ok := answer.(string)
		if !answerPresent || !ok || strings.TrimSpace(answerText) == "" {
			return fmt.Errorf("answer-ready result requires a non-empty payload.answer")
		}
		if len(findings) == 0 {
			return fmt.Errorf("answer-ready result requires cited findings")
		}
		if retrievalState != "gpu" || rerankingState != "gpu" || synthesisState != "gpu" {
			return fmt.Errorf("answer-ready result requires GPU retrieval, reranking, and synthesis")
		}
		if mode == "evidence" {
			return fmt.Errorf("evidence mode cannot return an answer-ready result")
		}
		if err := validateAnswerReadyGPUAttempts(attempts); err != nil {
			return err
		}
	case "failed":
		if mode != "answer" {
			return fmt.Errorf("failed inspection result is only valid in answer mode")
		}
		if answerReady || answerPresent || len(findings) != 0 {
			return fmt.Errorf("failed answer result must omit answer and synthesized findings")
		}
		if synthesisState != "failed" {
			return fmt.Errorf("failed answer result must record failed synthesis")
		}
		// An answer request may fail before any GPU operation is legitimate:
		// either there is no supported evidence, or no GPU retrieval/rerank
		// service is available and the worker returns lexical evidence only.
		noGPUFallback := retrievalState == "lexical_degraded" && rerankingState == "unavailable"
		if len(attempts) == 0 && !(len(evidence) == 0 && retrievalState == "failed") && !noGPUFallback {
			return fmt.Errorf("failed answer result requires complete runtime attempts")
		}
	default:
		return fmt.Errorf("payload.quality.result must be answer_ready, evidence_only, or failed")
	}

	if warnings, ok := payload["warnings"].([]any); !ok {
		return fmt.Errorf("payload.warnings must be an array")
	} else {
		for index, warning := range warnings {
			if _, ok := warning.(string); !ok {
				return fmt.Errorf("payload.warnings[%d] must be a string", index)
			}
		}
	}
	if _, ok := payload["provenance"].(map[string]any); !ok {
		return fmt.Errorf("payload.provenance must be an object")
	}
	return nil
}

func validateGPUAttempts(rawAttempts []any) ([]map[string]any, error) {
	attempts := make([]map[string]any, 0, len(rawAttempts))
	seenP40Synthesis := false
	seenV100Synthesis := false
	for index, raw := range rawAttempts {
		attempt, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("payload.runtime.attempts[%d] must be an object", index)
		}
		operation := strings.TrimSpace(stringField(attempt, "operation"))
		tier := strings.TrimSpace(stringField(attempt, "tier"))
		status := strings.TrimSpace(stringField(attempt, "status"))
		if !oneOf(operation, "semantic_retrieval", "rerank", "synthesis") {
			return nil, fmt.Errorf("payload.runtime.attempts[%d].operation is invalid", index)
		}
		if !oneOf(tier, "p40-retrieval", "p40-synthesis", "v100-reasoning", "a100-single", "a100-multigpu") {
			return nil, fmt.Errorf("payload.runtime.attempts[%d].tier is invalid", index)
		}
		if !oneOf(status, "succeeded", "failed", "degraded") {
			return nil, fmt.Errorf("payload.runtime.attempts[%d].status is invalid", index)
		}
		gpuCount, ok := integerField(attempt["gpu_count"])
		if !ok || gpuCount < 1 {
			return nil, fmt.Errorf("payload.runtime.attempts[%d].gpu_count must be positive", index)
		}
		expectedGPUCount := 1
		if tier == "v100-reasoning" || tier == "a100-multigpu" {
			expectedGPUCount = 4
		}
		if gpuCount != expectedGPUCount {
			return nil, fmt.Errorf("payload.runtime.attempts[%d] tier %s requires gpu_count=%d", index, tier, expectedGPUCount)
		}
		if operation == "semantic_retrieval" || operation == "rerank" {
			if tier != "p40-retrieval" {
				return nil, fmt.Errorf("payload.runtime.attempts[%d] retrieval and reranking require p40-retrieval", index)
			}
		} else {
			switch tier {
			case "p40-synthesis":
				seenP40Synthesis = true
			case "v100-reasoning":
				if !seenP40Synthesis {
					return nil, fmt.Errorf("payload.runtime.attempts[%d] records V100 before P40 synthesis", index)
				}
				seenV100Synthesis = true
			case "a100-single", "a100-multigpu":
				if !seenP40Synthesis || !seenV100Synthesis {
					return nil, fmt.Errorf("payload.runtime.attempts[%d] records A100 before P40 and V100 synthesis", index)
				}
			default:
				return nil, fmt.Errorf("payload.runtime.attempts[%d] has an invalid synthesis tier", index)
			}
		}
		attempts = append(attempts, attempt)
	}
	return attempts, nil
}

func validateAnswerReadyGPUAttempts(attempts []map[string]any) error {
	operationOrder := map[string]int{"semantic_retrieval": -1, "rerank": -1, "synthesis": -1}
	for index, attempt := range attempts {
		operation := stringField(attempt, "operation")
		if stringField(attempt, "status") != "succeeded" || operationOrder[operation] >= 0 {
			continue
		}
		if strings.TrimSpace(stringField(attempt, "model_profile")) == "" {
			return fmt.Errorf("answer-ready GPU attempt %s must record model_profile", operation)
		}
		if strings.TrimSpace(stringField(attempt, "slurm_job_id")) == "" {
			return fmt.Errorf("answer-ready GPU attempt %s must record slurm_job_id", operation)
		}
		operationOrder[operation] = index
	}
	if operationOrder["semantic_retrieval"] < 0 || operationOrder["rerank"] < 0 || operationOrder["synthesis"] < 0 {
		return fmt.Errorf("answer-ready result must record successful GPU retrieval, reranking, and synthesis attempts")
	}
	if !(operationOrder["semantic_retrieval"] < operationOrder["rerank"] && operationOrder["rerank"] < operationOrder["synthesis"]) {
		return fmt.Errorf("answer-ready GPU attempts must record retrieval, reranking, then synthesis in order")
	}
	return nil
}

func integerField(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int32:
		return int(typed), true
	case int64:
		return int(typed), true
	case float64:
		converted := int(typed)
		return converted, float64(converted) == typed
	default:
		return 0, false
	}
}

func stringField(value map[string]any, key string) string {
	text, _ := value[key].(string)
	return text
}

func oneOf(value string, expected ...string) bool {
	for _, candidate := range expected {
		if value == candidate {
			return true
		}
	}
	return false
}

func validatePatchProposalPack(taskType string, payload map[string]any) error {
	if taskType != "propose_patch" {
		return fmt.Errorf("schema patch_proposal_pack_v1 is incompatible with task type %q", taskType)
	}
	if _, ok := payload["summary"].(string); !ok {
		return fmt.Errorf("payload.summary must be a string")
	}
	if patches, exists := payload["patches"]; exists {
		if _, ok := patches.([]any); !ok {
			return fmt.Errorf("payload.patches must be an array")
		}
	}
	if validationSteps, exists := payload["validation_steps"]; exists {
		if _, ok := validationSteps.([]any); !ok {
			return fmt.Errorf("payload.validation_steps must be an array")
		}
	}
	return nil
}

func validateEvidencePackShape(payload map[string]any, requireQuery bool) error {
	if requireQuery {
		if _, ok := payload["query"].(string); !ok {
			return fmt.Errorf("payload.query must be a string")
		}
	}
	if retrieval, exists := payload["retrieval"]; exists {
		if _, ok := retrieval.(map[string]any); !ok {
			return fmt.Errorf("payload.retrieval must be an object")
		}
	}
	if retrievalPlan, exists := payload["retrieval_plan"]; exists {
		if _, ok := retrievalPlan.(map[string]any); !ok {
			return fmt.Errorf("payload.retrieval_plan must be an object")
		}
	}
	if retrievalTrace, exists := payload["retrieval_trace"]; exists {
		if _, ok := retrievalTrace.(map[string]any); !ok {
			return fmt.Errorf("payload.retrieval_trace must be an object")
		}
	}
	if evidence, exists := payload["evidence"]; exists {
		if _, ok := evidence.([]any); !ok {
			return fmt.Errorf("payload.evidence must be an array")
		}
	}
	if budget, exists := payload["budget"]; exists {
		if _, ok := budget.(map[string]any); !ok {
			return fmt.Errorf("payload.budget must be an object")
		}
	}
	return nil
}
