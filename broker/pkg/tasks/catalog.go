package tasks

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

const (
	MinInspectRepoFinalPackTokens = 2048
	MaxInspectRepoQueryBytes      = 2048
)

type Spec struct {
	Name        string
	HTTPPath    string
	SchemaName  string
	Description string
	Required    []string
	PromptField string
	Inputs      []string
	CacheExact  bool
}

var specs = []Spec{
	{
		Name:        "document_summary",
		SchemaName:  "document_summary_v1",
		Description: "Summarize a local document into a structured response.",
		Inputs:      []string{"file"},
		CacheExact:  true,
	},
	{
		Name:        "log_analysis",
		SchemaName:  "log_analysis_v1",
		Description: "Analyze local logs and return a structured summary.",
		Inputs:      []string{"file"},
		CacheExact:  true,
	},
	{
		Name:        "repo_summary",
		SchemaName:  "repo_summary_v1",
		Description: "Summarize a repository or directory into a structured response.",
		Inputs:      []string{"directory", "repo"},
		CacheExact:  true,
	},
	{
		Name:        "rag_compress",
		HTTPPath:    "/v1/rag/compressions",
		SchemaName:  "rag_evidence_pack_v1",
		Description: "Condense large local context into an agent-ready evidence pack with cited findings, answer-ready summary fields, and usage guidance for final responses.",
		Required:    []string{"query", "input_refs"},
		PromptField: "query",
		Inputs:      []string{"file", "repo", "log", "document", "artifact"},
		CacheExact:  true,
	},
	{
		Name:        "debug_with_local_context",
		HTTPPath:    "/v1/rag/debug-sessions",
		SchemaName:  "debug_evidence_pack_v1",
		Description: "Correlate logs, stack traces, tests, and repo paths into root-cause candidates with cited evidence and explicit next-step guidance.",
		Required:    []string{"problem", "input_refs"},
		PromptField: "problem",
		Inputs:      []string{"repo", "log", "artifact"},
	},
	{
		Name:        "summarize_logs",
		HTTPPath:    "/v1/logs:summarize",
		SchemaName:  "log_evidence_pack_v1",
		Description: "Cluster and compress large local logs into a concise failure summary with evidence refs that an agent can cite directly.",
		Required:    []string{"input_refs"},
		Inputs:      []string{"log"},
		CacheExact:  true,
	},
	{
		Name:        "inspect_repo",
		HTTPPath:    "/v1/repos:inspect",
		SchemaName:  "repo_inspection_v2",
		Description: "Inspect a local repository with GPU semantic retrieval, reranking, and cited synthesis; return lexical evidence only when GPU services are unavailable.",
		Required:    []string{"query", "input_refs"},
		PromptField: "query",
		Inputs:      []string{"repo", "directory"},
		CacheExact:  true,
	},
	{
		Name:        "propose_patch",
		HTTPPath:    "/v1/patches:propose",
		SchemaName:  "patch_proposal_pack_v1",
		Description: "Generate a small, evidence-backed patch proposal with cited rationale, scoped file targets, and validation guidance.",
		Required:    []string{"problem", "input_refs"},
		PromptField: "problem",
		Inputs:      []string{"repo", "artifact"},
	},
}

func Specs() []Spec {
	out := make([]Spec, len(specs))
	copy(out, specs)
	return out
}

func RAGAliasSpecs() []Spec {
	out := make([]Spec, 0, len(specs))
	for _, spec := range specs {
		if spec.HTTPPath != "" {
			out = append(out, spec)
		}
	}
	return out
}

func FindSpec(name string) (Spec, bool) {
	for _, spec := range specs {
		if spec.Name == name {
			return spec, true
		}
	}
	return Spec{}, false
}

func IsCacheableTask(name string) bool {
	spec, ok := FindSpec(name)
	return ok && spec.CacheExact
}

func CacheableTaskNames() []string {
	out := make([]string, 0, len(specs))
	for _, spec := range specs {
		if spec.CacheExact {
			out = append(out, spec.Name)
		}
	}
	return out
}

func DecodeSubmitRequest(raw json.RawMessage, spec Spec) (types.SubmitJobRequest, error) {
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return types.SubmitJobRequest{}, err
	}
	var req types.SubmitJobRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return types.SubmitJobRequest{}, err
	}
	req.TaskType = spec.Name
	req.OutputSchema = types.OutputSchemaRef{Name: spec.SchemaName}
	req.TaskParams = NormalizeTaskParams(req.TaskParams, payload, spec.Name)
	req = NormalizeSubmitRequest(req)
	if err := ValidateSubmitRequest(req); err != nil {
		return types.SubmitJobRequest{}, err
	}
	return req, nil
}

func NormalizeSubmitRequest(req types.SubmitJobRequest) types.SubmitJobRequest {
	if req.TaskType != "inspect_repo" {
		return req
	}
	params := make(map[string]any, len(req.TaskParams)+1)
	for key, value := range req.TaskParams {
		params[key] = value
	}
	mode := "auto"
	if value, ok := params["mode"].(string); ok && strings.TrimSpace(value) != "" {
		mode = strings.ToLower(strings.TrimSpace(value))
	}
	params["mode"] = mode
	req.TaskParams = params
	return req
}

func NormalizeTaskParams(taskParams map[string]any, payload map[string]any, taskType string) map[string]any {
	out := make(map[string]any, len(taskParams)+2)
	for k, v := range taskParams {
		out[k] = v
	}
	if value, ok := payload["query"]; ok {
		if text, ok := value.(string); ok && strings.TrimSpace(text) != "" {
			out["query"] = text
		}
	}
	if value, ok := payload["problem"]; ok {
		if text, ok := value.(string); ok && strings.TrimSpace(text) != "" {
			out["problem"] = text
		}
	}
	if value, ok := payload["retrieval_strategies"]; ok {
		if strategies, ok := value.([]any); ok && len(strategies) > 0 {
			out["retrieval_strategies"] = strategies
		}
	}
	if taskType == "inspect_repo" {
		mode := "auto"
		if value, ok := payload["mode"].(string); ok && strings.TrimSpace(value) != "" {
			mode = strings.ToLower(strings.TrimSpace(value))
		} else if value, ok := out["mode"].(string); ok && strings.TrimSpace(value) != "" {
			mode = strings.ToLower(strings.TrimSpace(value))
		}
		out["mode"] = mode
		if value, ok := payload["include_full_trace"].(bool); ok {
			out["include_full_trace"] = value
		}
	}
	if taskType == "debug_with_local_context" {
		if _, ok := out["problem"]; !ok {
			if text, ok := payload["problem"].(string); ok {
				out["problem"] = text
			}
		}
	}
	return out
}

// ValidateSubmitRequest enforces task contracts for both task-specific aliases
// and the generic job endpoint. Keeping this validation below the transport
// layer prevents callers from bypassing inspect_repo's answer-quality contract.
func ValidateSubmitRequest(req types.SubmitJobRequest) error {
	if req.TaskType != "inspect_repo" {
		return nil
	}
	if req.OutputSchema.Name != "repo_inspection_v2" {
		return fmt.Errorf("inspect_repo output_schema.name must be repo_inspection_v2")
	}
	query, _ := req.TaskParams["query"].(string)
	if strings.TrimSpace(query) == "" {
		return fmt.Errorf("inspect_repo query is required")
	}
	if len(query) > MaxInspectRepoQueryBytes {
		return fmt.Errorf("inspect_repo query cannot exceed %d UTF-8 bytes", MaxInspectRepoQueryBytes)
	}
	mode := "auto"
	if value, ok := req.TaskParams["mode"].(string); ok && strings.TrimSpace(value) != "" {
		mode = strings.ToLower(strings.TrimSpace(value))
	}
	switch mode {
	case "auto", "evidence", "answer":
	default:
		return fmt.Errorf("inspect_repo mode must be one of auto, evidence, or answer")
	}
	for key := range req.TaskParams {
		normalized := strings.ToLower(strings.TrimSpace(key))
		if strings.HasPrefix(normalized, "gpu_service") || normalized == "gpu_services" ||
			normalized == "service_endpoints" || normalized == "endpoint_selections" ||
			normalized == "runtime_services" || normalized == "service_selection" ||
			normalized == "index_cache_dir" || normalized == "repo_inspection_cache_path" {
			return fmt.Errorf("inspect_repo task parameter %q is broker-reserved", key)
		}
	}
	for name, value := range map[string]int{
		"retrieval_token_budget":         req.Constraints.RetrievalTokenBudget,
		"evidence_token_budget":          req.Constraints.EvidenceTokenBudget,
		"final_pack_token_budget":        req.Constraints.FinalPackTokenBudget,
		"synthesis_context_token_budget": req.Constraints.SynthesisContextTokenBudget,
	} {
		if value < 0 {
			return fmt.Errorf("inspect_repo %s must be non-negative", name)
		}
	}
	finalPackBudget := req.Constraints.FinalPackTokenBudget
	if finalPackBudget == 0 {
		finalPackBudget = req.Constraints.FinalEvidencePackBudget
	}
	if finalPackBudget > 0 && finalPackBudget < MinInspectRepoFinalPackTokens {
		return fmt.Errorf("inspect_repo final_pack_token_budget must be at least %d tokens when set", MinInspectRepoFinalPackTokens)
	}
	return nil
}
