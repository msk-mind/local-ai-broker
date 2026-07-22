package gpuservice

// UnavailableCapabilities describes the configured GPU tiers when the control
// plane cannot provide a live registry snapshot. Keep this contract stable:
// MCP exposes it as part of the public capabilities response.
func UnavailableCapabilities() map[string]any {
	type tier struct {
		name       string
		role       string
		gpuType    string
		gpuCount   int
		operations []string
		min        int
		max        int
	}
	tiers := []tier{
		{name: "p40-retrieval", role: "retrieval", gpuType: "p40", gpuCount: 1, operations: []string{"embeddings", "index_status", "index_upsert", "faiss_search", "rerank"}, min: 1, max: 2},
		{name: "p40-synthesis", role: "synthesis", gpuType: "p40", gpuCount: 1, operations: []string{"chat_completions"}, min: 1, max: 2},
		{name: "v100-reasoning", role: "synthesis", gpuType: "v100", gpuCount: 4, operations: []string{"chat_completions"}, min: 0, max: 1},
		{name: "a100-single", role: "synthesis", gpuType: "a100", gpuCount: 1, operations: []string{"chat_completions"}, min: 0, max: 1},
		{name: "a100-multigpu", role: "synthesis", gpuType: "a100", gpuCount: 4, operations: []string{"chat_completions"}, min: 0, max: 1},
	}
	result := make([]map[string]any, 0, len(tiers))
	for _, item := range tiers {
		result = append(result, map[string]any{
			"tier": item.name, "role": item.role, "model_profile": "", "context_limit_tokens": 0,
			"gpu":                  map[string]any{"type": item.gpuType, "count": item.gpuCount},
			"supported_operations": item.operations, "min_replicas": item.min, "max_replicas": item.max,
			"active_replicas": 0, "starting_replicas": 0, "queue_state": map[string]int{}, "endpoints": []any{},
		})
	}
	return map[string]any{"enabled": false, "healthy": false, "tiers": result}
}
