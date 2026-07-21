package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/msk-mind/local-ai-broker/broker/pkg/cache"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func main() {
	repo := os.Getenv("CACHE_KEY_HELPER_REPO")
	if repo == "" {
		fmt.Fprintln(os.Stderr, "CACHE_KEY_HELPER_REPO not set")
		os.Exit(2)
	}
	query := os.Getenv("CACHE_KEY_HELPER_QUERY")
	if query == "" {
		query = "Trace the retry_job service call chain"
	}
	mode := os.Getenv("CACHE_KEY_HELPER_MODE")
	if mode == "" {
		mode = "evidence"
	}
	classification := os.Getenv("CACHE_KEY_HELPER_CLASSIFICATION")
	if classification == "" {
		classification = "internal"
	}
	tier := os.Getenv("CACHE_KEY_HELPER_TIER")
	if tier == "" {
		tier = "cpu-rag-indexing"
	}
	runtime := os.Getenv("CACHE_KEY_HELPER_RUNTIME")
	if runtime == "" {
		runtime = "deterministic"
	}
	req := types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{Type: "repo", URI: "file://" + repo, Classification: classification},
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		TaskParams:   map[string]any{"query": query, "mode": mode},
		ExecutionProfile: types.ExecutionProfile{
			Tier:    tier,
			Runtime: runtime,
		},
	}
	key, cacheable, err := cache.KeyForRequest(req)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	payload := map[string]any{
		"cache_key":   key,
		"cacheable":   cacheable,
		"repo_path":   repo,
		"query":       query,
		"mode":        mode,
		"classification": classification,
		"tier":        tier,
		"runtime":     runtime,
		"schema_name": req.OutputSchema.Name,
		"task_type":   req.TaskType,
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(string(encoded))
}
