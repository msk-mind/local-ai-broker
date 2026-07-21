package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/service"
	"github.com/msk-mind/local-ai-broker/broker/pkg/tasks"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

const protocolVersion = "2025-11-25"

const (
	defaultWaitForResultMax = 15 * time.Minute
	defaultWaitPollInterval = 50 * time.Millisecond
	minimumWaitPollInterval = 10 * time.Millisecond
)

type Server struct {
	service          *service.Service
	gpuCapabilities  func(context.Context) (any, error)
	defaultPrincipal auth.Principal
	mu               sync.RWMutex
	sessionPrincipal auth.Principal
}

type messageFraming int

const (
	framingContentLength messageFraming = iota
	framingNDJSON
)

func NewServer(svc *service.Service, defaultPrincipal auth.Principal) *Server {
	return NewServerWithGPUCapabilities(svc, defaultPrincipal, nil)
}

func NewServerWithGPUCapabilities(
	svc *service.Service,
	defaultPrincipal auth.Principal,
	provider func(context.Context) (any, error),
) *Server {
	return &Server{
		service:          svc,
		gpuCapabilities:  provider,
		defaultPrincipal: defaultPrincipal,
	}
}

type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type response struct {
	JSONRPC string     `json:"jsonrpc"`
	ID      any        `json:"id,omitempty"`
	Result  any        `json:"result,omitempty"`
	Error   *respError `json:"error,omitempty"`
}

type respError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type toolsCallParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type initializeParams struct {
	Auth *struct {
		Actor string `json:"actor"`
		Role  string `json:"role"`
	} `json:"auth,omitempty"`
}

type waitForResultOptions struct {
	WaitForResult  bool `json:"wait_for_result,omitempty"`
	MaxWaitSeconds int  `json:"max_wait_seconds,omitempty"`
	PollIntervalMS int  `json:"poll_interval_ms,omitempty"`
}

type toolDefinition struct {
	name        string
	description string
	inputSchema map[string]any
}

type fetchJobLogsArgs struct {
	JobID    string `json:"job_id"`
	Stream   string `json:"stream"`
	MaxBytes int    `json:"max_bytes"`
}

func (s *Server) ServeStdio(ctx context.Context, in io.Reader, out io.Writer) error {
	reader := bufio.NewReader(in)
	writer := bufio.NewWriter(out)
	defer writer.Flush()

	for {
		payload, framing, err := readMessage(reader)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		var req request
		if err := json.Unmarshal(payload, &req); err != nil {
			if err := writeMessage(writer, response{
				JSONRPC: "2.0",
				Error:   &respError{Code: -32700, Message: "parse error"},
			}, framing); err != nil {
				return err
			}
			continue
		}

		resp := s.handleRequest(ctx, req)
		if req.ID == nil {
			continue
		}
		if err := writeMessage(writer, resp, framing); err != nil {
			return err
		}
	}
}

func (s *Server) handleRequest(ctx context.Context, req request) response {
	resp := response{
		JSONRPC: "2.0",
		ID:      req.ID,
	}

	switch req.Method {
	case "initialize":
		if _, err := s.initializePrincipal(ctx, req.Params); err != nil {
			resp.Error = &respError{Code: -32001, Message: err.Error()}
			return resp
		}
		resp.Result = map[string]any{
			"protocolVersion": protocolVersion,
			"serverInfo": map[string]any{
				"name":    "local-ai-compute-broker",
				"version": "0.1.0",
			},
			"capabilities": map[string]any{
				"tools": map[string]any{
					"listChanged": false,
				},
			},
			"instructions": `When this broker is active, use the local-ai-compute-broker MCP server by default for compute-intensive and context-heavy work.

Use broker tools first for:
- repository inspection
- log analysis
- retrieval/compression of local context
- local debugging passes over code or logs
- patch-oriented investigation when broker evidence is useful
- submitting jobs to local or cluster compute

Prefer inspect_repo, summarize_logs, debug_with_local_context, rag_compress, and propose_patch over direct ad hoc repository inspection when the task can be broker-mediated.

Only skip the broker when:
- the task is trivial and does not benefit from local retrieval or compression
- the broker is unavailable
- the broker result is degraded, non-authoritative, or clearly insufficient for the task

If the broker returns degraded retrieval quality, treat it as non-authoritative and verify directly in the repo before making strong claims.`,
		}
	case "ping":
		resp.Result = map[string]any{}
	case "tools/list":
		_, err := s.contextWithPrincipal(ctx)
		if err != nil {
			resp.Error = &respError{Code: -32001, Message: err.Error()}
			return resp
		}
		resp.Result = map[string]any{
			"tools": toolDefinitions(),
		}
	case "tools/call":
		var err error
		ctx, err = s.contextWithPrincipal(ctx)
		if err != nil {
			resp.Error = &respError{Code: -32001, Message: err.Error()}
			return resp
		}
		var params toolsCallParams
		if err := json.Unmarshal(req.Params, &params); err != nil {
			resp.Error = &respError{Code: -32602, Message: "invalid tools/call params"}
			return resp
		}
		result, err := s.callTool(ctx, params)
		if err != nil {
			resp.Error = &respError{Code: -32000, Message: err.Error()}
			return resp
		}
		resp.Result = result
	default:
		resp.Error = &respError{Code: -32601, Message: "method not found"}
	}

	return resp
}

func (s *Server) initializePrincipal(ctx context.Context, params json.RawMessage) (auth.Principal, error) {
	if principal := auth.PrincipalFromContext(ctx); principal.Actor != "" {
		s.setSessionPrincipal(principal)
		return principal, nil
	}

	var initParams initializeParams
	if len(params) > 0 {
		if err := json.Unmarshal(params, &initParams); err != nil {
			return auth.Principal{}, fmt.Errorf("invalid initialize params")
		}
	}
	if initParams.Auth != nil && strings.TrimSpace(initParams.Auth.Actor) != "" {
		principal := auth.Principal{
			Actor: strings.TrimSpace(initParams.Auth.Actor),
			Role:  defaultRole(initParams.Auth.Role),
		}
		s.setSessionPrincipal(principal)
		return principal, nil
	}
	if s.defaultPrincipal.Actor != "" {
		s.setSessionPrincipal(s.defaultPrincipal)
		return s.defaultPrincipal, nil
	}
	return auth.Principal{}, errors.New("mcp session identity is required; provide initialize.params.auth or BROKER_MCP_ACTOR")
}

func (s *Server) contextWithPrincipal(ctx context.Context) (context.Context, error) {
	if principal := auth.PrincipalFromContext(ctx); principal.Actor != "" {
		return ctx, nil
	}
	s.mu.RLock()
	principal := s.sessionPrincipal
	s.mu.RUnlock()
	if principal.Actor == "" && s.defaultPrincipal.Actor != "" {
		principal = s.defaultPrincipal
	}
	if principal.Actor == "" {
		return nil, errors.New("mcp session is not initialized with an identity")
	}
	return auth.WithPrincipal(ctx, principal), nil
}

func (s *Server) setSessionPrincipal(principal auth.Principal) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessionPrincipal = principal
}

func defaultRole(role string) string {
	role = strings.TrimSpace(role)
	if role == "" {
		return "user"
	}
	return role
}

func (s *Server) callTool(ctx context.Context, params toolsCallParams) (map[string]any, error) {
	switch params.Name {
	case "submit_local_job":
		return s.callSubmitJobTool(ctx, params.Arguments, "submit_local_job")
	case "submit_parallel_jobs":
		return s.callSubmitParallelJobsTool(ctx, params.Arguments)
	case "get_root_job_status":
		return s.callStringArgTool(params.Arguments, "root_job_id", "get_root_job_status", func(rootJobID string) (any, error) {
			return s.service.GetRootJobStatus(ctx, rootJobID)
		})
	case "retry_failed_root_shards":
		return callJSONTool(params.Arguments, "retry_failed_root_shards", func(req types.RetryFailedRootShardsRequest) bool {
			return strings.TrimSpace(req.RootJobID) != ""
		}, func(req types.RetryFailedRootShardsRequest) (any, error) {
			return s.service.RetryFailedRootShards(ctx, req)
		})
	case "rag_compress", "debug_with_local_context", "summarize_logs", "inspect_repo", "propose_patch":
		return s.callRAGTool(ctx, params.Name, params.Arguments)
	case "release_deferred_root_chunks":
		return callJSONTool(params.Arguments, "release_deferred_root_chunks", func(req types.ReleaseDeferredRootChunksRequest) bool {
			return strings.TrimSpace(req.RootJobID) != ""
		}, func(req types.ReleaseDeferredRootChunksRequest) (any, error) {
			return s.service.ReleaseDeferredRootChunks(ctx, req)
		})
	case "get_job_status":
		return s.callStringArgTool(params.Arguments, "job_id", "get_job_status", func(jobID string) (any, error) {
			return s.service.GetJob(ctx, jobID)
		})
	case "fetch_result":
		return s.callStringArgTool(params.Arguments, "job_id", "fetch_result", func(jobID string) (any, error) {
			return s.service.GetReleasedResult(ctx, jobID)
		})
	case "get_retry_recommendation":
		return s.callStringArgTool(params.Arguments, "job_id", "get_retry_recommendation", func(jobID string) (any, error) {
			return s.service.GetJobRetryRecommendation(ctx, jobID)
		})
	case "retry_with_recommended_profile":
		return s.callStringArgTool(params.Arguments, "job_id", "retry_with_recommended_profile", func(jobID string) (any, error) {
			return s.service.RetryJobWithRecommendation(ctx, jobID)
		})
	case "fetch_job_logs":
		return callJSONTool(params.Arguments, "fetch_job_logs", func(args fetchJobLogsArgs) bool {
			return strings.TrimSpace(args.JobID) != ""
		}, func(args fetchJobLogsArgs) (any, error) {
			return s.service.GetJobLogs(ctx, args.JobID, args.Stream, args.MaxBytes)
		})
	case "cancel_job":
		return s.callStringArgTool(params.Arguments, "job_id", "cancel_job", func(jobID string) (any, error) {
			return s.service.CancelJob(ctx, jobID)
		})
	case "list_local_capabilities":
		return toolResult(s.capabilitiesPayload(ctx)), nil
	default:
		return nil, fmt.Errorf("unknown tool: %s", params.Name)
	}
}

func (s *Server) callSubmitJobTool(ctx context.Context, raw json.RawMessage, toolName string) (map[string]any, error) {
	var req types.SubmitJobRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("invalid %s arguments", toolName)
	}
	return s.submitAndMaybeWait(ctx, raw, toolName, func(submitCtx context.Context) (types.SubmitJobResponse, error) {
		return s.service.SubmitJob(submitCtx, req)
	})
}

func (s *Server) callStringArgTool(raw json.RawMessage, field, toolName string, call func(string) (any, error)) (map[string]any, error) {
	value, err := decodeRequiredStringArg(raw, field)
	if err != nil {
		return nil, fmt.Errorf("invalid %s arguments", toolName)
	}
	payload, err := call(value)
	if err != nil {
		return nil, err
	}
	return toolResult(payload), nil
}

func callJSONTool[T any](raw json.RawMessage, toolName string, valid func(T) bool, call func(T) (any, error)) (map[string]any, error) {
	var req T
	if err := json.Unmarshal(raw, &req); err != nil || (valid != nil && !valid(req)) {
		return nil, fmt.Errorf("invalid %s arguments", toolName)
	}
	payload, err := call(req)
	if err != nil {
		return nil, err
	}
	return toolResult(payload), nil
}

func (s *Server) callSubmitParallelJobsTool(ctx context.Context, raw json.RawMessage) (map[string]any, error) {
	var req types.SubmitParallelJobsRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("invalid submit_parallel_jobs arguments")
	}
	resp, err := s.service.SubmitParallelJobs(ctx, req)
	if err != nil {
		return nil, err
	}
	return toolResult(resp), nil
}

func (s *Server) callRAGTool(ctx context.Context, toolName string, raw json.RawMessage) (map[string]any, error) {
	spec, ok := tasks.FindSpec(toolName)
	if !ok || spec.HTTPPath == "" {
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
	req, err := tasks.DecodeSubmitRequest(raw, spec)
	if err != nil {
		return nil, fmt.Errorf("invalid %s arguments", toolName)
	}
	return s.submitAndMaybeWait(ctx, raw, toolName, func(submitCtx context.Context) (types.SubmitJobResponse, error) {
		return s.service.SubmitJob(submitCtx, req)
	})
}

func (s *Server) submitAndMaybeWait(ctx context.Context, raw json.RawMessage, toolName string, submit func(context.Context) (types.SubmitJobResponse, error)) (map[string]any, error) {
	waitOpts, err := decodeWaitForResultOptions(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid %s arguments", toolName)
	}
	submitCtx := ctx
	if waitOpts.WaitForResult {
		submitCtx = service.WithPreferInlineLocalRelease(ctx)
	}
	resp, err := submit(submitCtx)
	if err != nil {
		return nil, err
	}
	if !waitOpts.WaitForResult {
		return toolResult(resp), nil
	}
	if resp.ReleasedResult != nil {
		return toolResult(*resp.ReleasedResult), nil
	}
	release, err := s.waitForReleasedResult(ctx, resp.JobID, waitOpts)
	if err != nil {
		return nil, err
	}
	return toolResult(release), nil
}

func decodeWaitForResultOptions(raw json.RawMessage) (waitForResultOptions, error) {
	var opts waitForResultOptions
	if len(raw) == 0 {
		return opts, nil
	}
	if err := json.Unmarshal(raw, &opts); err != nil {
		return waitForResultOptions{}, err
	}
	if opts.MaxWaitSeconds < 0 || opts.PollIntervalMS < 0 {
		return waitForResultOptions{}, fmt.Errorf("wait values must be non-negative")
	}
	return opts, nil
}

func (s *Server) waitForReleasedResult(ctx context.Context, jobID string, opts waitForResultOptions) (types.JobResultRelease, error) {
	maxWait := defaultWaitForResultMax
	if opts.MaxWaitSeconds > 0 {
		maxWait = time.Duration(opts.MaxWaitSeconds) * time.Second
	}
	waitCtx, cancel := context.WithTimeout(ctx, maxWait)
	defer cancel()

	pollInterval := defaultWaitPollInterval
	if opts.PollIntervalMS > 0 {
		pollInterval = time.Duration(opts.PollIntervalMS) * time.Millisecond
		if pollInterval < minimumWaitPollInterval {
			pollInterval = minimumWaitPollInterval
		}
	}

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		release, err := s.service.GetReleasedResult(service.WithSkipInspectRepoResultProbe(waitCtx), jobID)
		if err != nil {
			return types.JobResultRelease{}, err
		}
		if release.Result != nil || isTerminalJobState(release.State) {
			return release, nil
		}

		select {
		case <-waitCtx.Done():
			return types.JobResultRelease{}, fmt.Errorf("timed out waiting for job %q result", jobID)
		case <-ticker.C:
		}
	}
}

func isTerminalJobState(state types.JobState) bool {
	switch state {
	case types.JobStateSucceeded, types.JobStateFailed, types.JobStateCancelled, types.JobStatePreempted, types.JobStateTimedOut:
		return true
	default:
		return false
	}
}

func decodeRequiredStringArg(raw json.RawMessage, field string) (string, error) {
	var payload map[string]string
	if err := json.Unmarshal(raw, &payload); err != nil {
		return "", err
	}
	value := strings.TrimSpace(payload[field])
	if value == "" {
		return "", fmt.Errorf("%s is required", field)
	}
	return value, nil
}

func toolResult(payload any) map[string]any {
	textBytes, _ := json.MarshalIndent(payload, "", "  ")
	return map[string]any{
		"content": []map[string]any{
			{
				"type": "text",
				"text": string(textBytes),
			},
		},
		"structuredContent": payload,
	}
}

func toolDefinitions() []map[string]any {
	ragSpecs := tasks.RAGAliasSpecs()
	defs := make([]map[string]any, 0, len(ragSpecs)+12)
	for _, spec := range ragSpecs {
		properties := map[string]any{
			"input_refs":           inputRefsSchema(),
			"retrieval_strategies": retrievalStrategiesSchema(),
			"task_params":          map[string]any{"type": "object"},
			"constraints":          ragConstraintsSchema(),
			"execution_profile":    ragExecutionProfileSchema(),
			"idempotency_key":      map[string]any{"type": "string"},
			"wait_for_result":      map[string]any{"type": "boolean"},
			"max_wait_seconds":     map[string]any{"type": "integer"},
			"poll_interval_ms":     map[string]any{"type": "integer"},
		}
		if spec.PromptField != "" {
			properties[spec.PromptField] = map[string]any{"type": "string"}
		}
		if spec.Name == "inspect_repo" {
			properties["query"] = map[string]any{"type": "string", "maxLength": tasks.MaxInspectRepoQueryBytes}
			properties["constraints"] = inspectRepoConstraintsSchema()
			properties["mode"] = map[string]any{
				"type":    "string",
				"enum":    []string{"auto", "evidence", "answer"},
				"default": "auto",
			}
			properties["include_full_trace"] = map[string]any{"type": "boolean", "default": false}
		}
		defs = append(defs, ragToolDefinition(spec.Name, spec.Description, spec.Required, properties))
	}
	for _, spec := range coreToolDefinitions() {
		defs = append(defs, map[string]any{
			"name":        spec.name,
			"description": spec.description,
			"inputSchema": spec.inputSchema,
		})
	}
	return defs
}

func coreToolDefinitions() []toolDefinition {
	return []toolDefinition{
		{
			name:        "submit_local_job",
			description: "Submit a local broker task. Prefer the task-specific RAG tools and set wait_for_result=true when you want an answer-ready result instead of only a job id.",
			inputSchema: map[string]any{
				"type":     "object",
				"required": []string{"task_type", "input_refs", "output_schema"},
				"properties": map[string]any{
					"task_type":         map[string]any{"type": "string"},
					"input_refs":        inputRefsSchema(),
					"task_params":       map[string]any{"type": "object"},
					"constraints":       ragConstraintsSchema(),
					"execution_profile": ragExecutionProfileSchema(),
					"orchestration":     orchestrationSchema(),
					"output_schema":     outputSchemaRefSchema(),
					"idempotency_key":   map[string]any{"type": "string"},
					"wait_for_result":   map[string]any{"type": "boolean"},
					"max_wait_seconds":  map[string]any{"type": "integer"},
					"poll_interval_ms":  map[string]any{"type": "integer"},
				},
			},
		},
		{
			name:        "submit_parallel_jobs",
			description: "Submit many child jobs under one logical root investigation.",
			inputSchema: map[string]any{
				"type":     "object",
				"required": []string{"task_type", "children", "output_schema"},
				"properties": map[string]any{
					"task_type":         map[string]any{"type": "string"},
					"task_params":       map[string]any{"type": "object"},
					"constraints":       map[string]any{"type": "object"},
					"execution_profile": map[string]any{"type": "object"},
					"output_schema":     map[string]any{"type": "object"},
					"root_job_id":       map[string]any{"type": "string"},
					"parent_job_id":     map[string]any{"type": "string"},
					"strategy":          map[string]any{"type": "string"},
					"children":          map[string]any{"type": "array"},
					"reducer":           map[string]any{"type": "object"},
				},
			},
		},
		{
			name:        "get_job_status",
			description: "Retrieve the current state of a previously submitted local job.",
			inputSchema: simpleJobIDSchema(),
		},
		{
			name:        "get_root_job_status",
			description: "Retrieve aggregate status for a root investigation spanning many child jobs.",
			inputSchema: rootJobIDSchema(),
		},
		{
			name:        "retry_failed_root_shards",
			description: "Retry only the currently failed shards for a root investigation, optionally resubmitting its reducer.",
			inputSchema: map[string]any{
				"type":     "object",
				"required": []string{"root_job_id"},
				"properties": map[string]any{
					"root_job_id":       map[string]any{"type": "string"},
					"include_cancelled": map[string]any{"type": "boolean"},
					"resubmit_reducer":  map[string]any{"type": "boolean"},
				},
			},
		},
		{
			name:        "release_deferred_root_chunks",
			description: "Force immediate release of deferred child chunks for a root investigation, optionally bounded to a one-shot number of batches.",
			inputSchema: map[string]any{
				"type":     "object",
				"required": []string{"root_job_id"},
				"properties": map[string]any{
					"root_job_id":            map[string]any{"type": "string"},
					"max_additional_batches": map[string]any{"type": "integer"},
				},
			},
		},
		{
			name:        "fetch_result",
			description: "Fetch the structured result and artifacts for a completed local job.",
			inputSchema: simpleJobIDSchema(),
		},
		{
			name:        "get_retry_recommendation",
			description: "Return the broker-generated retry recommendation for a completed local job, if one exists.",
			inputSchema: simpleJobIDSchema(),
		},
		{
			name:        "retry_with_recommended_profile",
			description: "Submit a new job using the broker-recommended execution profile from a completed local job.",
			inputSchema: simpleJobIDSchema(),
		},
		{
			name:        "fetch_job_logs",
			description: "Fetch redacted stdout and stderr from a local worker run.",
			inputSchema: map[string]any{
				"type":     "object",
				"required": []string{"job_id"},
				"properties": map[string]any{
					"job_id":    map[string]any{"type": "string"},
					"stream":    map[string]any{"type": "string", "enum": []string{"stdout", "stderr", "combined"}},
					"max_bytes": map[string]any{"type": "integer"},
				},
			},
		},
		{
			name:        "cancel_job",
			description: "Cancel a queued or running local job.",
			inputSchema: simpleJobIDSchema(),
		},
		{
			name:        "list_local_capabilities",
			description: "List broker task types, schemas, execution modes, and backends so an agent can choose the smallest high-signal tool instead of ad hoc local inspection.",
			inputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}
}

func simpleJobIDSchema() map[string]any {
	return requiredStringFieldSchema("job_id")
}

func rootJobIDSchema() map[string]any {
	return requiredStringFieldSchema("root_job_id")
}

func requiredStringFieldSchema(field string) map[string]any {
	return map[string]any{
		"type":     "object",
		"required": []string{field},
		"properties": map[string]any{
			field: map[string]any{"type": "string"},
		},
	}
}

func (s *Server) capabilitiesPayload(ctx context.Context) map[string]any {
	taskSpecs := tasks.Specs()
	taskTypes := make([]map[string]any, 0, len(taskSpecs))
	coreTools := coreToolDefinitions()
	tools := make([]string, 0, len(taskSpecs)+len(coreTools))
	for _, spec := range taskSpecs {
		taskTypes = append(taskTypes, map[string]any{
			"name":   spec.Name,
			"schema": spec.SchemaName,
			"inputs": spec.Inputs,
		})
		if spec.HTTPPath != "" {
			tools = append(tools, spec.Name)
		}
	}
	for _, tool := range coreTools {
		tools = append(tools, tool.name)
	}
	payload := map[string]any{
		"task_types": taskTypes,
		"backends": []map[string]any{
			{
				"name":         "slurm",
				"modes":        []string{"stub", "command"},
				"default_mode": "stub",
			},
			{
				"name":         "local",
				"modes":        []string{"stub", "command"},
				"default_mode": "command",
			},
		},
		"tools": tools,
		"orchestration": map[string]any{
			"independent_parallel_jobs":  true,
			"parent_child_metadata":      true,
			"client_orchestrated_fanout": true,
			"aggregator_jobs":            true,
			"server_side_dag_scheduler":  false,
		},
		"cache": map[string]any{
			"exact_match_tasks": tasks.CacheableTaskNames(),
		},
	}
	payload["gpu_services"] = unavailableGPUServiceCapabilities()
	if s.gpuCapabilities != nil {
		if snapshot, err := s.gpuCapabilities(ctx); err == nil {
			payload["gpu_services"] = snapshot
		} else {
			payload["gpu_services"] = map[string]any{
				"enabled": false,
				"healthy": false,
				"error":   err.Error(),
				"tiers":   unavailableGPUServiceCapabilities()["tiers"],
			}
		}
	}
	return payload
}

func unavailableGPUServiceCapabilities() map[string]any {
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
			"tier":                 item.name,
			"role":                 item.role,
			"model_profile":        "",
			"context_limit_tokens": 0,
			"gpu":                  map[string]any{"type": item.gpuType, "count": item.gpuCount},
			"supported_operations": item.operations,
			"min_replicas":         item.min,
			"max_replicas":         item.max,
			"active_replicas":      0,
			"starting_replicas":    0,
			"queue_state":          map[string]int{},
			"endpoints":            []any{},
		})
	}
	return map[string]any{
		"enabled": false,
		"healthy": false,
		"tiers":   result,
	}
}

func ragToolDefinition(name, description string, required []string, properties map[string]any) map[string]any {
	return map[string]any{
		"name":        name,
		"description": description,
		"inputSchema": map[string]any{
			"type":                 "object",
			"required":             required,
			"additionalProperties": false,
			"properties":           properties,
		},
	}
}

func inputRefsSchema() map[string]any {
	return map[string]any{
		"type": "array",
		"items": map[string]any{
			"type":                 "object",
			"required":             []string{"type", "uri"},
			"additionalProperties": false,
			"properties": map[string]any{
				"type": map[string]any{
					"type": "string",
					"enum": []string{"file", "repo", "log", "document", "artifact", "directory"},
				},
				"uri":            map[string]any{"type": "string"},
				"content_hash":   map[string]any{"type": "string"},
				"classification": map[string]any{"type": "string"},
				"metadata":       map[string]any{"type": "object"},
			},
		},
	}
}

func orchestrationSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"parent_job_id":      map[string]any{"type": "string"},
			"root_job_id":        map[string]any{"type": "string"},
			"strategy":           map[string]any{"type": "string"},
			"shard_key":          map[string]any{"type": "string"},
			"shard_index":        map[string]any{"type": "integer"},
			"shard_count":        map[string]any{"type": "integer"},
			"aggregation_key":    map[string]any{"type": "string"},
			"depends_on_job_ids": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
		},
	}
}

func outputSchemaRefSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"required":             []string{"name"},
		"additionalProperties": false,
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}
}

func ragConstraintsSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"retrieval_token_budget":         map[string]any{"type": "integer", "minimum": 1},
			"evidence_token_budget":          map[string]any{"type": "integer", "minimum": 1},
			"final_pack_token_budget":        map[string]any{"type": "integer", "minimum": 1},
			"synthesis_context_token_budget": map[string]any{"type": "integer", "minimum": 1},
			"max_runtime_seconds":            map[string]any{"type": "integer"},
			"confidentiality":                map[string]any{"type": "string"},
		},
	}
}

func inspectRepoConstraintsSchema() map[string]any {
	schema := ragConstraintsSchema()
	properties := schema["properties"].(map[string]any)
	properties["final_pack_token_budget"] = map[string]any{
		"type": "integer", "minimum": tasks.MinInspectRepoFinalPackTokens,
	}
	return schema
}

func retrievalStrategiesSchema() map[string]any {
	return map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "string",
			"enum": []string{"ripgrep", "bm25", "tree_sitter", "embeddings", "stack_trace_path", "git_diff_history", "artifact_context"},
		},
	}
}

func ragExecutionProfileSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"backend": map[string]any{"type": "string"},
			"tier": map[string]any{
				"type": "string",
				"enum": []string{
					"cpu-rag-indexing", "p40-rag-compression", "a100-reasoning",
					"p40-retrieval", "p40-synthesis", "v100-reasoning", "a100-single", "a100-multigpu",
				},
			},
			"model": map[string]any{"type": "string"},
			"runtime": map[string]any{
				"type": "string",
				"enum": []string{"llama.cpp", "vllm", "sglang", "deterministic"},
			},
			"qos":        map[string]any{"type": "string"},
			"nodelist":   map[string]any{"type": "string"},
			"constraint": map[string]any{"type": "string"},
			"gpu_count":  map[string]any{"type": "integer", "minimum": 1, "maximum": 4},
		},
	}
}

func readMessage(r *bufio.Reader) ([]byte, messageFraming, error) {
	length := 0
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, framingContentLength, err
		}
		trimmed := strings.TrimRight(line, "\r\n")
		if length == 0 && strings.HasPrefix(strings.TrimSpace(trimmed), "{") {
			return []byte(strings.TrimSpace(trimmed)), framingNDJSON, nil
		}
		line = trimmed
		if line == "" {
			break
		}
		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			_, err := fmt.Sscanf(line, "Content-Length: %d", &length)
			if err != nil {
				_, err = fmt.Sscanf(line, "content-length: %d", &length)
				if err != nil {
					return nil, framingContentLength, err
				}
			}
		}
	}
	if length <= 0 {
		return nil, framingContentLength, fmt.Errorf("missing content length")
	}
	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, framingContentLength, err
	}
	return payload, framingContentLength, nil
}

func writeMessage(w *bufio.Writer, payload any, framing messageFraming) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if framing == framingNDJSON {
		if _, err := w.Write(append(data, '\n')); err != nil {
			return err
		}
		return w.Flush()
	}
	if _, err := fmt.Fprintf(w, "Content-Length: %d\r\n\r\n", len(data)); err != nil {
		return err
	}
	if _, err := w.Write(data); err != nil {
		return err
	}
	return w.Flush()
}
