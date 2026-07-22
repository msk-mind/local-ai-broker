package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/authz"
	"github.com/msk-mind/local-ai-broker/broker/pkg/policy"
	"github.com/msk-mind/local-ai-broker/broker/pkg/service"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/tasks"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type Handler struct {
	service       *service.Service
	authenticator *auth.Authenticator
	auditLogPath  string
	mux           *http.ServeMux
}

type pathAction struct {
	suffix  string
	handler func(http.ResponseWriter, *http.Request, string)
}

type releasedResultWaitDiagnostics struct {
	requestStartedUnixNS  int64
	initialFetchUnixNS    int64
	releaseObservedUnixNS int64
	responseReadyUnixNS   int64
	pollCount             int
}

const (
	defaultReleasedResultWaitPollInterval = 10 * time.Millisecond
	minimumReleasedResultWaitPollInterval = 2 * time.Millisecond
	maximumReleasedResultWaitPollInterval = 4 * time.Millisecond
)

func NewHandler(svc *service.Service, authenticator *auth.Authenticator) *Handler {
	return NewHandlerWithAudit(svc, authenticator, "")
}

func NewHandlerWithAudit(svc *service.Service, authenticator *auth.Authenticator, auditLogPath string) *Handler {
	h := &Handler{
		service:       svc,
		authenticator: authenticator,
		auditLogPath:  auditLogPath,
		mux:           http.NewServeMux(),
	}
	h.routes()
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/healthz" {
		h.mux.ServeHTTP(w, r)
		return
	}
	principal, err := h.authenticator.Authenticate(r)
	if err != nil {
		if errors.Is(err, auth.ErrUnauthenticated) {
			writeError(w, http.StatusUnauthorized, "UNAUTHENTICATED", err.Error())
			return
		}
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
		return
	}
	h.mux.ServeHTTP(w, r.WithContext(auth.WithPrincipal(r.Context(), principal)))
}

func (h *Handler) routes() {
	h.mux.HandleFunc("/healthz", h.handleHealth)
	h.mux.HandleFunc("/v1/system/audit-health", h.handleAuditHealth)
	h.mux.HandleFunc("/v1/jobs", h.handleJobs)
	h.mux.HandleFunc("/v1/jobs/", h.handleJobByID)
	h.mux.HandleFunc("/v1/roots/", h.handleRootByID)
	for _, spec := range tasks.RAGAliasSpecs() {
		h.mux.HandleFunc(spec.HTTPPath, h.handleRAGAlias(spec))
	}
	h.mux.HandleFunc("/v1/rag/evidence-packs/", h.handleRAGEvidencePackMetadata)
	h.mux.HandleFunc("/v1/rag/indexes/", h.handleRAGIndexMetadata)
	h.mux.HandleFunc("/v1/rag/cache:lookup", h.handleRAGCacheLookup)
}

func (h *Handler) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
	})
}

func (h *Handler) handleRAGAlias(spec tasks.Spec) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
			return
		}
		var raw json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid job request body")
			return
		}
		req, err := tasks.DecodeSubmitRequest(raw, spec)
		if err != nil {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid job request body")
			return
		}
		submitStartedUnixNS := time.Now().UnixNano()
		submitCtx := r.Context()
		if req.TaskType == "inspect_repo" {
			submitCtx = service.WithPreferInlineLocalRelease(submitCtx)
		}
		resp, err := h.service.SubmitJob(submitCtx, req)
		if err != nil {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
			return
		}
		if resp.ReleasedResult != nil {
			annotateReleaseBrokerLifecycle(resp.ReleasedResult, map[string]any{
				"broker_submit_request_started_unix_ns": submitStartedUnixNS,
				"broker_submit_response_ready_unix_ns":  time.Now().UnixNano(),
				"broker_submit_inline_release":          true,
			})
		}
		writeJSON(w, http.StatusAccepted, resp)
	}
}

func (h *Handler) handleAuditHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}
	principal := auth.PrincipalFromContext(r.Context())
	if !auth.IsAdmin(principal) {
		writeError(w, http.StatusForbidden, "FORBIDDEN", "audit health requires admin role")
		return
	}
	if h.auditLogPath == "" {
		writeError(w, http.StatusNotImplemented, "NOT_CONFIGURED", "audit log path is not configured")
		return
	}
	result, err := audit.VerifyFile(h.auditLogPath)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
		return
	}
	status := http.StatusOK
	if !result.Valid {
		status = http.StatusServiceUnavailable
	}
	writeJSON(w, status, result)
}

func (h *Handler) handleJobs(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid JSON request body")
			return
		}
		if len(bytes.TrimSpace(raw)) == 0 {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid JSON request body")
			return
		}
		var envelope struct {
			Children json.RawMessage `json:"children"`
		}
		if err := json.Unmarshal(raw, &envelope); err != nil {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid JSON request body")
			return
		}
		if envelope.Children != nil {
			var req types.SubmitParallelJobsRequest
			if err := json.Unmarshal(raw, &req); err != nil {
				writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid parallel job request body")
				return
			}
			resp, err := h.service.SubmitParallelJobs(r.Context(), req)
			if err != nil {
				writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
				return
			}
			writeJSON(w, http.StatusAccepted, resp)
			return
		}
		var req types.SubmitJobRequest
		if err := json.Unmarshal(raw, &req); err != nil {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid job request body")
			return
		}

		submitStartedUnixNS := time.Now().UnixNano()
		submitCtx := r.Context()
		if req.TaskType == "inspect_repo" {
			submitCtx = service.WithPreferInlineLocalRelease(submitCtx)
		}
		resp, err := h.service.SubmitJob(submitCtx, req)
		if err != nil {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
			return
		}
		if resp.ReleasedResult != nil {
			annotateReleaseBrokerLifecycle(resp.ReleasedResult, map[string]any{
				"broker_submit_request_started_unix_ns": submitStartedUnixNS,
				"broker_submit_response_ready_unix_ns":  time.Now().UnixNano(),
				"broker_submit_inline_release":          true,
			})
		}
		writeJSON(w, http.StatusAccepted, resp)
	case http.MethodGet:
		h.handleListJobs(w, r)
	default:
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
	}
}

func (h *Handler) handleJobByID(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/v1/jobs/")
	if dispatchPathAction(w, r, path, []pathAction{
		{suffix: ":cancel", handler: h.handleCancelJob},
		{suffix: ":retry-recommended", handler: h.handleRetryRecommendedJob},
		{suffix: "/retry-recommendation", handler: h.handleGetRetryRecommendation},
		{suffix: "/result", handler: h.handleFetchResult},
		{suffix: "/logs", handler: h.handleFetchLogs},
	}) {
		return
	}
	h.handleGetJob(w, r, path)
}

func (h *Handler) handleRAGEvidencePackMetadata(w http.ResponseWriter, r *http.Request) {
	h.handleArtifactMetadata(w, r, "/v1/rag/evidence-packs/", map[string]struct{}{"evidence_pack": {}})
}

func (h *Handler) handleRAGIndexMetadata(w http.ResponseWriter, r *http.Request) {
	h.handleArtifactMetadata(w, r, "/v1/rag/indexes/", map[string]struct{}{"retrieval_result": {}, "chunk_manifest": {}})
}

func (h *Handler) handleArtifactMetadata(w http.ResponseWriter, r *http.Request, prefix string, allowedTypes map[string]struct{}) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}
	path := strings.TrimPrefix(r.URL.Path, prefix)
	if !strings.HasSuffix(path, "/metadata") {
		writeError(w, http.StatusNotFound, "NOT_FOUND", "endpoint not found")
		return
	}
	artifactID := strings.TrimSuffix(path, "/metadata")
	meta, err := h.service.GetArtifactMetadata(r.Context(), artifactID, allowedTypes)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, meta)
}

func (h *Handler) handleRAGCacheLookup(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}
	var req types.SubmitJobRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid cache lookup request body")
		return
	}
	resp, err := h.service.LookupCache(r.Context(), req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) handleGetRetryRecommendation(w http.ResponseWriter, r *http.Request, jobID string) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}
	rec, err := h.service.GetJobRetryRecommendation(r.Context(), jobID)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, rec)
}

func (h *Handler) handleRetryRecommendedJob(w http.ResponseWriter, r *http.Request, jobID string) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}
	resp, err := h.service.RetryJobWithRecommendation(r.Context(), jobID)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, resp)
}

func (h *Handler) handleRootByID(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/v1/roots/")
	if dispatchPathAction(w, r, path, []pathAction{
		{suffix: ":retry-failed", handler: h.handleRetryFailedRootShards},
		{suffix: ":release-deferred", handler: h.handleReleaseDeferredRootChunks},
	}) {
		return
	}
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}
	rootJobID := path
	status, err := h.service.GetRootJobStatus(r.Context(), rootJobID)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, status)
}

func (h *Handler) handleRetryFailedRootShards(w http.ResponseWriter, r *http.Request, rootJobID string) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}
	req := types.RetryFailedRootShardsRequest{
		RootJobID:       rootJobID,
		ResubmitReducer: true,
	}
	if err := decodeOptionalJSONBody(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid retry request body")
		return
	}
	if strings.TrimSpace(req.RootJobID) == "" {
		req.RootJobID = rootJobID
	}
	resp, err := h.service.RetryFailedRootShards(r.Context(), req)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, resp)
}

func (h *Handler) handleReleaseDeferredRootChunks(w http.ResponseWriter, r *http.Request, rootJobID string) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}
	req := types.ReleaseDeferredRootChunksRequest{
		RootJobID: rootJobID,
	}
	if err := decodeOptionalJSONBody(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid release request body")
		return
	}
	if strings.TrimSpace(req.RootJobID) == "" {
		req.RootJobID = rootJobID
	}
	resp, err := h.service.ReleaseDeferredRootChunks(r.Context(), req)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, resp)
}

func (h *Handler) handleGetJob(w http.ResponseWriter, r *http.Request, jobID string) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	job, err := h.service.GetJob(r.Context(), jobID)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (h *Handler) handleListJobs(w http.ResponseWriter, r *http.Request) {
	jobs, err := h.service.ListJobs(r.Context())
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"jobs":  jobs,
		"count": len(jobs),
	})
}

func (h *Handler) handleFetchResult(w http.ResponseWriter, r *http.Request, jobID string) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	waitMS, err := parseNonNegativeIntQuery(r, "wait_ms")
	if err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}
	pollIntervalMS, err := parseNonNegativeIntQuery(r, "poll_interval_ms")
	if err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	release, diagnostics, err := h.waitForReleasedResult(r.Context(), jobID, waitMS, pollIntervalMS)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	diagnostics.responseReadyUnixNS = time.Now().UnixNano()
	annotateReleaseBrokerLifecycle(&release, map[string]any{
		"broker_result_request_started_unix_ns":  diagnostics.requestStartedUnixNS,
		"broker_result_initial_fetch_unix_ns":    diagnostics.initialFetchUnixNS,
		"broker_result_release_observed_unix_ns": diagnostics.releaseObservedUnixNS,
		"broker_result_response_ready_unix_ns":   diagnostics.responseReadyUnixNS,
		"broker_result_poll_count":               diagnostics.pollCount,
	})

	writeJSON(w, http.StatusOK, release)
}

func (h *Handler) waitForReleasedResult(ctx context.Context, jobID string, waitMS, pollIntervalMS int) (types.JobResultRelease, releasedResultWaitDiagnostics, error) {
	diagnostics := releasedResultWaitDiagnostics{
		requestStartedUnixNS: time.Now().UnixNano(),
	}
	initialFetchCtx := ctx
	pollFetchCtx := service.WithSkipInspectRepoResultProbe(ctx)
	if waitMS <= 0 {
		initialFetchCtx = pollFetchCtx
	}
	diagnostics.initialFetchUnixNS = time.Now().UnixNano()
	release, err := h.service.GetReleasedResult(initialFetchCtx, jobID)
	if err != nil {
		return types.JobResultRelease{}, diagnostics, err
	}
	if waitMS <= 0 || release.Result != nil || isTerminalReleaseState(release.State) {
		diagnostics.releaseObservedUnixNS = time.Now().UnixNano()
		return release, diagnostics, nil
	}
	current := release

	waitCtx, cancel := context.WithTimeout(ctx, time.Duration(waitMS)*time.Millisecond)
	defer cancel()

	interval := releasedResultWaitPollInterval(pollIntervalMS)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-waitCtx.Done():
			return current, diagnostics, nil
		case <-ticker.C:
			diagnostics.pollCount++
			current, err = h.service.GetReleasedResult(pollFetchCtx, jobID)
			if err != nil {
				return types.JobResultRelease{}, diagnostics, err
			}
			if current.Result != nil || isTerminalReleaseState(current.State) {
				diagnostics.releaseObservedUnixNS = time.Now().UnixNano()
				return current, diagnostics, nil
			}
		}
	}
}

func releasedResultWaitPollInterval(pollIntervalMS int) time.Duration {
	interval := defaultReleasedResultWaitPollInterval
	if pollIntervalMS > 0 {
		interval = time.Duration(pollIntervalMS) * time.Millisecond
	}
	if interval < minimumReleasedResultWaitPollInterval {
		interval = minimumReleasedResultWaitPollInterval
	}
	if interval > maximumReleasedResultWaitPollInterval {
		interval = maximumReleasedResultWaitPollInterval
	}
	return interval
}

func annotateReleaseBrokerLifecycle(release *types.JobResultRelease, fields map[string]any) {
	if release == nil || release.Result == nil || len(fields) == 0 {
		return
	}
	payload := cloneAnyMap(release.Result.Payload)
	runtime := cloneAnyMap(mapValue(payload["runtime"]))
	lifecycle := cloneAnyMap(mapValue(runtime["broker_lifecycle"]))
	for key, value := range fields {
		lifecycle[key] = value
	}
	runtime["broker_lifecycle"] = lifecycle
	payload["runtime"] = runtime
	release.Result.Payload = payload
}

func cloneAnyMap(input map[string]any) map[string]any {
	if len(input) == 0 {
		return map[string]any{}
	}
	cloned := make(map[string]any, len(input))
	for key, value := range input {
		cloned[key] = cloneAnyValue(value)
	}
	return cloned
}

func cloneAnyValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return cloneAnyMap(typed)
	case []any:
		cloned := make([]any, len(typed))
		for i, item := range typed {
			cloned[i] = cloneAnyValue(item)
		}
		return cloned
	case []string:
		cloned := make([]string, len(typed))
		copy(cloned, typed)
		return cloned
	default:
		return typed
	}
}

func mapValue(value any) map[string]any {
	typed, _ := value.(map[string]any)
	return typed
}

func parseNonNegativeIntQuery(r *http.Request, key string) (int, error) {
	raw := strings.TrimSpace(r.URL.Query().Get(key))
	if raw == "" {
		return 0, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return 0, fmt.Errorf("invalid %s", key)
	}
	return value, nil
}

func isTerminalReleaseState(state types.JobState) bool {
	switch state {
	case types.JobStateSucceeded, types.JobStateFailed, types.JobStateCancelled, types.JobStatePreempted, types.JobStateTimedOut:
		return true
	default:
		return false
	}
}

func (h *Handler) handleFetchLogs(w http.ResponseWriter, r *http.Request, jobID string) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	stream := r.URL.Query().Get("stream")
	maxBytes := 0
	if raw := r.URL.Query().Get("max_bytes"); raw != "" {
		if _, err := fmt.Sscanf(raw, "%d", &maxBytes); err != nil {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", "invalid max_bytes")
			return
		}
	}

	logs, err := h.service.GetJobLogs(r.Context(), jobID, stream, maxBytes)
	if err != nil {
		if strings.Contains(err.Error(), "unsupported log stream") {
			writeError(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
			return
		}
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, logs)
}

func (h *Handler) handleCancelJob(w http.ResponseWriter, r *http.Request, jobID string) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "method not allowed")
		return
	}

	resp, err := h.service.CancelJob(r.Context(), jobID)
	if err != nil {
		handleServiceError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func handleServiceError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, store.ErrNotFound):
		writeError(w, http.StatusNotFound, "NOT_FOUND", err.Error())
	case errors.Is(err, authz.ErrForbidden):
		writeError(w, http.StatusForbidden, "FORBIDDEN", err.Error())
	case errors.Is(err, policy.ErrPolicyDenied):
		writeError(w, http.StatusForbidden, "POLICY_DENIED", err.Error())
	default:
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", err.Error())
	}
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"code":    code,
			"message": message,
		},
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func dispatchPathAction(w http.ResponseWriter, r *http.Request, path string, actions []pathAction) bool {
	for _, action := range actions {
		if strings.HasSuffix(path, action.suffix) {
			action.handler(w, r, strings.TrimSuffix(path, action.suffix))
			return true
		}
	}
	return false
}

func decodeOptionalJSONBody(r *http.Request, dst any) error {
	if r.Body == nil || r.ContentLength == 0 {
		return nil
	}
	return json.NewDecoder(r.Body).Decode(dst)
}
