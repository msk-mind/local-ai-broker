package gpuservice

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	controlRequestSchema    = "gpu_service_demand_v1"
	controlResponseSchema   = "gpu_service_demand_response_v1"
	controlFailureSchema    = "gpu_service_failure_report_v1"
	maxControlFileBytes     = 64 * 1024
	orphanResponseRetention = 5 * time.Minute
)

type ControlRequest struct {
	Schema          string          `json:"schema"`
	RequestID       string          `json:"request_id"`
	Tier            Tier            `json:"tier"`
	FailureCategory FailureCategory `json:"failure_category,omitempty"`
	Reason          string          `json:"reason,omitempty"`
	RequestedAt     string          `json:"requested_at"`
	Deadline        string          `json:"deadline"`
	Nonce           string          `json:"nonce"`
	Signature       string          `json:"signature"`
}

type EndpointAccess struct {
	ID                 string       `json:"id"`
	Tier               Tier         `json:"tier"`
	Endpoint           string       `json:"endpoint"`
	EndpointAuth       EndpointAuth `json:"endpoint_auth"`
	ModelProfile       string       `json:"model_profile"`
	Model              string       `json:"model"`
	Capabilities       []string     `json:"capabilities"`
	ContextLimitTokens int          `json:"context_limit_tokens"`
	GPU                GPU          `json:"gpu"`
	SlurmJobID         string       `json:"slurm_job_id,omitempty"`
	HeartbeatAt        time.Time    `json:"heartbeat_at"`
	LeaseExpiresAt     time.Time    `json:"lease_expires_at"`
}

type ControlResponse struct {
	Schema            string             `json:"schema"`
	RequestID         string             `json:"request_id"`
	DemandID          string             `json:"demand_id,omitempty"`
	State             DemandState        `json:"state"`
	FailureCategory   FailureCategory    `json:"failure_category,omitempty"`
	UpdatedAt         string             `json:"updated_at"`
	Error             string             `json:"error,omitempty"`
	Service           *EndpointAccess    `json:"service,omitempty"`
	ServiceDiagnostic *ServiceDiagnostic `json:"service_diagnostics,omitempty"`
	Signature         string             `json:"signature"`
}

type ControlFailureReport struct {
	Schema          string          `json:"schema"`
	ReportID        string          `json:"report_id"`
	ServiceID       string          `json:"service_id"`
	Tier            Tier            `json:"tier"`
	FailureCategory FailureCategory `json:"failure_category"`
	Reason          string          `json:"reason"`
	ReportedAt      string          `json:"reported_at"`
	Nonce           string          `json:"nonce"`
	Signature       string          `json:"signature"`
}

// ControlSpool is the cross-language request protocol. Writers create an
// authenticated request file atomically; only the reconciler imports it and
// mutates scheduler state.
type ControlSpool struct {
	dir   string
	token []byte
	now   func() time.Time
}

func NewControlSpool(dir, controlToken string) (*ControlSpool, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, errors.New("GPU service control request directory is required")
	}
	if strings.TrimSpace(controlToken) == "" {
		return nil, errors.New("GPU service control token is required")
	}
	return &ControlSpool{dir: filepath.Clean(dir), token: []byte(controlToken), now: time.Now}, nil
}

func (s *ControlSpool) Dir() string { return s.dir }

func (s *ControlSpool) Submit(ctx context.Context, request DemandRequest) (ControlRequest, error) {
	if err := ctx.Err(); err != nil {
		return ControlRequest{}, err
	}
	if !request.Tier.Valid() || request.TTL <= 0 {
		return ControlRequest{}, errors.New("valid demand tier and positive TTL are required")
	}
	idSuffix, err := randomHex(12)
	if err != nil {
		return ControlRequest{}, err
	}
	nonce, err := randomHex(16)
	if err != nil {
		return ControlRequest{}, err
	}
	now := s.now().UTC()
	control := ControlRequest{
		Schema:          controlRequestSchema,
		RequestID:       "gpu-request-" + idSuffix,
		Tier:            request.Tier,
		FailureCategory: request.FailureCategory,
		Reason:          strings.TrimSpace(request.Reason),
		RequestedAt:     now.Format(time.RFC3339Nano),
		Deadline:        now.Add(request.TTL).Format(time.RFC3339Nano),
		Nonce:           nonce,
	}
	control.Signature = s.signRequest(control)
	if err := s.writeJSON(control.RequestID+".request.json", control); err != nil {
		return ControlRequest{}, err
	}
	return control, nil
}

// SubmitFailureReport is fire-and-forget. It writes an authenticated action
// for the reconciler and never schedules or waits for replacement capacity.
func (s *ControlSpool) SubmitFailureReport(ctx context.Context, failure ServiceFailure) (ControlFailureReport, error) {
	if err := ctx.Err(); err != nil {
		return ControlFailureReport{}, err
	}
	if !validControlID(failure.ServiceID) ||
		(failure.Tier != TierP40Retrieval && failure.Tier != TierP40Synthesis) ||
		!reportableServiceFailure(failure.FailureCategory) {
		return ControlFailureReport{}, ErrFailureReportRejected
	}
	idSuffix, err := randomHex(12)
	if err != nil {
		return ControlFailureReport{}, err
	}
	nonce, err := randomHex(16)
	if err != nil {
		return ControlFailureReport{}, err
	}
	report := ControlFailureReport{
		Schema:          controlFailureSchema,
		ReportID:        "gpu-failure-" + idSuffix,
		ServiceID:       failure.ServiceID,
		Tier:            failure.Tier,
		FailureCategory: failure.FailureCategory,
		Reason:          strings.TrimSpace(failure.Reason),
		ReportedAt:      s.now().UTC().Format(time.RFC3339Nano),
		Nonce:           nonce,
	}
	report.Signature = s.signFailureReport(report)
	if err := s.writeJSON(report.ReportID+".failure.json", report); err != nil {
		return ControlFailureReport{}, err
	}
	return report, nil
}

func (s *ControlSpool) Import(ctx context.Context, registry Registry) error {
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return err
	}
	paths, err := filepath.Glob(filepath.Join(s.dir, "*.request.json"))
	if err != nil {
		return err
	}
	sort.Strings(paths)
	for _, path := range paths {
		if err := ctx.Err(); err != nil {
			return err
		}
		request, err := s.readRequest(path)
		if err != nil {
			s.rejectRequest(path)
			continue
		}
		deadline, _ := time.Parse(time.RFC3339Nano, request.Deadline)
		ttl := deadline.Sub(s.now().UTC())
		if ttl <= 0 {
			if err := s.writeFailureResponse(request.RequestID, FailureQueueDelay, "demand deadline expired"); err != nil {
				return err
			}
			if err := s.removeRequest(path); err != nil {
				return err
			}
			continue
		}
		demand, err := registry.RequestDemand(ctx, string(s.token), DemandRequest{
			Tier:            request.Tier,
			FailureCategory: request.FailureCategory,
			Reason:          request.Reason,
			TTL:             ttl,
		})
		if err != nil {
			if err := s.writeFailureResponse(request.RequestID, FailureService, err.Error()); err != nil {
				return err
			}
			if err := s.removeRequest(path); err != nil {
				return err
			}
			continue
		}
		response := ControlResponse{
			Schema:    controlResponseSchema,
			RequestID: request.RequestID,
			DemandID:  demand.ID,
			State:     demand.State,
			UpdatedAt: s.now().UTC().Format(time.RFC3339Nano),
		}
		if err := s.writeResponse(response); err != nil {
			return err
		}
		if err := s.removeRequest(path); err != nil {
			return err
		}
	}
	return s.importFailureReports(ctx, registry)
}

func (s *ControlSpool) rejectRequest(path string) { _ = os.Rename(path, path+".rejected") }

func (s *ControlSpool) removeRequest(path string) error {
	err := os.Remove(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func (s *ControlSpool) writeFailureResponse(requestID string, category FailureCategory, message string) error {
	return s.writeResponse(ControlResponse{
		Schema: controlResponseSchema, RequestID: requestID, State: DemandFailed,
		FailureCategory: category, UpdatedAt: s.now().UTC().Format(time.RFC3339Nano), Error: message,
	})
}

func (s *ControlSpool) importFailureReports(ctx context.Context, registry Registry) error {
	paths, err := filepath.Glob(filepath.Join(s.dir, "*.failure.json"))
	if err != nil {
		return err
	}
	sort.Strings(paths)
	for _, path := range paths {
		if err := ctx.Err(); err != nil {
			return err
		}
		report, err := s.readFailureReport(path)
		if err != nil {
			_ = os.Rename(path, path+".rejected")
			continue
		}
		err = registry.ReportServiceFailure(ctx, string(s.token), ServiceFailure{
			ServiceID:       report.ServiceID,
			Tier:            report.Tier,
			FailureCategory: report.FailureCategory,
			Reason:          report.Reason,
		})
		if err != nil && !errors.Is(err, ErrRecordNotFound) && !errors.Is(err, ErrFailureReportRejected) {
			return err
		}
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func (s *ControlSpool) Sync(ctx context.Context, registry Registry) error {
	now := s.now().UTC()
	demands, err := registry.ListDemands(ctx)
	if err != nil {
		return err
	}
	demandByID := make(map[string]Demand, len(demands))
	for _, demand := range demands {
		demandByID[demand.ID] = demand
	}
	records, err := registry.List(ctx)
	if err != nil {
		return err
	}
	recordByID := make(map[string]Record, len(records))
	for _, record := range records {
		recordByID[record.ID] = record
	}
	paths, err := filepath.Glob(filepath.Join(s.dir, "*.response.json"))
	if err != nil {
		return err
	}
	for _, path := range paths {
		response, err := s.readResponse(path)
		if err != nil {
			_ = os.Remove(path)
			continue
		}
		demand, ok := demandByID[response.DemandID]
		if !ok {
			if response.Service != nil || responseOlderThan(response, now, orphanResponseRetention) {
				_ = os.Remove(path)
			}
			continue
		}
		if (demand.State == DemandReady || demand.State == DemandFailed) && !demand.Deadline.After(now) {
			_ = os.Remove(path)
			continue
		}
		// Terminal responses are immutable capabilities. Writing them on every
		// heartbeat creates a race that could recreate a credential file after
		// its consumer removes it. Rewrite only when terminal state changes.
		if response.State == demand.State && response.State == DemandFailed &&
			(response.ServiceDiagnostic != nil || demand.ServiceDiagnostic == nil) {
			continue
		}
		if response.State == demand.State && response.State == DemandReady && response.Service != nil {
			record, exists := recordByID[demand.ServiceID]
			if exists && record.State == StateReady && record.EffectiveLeaseExpiresAt().After(now) && response.Service.ID == record.ID {
				continue
			}
		}
		response.State = demand.State
		response.FailureCategory = demand.FailureCategory
		response.Error = demand.Error
		response.UpdatedAt = now.Format(time.RFC3339Nano)
		response.Service = nil
		response.ServiceDiagnostic = nil
		if demand.State == DemandReady {
			record, exists := recordByID[demand.ServiceID]
			if !exists || record.State != StateReady || !record.EffectiveLeaseExpiresAt().After(now) {
				response.State = DemandFailed
				response.FailureCategory = FailureService
				response.Error = "bound service lease is unavailable"
			} else {
				response.Service = endpointAccess(record)
			}
		}
		if response.State == DemandFailed {
			if demand.ServiceDiagnostic != nil {
				diagnostic := *demand.ServiceDiagnostic
				response.ServiceDiagnostic = &diagnostic
			} else if record, exists := recordByID[demand.ServiceID]; exists {
				diagnostic := record.Diagnostic()
				response.ServiceDiagnostic = &diagnostic
			}
		}
		if err := s.writeResponse(response); err != nil {
			return err
		}
	}
	return nil
}

func responseOlderThan(response ControlResponse, now time.Time, age time.Duration) bool {
	updatedAt, err := time.Parse(time.RFC3339Nano, response.UpdatedAt)
	return err != nil || now.Sub(updatedAt) >= age
}

func (s *ControlSpool) Await(ctx context.Context, requestID string, pollInterval time.Duration) (EndpointAccess, error) {
	if !validControlID(requestID) {
		return EndpointAccess{}, errors.New("invalid control request id")
	}
	if pollInterval <= 0 {
		pollInterval = 250 * time.Millisecond
	}
	path := filepath.Join(s.dir, requestID+".response.json")
	defer os.Remove(path) // terminal credentials are single-consumer
	for {
		response, err := s.readResponse(path)
		if err == nil {
			switch response.State {
			case DemandReady:
				if response.Service == nil {
					return EndpointAccess{}, fmt.Errorf("%w: ready response has no service", ErrDemandFailed)
				}
				return *response.Service, nil
			case DemandFailed:
				return EndpointAccess{}, fmt.Errorf("%w: %s", ErrDemandFailed, response.Error)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return EndpointAccess{}, err
		}
		timer := time.NewTimer(pollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return EndpointAccess{}, ctx.Err()
		case <-timer.C:
		}
	}
}

func (s *ControlSpool) readRequest(path string) (ControlRequest, error) {
	var request ControlRequest
	if err := readControlJSON(path, &request); err != nil {
		return request, err
	}
	if request.Schema != controlRequestSchema || !validControlID(request.RequestID) {
		return request, errors.New("invalid GPU service control request")
	}
	if filepath.Base(path) != request.RequestID+".request.json" {
		return request, errors.New("control request filename does not match request id")
	}
	if !request.Tier.Valid() || request.Nonce == "" || !hmac.Equal([]byte(request.Signature), []byte(s.signRequest(request))) {
		return request, ErrControlDenied
	}
	requestedAt, err := time.Parse(time.RFC3339Nano, request.RequestedAt)
	if err != nil {
		return request, err
	}
	deadline, err := time.Parse(time.RFC3339Nano, request.Deadline)
	if err != nil {
		return request, err
	}
	if !deadline.After(requestedAt) || deadline.Sub(requestedAt) > 24*time.Hour {
		return request, errors.New("invalid GPU service demand deadline")
	}
	return request, nil
}

func (s *ControlSpool) readResponse(path string) (ControlResponse, error) {
	var response ControlResponse
	if err := readControlJSON(path, &response); err != nil {
		return response, err
	}
	if response.Schema != controlResponseSchema || !validControlID(response.RequestID) ||
		!hmac.Equal([]byte(response.Signature), []byte(s.signResponse(response))) {
		return response, ErrControlDenied
	}
	return response, nil
}

func (s *ControlSpool) readFailureReport(path string) (ControlFailureReport, error) {
	var report ControlFailureReport
	if err := readControlJSON(path, &report); err != nil {
		return report, err
	}
	if report.Schema != controlFailureSchema || !validControlID(report.ReportID) || !validControlID(report.ServiceID) {
		return report, errors.New("invalid GPU service failure report")
	}
	if filepath.Base(path) != report.ReportID+".failure.json" {
		return report, errors.New("failure report filename does not match report id")
	}
	if (report.Tier != TierP40Retrieval && report.Tier != TierP40Synthesis) ||
		!reportableServiceFailure(report.FailureCategory) || report.Nonce == "" ||
		!hmac.Equal([]byte(report.Signature), []byte(s.signFailureReport(report))) {
		return report, ErrControlDenied
	}
	reportedAt, err := time.Parse(time.RFC3339Nano, report.ReportedAt)
	if err != nil {
		return report, err
	}
	now := s.now().UTC()
	if reportedAt.After(now.Add(5*time.Minute)) || now.Sub(reportedAt) > 24*time.Hour {
		return report, errors.New("GPU service failure report timestamp is outside the accepted window")
	}
	return report, nil
}

// Request signing input is the NUL-separated sequence documented here so
// Python and other workers can implement it without canonical-JSON ambiguity:
// schema, request_id, tier, failure_category, base64url(reason),
// requested_at, deadline, nonce.
func (s *ControlSpool) signRequest(request ControlRequest) string {
	message := strings.Join([]string{
		request.Schema,
		request.RequestID,
		string(request.Tier),
		string(request.FailureCategory),
		base64.RawURLEncoding.EncodeToString([]byte(request.Reason)),
		request.RequestedAt,
		request.Deadline,
		request.Nonce,
	}, "\x00")
	return signControl(s.token, message)
}

// Response signing extends the common fields with either full ready-service
// access or the marker and sanitized failure identity fields. The latter are:
// service_diagnostics, tier, Slurm job ID, GPU type, GPU count, model profile.
func (s *ControlSpool) signResponse(response ControlResponse) string {
	fields := []string{
		response.Schema,
		response.RequestID,
		response.DemandID,
		string(response.State),
		string(response.FailureCategory),
		response.UpdatedAt,
		base64.RawURLEncoding.EncodeToString([]byte(response.Error)),
	}
	if response.Service != nil {
		capabilities := append([]string(nil), response.Service.Capabilities...)
		sort.Strings(capabilities)
		fields = append(fields,
			response.Service.ID,
			string(response.Service.Tier),
			response.Service.Endpoint,
			response.Service.EndpointAuth.Type,
			response.Service.EndpointAuth.BearerToken,
			response.Service.ModelProfile,
			response.Service.Model,
			strings.Join(capabilities, ","),
			strconv.Itoa(response.Service.ContextLimitTokens),
			response.Service.GPU.Type,
			strconv.Itoa(response.Service.GPU.Count),
			response.Service.SlurmJobID,
			response.Service.HeartbeatAt.UTC().Format(time.RFC3339Nano),
			response.Service.LeaseExpiresAt.UTC().Format(time.RFC3339Nano),
		)
	}
	if response.ServiceDiagnostic != nil {
		fields = append(fields,
			"service_diagnostics",
			string(response.ServiceDiagnostic.Tier),
			response.ServiceDiagnostic.SlurmJobID,
			response.ServiceDiagnostic.GPU.Type,
			strconv.Itoa(response.ServiceDiagnostic.GPU.Count),
			response.ServiceDiagnostic.ModelProfile,
		)
	}
	return signControl(s.token, strings.Join(fields, "\x00"))
}

// Failure-report signing input is NUL-separated to avoid canonical-JSON
// ambiguity: schema, report_id, service_id, tier, failure_category,
// base64url(reason), reported_at, nonce.
func (s *ControlSpool) signFailureReport(report ControlFailureReport) string {
	return signControl(s.token, strings.Join([]string{
		report.Schema,
		report.ReportID,
		report.ServiceID,
		string(report.Tier),
		string(report.FailureCategory),
		base64.RawURLEncoding.EncodeToString([]byte(report.Reason)),
		report.ReportedAt,
		report.Nonce,
	}, "\x00"))
}

func signControl(token []byte, message string) string {
	mac := hmac.New(sha256.New, token)
	_, _ = mac.Write([]byte(message))
	return hex.EncodeToString(mac.Sum(nil))
}

func endpointAccess(record Record) *EndpointAccess {
	return &EndpointAccess{
		ID:                 record.ID,
		Tier:               record.Tier,
		Endpoint:           record.Endpoint,
		EndpointAuth:       record.EndpointAuth,
		ModelProfile:       record.ModelProfile,
		Model:              record.Model,
		Capabilities:       append([]string(nil), record.Capabilities...),
		ContextLimitTokens: record.ContextLimitTokens,
		GPU:                record.GPU,
		SlurmJobID:         record.SlurmJobID,
		HeartbeatAt:        record.HeartbeatAt,
		LeaseExpiresAt:     record.EffectiveLeaseExpiresAt(),
	}
}

func (s *ControlSpool) writeResponse(response ControlResponse) error {
	response.Schema = controlResponseSchema
	response.Signature = ""
	response.Signature = s.signResponse(response)
	return s.writeJSON(response.RequestID+".response.json", response)
}

func (s *ControlSpool) writeJSON(name string, value any) error {
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return err
	}
	content, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.dir, ".gpu-control-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(content); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, filepath.Join(s.dir, name))
}

func readControlJSON(path string, destination any) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if info.Size() > maxControlFileBytes {
		return errors.New("GPU control file exceeds size limit")
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(content, destination); err != nil {
		return fmt.Errorf("decode GPU control file: %w", err)
	}
	return nil
}

func validControlID(value string) bool {
	if len(value) < 8 || len(value) > 128 {
		return false
	}
	for _, char := range value {
		if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') ||
			(char >= '0' && char <= '9') || char == '-' || char == '_' {
			continue
		}
		return false
	}
	return true
}
