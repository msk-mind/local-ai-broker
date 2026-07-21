// Package gpuservice manages scheduler-backed GPU model services independently
// from broker inspection request workers.
package gpuservice

import (
	"errors"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"time"
)

type Tier string

const (
	TierP40Retrieval  Tier = "p40-retrieval"
	TierP40Synthesis  Tier = "p40-synthesis"
	TierV100Reasoning Tier = "v100-reasoning"
	TierA100Single    Tier = "a100-single"
	TierA100Multigpu  Tier = "a100-multigpu"
)

var allTiers = []Tier{
	TierP40Retrieval,
	TierP40Synthesis,
	TierV100Reasoning,
	TierA100Single,
	TierA100Multigpu,
}

func AllTiers() []Tier {
	return append([]Tier(nil), allTiers...)
}

func (t Tier) Valid() bool {
	for _, candidate := range allTiers {
		if t == candidate {
			return true
		}
	}
	return false
}

type Role string

const (
	RoleRetrieval Role = "retrieval"
	RoleSynthesis Role = "synthesis"
)

const (
	OperationEmbeddings      = "embeddings"
	OperationIndexStatus     = "index_status"
	OperationIndexUpsert     = "index_upsert"
	OperationVectorSearch    = "faiss_search"
	OperationRerank          = "rerank"
	OperationChatCompletions = "chat_completions"
)

type ServiceState string

const (
	StateStarting  ServiceState = "starting"
	StateReady     ServiceState = "ready"
	StateUnhealthy ServiceState = "unhealthy"
)

type FailureCategory string

const (
	FailureNone             FailureCategory = ""
	FailureUnavailable      FailureCategory = "availability"
	FailureQueueDelay       FailureCategory = "queue_delay"
	FailureTimeout          FailureCategory = "timeout"
	FailureService          FailureCategory = "service_failure"
	FailureAuthentication   FailureCategory = "authentication"
	FailureOOM              FailureCategory = "oom"
	FailureContextOverflow  FailureCategory = "context_overflow"
	FailureModelLimit       FailureCategory = "model_limit"
	FailureInvalidSynthesis FailureCategory = "invalid_synthesis"
	// FailureRepeatedInvalidOutput is retained as a descriptive alias for the
	// second invalid-synthesis result that triggers multigpu escalation.
	FailureRepeatedInvalidOutput                 = FailureInvalidSynthesis
	FailureLeaseExpired          FailureCategory = "lease_expired"
	FailureHeartbeatLost         FailureCategory = "heartbeat_lost"
	FailureStartupTimeout        FailureCategory = "startup_timeout"
	FailureEndpointUnhealthy     FailureCategory = "endpoint_unhealthy"
)

type DemandState string

const (
	DemandPending   DemandState = "pending"
	DemandLaunching DemandState = "launching"
	DemandReady     DemandState = "ready"
	DemandFailed    DemandState = "failed"
)

type DemandRequest struct {
	Tier            Tier            `json:"tier"`
	FailureCategory FailureCategory `json:"failure_category,omitempty"`
	Reason          string          `json:"reason,omitempty"`
	TTL             time.Duration   `json:"-"`
}

type Demand struct {
	ID                string             `json:"id"`
	Tier              Tier               `json:"tier"`
	State             DemandState        `json:"state"`
	FailureCategory   FailureCategory    `json:"failure_category,omitempty"`
	Reason            string             `json:"reason,omitempty"`
	RequestedAt       time.Time          `json:"requested_at"`
	Deadline          time.Time          `json:"deadline"`
	ServiceID         string             `json:"service_id,omitempty"`
	ServiceDiagnostic *ServiceDiagnostic `json:"service_diagnostics,omitempty"`
	Error             string             `json:"error,omitempty"`
}

type DemandUpdate struct {
	State           DemandState
	ServiceID       string
	FailureCategory FailureCategory
	Error           string
}

// ServiceFailure is an authenticated fire-and-forget report about one known
// P40 lease. It never requests or schedules replacement capacity itself.
type ServiceFailure struct {
	ServiceID       string
	Tier            Tier
	FailureCategory FailureCategory
	Reason          string
}

type GPU struct {
	Type  string `json:"type"`
	Count int    `json:"count"`
}

// ServiceDiagnostic is the non-secret identity of a scheduler-managed model
// service. It is safe to retain on a failed demand: unlike EndpointAccess it
// contains no endpoint, bearer token, model artifact path, or runtime args.
type ServiceDiagnostic struct {
	Tier         Tier   `json:"tier"`
	SlurmJobID   string `json:"slurm_job_id,omitempty"`
	GPU          GPU    `json:"gpu"`
	ModelProfile string `json:"model_profile"`
}

type EndpointAuth struct {
	Type        string `json:"type"`
	BearerToken string `json:"bearer_token"`
}

type DeploymentProfile struct {
	Name               string   `json:"name"`
	Model              string   `json:"model"`
	Quantization       string   `json:"quantization"`
	ContextLimitTokens int      `json:"context_limit_tokens"`
	Runtime            string   `json:"runtime"`
	RuntimeArgs        []string `json:"runtime_args"`
}

type Placement struct {
	Partition  string `json:"partition,omitempty"`
	GPU        GPU    `json:"gpu"`
	NodeList   string `json:"nodelist,omitempty"`
	Constraint string `json:"constraint,omitempty"`
	QOS        string `json:"qos,omitempty"`
}

type Profile struct {
	Tier                Tier              `json:"tier"`
	Role                Role              `json:"role"`
	SupportedOperations []string          `json:"supported_operations"`
	Deployment          DeploymentProfile `json:"deployment"`
	Placement           Placement         `json:"placement"`
	MinReplicas         int               `json:"min_replicas"`
	MaxReplicas         int               `json:"max_replicas"`
}

func (p Profile) Validate() error {
	if !p.Tier.Valid() {
		return fmt.Errorf("unsupported GPU service tier %q", p.Tier)
	}
	if p.Role != RoleRetrieval && p.Role != RoleSynthesis {
		return fmt.Errorf("tier %s has invalid role %q", p.Tier, p.Role)
	}
	if strings.TrimSpace(p.Deployment.Name) == "" {
		return fmt.Errorf("tier %s requires a deployment profile name", p.Tier)
	}
	if strings.TrimSpace(p.Deployment.Model) == "" {
		return fmt.Errorf("tier %s requires an exact model path", p.Tier)
	}
	if strings.TrimSpace(p.Deployment.Quantization) == "" {
		return fmt.Errorf("tier %s requires an explicit quantization setting", p.Tier)
	}
	if p.Deployment.ContextLimitTokens <= 0 {
		return fmt.Errorf("tier %s requires a positive context limit", p.Tier)
	}
	if strings.TrimSpace(p.Deployment.Runtime) == "" {
		return fmt.Errorf("tier %s requires a runtime", p.Tier)
	}
	if len(p.Deployment.RuntimeArgs) == 0 {
		return fmt.Errorf("tier %s requires explicit runtime arguments", p.Tier)
	}
	for i, arg := range p.Deployment.RuntimeArgs {
		if strings.TrimSpace(arg) == "" {
			return fmt.Errorf("tier %s has an empty runtime argument at index %d", p.Tier, i)
		}
	}
	if len(p.SupportedOperations) == 0 {
		return fmt.Errorf("tier %s requires at least one supported operation", p.Tier)
	}
	if strings.TrimSpace(p.Placement.GPU.Type) == "" {
		return fmt.Errorf("tier %s requires a GPU type", p.Tier)
	}
	wantGPUCount := 1
	if p.Tier == TierV100Reasoning || p.Tier == TierA100Multigpu {
		wantGPUCount = 4
	}
	if p.Placement.GPU.Count != wantGPUCount {
		return fmt.Errorf("tier %s requires exactly %d GPU(s), got %d", p.Tier, wantGPUCount, p.Placement.GPU.Count)
	}
	if p.MinReplicas < 0 || p.MaxReplicas < 1 || p.MinReplicas > p.MaxReplicas {
		return fmt.Errorf("tier %s has invalid replica limits %d..%d", p.Tier, p.MinReplicas, p.MaxReplicas)
	}
	switch p.Tier {
	case TierP40Retrieval, TierP40Synthesis:
		if p.MinReplicas < 1 || p.MaxReplicas > 2 {
			return fmt.Errorf("tier %s must keep 1..2 replicas", p.Tier)
		}
	default:
		if p.MinReplicas != 0 || p.MaxReplicas != 1 {
			return fmt.Errorf("tier %s must scale from 0..1 replicas", p.Tier)
		}
	}
	return nil
}

type Record struct {
	ID                      string          `json:"id"`
	Tier                    Tier            `json:"tier"`
	Role                    Role            `json:"role"`
	State                   ServiceState    `json:"state"`
	Endpoint                string          `json:"endpoint,omitempty"`
	EndpointAuth            EndpointAuth    `json:"endpoint_auth,omitempty"`
	ModelProfile            string          `json:"model_profile"`
	Model                   string          `json:"model"`
	Capabilities            []string        `json:"capabilities"`
	ContextLimitTokens      int             `json:"context_limit_tokens"`
	GPU                     GPU             `json:"gpu"`
	SlurmJobID              string          `json:"slurm_job_id,omitempty"`
	SchedulerState          string          `json:"scheduler_state,omitempty"`
	CreatedAt               time.Time       `json:"created_at"`
	StartupDeadline         time.Time       `json:"startup_deadline"`
	HeartbeatAt             time.Time       `json:"heartbeat_at,omitempty"`
	LeaseExpiresAt          time.Time       `json:"lease_expires_at"`
	AbsoluteLeaseExpiresAt  time.Time       `json:"absolute_lease_expires_at,omitempty"`
	LastHealthCheckAt       time.Time       `json:"last_health_check_at,omitempty"`
	HealthError             string          `json:"health_error,omitempty"`
	FailureCategory         FailureCategory `json:"failure_category,omitempty"`
	CancelRequested         bool            `json:"cancel_requested,omitempty"`
	RegistrationTokenSHA256 string          `json:"registration_token_sha256"`
}

func (r Record) Routable(now time.Time, heartbeatTimeout time.Duration) bool {
	if r.State != StateReady || strings.TrimSpace(r.Endpoint) == "" {
		return false
	}
	if strings.ToLower(strings.TrimSpace(r.EndpointAuth.Type)) != "bearer" || strings.TrimSpace(r.EndpointAuth.BearerToken) == "" {
		return false
	}
	if r.HeartbeatAt.IsZero() || !r.EffectiveLeaseExpiresAt().After(now) {
		return false
	}
	return heartbeatTimeout <= 0 || now.Sub(r.HeartbeatAt) <= heartbeatTimeout
}

// EffectiveLeaseExpiresAt caps scale-from-zero services at their absolute
// lease while allowing warm P40 leases to renew indefinitely.
func (r Record) EffectiveLeaseExpiresAt() time.Time {
	if !r.AbsoluteLeaseExpiresAt.IsZero() && r.AbsoluteLeaseExpiresAt.Before(r.LeaseExpiresAt) {
		return r.AbsoluteLeaseExpiresAt
	}
	return r.LeaseExpiresAt
}

func (r Record) Diagnostic() ServiceDiagnostic {
	return ServiceDiagnostic{
		Tier:         r.Tier,
		SlurmJobID:   r.SlurmJobID,
		GPU:          r.GPU,
		ModelProfile: r.ModelProfile,
	}
}

func (r Record) Sanitized(now time.Time, heartbeatTimeout time.Duration) EndpointStatus {
	return EndpointStatus{
		ID:                r.ID,
		State:             r.State,
		Endpoint:          r.Endpoint,
		Healthy:           r.Routable(now, heartbeatTimeout) && r.HealthError == "",
		SlurmJobID:        r.SlurmJobID,
		SchedulerState:    r.SchedulerState,
		HeartbeatAt:       r.HeartbeatAt,
		LeaseExpiresAt:    r.EffectiveLeaseExpiresAt(),
		LastHealthCheckAt: r.LastHealthCheckAt,
		HealthError:       r.HealthError,
		FailureCategory:   r.FailureCategory,
	}
}

type Reservation struct {
	Tier            Tier
	Role            Role
	ModelProfile    string
	Model           string
	Capabilities    []string
	ContextLimit    int
	GPU             GPU
	StartupDeadline time.Time
}

type Publication struct {
	ID                 string
	Tier               Tier
	Endpoint           string
	EndpointAuth       EndpointAuth
	ModelProfile       string
	Model              string
	Capabilities       []string
	ContextLimitTokens int
	GPU                GPU
	SlurmJobID         string
}

func (p Publication) validate() error {
	if strings.TrimSpace(p.ID) == "" {
		return errors.New("service id is required")
	}
	parsed, err := url.Parse(p.Endpoint)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return fmt.Errorf("service endpoint must be an absolute HTTP(S) URL")
	}
	if strings.ToLower(strings.TrimSpace(p.EndpointAuth.Type)) != "bearer" || strings.TrimSpace(p.EndpointAuth.BearerToken) == "" {
		return errors.New("service endpoint requires bearer authentication")
	}
	return nil
}

type ControlUpdate struct {
	State           *ServiceState
	SchedulerState  *string
	HealthCheckedAt *time.Time
	HealthError     *string
	FailureCategory *FailureCategory
	LeaseExpiresAt  *time.Time
	CancelRequested *bool
}

type ServiceJobState string

const (
	JobStateQueued  ServiceJobState = "queued"
	JobStateRunning ServiceJobState = "running"
	JobStateStopped ServiceJobState = "stopped"
	JobStateFailed  ServiceJobState = "failed"
	JobStateUnknown ServiceJobState = "unknown"
)

func (s ServiceJobState) Terminal() bool {
	return s == JobStateStopped || s == JobStateFailed
}

// ServiceJobStatus preserves scheduler failure semantics that determine the
// next adaptive tier. RawState is diagnostic; FailureCategory is the stable
// cross-process contract.
type ServiceJobStatus struct {
	State           ServiceJobState `json:"state"`
	RawState        string          `json:"raw_state,omitempty"`
	FailureCategory FailureCategory `json:"failure_category,omitempty"`
}

func (s ServiceJobStatus) Terminal() bool { return s.State.Terminal() }

type LaunchRequest struct {
	ServiceID                string            `json:"service_id"`
	Tier                     Tier              `json:"tier"`
	Role                     Role              `json:"role"`
	RegistryPath             string            `json:"registry_path"`
	RegistrationToken        string            `json:"registration_token"`
	HeartbeatIntervalSeconds int               `json:"heartbeat_interval_seconds"`
	LeaseDurationSeconds     int               `json:"lease_duration_seconds"`
	Capabilities             []string          `json:"capabilities"`
	Deployment               DeploymentProfile `json:"deployment"`
	Placement                Placement         `json:"placement"`
}

type EndpointStatus struct {
	ID                string          `json:"id"`
	State             ServiceState    `json:"state"`
	Endpoint          string          `json:"endpoint,omitempty"`
	Healthy           bool            `json:"healthy"`
	SlurmJobID        string          `json:"slurm_job_id,omitempty"`
	SchedulerState    string          `json:"scheduler_state,omitempty"`
	HeartbeatAt       time.Time       `json:"heartbeat_at,omitempty"`
	LeaseExpiresAt    time.Time       `json:"lease_expires_at"`
	LastHealthCheckAt time.Time       `json:"last_health_check_at,omitempty"`
	HealthError       string          `json:"health_error,omitempty"`
	FailureCategory   FailureCategory `json:"failure_category,omitempty"`
}

type TierCapabilities struct {
	Tier                Tier             `json:"tier"`
	Role                Role             `json:"role"`
	ModelProfile        string           `json:"model_profile"`
	ContextLimitTokens  int              `json:"context_limit_tokens"`
	GPU                 GPU              `json:"gpu"`
	SupportedOperations []string         `json:"supported_operations"`
	MinReplicas         int              `json:"min_replicas"`
	MaxReplicas         int              `json:"max_replicas"`
	ActiveReplicas      int              `json:"active_replicas"`
	StartingReplicas    int              `json:"starting_replicas"`
	QueueState          map[string]int   `json:"queue_state"`
	Endpoints           []EndpointStatus `json:"endpoints"`
}

type CapabilitiesSnapshot struct {
	Enabled     bool               `json:"enabled"`
	Healthy     bool               `json:"healthy"`
	GeneratedAt time.Time          `json:"generated_at"`
	Tiers       []TierCapabilities `json:"tiers"`
}

func sortedProfiles(profiles map[Tier]Profile) []Profile {
	result := make([]Profile, 0, len(profiles))
	for _, profile := range profiles {
		result = append(result, profile)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Tier < result[j].Tier })
	return result
}
