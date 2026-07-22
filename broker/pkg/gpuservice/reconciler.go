package gpuservice

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

type Scheduler interface {
	SubmitGPUService(context.Context, LaunchRequest) (string, error)
	GPUServiceStatus(context.Context, string) (ServiceJobStatus, error)
	CancelGPUService(context.Context, string) error
}

type HealthChecker interface {
	Check(context.Context, Record) error
}

type ManagerOptions struct {
	Profiles     []Profile
	Timing       Timing
	ControlToken string
	ControlSpool *ControlSpool
}

// Manager is the single owner of scheduler mutations. Request workers can only
// create authenticated demands and await a healthy lease.
type Manager struct {
	registry     Registry
	scheduler    Scheduler
	health       HealthChecker
	profiles     map[Tier]Profile
	timing       Timing
	controlToken string
	controlSpool *ControlSpool
	now          func() time.Time
	mu           sync.Mutex
}

func NewManager(registry Registry, scheduler Scheduler, health HealthChecker, options ManagerOptions) (*Manager, error) {
	if registry == nil || scheduler == nil || health == nil {
		return nil, errors.New("GPU service registry, scheduler, and health checker are required")
	}
	if options.Timing.LeaseDuration <= 0 || options.Timing.HealthInterval <= 0 ||
		options.Timing.HeartbeatTimeout <= 0 || options.Timing.StartupTimeout <= 0 {
		return nil, errors.New("GPU service timings must be positive")
	}
	if strings.TrimSpace(options.ControlToken) == "" {
		return nil, errors.New("GPU service control token is required")
	}
	profiles := make(map[Tier]Profile, len(options.Profiles))
	for _, profile := range options.Profiles {
		if err := profile.Validate(); err != nil {
			return nil, err
		}
		if _, exists := profiles[profile.Tier]; exists {
			return nil, fmt.Errorf("duplicate GPU service profile %s", profile.Tier)
		}
		profiles[profile.Tier] = profile
	}
	if len(profiles) == 0 {
		return nil, errors.New("at least one GPU service profile is required")
	}
	return &Manager{
		registry:     registry,
		scheduler:    scheduler,
		health:       health,
		profiles:     profiles,
		timing:       options.Timing,
		controlToken: options.ControlToken,
		controlSpool: options.ControlSpool,
		now:          time.Now,
	}, nil
}

// Run performs restart recovery immediately and then maintains leases at the
// configured health interval until the context is cancelled.
func (m *Manager) Run(ctx context.Context) error {
	if err := m.Reconcile(ctx); err != nil {
		return err
	}
	ticker := time.NewTicker(m.timing.HealthInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := m.Reconcile(ctx); err != nil {
				return err
			}
		}
	}
}

// Shutdown eagerly cancels and removes every registered GPU service lease
// owned by this manager's registry. This is intended for broker shutdown so
// warm service jobs do not linger until lease expiry.
func (m *Manager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	records, err := m.registry.List(ctx)
	if err != nil {
		return err
	}
	var firstErr error
	for _, record := range records {
		if record.SlurmJobID != "" {
			if err := m.scheduler.CancelGPUService(ctx, record.SlurmJobID); err != nil && firstErr == nil {
				firstErr = fmt.Errorf("cancel GPU service %s (%s): %w", record.ID, record.SlurmJobID, err)
			}
		}
		if err := m.registry.FailDemandsForService(ctx, record.ID, FailureService, "broker shutdown"); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("fail GPU service demands for %s: %w", record.ID, err)
		}
		if err := m.registry.Delete(ctx, record.ID); err != nil && !errors.Is(err, ErrRecordNotFound) && firstErr == nil {
			firstErr = fmt.Errorf("delete GPU service %s: %w", record.ID, err)
		}
	}
	return firstErr
}

// RequestTier records scale demand only. The next reconciler pass owns the
// scheduler launch, which keeps model startup outside inspection workers.
func (m *Manager) RequestTier(ctx context.Context, request DemandRequest) (Demand, error) {
	return m.registry.RequestDemand(ctx, m.controlToken, request)
}

func (m *Manager) AwaitTier(ctx context.Context, demandID string, pollInterval time.Duration) (Record, error) {
	record, err := m.registry.AwaitDemand(ctx, m.controlToken, demandID, pollInterval)
	if err != nil {
		return Record{}, err
	}
	if !record.Routable(m.now().UTC(), m.timing.HeartbeatTimeout) {
		return Record{}, fmt.Errorf("%w: service lease is no longer healthy", ErrDemandFailed)
	}
	return record, nil
}

// Endpoint returns endpoint credentials only for a currently routable lease.
// Capabilities uses a separate sanitized projection and never calls this API.
func (m *Manager) Endpoint(ctx context.Context, tier Tier) (EndpointAccess, error) {
	if !tier.Valid() {
		return EndpointAccess{}, fmt.Errorf("invalid GPU service tier %q", tier)
	}
	records, err := m.registry.List(ctx)
	if err != nil {
		return EndpointAccess{}, err
	}
	now := m.now().UTC()
	for _, record := range records {
		if record.Tier == tier && record.Routable(now, m.timing.HeartbeatTimeout) && record.HealthError == "" {
			return *endpointAccess(record), nil
		}
	}
	return EndpointAccess{}, fmt.Errorf("%w for tier %s", ErrNoHealthyService, tier)
}

func (m *Manager) Reconcile(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.controlSpool != nil {
		if err := m.controlSpool.Import(ctx, m.registry); err != nil {
			return fmt.Errorf("import GPU service demands: %w", err)
		}
	}
	records, err := m.registry.List(ctx)
	if err != nil {
		return err
	}
	now := m.now().UTC()
	for _, record := range records {
		if err := m.reconcileRecord(ctx, record, now); err != nil {
			return err
		}
	}
	if err := m.enforceReplicaLimits(ctx, now); err != nil {
		return err
	}
	if err := m.reconcileDemands(ctx, now); err != nil {
		return err
	}
	if err := m.maintainMinimums(ctx, now); err != nil {
		return err
	}
	if m.controlSpool != nil {
		if err := m.controlSpool.Sync(ctx, m.registry); err != nil {
			return fmt.Errorf("sync GPU service demand responses: %w", err)
		}
	}
	if _, err := m.registry.PruneExpiredDemands(ctx, now); err != nil {
		return fmt.Errorf("prune GPU service demands: %w", err)
	}
	return nil
}

func (m *Manager) reconcileRecord(ctx context.Context, record Record, now time.Time) error {
	if record.State == StateUnhealthy {
		return m.reconcileUnhealthy(ctx, record, now)
	}
	if _, ok := m.profiles[record.Tier]; !ok {
		return m.retire(ctx, record, FailureService, "deployment profile was removed", now)
	}
	if record.SlurmJobID != "" {
		status, err := m.scheduler.GPUServiceStatus(ctx, record.SlurmJobID)
		if err != nil {
			unknown := string(JobStateUnknown)
			message := "scheduler status: " + err.Error()
			if err := m.registry.UpdateControl(ctx, record.ID, ControlUpdate{
				SchedulerState: &unknown,
				HealthError:    &message,
			}); err != nil {
				return err
			}
		} else {
			rawState := strings.TrimSpace(status.RawState)
			if rawState == "" {
				rawState = string(status.State)
			}
			if err := m.registry.UpdateControl(ctx, record.ID, ControlUpdate{SchedulerState: &rawState}); err != nil {
				return err
			}
			if status.Terminal() {
				category := status.FailureCategory
				if category == FailureNone {
					category = FailureService
				}
				return m.retireTerminal(ctx, record, category, "scheduler service job is "+rawState, now)
			}
		}
	}
	if record.State == StateStarting {
		if !record.StartupDeadline.After(now) {
			return m.retire(ctx, record, FailureStartupTimeout, "service startup deadline exceeded", now)
		}
		return nil
	}
	if !record.EffectiveLeaseExpiresAt().After(now) {
		return m.retire(ctx, record, FailureLeaseExpired, "service lease expired", now)
	}
	if record.HeartbeatAt.IsZero() || now.Sub(record.HeartbeatAt) > m.timing.HeartbeatTimeout {
		return m.retire(ctx, record, FailureHeartbeatLost, "service heartbeat is stale", now)
	}
	if err := m.health.Check(ctx, record); err != nil {
		return m.retire(ctx, record, FailureEndpointUnhealthy, err.Error(), now)
	}
	empty := ""
	return m.registry.UpdateControl(ctx, record.ID, ControlUpdate{
		HealthCheckedAt: &now,
		HealthError:     &empty,
	})
}

func (m *Manager) retire(ctx context.Context, record Record, category FailureCategory, reason string, now time.Time) error {
	return m.retireService(ctx, record, category, reason, now, true)
}

func (m *Manager) retireTerminal(ctx context.Context, record Record, category FailureCategory, reason string, now time.Time) error {
	return m.retireService(ctx, record, category, reason, now, false)
}

func (m *Manager) retireService(ctx context.Context, record Record, category FailureCategory, reason string, now time.Time, cancel bool) error {
	state := StateUnhealthy
	cancelDone := !cancel || record.SlurmJobID == ""
	if err := m.registry.UpdateControl(ctx, record.ID, ControlUpdate{
		State:           &state,
		HealthCheckedAt: &now,
		HealthError:     &reason,
		FailureCategory: &category,
		LeaseExpiresAt:  &now,
		CancelRequested: &cancelDone,
	}); err != nil {
		return err
	}
	if err := m.registry.FailDemandsForService(ctx, record.ID, category, reason); err != nil {
		return err
	}
	if cancel && record.SlurmJobID != "" {
		if err := m.scheduler.CancelGPUService(ctx, record.SlurmJobID); err != nil {
			reason += "; cancel failed: " + err.Error()
			_ = m.registry.UpdateControl(ctx, record.ID, ControlUpdate{HealthError: &reason})
			return nil
		}
		cancelDone = true
		return m.registry.UpdateControl(ctx, record.ID, ControlUpdate{CancelRequested: &cancelDone})
	}
	return nil
}

func (m *Manager) reconcileUnhealthy(ctx context.Context, record Record, now time.Time) error {
	category := record.FailureCategory
	if category == FailureNone {
		category = FailureService
	}
	reason := strings.TrimSpace(record.HealthError)
	if reason == "" {
		reason = "GPU service became unhealthy"
	}
	cancelDone := record.CancelRequested || record.SlurmJobID == ""
	terminal := false
	if record.SlurmJobID != "" {
		status, statusErr := m.scheduler.GPUServiceStatus(ctx, record.SlurmJobID)
		if statusErr == nil {
			rawState := strings.TrimSpace(status.RawState)
			if rawState == "" {
				rawState = string(status.State)
			}
			if err := m.registry.UpdateControl(ctx, record.ID, ControlUpdate{SchedulerState: &rawState}); err != nil {
				return err
			}
			terminal = status.Terminal()
			if terminal {
				cancelDone = true
				schedulerCategory := schedulerTerminalFailure(status)
				category = mergeTerminalFailure(category, schedulerCategory)
				schedulerReason := "scheduler service job is " + rawState
				if !strings.Contains(reason, schedulerReason) {
					reason += "; " + schedulerReason
				}
				if err := m.registry.UpdateControl(ctx, record.ID, ControlUpdate{
					FailureCategory: &category,
					HealthError:     &reason,
					CancelRequested: &cancelDone,
				}); err != nil {
					return err
				}
			}
		}
	}

	// The launcher can mark a scale-zero lease unhealthy just before its Slurm
	// job reaches a terminal state. Keep the bound demand open long enough to
	// recover scheduler-owned OOM/timeout classification instead of finalizing
	// the launcher's generic service_failure and choosing the wrong A100 tier.
	if !terminal && awaitsTerminalSchedulerFailure(record) {
		active, err := m.hasActiveBoundDemand(ctx, record.ID)
		if err != nil {
			return err
		}
		if active {
			return nil
		}
	}

	if err := m.registry.FailDemandsForService(ctx, record.ID, category, reason); err != nil {
		return err
	}
	if !cancelDone {
		if record.SlurmJobID != "" {
			if err := m.scheduler.CancelGPUService(ctx, record.SlurmJobID); err != nil {
				reason += "; cancel failed: " + err.Error()
				return m.registry.UpdateControl(ctx, record.ID, ControlUpdate{HealthError: &reason})
			}
			cancelDone = true
		}
		if err := m.registry.UpdateControl(ctx, record.ID, ControlUpdate{CancelRequested: &cancelDone}); err != nil {
			return err
		}
	}
	if cancelDone && !record.LeaseExpiresAt.After(now) {
		if err := m.registry.Delete(ctx, record.ID); err != nil && !errors.Is(err, ErrRecordNotFound) {
			return err
		}
	}
	return nil
}

func schedulerTerminalFailure(status ServiceJobStatus) FailureCategory {
	rawState := strings.ToUpper(strings.TrimSpace(status.RawState))
	switch {
	case strings.Contains(rawState, "OUT_OF_MEMORY"):
		return FailureOOM
	case strings.Contains(rawState, "TIMEOUT"):
		return FailureTimeout
	}
	if status.FailureCategory != FailureNone {
		return status.FailureCategory
	}
	return FailureService
}

func mergeTerminalFailure(current, scheduler FailureCategory) FailureCategory {
	if scheduler == FailureOOM {
		return FailureOOM
	}
	if current == FailureNone || current == FailureService {
		if scheduler != FailureNone {
			return scheduler
		}
		return FailureService
	}
	return current
}

func awaitsTerminalSchedulerFailure(record Record) bool {
	if record.SlurmJobID == "" || (record.FailureCategory != FailureNone && record.FailureCategory != FailureService) {
		return false
	}
	switch record.Tier {
	case TierV100Reasoning, TierA100Single, TierA100Multigpu:
		return true
	default:
		return false
	}
}

func (m *Manager) hasActiveBoundDemand(ctx context.Context, serviceID string) (bool, error) {
	demands, err := m.registry.ListDemands(ctx)
	if err != nil {
		return false, err
	}
	for _, demand := range demands {
		if demand.ServiceID == serviceID && (demand.State == DemandLaunching || demand.State == DemandReady) {
			return true, nil
		}
	}
	return false, nil
}

func (m *Manager) enforceReplicaLimits(ctx context.Context, now time.Time) error {
	records, err := m.registry.List(ctx)
	if err != nil {
		return err
	}
	for tier, profile := range m.profiles {
		active := recordsForTier(records, tier, now)
		sort.Slice(active, func(i, j int) bool { return active[i].CreatedAt.Before(active[j].CreatedAt) })
		if len(active) <= profile.MaxReplicas {
			continue
		}
		for _, record := range active[profile.MaxReplicas:] {
			if err := m.retire(ctx, record, FailureService, "configured replica maximum exceeded", now); err != nil {
				return err
			}
		}
	}
	return nil
}

func (m *Manager) reconcileDemands(ctx context.Context, now time.Time) error {
	demands, err := m.registry.PendingDemands(ctx)
	if err != nil {
		return err
	}
demandLoop:
	for _, demand := range demands {
		profile, ok := m.profiles[demand.Tier]
		if !ok {
			if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{State: DemandFailed, FailureCategory: FailureService, Error: "tier is not configured"}); err != nil {
				return err
			}
			continue
		}
		records, err := m.registry.List(ctx)
		if err != nil {
			return err
		}
		if demand.ServiceID != "" {
			for _, record := range records {
				if record.ID != demand.ServiceID {
					continue
				}
				switch {
				case record.State == StateUnhealthy:
					if awaitsTerminalSchedulerFailure(record) {
						continue demandLoop
					}
					category := record.FailureCategory
					if category == FailureNone {
						category = FailureService
					}
					if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{
						State: DemandFailed, FailureCategory: category, Error: record.HealthError,
					}); err != nil {
						return err
					}
				case record.Routable(now, m.timing.HeartbeatTimeout) && record.HealthError == "":
					if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{State: DemandReady, ServiceID: record.ID}); err != nil {
						return err
					}
				case record.State == StateStarting && record.EffectiveLeaseExpiresAt().After(now):
					// Keep waiting on the exact service originally bound to this
					// demand. A terminal result must fail instead of relaunching.
				default:
					if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{
						State: DemandFailed, FailureCategory: FailureService, Error: "bound GPU service is unavailable",
					}); err != nil {
						return err
					}
				}
				continue demandLoop
			}
			if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{
				State: DemandFailed, FailureCategory: FailureService, Error: "bound GPU service record disappeared",
			}); err != nil {
				return err
			}
			continue
		}
		var starting *Record
		for i := range records {
			record := records[i]
			if record.Tier != demand.Tier {
				continue
			}
			if record.Routable(now, m.timing.HeartbeatTimeout) && record.HealthError == "" {
				if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{State: DemandReady, ServiceID: record.ID}); err != nil {
					return err
				}
				starting = nil
				break
			}
			if record.State == StateStarting && record.EffectiveLeaseExpiresAt().After(now) {
				copy := record
				starting = &copy
			}
		}
		updated, err := m.registry.GetDemand(ctx, demand.ID)
		if err != nil {
			return err
		}
		if updated.State == DemandReady {
			continue
		}
		if starting != nil {
			if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{State: DemandLaunching, ServiceID: starting.ID}); err != nil {
				return err
			}
			continue
		}
		record, err := m.launch(ctx, profile, now)
		if err != nil {
			if errors.Is(err, ErrReplicaLimit) {
				continue
			}
			if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{State: DemandFailed, FailureCategory: FailureService, Error: err.Error()}); err != nil {
				return err
			}
			continue
		}
		if err := m.registry.UpdateDemand(ctx, demand.ID, DemandUpdate{State: DemandLaunching, ServiceID: record.ID}); err != nil {
			return err
		}
	}
	return nil
}

func (m *Manager) maintainMinimums(ctx context.Context, now time.Time) error {
	records, err := m.registry.List(ctx)
	if err != nil {
		return err
	}
	for _, profile := range sortedProfiles(m.profiles) {
		count := len(recordsForTier(records, profile.Tier, now))
		for count < profile.MinReplicas {
			if _, err := m.launch(ctx, profile, now); err != nil {
				return fmt.Errorf("launch minimum replica for %s: %w", profile.Tier, err)
			}
			count++
		}
	}
	return nil
}

func (m *Manager) launch(ctx context.Context, profile Profile, now time.Time) (Record, error) {
	reservation := Reservation{
		Tier:            profile.Tier,
		Role:            profile.Role,
		ModelProfile:    profile.Deployment.Name,
		Model:           profile.Deployment.Model,
		Capabilities:    append([]string(nil), profile.SupportedOperations...),
		ContextLimit:    profile.Deployment.ContextLimitTokens,
		GPU:             profile.Placement.GPU,
		StartupDeadline: now.Add(m.timing.StartupTimeout),
	}
	record, registrationToken, err := m.registry.Reserve(ctx, reservation, profile.MaxReplicas)
	if err != nil {
		return Record{}, err
	}
	request := LaunchRequest{
		ServiceID:                record.ID,
		Tier:                     profile.Tier,
		Role:                     profile.Role,
		RegistryPath:             m.registry.Path(),
		RegistrationToken:        registrationToken,
		HeartbeatIntervalSeconds: max(1, int(m.timing.HealthInterval/time.Second)),
		LeaseDurationSeconds:     max(1, int(m.timing.LeaseDuration/time.Second)),
		Capabilities:             append([]string(nil), profile.SupportedOperations...),
		Deployment:               profile.Deployment,
		Placement:                profile.Placement,
	}
	jobID, err := m.scheduler.SubmitGPUService(ctx, request)
	if err != nil {
		_ = m.registry.Delete(ctx, record.ID)
		return Record{}, fmt.Errorf("submit %s service: %w", profile.Tier, err)
	}
	if err := m.registry.AttachJob(ctx, record.ID, jobID); err != nil {
		_ = m.scheduler.CancelGPUService(ctx, jobID)
		_ = m.registry.Delete(ctx, record.ID)
		return Record{}, err
	}
	queued := string(JobStateQueued)
	if err := m.registry.UpdateControl(ctx, record.ID, ControlUpdate{SchedulerState: &queued}); err != nil {
		return Record{}, err
	}
	record.SlurmJobID = jobID
	record.SchedulerState = queued
	return record, nil
}

func recordsForTier(records []Record, tier Tier, now time.Time) []Record {
	result := make([]Record, 0)
	for _, record := range records {
		if record.Tier == tier && (record.State == StateStarting || record.State == StateReady) && record.EffectiveLeaseExpiresAt().After(now) {
			result = append(result, record)
		}
	}
	return result
}

// Capabilities is read-only and never exposes endpoint bearer credentials.
func (m *Manager) Capabilities(ctx context.Context) (CapabilitiesSnapshot, error) {
	records, err := m.registry.List(ctx)
	if err != nil {
		return CapabilitiesSnapshot{}, err
	}
	now := m.now().UTC()
	snapshot := CapabilitiesSnapshot{Enabled: true, GeneratedAt: now}
	warmHealthy := map[Tier]bool{
		TierP40Retrieval: false,
		TierP40Synthesis: false,
	}
	for _, profile := range sortedProfiles(m.profiles) {
		tier := TierCapabilities{
			Tier:                profile.Tier,
			Role:                profile.Role,
			ModelProfile:        profile.Deployment.Name,
			ContextLimitTokens:  profile.Deployment.ContextLimitTokens,
			GPU:                 profile.Placement.GPU,
			SupportedOperations: append([]string(nil), profile.SupportedOperations...),
			MinReplicas:         profile.MinReplicas,
			MaxReplicas:         profile.MaxReplicas,
			QueueState:          map[string]int{},
		}
		for _, record := range records {
			if record.Tier != profile.Tier {
				continue
			}
			if record.Routable(now, m.timing.HeartbeatTimeout) && record.HealthError == "" {
				tier.ActiveReplicas++
			}
			if record.State == StateStarting && record.EffectiveLeaseExpiresAt().After(now) {
				tier.StartingReplicas++
			}
			queueState := record.SchedulerState
			if queueState == "" {
				queueState = string(record.State)
			}
			tier.QueueState[queueState]++
			tier.Endpoints = append(tier.Endpoints, record.Sanitized(now, m.timing.HeartbeatTimeout))
		}
		sort.Slice(tier.Endpoints, func(i, j int) bool { return tier.Endpoints[i].ID < tier.Endpoints[j].ID })
		if tier.ActiveReplicas > 0 {
			if _, required := warmHealthy[profile.Tier]; required {
				warmHealthy[profile.Tier] = true
			}
		}
		snapshot.Tiers = append(snapshot.Tiers, tier)
	}
	snapshot.Healthy = warmHealthy[TierP40Retrieval] && warmHealthy[TierP40Synthesis]
	return snapshot, nil
}
