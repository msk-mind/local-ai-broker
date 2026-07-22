package gpuservice

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
)

var (
	ErrReplicaLimit          = errors.New("GPU service replica limit reached")
	ErrRecordNotFound        = errors.New("GPU service record not found")
	ErrRegistrationDenied    = errors.New("GPU service registration denied")
	ErrControlDenied         = errors.New("GPU service control request denied")
	ErrDemandFailed          = errors.New("GPU service demand failed")
	ErrNoHealthyService      = errors.New("no healthy GPU service lease")
	ErrFailureReportRejected = errors.New("GPU service failure report rejected")
)

type Registry interface {
	Path() string
	Reserve(context.Context, Reservation, int) (Record, string, error)
	Publish(context.Context, string, Publication) (Record, error)
	Renew(context.Context, string, string) (Record, error)
	AttachJob(context.Context, string, string) error
	UpdateControl(context.Context, string, ControlUpdate) error
	List(context.Context) ([]Record, error)
	Delete(context.Context, string) error
	RequestDemand(context.Context, string, DemandRequest) (Demand, error)
	PendingDemands(context.Context) ([]Demand, error)
	ListDemands(context.Context) ([]Demand, error)
	GetDemand(context.Context, string) (Demand, error)
	UpdateDemand(context.Context, string, DemandUpdate) error
	FailDemandsForService(context.Context, string, FailureCategory, string) error
	PruneExpiredDemands(context.Context, time.Time) (int, error)
	ReportServiceFailure(context.Context, string, ServiceFailure) error
	AwaitDemand(context.Context, string, string, time.Duration) (Record, error)
}

type registryFile struct {
	Schema    string    `json:"schema"`
	UpdatedAt time.Time `json:"updated_at"`
	Records   []Record  `json:"records"`
	Demands   []Demand  `json:"demands,omitempty"`
}

const registrySchema = "gpu_service_registry_v1"

// FileRegistry is an atomic, process-safe registry intended for a shared
// filesystem. A separate flock file serializes read/modify/write operations,
// and the registry and lock are mode 0600 because endpoint credentials live in
// the internal record.
type FileRegistry struct {
	path               string
	leaseDuration      time.Duration
	controlTokenSHA256 string
	mu                 sync.Mutex
	now                func() time.Time
}

func NewFileRegistry(path string, leaseDuration time.Duration) (*FileRegistry, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, errors.New("GPU service registry path is required")
	}
	if leaseDuration <= 0 {
		return nil, errors.New("GPU service lease duration must be positive")
	}
	return &FileRegistry{
		path:          filepath.Clean(path),
		leaseDuration: leaseDuration,
		now:           time.Now,
	}, nil
}

func NewAuthenticatedFileRegistry(path string, leaseDuration time.Duration, controlToken string) (*FileRegistry, error) {
	registry, err := NewFileRegistry(path, leaseDuration)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(controlToken) == "" {
		return nil, errors.New("GPU service control token is required")
	}
	hash := sha256.Sum256([]byte(controlToken))
	registry.controlTokenSHA256 = hex.EncodeToString(hash[:])
	return registry, nil
}

func (r *FileRegistry) Path() string { return r.path }

func (r *FileRegistry) Reserve(ctx context.Context, reservation Reservation, maxReplicas int) (Record, string, error) {
	if err := validateReservation(reservation); err != nil {
		return Record{}, "", err
	}
	if maxReplicas < 1 {
		return Record{}, "", errors.New("maximum replicas must be positive")
	}

	var created Record
	var token string
	err := r.mutate(ctx, func(data *registryFile) error {
		now := r.now().UTC()
		active := 0
		for _, record := range data.Records {
			if record.Tier == reservation.Tier && (record.State == StateStarting || record.State == StateReady) && record.EffectiveLeaseExpiresAt().After(now) {
				active++
			}
		}
		if active >= maxReplicas {
			return fmt.Errorf("%w for tier %s (%d)", ErrReplicaLimit, reservation.Tier, maxReplicas)
		}

		idSuffix, err := randomHex(8)
		if err != nil {
			return fmt.Errorf("generate service id: %w", err)
		}
		token, err = randomHex(32)
		if err != nil {
			return fmt.Errorf("generate registration token: %w", err)
		}
		hash := sha256.Sum256([]byte(token))
		created = Record{
			ID:                      "gpu-" + string(reservation.Tier) + "-" + idSuffix,
			Tier:                    reservation.Tier,
			Role:                    reservation.Role,
			State:                   StateStarting,
			ModelProfile:            reservation.ModelProfile,
			Model:                   reservation.Model,
			Capabilities:            append([]string(nil), reservation.Capabilities...),
			ContextLimitTokens:      reservation.ContextLimit,
			GPU:                     reservation.GPU,
			CreatedAt:               now,
			StartupDeadline:         reservation.StartupDeadline.UTC(),
			LeaseExpiresAt:          now.Add(r.leaseDuration),
			RegistrationTokenSHA256: hex.EncodeToString(hash[:]),
		}
		if reservation.Tier != TierP40Retrieval && reservation.Tier != TierP40Synthesis {
			created.AbsoluteLeaseExpiresAt = created.LeaseExpiresAt
		}
		data.Records = append(data.Records, created)
		return nil
	})
	return cloneRecord(created), token, err
}

func (r *FileRegistry) Publish(ctx context.Context, registrationToken string, publication Publication) (Record, error) {
	if err := publication.validate(); err != nil {
		return Record{}, err
	}
	var published Record
	err := r.mutate(ctx, func(data *registryFile) error {
		record, err := findRecord(data, publication.ID)
		if err != nil {
			return err
		}
		if !registrationTokenMatches(*record, registrationToken) {
			return ErrRegistrationDenied
		}
		if record.State == StateUnhealthy {
			return errors.New("cannot publish an unhealthy service reservation")
		}
		if publication.Tier != record.Tier || publication.ModelProfile != record.ModelProfile || publication.Model != record.Model ||
			publication.ContextLimitTokens != record.ContextLimitTokens || publication.GPU != record.GPU ||
			!equalStringSet(publication.Capabilities, record.Capabilities) {
			return fmt.Errorf("%w: published service metadata does not match its reservation", ErrRegistrationDenied)
		}
		now := r.now().UTC()
		record.State = StateReady
		record.Endpoint = publication.Endpoint
		record.EndpointAuth = publication.EndpointAuth
		record.SlurmJobID = firstNonEmpty(record.SlurmJobID, publication.SlurmJobID)
		record.HeartbeatAt = now
		record.LeaseExpiresAt = boundedLeaseExpiry(*record, now.Add(r.leaseDuration))
		record.HealthError = ""
		record.FailureCategory = FailureNone
		published = cloneRecord(*record)
		return nil
	})
	return published, err
}

func (r *FileRegistry) Renew(ctx context.Context, id, registrationToken string) (Record, error) {
	var renewed Record
	err := r.mutate(ctx, func(data *registryFile) error {
		record, err := findRecord(data, id)
		if err != nil {
			return err
		}
		if !registrationTokenMatches(*record, registrationToken) {
			return ErrRegistrationDenied
		}
		if record.State != StateReady {
			return fmt.Errorf("service %s is not ready", id)
		}
		now := r.now().UTC()
		record.HeartbeatAt = now
		record.LeaseExpiresAt = boundedLeaseExpiry(*record, now.Add(r.leaseDuration))
		renewed = cloneRecord(*record)
		return nil
	})
	return renewed, err
}

func boundedLeaseExpiry(record Record, candidate time.Time) time.Time {
	if !record.AbsoluteLeaseExpiresAt.IsZero() && record.AbsoluteLeaseExpiresAt.Before(candidate) {
		return record.AbsoluteLeaseExpiresAt
	}
	return candidate
}

func (r *FileRegistry) AttachJob(ctx context.Context, id, slurmJobID string) error {
	slurmJobID = strings.TrimSpace(slurmJobID)
	if slurmJobID == "" {
		return errors.New("Slurm job id is required")
	}
	return r.mutate(ctx, func(data *registryFile) error {
		record, err := findRecord(data, id)
		if err != nil {
			return err
		}
		if record.SlurmJobID != "" && record.SlurmJobID != slurmJobID {
			return fmt.Errorf("service %s is already attached to Slurm job %s", id, record.SlurmJobID)
		}
		record.SlurmJobID = slurmJobID
		return nil
	})
}

func (r *FileRegistry) UpdateControl(ctx context.Context, id string, update ControlUpdate) error {
	return r.mutate(ctx, func(data *registryFile) error {
		record, err := findRecord(data, id)
		if err != nil {
			return err
		}
		if update.State != nil {
			record.State = *update.State
		}
		if update.SchedulerState != nil {
			record.SchedulerState = *update.SchedulerState
		}
		if update.HealthCheckedAt != nil {
			record.LastHealthCheckAt = update.HealthCheckedAt.UTC()
		}
		if update.HealthError != nil {
			record.HealthError = *update.HealthError
		}
		if update.FailureCategory != nil {
			record.FailureCategory = *update.FailureCategory
		}
		if update.LeaseExpiresAt != nil {
			record.LeaseExpiresAt = update.LeaseExpiresAt.UTC()
		}
		if update.CancelRequested != nil {
			record.CancelRequested = *update.CancelRequested
		}
		return nil
	})
}

func (r *FileRegistry) List(ctx context.Context) ([]Record, error) {
	var result []Record
	err := r.read(ctx, func(data registryFile) error {
		result = make([]Record, 0, len(data.Records))
		for _, record := range data.Records {
			result = append(result, cloneRecord(record))
		}
		sort.Slice(result, func(i, j int) bool {
			if result[i].Tier == result[j].Tier {
				return result[i].CreatedAt.Before(result[j].CreatedAt)
			}
			return result[i].Tier < result[j].Tier
		})
		return nil
	})
	return result, err
}

func (r *FileRegistry) Delete(ctx context.Context, id string) error {
	return r.mutate(ctx, func(data *registryFile) error {
		for i := range data.Records {
			if data.Records[i].ID == id {
				data.Records = append(data.Records[:i], data.Records[i+1:]...)
				return nil
			}
		}
		return ErrRecordNotFound
	})
}

// RequestDemand is the authenticated, scheduler-free scale-from-zero protocol.
// Concurrent requests for the same tier coalesce into one pending demand.
func (r *FileRegistry) RequestDemand(ctx context.Context, controlToken string, request DemandRequest) (Demand, error) {
	if !r.controlTokenMatches(controlToken) {
		return Demand{}, ErrControlDenied
	}
	if !request.Tier.Valid() {
		return Demand{}, fmt.Errorf("invalid demand tier %q", request.Tier)
	}
	if request.TTL <= 0 {
		return Demand{}, errors.New("demand TTL must be positive")
	}

	var result Demand
	err := r.mutate(ctx, func(data *registryFile) error {
		now := r.now().UTC()
		for i := range data.Demands {
			demand := &data.Demands[i]
			if demand.Tier == request.Tier && demand.Deadline.After(now) &&
				(demand.State == DemandPending || demand.State == DemandLaunching) {
				result = *demand
				return nil
			}
		}
		suffix, err := randomHex(8)
		if err != nil {
			return err
		}
		result = Demand{
			ID:              "gpu-demand-" + suffix,
			Tier:            request.Tier,
			State:           DemandPending,
			FailureCategory: request.FailureCategory,
			Reason:          strings.TrimSpace(request.Reason),
			RequestedAt:     now,
			Deadline:        now.Add(request.TTL),
		}
		data.Demands = append(data.Demands, result)
		return nil
	})
	return result, err
}

func (r *FileRegistry) PendingDemands(ctx context.Context) ([]Demand, error) {
	var pending []Demand
	err := r.mutate(ctx, func(data *registryFile) error {
		now := r.now().UTC()
		for i := range data.Demands {
			demand := &data.Demands[i]
			if (demand.State == DemandPending || demand.State == DemandLaunching) && !demand.Deadline.After(now) {
				demand.State = DemandFailed
				demand.FailureCategory = FailureQueueDelay
				demand.Error = "demand deadline expired"
			}
			if demand.State == DemandPending || demand.State == DemandLaunching {
				pending = append(pending, *demand)
			}
		}
		sort.Slice(pending, func(i, j int) bool { return pending[i].RequestedAt.Before(pending[j].RequestedAt) })
		return nil
	})
	return pending, err
}

func (r *FileRegistry) ListDemands(ctx context.Context) ([]Demand, error) {
	var demands []Demand
	err := r.read(ctx, func(data registryFile) error {
		demands = append([]Demand(nil), data.Demands...)
		sort.Slice(demands, func(i, j int) bool { return demands[i].RequestedAt.Before(demands[j].RequestedAt) })
		return nil
	})
	return demands, err
}

func (r *FileRegistry) GetDemand(ctx context.Context, id string) (Demand, error) {
	var result Demand
	err := r.read(ctx, func(data registryFile) error {
		for _, demand := range data.Demands {
			if demand.ID == id {
				result = demand
				return nil
			}
		}
		return ErrRecordNotFound
	})
	return result, err
}

func (r *FileRegistry) UpdateDemand(ctx context.Context, id string, update DemandUpdate) error {
	if update.State != DemandPending && update.State != DemandLaunching && update.State != DemandReady && update.State != DemandFailed {
		return fmt.Errorf("invalid demand state %q", update.State)
	}
	return r.mutate(ctx, func(data *registryFile) error {
		for i := range data.Demands {
			if data.Demands[i].ID != id {
				continue
			}
			data.Demands[i].State = update.State
			if serviceID := strings.TrimSpace(update.ServiceID); serviceID != "" {
				data.Demands[i].ServiceID = serviceID
				for _, record := range data.Records {
					if record.ID == serviceID {
						diagnostic := record.Diagnostic()
						data.Demands[i].ServiceDiagnostic = &diagnostic
						break
					}
				}
			}
			if update.FailureCategory != FailureNone {
				data.Demands[i].FailureCategory = update.FailureCategory
			}
			data.Demands[i].Error = strings.TrimSpace(update.Error)
			return nil
		}
		return ErrRecordNotFound
	})
}

func (r *FileRegistry) FailDemandsForService(ctx context.Context, serviceID string, category FailureCategory, reason string) error {
	serviceID = strings.TrimSpace(serviceID)
	if serviceID == "" {
		return errors.New("service id is required")
	}
	if category == FailureNone {
		category = FailureService
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "bound GPU service failed"
	}
	return r.mutate(ctx, func(data *registryFile) error {
		var serviceDiagnostic *ServiceDiagnostic
		for _, record := range data.Records {
			if record.ID == serviceID {
				diagnostic := record.Diagnostic()
				serviceDiagnostic = &diagnostic
				break
			}
		}
		for i := range data.Demands {
			demand := &data.Demands[i]
			if demand.ServiceID != serviceID || (demand.State != DemandLaunching && demand.State != DemandReady) {
				continue
			}
			demand.State = DemandFailed
			demand.FailureCategory = category
			demand.Error = reason
			if serviceDiagnostic != nil {
				diagnostic := *serviceDiagnostic
				demand.ServiceDiagnostic = &diagnostic
			}
		}
		return nil
	})
}

func (r *FileRegistry) PruneExpiredDemands(ctx context.Context, before time.Time) (int, error) {
	before = before.UTC()
	pruned := 0
	err := r.mutate(ctx, func(data *registryFile) error {
		kept := data.Demands[:0]
		for _, demand := range data.Demands {
			terminal := demand.State == DemandReady || demand.State == DemandFailed
			if terminal && !demand.Deadline.After(before) {
				pruned++
				continue
			}
			kept = append(kept, demand)
		}
		data.Demands = kept
		return nil
	})
	return pruned, err
}

func (r *FileRegistry) ReportServiceFailure(ctx context.Context, controlToken string, failure ServiceFailure) error {
	if !r.controlTokenMatches(controlToken) {
		return ErrControlDenied
	}
	if failure.Tier != TierP40Retrieval && failure.Tier != TierP40Synthesis {
		return fmt.Errorf("%w: only P40 leases may be reported", ErrFailureReportRejected)
	}
	if !reportableServiceFailure(failure.FailureCategory) {
		return fmt.Errorf("%w: unsupported failure category %q", ErrFailureReportRejected, failure.FailureCategory)
	}
	reason := strings.TrimSpace(failure.Reason)
	if reason == "" {
		reason = "request worker reported GPU service failure"
	}
	if len(reason) > 4096 {
		reason = reason[:4096]
	}
	return r.mutate(ctx, func(data *registryFile) error {
		record, err := findRecord(data, strings.TrimSpace(failure.ServiceID))
		if err != nil {
			return err
		}
		if record.Tier != failure.Tier {
			return fmt.Errorf("%w: service tier does not match", ErrFailureReportRejected)
		}
		if record.State == StateUnhealthy {
			return nil
		}
		if record.State != StateReady {
			return fmt.Errorf("%w: service lease is not ready", ErrFailureReportRejected)
		}
		now := r.now().UTC()
		record.State = StateUnhealthy
		record.HealthError = reason
		record.FailureCategory = failure.FailureCategory
		record.LastHealthCheckAt = now
		record.LeaseExpiresAt = now
		record.CancelRequested = false
		return nil
	})
}

func reportableServiceFailure(category FailureCategory) bool {
	switch category {
	case FailureUnavailable, FailureTimeout, FailureService, FailureAuthentication,
		FailureOOM, FailureHeartbeatLost, FailureEndpointUnhealthy:
		return true
	default:
		return false
	}
}

// AwaitDemand lets an authenticated caller wait for the reconciler to bind a
// healthy lease. It never submits a scheduler job itself.
func (r *FileRegistry) AwaitDemand(ctx context.Context, controlToken, demandID string, pollInterval time.Duration) (Record, error) {
	if !r.controlTokenMatches(controlToken) {
		return Record{}, ErrControlDenied
	}
	if pollInterval <= 0 {
		pollInterval = 250 * time.Millisecond
	}
	for {
		var demand Demand
		var record Record
		err := r.read(ctx, func(data registryFile) error {
			found := false
			for _, candidate := range data.Demands {
				if candidate.ID == demandID {
					demand = candidate
					found = true
					break
				}
			}
			if !found {
				return ErrRecordNotFound
			}
			if demand.State == DemandReady {
				for _, candidate := range data.Records {
					if candidate.ID == demand.ServiceID {
						record = cloneRecord(candidate)
						return nil
					}
				}
				return ErrRecordNotFound
			}
			return nil
		})
		if err != nil {
			return Record{}, err
		}
		switch demand.State {
		case DemandReady:
			if record.State == StateReady && record.EffectiveLeaseExpiresAt().After(r.now().UTC()) {
				return record, nil
			}
			return Record{}, fmt.Errorf("%w: bound service is not ready", ErrDemandFailed)
		case DemandFailed:
			return Record{}, fmt.Errorf("%w: %s", ErrDemandFailed, demand.Error)
		}
		if !demand.Deadline.After(r.now().UTC()) {
			return Record{}, fmt.Errorf("%w: demand deadline expired", ErrDemandFailed)
		}
		timer := time.NewTimer(pollInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return Record{}, ctx.Err()
		case <-timer.C:
		}
	}
}

func validateReservation(reservation Reservation) error {
	if !reservation.Tier.Valid() {
		return fmt.Errorf("invalid tier %q", reservation.Tier)
	}
	if reservation.Role != RoleRetrieval && reservation.Role != RoleSynthesis {
		return fmt.Errorf("invalid role %q", reservation.Role)
	}
	if strings.TrimSpace(reservation.ModelProfile) == "" || strings.TrimSpace(reservation.Model) == "" {
		return errors.New("model profile and model are required")
	}
	if reservation.ContextLimit <= 0 || reservation.GPU.Count <= 0 || strings.TrimSpace(reservation.GPU.Type) == "" {
		return errors.New("context limit and GPU placement are required")
	}
	if len(reservation.Capabilities) == 0 {
		return errors.New("capabilities are required")
	}
	if reservation.StartupDeadline.IsZero() {
		return errors.New("startup deadline is required")
	}
	return nil
}

func registrationTokenMatches(record Record, token string) bool {
	hash := sha256.Sum256([]byte(token))
	actual := hex.EncodeToString(hash[:])
	return token != "" && subtle.ConstantTimeCompare([]byte(actual), []byte(record.RegistrationTokenSHA256)) == 1
}

func (r *FileRegistry) controlTokenMatches(token string) bool {
	if token == "" || r.controlTokenSHA256 == "" {
		return false
	}
	hash := sha256.Sum256([]byte(token))
	actual := hex.EncodeToString(hash[:])
	return subtle.ConstantTimeCompare([]byte(actual), []byte(r.controlTokenSHA256)) == 1
}

func findRecord(data *registryFile, id string) (*Record, error) {
	for i := range data.Records {
		if data.Records[i].ID == id {
			return &data.Records[i], nil
		}
	}
	return nil, ErrRecordNotFound
}

func (r *FileRegistry) read(ctx context.Context, fn func(registryFile) error) error {
	return r.withLock(ctx, false, func(data *registryFile) error { return fn(*data) })
}

func (r *FileRegistry) mutate(ctx context.Context, fn func(*registryFile) error) error {
	return r.withLock(ctx, true, fn)
}

func (r *FileRegistry) withLock(ctx context.Context, write bool, fn func(*registryFile) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	dir := filepath.Dir(r.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create registry directory: %w", err)
	}
	lock, err := os.OpenFile(r.path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("open registry lock: %w", err)
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		return fmt.Errorf("lock registry: %w", err)
	}
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN) //nolint:errcheck
	if err := ctx.Err(); err != nil {
		return err
	}

	data, err := r.load()
	if err != nil {
		return err
	}
	if err := fn(&data); err != nil {
		return err
	}
	if !write {
		return nil
	}
	data.Schema = registrySchema
	data.UpdatedAt = r.now().UTC()
	return r.save(data)
}

func (r *FileRegistry) load() (registryFile, error) {
	content, err := os.ReadFile(r.path)
	if errors.Is(err, os.ErrNotExist) {
		return registryFile{Schema: registrySchema, Records: []Record{}, Demands: []Demand{}}, nil
	}
	if err != nil {
		return registryFile{}, fmt.Errorf("read GPU service registry: %w", err)
	}
	var data registryFile
	if err := json.Unmarshal(content, &data); err != nil {
		return registryFile{}, fmt.Errorf("decode GPU service registry: %w", err)
	}
	if data.Schema != registrySchema {
		return registryFile{}, fmt.Errorf("unsupported GPU service registry schema %q", data.Schema)
	}
	return data, nil
}

func (r *FileRegistry) save(data registryFile) error {
	content, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("encode GPU service registry: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(r.path), ".gpu-services-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary GPU service registry: %w", err)
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
	if err := os.Rename(tmpPath, r.path); err != nil {
		return fmt.Errorf("replace GPU service registry: %w", err)
	}
	return os.Chmod(r.path, 0o600)
}

func randomHex(byteCount int) (string, error) {
	value := make([]byte, byteCount)
	if _, err := rand.Read(value); err != nil {
		return "", err
	}
	return hex.EncodeToString(value), nil
}

func equalStringSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	left := append([]string(nil), a...)
	right := append([]string(nil), b...)
	sort.Strings(left)
	sort.Strings(right)
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func cloneRecord(record Record) Record {
	record.Capabilities = append([]string(nil), record.Capabilities...)
	return record
}
