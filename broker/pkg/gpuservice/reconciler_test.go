package gpuservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeServiceScheduler struct {
	mu        sync.Mutex
	launches  []LaunchRequest
	statuses  map[string]ServiceJobStatus
	cancelled []string
}

func newFakeServiceScheduler() *fakeServiceScheduler {
	return &fakeServiceScheduler{statuses: map[string]ServiceJobStatus{}}
}

func (f *fakeServiceScheduler) SubmitGPUService(_ context.Context, request LaunchRequest) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	jobID := fmt.Sprintf("job-%d", len(f.launches)+1)
	f.launches = append(f.launches, request)
	f.statuses[jobID] = ServiceJobStatus{State: JobStateQueued, RawState: "PENDING"}
	return jobID, nil
}

func (f *fakeServiceScheduler) GPUServiceStatus(_ context.Context, jobID string) (ServiceJobStatus, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	state, ok := f.statuses[jobID]
	if !ok {
		return ServiceJobStatus{State: JobStateUnknown}, errors.New("unknown job")
	}
	return state, nil
}

func (f *fakeServiceScheduler) CancelGPUService(_ context.Context, jobID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cancelled = append(f.cancelled, jobID)
	f.statuses[jobID] = ServiceJobStatus{State: JobStateStopped, RawState: "CANCELLED", FailureCategory: FailureService}
	return nil
}

func (f *fakeServiceScheduler) launchCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.launches)
}

func (f *fakeServiceScheduler) launch(index int) LaunchRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.launches[index]
}

func (f *fakeServiceScheduler) setStatus(jobID string, status ServiceJobStatus) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.statuses[jobID] = status
}

type fakeHealthChecker struct {
	mu     sync.Mutex
	errors map[string]error
}

func (f *fakeHealthChecker) Check(_ context.Context, record Record) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.errors[record.ID]
}

func (f *fakeHealthChecker) fail(id string, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.errors[id] = err
}

func TestManagerMaintainsWarmP40ServicesAndRecoversAfterRestart(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	profiles := []Profile{
		testProfile(TierP40Retrieval, 1, 2),
		testProfile(TierP40Synthesis, 1, 2),
	}
	manager := testManager(t, registry, scheduler, health, profiles, &now)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatalf("initial reconcile: %v", err)
	}
	if scheduler.launchCount() != 2 {
		t.Fatalf("expected two warm service launches, got %d", scheduler.launchCount())
	}

	// A new manager instance must adopt the shared starting leases instead of
	// launching duplicate models.
	restarted := testManager(t, registry, scheduler, health, profiles, &now)
	if err := restarted.Reconcile(context.Background()); err != nil {
		t.Fatalf("restart reconcile: %v", err)
	}
	if scheduler.launchCount() != 2 {
		t.Fatalf("restart launched duplicates: %d", scheduler.launchCount())
	}

	records, err := registry.List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range records {
		var request LaunchRequest
		for i := 0; i < scheduler.launchCount(); i++ {
			candidate := scheduler.launch(i)
			if candidate.ServiceID == record.ID {
				request = candidate
				break
			}
		}
		publication := testPublication(record)
		publication.SlurmJobID = record.SlurmJobID
		if _, err := registry.Publish(context.Background(), request.RegistrationToken, publication); err != nil {
			t.Fatalf("publish %s: %v", record.ID, err)
		}
	}
	if err := restarted.Reconcile(context.Background()); err != nil {
		t.Fatalf("healthy reconcile: %v", err)
	}
	snapshot, err := restarted.Capabilities(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !snapshot.Enabled || !snapshot.Healthy || len(snapshot.Tiers) != 2 {
		t.Fatalf("unexpected capabilities: %#v", snapshot)
	}
	encoded, _ := json.Marshal(snapshot)
	if strings.Contains(string(encoded), "endpoint-secret") || strings.Contains(string(encoded), "/models/") {
		t.Fatalf("capabilities leaked endpoint credentials or model paths: %s", encoded)
	}
}

func TestManagerShutdownCancelsWarmServicesAndDeletesLeases(t *testing.T) {
	now := time.Date(2026, 7, 13, 12, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	profiles := []Profile{
		testProfile(TierP40Retrieval, 1, 1),
		testProfile(TierP40Synthesis, 1, 1),
	}
	manager := testManager(t, registry, scheduler, health, profiles, &now)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatalf("initial reconcile: %v", err)
	}
	if scheduler.launchCount() != 2 {
		t.Fatalf("expected two warm service launches, got %d", scheduler.launchCount())
	}
	records, err := registry.List(context.Background())
	if err != nil {
		t.Fatalf("list records: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("expected two registry records, got %d", len(records))
	}

	if err := manager.Shutdown(context.Background()); err != nil {
		t.Fatalf("shutdown: %v", err)
	}

	records, err = registry.List(context.Background())
	if err != nil {
		t.Fatalf("list records after shutdown: %v", err)
	}
	if len(records) != 0 {
		t.Fatalf("expected shutdown to delete all records, got %#v", records)
	}
	if len(scheduler.cancelled) != 2 {
		t.Fatalf("expected shutdown to cancel both jobs, got %#v", scheduler.cancelled)
	}
}

func TestManagerReplacesUnhealthyWarmP40ServiceWithinReplicaLimit(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	manager := testManager(t, registry, scheduler, health, []Profile{testProfile(TierP40Retrieval, 1, 2)}, &now)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	record := onlyRecord(t, registry)
	request := scheduler.launch(0)
	if _, err := registry.Publish(context.Background(), request.RegistrationToken, testPublication(record)); err != nil {
		t.Fatal(err)
	}
	health.fail(record.ID, errors.New("connection refused"))
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatalf("replace unhealthy service: %v", err)
	}
	if scheduler.launchCount() != 2 {
		t.Fatalf("expected one replacement, got %d launches", scheduler.launchCount())
	}
	if len(scheduler.cancelled) != 1 || scheduler.cancelled[0] != record.SlurmJobID {
		t.Fatalf("unhealthy job was not cancelled: %#v", scheduler.cancelled)
	}
	records, _ := registry.List(context.Background())
	starting, unhealthy := 0, 0
	for _, current := range records {
		switch current.State {
		case StateStarting:
			starting++
		case StateUnhealthy:
			unhealthy++
		}
	}
	if starting != 1 || unhealthy != 1 {
		t.Fatalf("unexpected replacement records: %#v", records)
	}
}

func TestManagerReplacesLostHeartbeatAndStartupTimeout(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	manager := testManager(t, registry, scheduler, health, []Profile{testProfile(TierP40Synthesis, 1, 2)}, &now)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	first := onlyRecord(t, registry)
	if _, err := registry.Publish(context.Background(), scheduler.launch(0).RegistrationToken, testPublication(first)); err != nil {
		t.Fatal(err)
	}
	now = now.Add(2 * time.Minute)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatalf("heartbeat reconcile: %v", err)
	}
	if scheduler.launchCount() != 2 {
		t.Fatalf("stale heartbeat was not replaced: %d", scheduler.launchCount())
	}

	// Let the replacement remain in starting state beyond its deadline.
	now = now.Add(2 * time.Minute)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatalf("startup timeout reconcile: %v", err)
	}
	if scheduler.launchCount() != 3 {
		t.Fatalf("startup timeout was not replaced: %d", scheduler.launchCount())
	}
}

func TestManagerReplacesExpiredLease(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	manager := testManager(t, registry, scheduler, health, []Profile{testProfile(TierP40Retrieval, 1, 2)}, &now)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	first := onlyRecord(t, registry)
	if _, err := registry.Publish(context.Background(), scheduler.launch(0).RegistrationToken, testPublication(first)); err != nil {
		t.Fatal(err)
	}
	now = now.Add(5 * time.Hour)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatalf("lease expiry reconcile: %v", err)
	}
	if scheduler.launchCount() != 2 {
		t.Fatalf("expired lease was not replaced: %d", scheduler.launchCount())
	}
	records, _ := registry.List(context.Background())
	foundExpired := false
	for _, record := range records {
		if record.ID == first.ID && record.FailureCategory == FailureLeaseExpired {
			foundExpired = true
		}
	}
	if !foundExpired {
		t.Fatalf("expired lease failure was not recorded: %#v", records)
	}
}

func TestManagerDemandScalesV100FromZeroOnceAndReturnsHealthyLease(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	manager := testManager(t, registry, scheduler, health, []Profile{testProfile(TierV100Reasoning, 0, 1)}, &now)
	first, err := manager.RequestTier(context.Background(), DemandRequest{Tier: TierV100Reasoning, TTL: 5 * time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	second, err := manager.RequestTier(context.Background(), DemandRequest{Tier: TierV100Reasoning, TTL: 5 * time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID {
		t.Fatalf("demands did not coalesce: %s != %s", first.ID, second.ID)
	}
	if scheduler.launchCount() != 0 {
		t.Fatal("request path launched a model instead of waiting for reconciler")
	}
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if scheduler.launchCount() != 1 {
		t.Fatalf("demand exceeded scale limit: %d", scheduler.launchCount())
	}
	record := onlyRecord(t, registry)
	if record.GPU.Count != 4 {
		t.Fatalf("V100 demand did not request four GPUs: %#v", record.GPU)
	}
	if _, err := registry.Publish(context.Background(), scheduler.launch(0).RegistrationToken, testPublication(record)); err != nil {
		t.Fatal(err)
	}
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	ready, err := manager.AwaitTier(context.Background(), first.ID, time.Millisecond)
	if err != nil {
		t.Fatalf("await tier: %v", err)
	}
	if ready.ID != record.ID || ready.EndpointAuth.BearerToken == "" {
		t.Fatalf("unexpected ready lease: %#v", ready)
	}
}

func TestManagerFailsBoundDemandWithTerminalOOMWithoutRelaunch(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	manager := testManager(t, registry, scheduler, health, []Profile{testProfile(TierV100Reasoning, 0, 1)}, &now)
	demand, err := manager.RequestTier(context.Background(), DemandRequest{Tier: TierV100Reasoning, TTL: 5 * time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	record := onlyRecord(t, registry)
	scheduler.setStatus(record.SlurmJobID, ServiceJobStatus{
		State: JobStateFailed, RawState: "OUT_OF_MEMORY", FailureCategory: FailureOOM,
	})
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if scheduler.launchCount() != 1 {
		t.Fatalf("terminal V100 was relaunched instead of failing its demand: %d launches", scheduler.launchCount())
	}
	failed, err := registry.GetDemand(context.Background(), demand.ID)
	if err != nil {
		t.Fatal(err)
	}
	if failed.State != DemandFailed || failed.FailureCategory != FailureOOM {
		t.Fatalf("OOM was not preserved on bound demand: %#v", failed)
	}
	if len(scheduler.cancelled) != 0 {
		t.Fatalf("terminal Slurm job should not be cancelled again: %#v", scheduler.cancelled)
	}
}

func TestManagerLauncherUnhealthyWaitsForTerminalSlurmOOM(t *testing.T) {
	now := time.Date(2026, 7, 10, 13, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	manager := testManager(t, registry, scheduler, health, []Profile{testProfile(TierV100Reasoning, 0, 1)}, &now)
	demand, err := manager.RequestTier(context.Background(), DemandRequest{Tier: TierV100Reasoning, TTL: 5 * time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	record := onlyRecord(t, registry)

	// Mirror RegistryPublisher.mark_unhealthy: it can only observe the runtime
	// process failure and therefore records a generic service category before
	// Slurm/sacct publishes the terminal GPU reason.
	unhealthy := StateUnhealthy
	generic := FailureService
	reason := "GPU runtime exited during startup"
	if err := registry.UpdateControl(context.Background(), record.ID, ControlUpdate{
		State:           &unhealthy,
		HealthCheckedAt: &now,
		HealthError:     &reason,
		FailureCategory: &generic,
		LeaseExpiresAt:  &now,
	}); err != nil {
		t.Fatal(err)
	}
	scheduler.setStatus(record.SlurmJobID, ServiceJobStatus{State: JobStateRunning, RawState: "RUNNING"})
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	waiting, err := registry.GetDemand(context.Background(), demand.ID)
	if err != nil {
		t.Fatal(err)
	}
	if waiting.State != DemandLaunching || waiting.FailureCategory == FailureOOM {
		t.Fatalf("generic launcher failure was finalized before Slurm became terminal: %#v", waiting)
	}
	if len(scheduler.cancelled) != 0 {
		t.Fatalf("manager cancelled the job before terminal classification was available: %#v", scheduler.cancelled)
	}

	scheduler.setStatus(record.SlurmJobID, ServiceJobStatus{
		State: JobStateFailed, RawState: "OUT_OF_MEMORY", FailureCategory: FailureOOM,
	})
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	failed, err := registry.GetDemand(context.Background(), demand.ID)
	if err != nil {
		t.Fatal(err)
	}
	if failed.State != DemandFailed || failed.FailureCategory != FailureOOM ||
		!strings.Contains(failed.Error, "OUT_OF_MEMORY") {
		t.Fatalf("terminal Slurm OOM did not override launcher service_failure: %#v", failed)
	}
	if failed.ServiceDiagnostic == nil || failed.ServiceDiagnostic.SlurmJobID != record.SlurmJobID {
		t.Fatalf("failed demand lost launched service diagnostics: %#v", failed.ServiceDiagnostic)
	}
	if scheduler.launchCount() != 1 || len(scheduler.cancelled) != 0 {
		t.Fatalf("terminal OOM was cancelled or relaunched: launches=%d cancelled=%#v", scheduler.launchCount(), scheduler.cancelled)
	}
}

func TestManagerConsumesP40FailureReportAndReplacesLease(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	registry, err := NewAuthenticatedFileRegistry(root+"/registry.json", 4*time.Hour, "control-token")
	if err != nil {
		t.Fatal(err)
	}
	registry.now = func() time.Time { return now }
	spool, err := NewControlSpool(root+"/requests", "control-token")
	if err != nil {
		t.Fatal(err)
	}
	spool.now = func() time.Time { return now }
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	manager := testManager(t, registry, scheduler, health, []Profile{testProfile(TierP40Synthesis, 1, 2)}, &now)
	manager.controlSpool = spool
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	failedLease := onlyRecord(t, registry)
	if _, err := registry.Publish(context.Background(), scheduler.launch(0).RegistrationToken, testPublication(failedLease)); err != nil {
		t.Fatal(err)
	}
	report, err := spool.SubmitFailureReport(context.Background(), ServiceFailure{
		ServiceID: failedLease.ID, Tier: TierP40Synthesis,
		FailureCategory: FailureEndpointUnhealthy, Reason: "authenticated endpoint refused request",
	})
	if err != nil {
		t.Fatal(err)
	}
	if scheduler.launchCount() != 1 {
		t.Fatal("fire-and-forget report scheduled work directly")
	}
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if scheduler.launchCount() != 2 || len(scheduler.cancelled) != 1 || scheduler.cancelled[0] != failedLease.SlurmJobID {
		t.Fatalf("reported P40 lease was not cancelled and replaced: launches=%d cancelled=%#v", scheduler.launchCount(), scheduler.cancelled)
	}
	if _, err := os.Stat(root + "/requests/" + report.ReportID + ".failure.json"); !os.IsNotExist(err) {
		t.Fatalf("consumed failure report was not removed: %v", err)
	}
}

func TestManagerStopsScaleZeroServiceAtAbsoluteLease(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	registry := testRegistry(t, &now)
	scheduler := newFakeServiceScheduler()
	health := &fakeHealthChecker{errors: map[string]error{}}
	manager := testManager(t, registry, scheduler, health, []Profile{testProfile(TierA100Single, 0, 1)}, &now)
	if _, err := manager.RequestTier(context.Background(), DemandRequest{Tier: TierA100Single, TTL: 5 * time.Minute}); err != nil {
		t.Fatal(err)
	}
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	record := onlyRecord(t, registry)
	if _, err := registry.Publish(context.Background(), scheduler.launch(0).RegistrationToken, testPublication(record)); err != nil {
		t.Fatal(err)
	}
	// Simulate a launcher that keeps extending the renewable heartbeat field.
	extended := now.Add(8 * time.Hour)
	if err := registry.UpdateControl(context.Background(), record.ID, ControlUpdate{LeaseExpiresAt: &extended}); err != nil {
		t.Fatal(err)
	}
	now = record.AbsoluteLeaseExpiresAt.Add(time.Second)
	if err := manager.Reconcile(context.Background()); err != nil {
		t.Fatal(err)
	}
	if scheduler.launchCount() != 1 || len(scheduler.cancelled) != 1 {
		t.Fatalf("scale-zero service outlived absolute lease: launches=%d cancelled=%#v", scheduler.launchCount(), scheduler.cancelled)
	}
}

func testRegistry(t *testing.T, now *time.Time) *FileRegistry {
	t.Helper()
	registry, err := NewAuthenticatedFileRegistry(t.TempDir()+"/registry.json", 4*time.Hour, "control-token")
	if err != nil {
		t.Fatal(err)
	}
	registry.now = func() time.Time { return *now }
	return registry
}

func testManager(t *testing.T, registry Registry, scheduler Scheduler, health HealthChecker, profiles []Profile, now *time.Time) *Manager {
	t.Helper()
	manager, err := NewManager(registry, scheduler, health, ManagerOptions{
		Profiles:     profiles,
		ControlToken: "control-token",
		Timing: Timing{
			LeaseDuration:    4 * time.Hour,
			HealthInterval:   15 * time.Second,
			HeartbeatTimeout: time.Minute,
			StartupTimeout:   time.Minute,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	manager.now = func() time.Time { return *now }
	return manager
}

func testProfile(tier Tier, minReplicas, maxReplicas int) Profile {
	role := RoleSynthesis
	operations := []string{OperationChatCompletions}
	gpu := GPU{Type: "p40", Count: 1}
	switch tier {
	case TierP40Retrieval:
		role = RoleRetrieval
		operations = []string{OperationEmbeddings, OperationVectorSearch, OperationRerank}
	case TierV100Reasoning:
		gpu = GPU{Type: "v100", Count: 4}
	case TierA100Single:
		gpu = GPU{Type: "a100", Count: 1}
	case TierA100Multigpu:
		gpu = GPU{Type: "a100", Count: 4}
	}
	return Profile{
		Tier:                tier,
		Role:                role,
		SupportedOperations: operations,
		Deployment: DeploymentProfile{
			Name:               string(tier) + "-profile",
			Model:              "/models/" + string(tier),
			Quantization:       "bf16",
			ContextLimitTokens: 32768,
			Runtime:            "vllm",
			RuntimeArgs:        []string{"--served-model-name", string(tier)},
		},
		Placement:   Placement{GPU: gpu},
		MinReplicas: minReplicas,
		MaxReplicas: maxReplicas,
	}
}

func onlyRecord(t *testing.T, registry Registry) Record {
	t.Helper()
	records, err := registry.List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var active []Record
	for _, record := range records {
		if record.State == StateStarting || record.State == StateReady {
			active = append(active, record)
		}
	}
	if len(active) != 1 {
		t.Fatalf("expected one active record, got %#v", records)
	}
	return active[0]
}
