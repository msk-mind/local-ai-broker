package gpuservice

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestFileRegistryAuthenticatedPublicationRenewalAndRestartRecovery(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	path := t.TempDir() + "/gpu-services.json"
	registry, err := NewAuthenticatedFileRegistry(path, 4*time.Hour, "control-token")
	if err != nil {
		t.Fatal(err)
	}
	registry.now = func() time.Time { return now }

	reservation := testReservation(TierP40Retrieval, now.Add(10*time.Minute))
	record, registrationToken, err := registry.Reserve(context.Background(), reservation, 2)
	if err != nil {
		t.Fatalf("reserve: %v", err)
	}
	if registrationToken == "" || strings.Contains(record.RegistrationTokenSHA256, registrationToken) {
		t.Fatal("registration token was not hashed")
	}
	if _, err := registry.Publish(context.Background(), "wrong-token", testPublication(record)); !errors.Is(err, ErrRegistrationDenied) {
		t.Fatalf("expected registration denial, got %v", err)
	}
	published, err := registry.Publish(context.Background(), registrationToken, testPublication(record))
	if err != nil {
		t.Fatalf("publish: %v", err)
	}
	if !published.Routable(now, time.Minute) {
		t.Fatalf("published record is not routable: %#v", published)
	}

	now = now.Add(3 * time.Hour)
	renewed, err := registry.Renew(context.Background(), record.ID, registrationToken)
	if err != nil {
		t.Fatalf("renew: %v", err)
	}
	if want := now.Add(4 * time.Hour); !renewed.LeaseExpiresAt.Equal(want) {
		t.Fatalf("lease expiry %s, want %s", renewed.LeaseExpiresAt, want)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("registry permissions are %o", info.Mode().Perm())
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(content), registrationToken) {
		t.Fatal("registry persisted a plaintext registration token")
	}

	restarted, err := NewAuthenticatedFileRegistry(path, 4*time.Hour, "control-token")
	if err != nil {
		t.Fatal(err)
	}
	restarted.now = func() time.Time { return now }
	records, err := restarted.List(context.Background())
	if err != nil || len(records) != 1 || records[0].ID != record.ID {
		t.Fatalf("restart recovery records=%#v err=%v", records, err)
	}
}

func TestFileRegistryEnforcesReplicaLimitAndCoalescesConcurrentDemand(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	registry, err := NewAuthenticatedFileRegistry(t.TempDir()+"/registry.json", time.Hour, "control-token")
	if err != nil {
		t.Fatal(err)
	}
	registry.now = func() time.Time { return now }
	if _, _, err := registry.Reserve(context.Background(), testReservation(TierV100Reasoning, now.Add(time.Minute)), 1); err != nil {
		t.Fatal(err)
	}
	if _, _, err := registry.Reserve(context.Background(), testReservation(TierV100Reasoning, now.Add(time.Minute)), 1); !errors.Is(err, ErrReplicaLimit) {
		t.Fatalf("expected replica limit, got %v", err)
	}
	if _, err := registry.RequestDemand(context.Background(), "wrong", DemandRequest{Tier: TierV100Reasoning, TTL: time.Minute}); !errors.Is(err, ErrControlDenied) {
		t.Fatalf("expected control denial, got %v", err)
	}

	const callers = 16
	ids := make(chan string, callers)
	errs := make(chan error, callers)
	var wg sync.WaitGroup
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			demand, err := registry.RequestDemand(context.Background(), "control-token", DemandRequest{
				Tier:   TierV100Reasoning,
				Reason: "fallback",
				TTL:    5 * time.Minute,
			})
			if err != nil {
				errs <- err
				return
			}
			ids <- demand.ID
		}()
	}
	wg.Wait()
	close(ids)
	close(errs)
	for err := range errs {
		t.Fatalf("request demand: %v", err)
	}
	var first string
	for id := range ids {
		if first == "" {
			first = id
		}
		if id != first {
			t.Fatalf("concurrent demands did not coalesce: %q != %q", id, first)
		}
	}
}

func TestScaleZeroLeaseCannotRenewPastAbsoluteExpiry(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	registry, err := NewAuthenticatedFileRegistry(t.TempDir()+"/registry.json", 4*time.Hour, "control-token")
	if err != nil {
		t.Fatal(err)
	}
	registry.now = func() time.Time { return now }
	record, token, err := registry.Reserve(context.Background(), testReservation(TierV100Reasoning, now.Add(time.Minute)), 1)
	if err != nil {
		t.Fatal(err)
	}
	absolute := now.Add(4 * time.Hour)
	if !record.AbsoluteLeaseExpiresAt.Equal(absolute) {
		t.Fatalf("missing absolute scale-zero lease: %#v", record)
	}
	if _, err := registry.Publish(context.Background(), token, testPublication(record)); err != nil {
		t.Fatal(err)
	}
	now = now.Add(3 * time.Hour)
	renewed, err := registry.Renew(context.Background(), record.ID, token)
	if err != nil {
		t.Fatal(err)
	}
	if !renewed.LeaseExpiresAt.Equal(absolute) {
		t.Fatalf("scale-zero renewal escaped absolute lease: %s != %s", renewed.LeaseExpiresAt, absolute)
	}
	now = absolute.Add(time.Second)
	if renewed.Routable(now, 4*time.Hour) {
		t.Fatal("expired scale-zero service remained routable")
	}
}

func testReservation(tier Tier, deadline time.Time) Reservation {
	role := RoleSynthesis
	capabilities := []string{OperationChatCompletions}
	gpu := GPU{Type: "v100", Count: 4}
	if tier == TierP40Retrieval {
		role = RoleRetrieval
		capabilities = []string{OperationEmbeddings, OperationVectorSearch, OperationRerank}
		gpu = GPU{Type: "p40", Count: 1}
	}
	return Reservation{
		Tier:            tier,
		Role:            role,
		ModelProfile:    string(tier) + "-profile",
		Model:           "/models/" + string(tier),
		Capabilities:    capabilities,
		ContextLimit:    32768,
		GPU:             gpu,
		StartupDeadline: deadline,
	}
}

func testPublication(record Record) Publication {
	return Publication{
		ID:                 record.ID,
		Tier:               record.Tier,
		Endpoint:           "http://gpu-service.internal:8000/v1",
		EndpointAuth:       EndpointAuth{Type: "bearer", BearerToken: "endpoint-secret"},
		ModelProfile:       record.ModelProfile,
		Model:              record.Model,
		Capabilities:       append([]string(nil), record.Capabilities...),
		ContextLimitTokens: record.ContextLimitTokens,
		GPU:                record.GPU,
		SlurmJobID:         "12345",
	}
}
