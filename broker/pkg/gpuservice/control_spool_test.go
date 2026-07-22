package gpuservice

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestControlSpoolImportsSignedDemandAndReturnsSignedReadyEndpoint(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
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

	request, err := spool.Submit(context.Background(), DemandRequest{
		Tier:            TierV100Reasoning,
		FailureCategory: FailureService,
		Reason:          "P40 endpoint failed",
		TTL:             5 * time.Minute,
	})
	if err != nil {
		t.Fatalf("submit control request: %v", err)
	}
	if err := spool.Import(context.Background(), registry); err != nil {
		t.Fatalf("import control request: %v", err)
	}
	if _, err := os.Stat(root + "/requests/" + request.RequestID + ".request.json"); !os.IsNotExist(err) {
		t.Fatalf("request file was not consumed: %v", err)
	}
	demands, err := registry.ListDemands(context.Background())
	if err != nil || len(demands) != 1 {
		t.Fatalf("demands=%#v err=%v", demands, err)
	}

	record, registrationToken, err := registry.Reserve(context.Background(), testReservation(TierV100Reasoning, now.Add(time.Minute)), 1)
	if err != nil {
		t.Fatal(err)
	}
	published, err := registry.Publish(context.Background(), registrationToken, testPublication(record))
	if err != nil {
		t.Fatal(err)
	}
	if err := registry.UpdateDemand(context.Background(), demands[0].ID, DemandUpdate{
		State:     DemandReady,
		ServiceID: published.ID,
	}); err != nil {
		t.Fatal(err)
	}
	if err := spool.Sync(context.Background(), registry); err != nil {
		t.Fatalf("sync response: %v", err)
	}
	responsePath := root + "/requests/" + request.RequestID + ".response.json"
	info, err := os.Stat(responsePath)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("response permissions are %o", info.Mode().Perm())
	}
	endpoint, err := spool.Await(context.Background(), request.RequestID, time.Millisecond)
	if err != nil {
		t.Fatalf("await response: %v", err)
	}
	if endpoint.EndpointAuth.BearerToken != "endpoint-secret" || endpoint.GPU.Count != 4 || endpoint.Tier != TierV100Reasoning {
		t.Fatalf("unexpected endpoint access: %#v", endpoint)
	}
	if _, err := os.Stat(responsePath); !os.IsNotExist(err) {
		t.Fatalf("consumed response was not removed: %v", err)
	}
}

func TestControlSpoolRejectsTamperedRequest(t *testing.T) {
	now := time.Date(2026, 7, 9, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	registry, _ := NewAuthenticatedFileRegistry(root+"/registry.json", time.Hour, "control-token")
	registry.now = func() time.Time { return now }
	spool, _ := NewControlSpool(root+"/requests", "control-token")
	spool.now = func() time.Time { return now }
	request, err := spool.Submit(context.Background(), DemandRequest{Tier: TierA100Single, TTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	path := root + "/requests/" + request.RequestID + ".request.json"
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	content = []byte(strings.Replace(string(content), `"a100-single"`, `"a100-multigpu"`, 1))
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := spool.Import(context.Background(), registry); err != nil {
		t.Fatalf("tampered request should be quarantined without stopping reconciliation: %v", err)
	}
	demands, err := registry.ListDemands(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(demands) != 0 {
		t.Fatalf("tampered demand was imported: %#v", demands)
	}
	if _, err := os.Stat(path + ".rejected"); err != nil {
		t.Fatalf("tampered request was not quarantined: %v", err)
	}
}

func TestControlSpoolSignsTerminalFailureCategory(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	registry, _ := NewAuthenticatedFileRegistry(root+"/registry.json", time.Hour, "control-token")
	registry.now = func() time.Time { return now }
	spool, _ := NewControlSpool(root+"/requests", "control-token")
	spool.now = func() time.Time { return now }
	request, err := spool.Submit(context.Background(), DemandRequest{Tier: TierV100Reasoning, TTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	if err := spool.Import(context.Background(), registry); err != nil {
		t.Fatal(err)
	}
	demands, _ := registry.ListDemands(context.Background())
	record, _, err := registry.Reserve(
		context.Background(),
		testReservation(TierV100Reasoning, now.Add(time.Minute)),
		1,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := registry.AttachJob(context.Background(), record.ID, "v100-job-42"); err != nil {
		t.Fatal(err)
	}
	if err := registry.UpdateDemand(context.Background(), demands[0].ID, DemandUpdate{
		State: DemandLaunching, ServiceID: record.ID,
	}); err != nil {
		t.Fatal(err)
	}
	if err := registry.FailDemandsForService(
		context.Background(), record.ID, FailureOOM, "Slurm OUT_OF_MEMORY",
	); err != nil {
		t.Fatal(err)
	}
	// The diagnostic is retained on the demand so a broker restart or service
	// cleanup cannot erase the identity before the worker consumes the failure.
	if err := registry.Delete(context.Background(), record.ID); err != nil {
		t.Fatal(err)
	}
	if err := spool.Sync(context.Background(), registry); err != nil {
		t.Fatal(err)
	}
	responsePath := root + "/requests/" + request.RequestID + ".response.json"
	response, err := spool.readResponse(responsePath)
	if err != nil {
		t.Fatal(err)
	}
	if response.FailureCategory != FailureOOM {
		t.Fatalf("failure category was not signed into response: %#v", response)
	}
	if response.Service != nil {
		t.Fatalf("failed response leaked endpoint access: %#v", response.Service)
	}
	if response.ServiceDiagnostic == nil ||
		response.ServiceDiagnostic.Tier != TierV100Reasoning ||
		response.ServiceDiagnostic.SlurmJobID != "v100-job-42" ||
		response.ServiceDiagnostic.GPU != (GPU{Type: "v100", Count: 4}) ||
		response.ServiceDiagnostic.ModelProfile != "v100-reasoning-profile" {
		t.Fatalf("failed response lost sanitized service identity: %#v", response.ServiceDiagnostic)
	}
	content, err := os.ReadFile(responsePath)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"endpoint-secret", "/models/v100-reasoning", "gpu-service.internal"} {
		if strings.Contains(string(content), secret) {
			t.Fatalf("failed response leaked %q: %s", secret, content)
		}
	}
	tampered := *response.ServiceDiagnostic
	tampered.SlurmJobID = "tampered-job"
	response.ServiceDiagnostic = &tampered
	tamperedContent, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(responsePath, tamperedContent, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := spool.readResponse(responsePath); !errors.Is(err, ErrControlDenied) {
		t.Fatalf("tampered service diagnostics passed signature verification: %v", err)
	}
	tampered.SlurmJobID = "v100-job-42"
	response.ServiceDiagnostic = &tampered
	if err := spool.writeResponse(response); err != nil {
		t.Fatal(err)
	}
	if _, err := spool.Await(context.Background(), request.RequestID, time.Millisecond); err == nil {
		t.Fatal("expected terminal demand error")
	}
	if _, err := os.Stat(responsePath); !os.IsNotExist(err) {
		t.Fatalf("failed response was not removed after consumption: %v", err)
	}
}

func TestControlSpoolPrunesExpiredDemandAndCredentialResponses(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	registry, _ := NewAuthenticatedFileRegistry(root+"/registry.json", time.Hour, "control-token")
	registry.now = func() time.Time { return now }
	spool, _ := NewControlSpool(root+"/requests", "control-token")
	spool.now = func() time.Time { return now }
	request, err := spool.Submit(context.Background(), DemandRequest{Tier: TierA100Single, TTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	if err := spool.Import(context.Background(), registry); err != nil {
		t.Fatal(err)
	}
	demands, _ := registry.ListDemands(context.Background())
	if err := registry.UpdateDemand(context.Background(), demands[0].ID, DemandUpdate{State: DemandFailed, FailureCategory: FailureTimeout, Error: "timed out"}); err != nil {
		t.Fatal(err)
	}
	if err := spool.Sync(context.Background(), registry); err != nil {
		t.Fatal(err)
	}
	now = now.Add(2 * time.Minute)
	if err := spool.Sync(context.Background(), registry); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.PruneExpiredDemands(context.Background(), now); err != nil {
		t.Fatal(err)
	}
	if demands, _ := registry.ListDemands(context.Background()); len(demands) != 0 {
		t.Fatalf("expired terminal demands were retained: %#v", demands)
	}
	if _, err := os.Stat(root + "/requests/" + request.RequestID + ".response.json"); !os.IsNotExist(err) {
		t.Fatalf("expired response was retained: %v", err)
	}

	orphan := ControlResponse{
		RequestID: "orphan-response", DemandID: "missing-demand", State: DemandReady,
		UpdatedAt: now.Format(time.RFC3339Nano),
		Service:   &EndpointAccess{ID: "gpu-orphan", Tier: TierV100Reasoning, EndpointAuth: EndpointAuth{Type: "bearer", BearerToken: "secret"}},
	}
	if err := spool.writeResponse(orphan); err != nil {
		t.Fatal(err)
	}
	if err := spool.Sync(context.Background(), registry); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(root + "/requests/orphan-response.response.json"); !os.IsNotExist(err) {
		t.Fatalf("orphan credential response was retained: %v", err)
	}
}

func TestControlSpoolRejectsTamperedP40FailureReport(t *testing.T) {
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	registry, _ := NewAuthenticatedFileRegistry(root+"/registry.json", time.Hour, "control-token")
	registry.now = func() time.Time { return now }
	record, token, err := registry.Reserve(context.Background(), testReservation(TierP40Retrieval, now.Add(time.Minute)), 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Publish(context.Background(), token, testPublication(record)); err != nil {
		t.Fatal(err)
	}
	spool, _ := NewControlSpool(root+"/requests", "control-token")
	spool.now = func() time.Time { return now }
	report, err := spool.SubmitFailureReport(context.Background(), ServiceFailure{
		ServiceID: record.ID, Tier: TierP40Retrieval, FailureCategory: FailureService, Reason: "connection failed",
	})
	if err != nil {
		t.Fatal(err)
	}
	path := root + "/requests/" + report.ReportID + ".failure.json"
	content, _ := os.ReadFile(path)
	content = []byte(strings.Replace(string(content), "connection failed", "tampered", 1))
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := spool.Import(context.Background(), registry); err != nil {
		t.Fatal(err)
	}
	current, _ := registry.List(context.Background())
	if len(current) != 1 || current[0].State != StateReady {
		t.Fatalf("tampered failure report changed lease: %#v", current)
	}
	if _, err := os.Stat(path + ".rejected"); err != nil {
		t.Fatalf("tampered failure report was not quarantined: %v", err)
	}
}
