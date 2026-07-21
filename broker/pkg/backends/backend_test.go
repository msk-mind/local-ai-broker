package backends

import "testing"

func TestStubHelpers(t *testing.T) {
	resp := StubSubmitResponse("local", "run-1")
	if resp.BackendKind != "local" || resp.BackendRunID != "run-1" {
		t.Fatalf("unexpected submit response: %#v", resp)
	}
	if resp.InitialState != "queued" {
		t.Fatalf("unexpected initial state: %#v", resp)
	}

	status := StubRunStatus("run-2")
	if status.BackendRunID != "run-2" || status.RawState != "STUB" {
		t.Fatalf("unexpected run status: %#v", status)
	}
}

func TestIndexedStubResponses(t *testing.T) {
	next := uint64(41)
	responses := IndexedStubResponses("slurm", "batch", 3, func() uint64 {
		next++
		return next
	})

	if len(responses) != 3 {
		t.Fatalf("expected 3 responses, got %d", len(responses))
	}
	if responses[0].BackendRunID != "batch-000042" || responses[2].BackendRunID != "batch-000044" {
		t.Fatalf("unexpected run IDs: %#v", responses)
	}
}
