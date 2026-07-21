package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestFormatWatchLineIncludesProgress(t *testing.T) {
	ts := time.Date(2026, 6, 26, 12, 0, 0, 0, time.UTC)
	job := types.Job{
		ID:           "job_123",
		State:        types.JobStateRunning,
		BackendState: "RUNNING",
		Progress: &types.ProgressInfo{
			State:       "running",
			Phase:       "preprocessing",
			Percent:     35,
			Message:     "Loading source document",
			Timestamp:   &ts,
			LastUpdated: &ts,
		},
	}

	got := formatWatchLine(job)
	want := `job=job_123 state=running phase=preprocessing percent=35 message="Loading source document" backend=RUNNING`
	if got != want {
		t.Fatalf("unexpected watch line:\nwant: %s\ngot:  %s", want, got)
	}
}

func TestFormatWatchLineIncludesError(t *testing.T) {
	job := types.Job{
		ID:          "job_456",
		State:       types.JobStateFailed,
		ResultError: "schema validation failed",
	}

	got := formatWatchLine(job)
	want := `job=job_456 state=failed error="schema validation failed"`
	if got != want {
		t.Fatalf("unexpected watch line:\nwant: %s\ngot:  %s", want, got)
	}
}

func TestIsTerminalState(t *testing.T) {
	if !isTerminalState(types.JobStateSucceeded) {
		t.Fatal("expected succeeded to be terminal")
	}
	if isTerminalState(types.JobStateRunning) {
		t.Fatal("expected running to be non-terminal")
	}
}

func TestVerifyAuditFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	logger := audit.NewFileLogger(path)
	if err := logger.Log(context.Background(), audit.Event{Actor: "alice", Action: "job.submit", Outcome: "success"}); err != nil {
		t.Fatalf("log event: %v", err)
	}

	result, err := audit.VerifyFile(path)
	if err != nil {
		t.Fatalf("verify file: %v", err)
	}
	if !result.Valid {
		t.Fatalf("expected valid audit result, got %#v", result)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read audit file: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("expected audit file content")
	}
}

func TestQuoteHelpersAndEnvOrDefault(t *testing.T) {
	if got := quoteIfNeeded("plain"); got != "plain" {
		t.Fatalf("unexpected plain quote behavior: %q", got)
	}
	if got := quoteIfNeeded("two words"); got != `"two words"` {
		t.Fatalf("unexpected spaced quote behavior: %q", got)
	}

	t.Setenv("BROKER_CLI_TEST_ENV", "set")
	if got := envOrDefault("BROKER_CLI_TEST_ENV", "fallback"); got != "set" {
		t.Fatalf("unexpected env override: %q", got)
	}
	if got := envOrDefault("BROKER_CLI_TEST_ENV_MISSING", "fallback"); got != "fallback" {
		t.Fatalf("unexpected env fallback: %q", got)
	}
}

func TestDoJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		if string(body) != `{"ok":true}` {
			t.Fatalf("unexpected request body: %q", string(body))
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"job_id":"job_123"}`))
	}))
	defer server.Close()

	got := doJSON(server.Client(), http.MethodPost, server.URL, []byte(`{"ok":true}`))
	if got != `{"job_id":"job_123"}` {
		t.Fatalf("unexpected response body: %q", got)
	}
}
