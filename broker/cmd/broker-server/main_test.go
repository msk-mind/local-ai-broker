package main

import (
	"io"
	"log"
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
)

func TestBuildAuthenticator(t *testing.T) {
	authenticator, err := buildAuthenticator(config.Config{AuthMode: "header"})
	if err != nil {
		t.Fatalf("build header authenticator: %v", err)
	}
	if authenticator == nil {
		t.Fatal("expected header authenticator")
	}

	authenticator, err = buildAuthenticator(config.Config{
		AuthMode:     "static_tokens",
		StaticTokens: "secret=alice:admin",
	})
	if err != nil {
		t.Fatalf("build static token authenticator: %v", err)
	}
	if authenticator == nil {
		t.Fatal("expected static token authenticator")
	}

	if _, err := buildAuthenticator(config.Config{AuthMode: "static_tokens"}); err == nil {
		t.Fatal("expected missing static token config error")
	}
	if _, err := buildAuthenticator(config.Config{AuthMode: "bogus"}); err == nil {
		t.Fatal("expected unsupported auth mode error")
	}
}

func TestBuildBackend(t *testing.T) {
	backend, err := buildBackend(config.Config{BackendKind: "local"})
	if err != nil {
		t.Fatalf("build local backend: %v", err)
	}
	if backend.Name() != "local" {
		t.Fatalf("unexpected backend: %q", backend.Name())
	}

	backend, err = buildBackend(config.Config{BackendKind: "slurm"})
	if err != nil {
		t.Fatalf("build slurm backend: %v", err)
	}
	if backend.Name() != "slurm" {
		t.Fatalf("unexpected backend: %q", backend.Name())
	}

	if _, err := buildBackend(config.Config{BackendKind: "bogus"}); err == nil {
		t.Fatal("expected unsupported backend error")
	}
}

func TestStartAuditMaintenanceNoopWhenDisabled(t *testing.T) {
	stop := startAuditMaintenance(log.New(io.Discard, "", 0), config.Config{})
	stop()
}
