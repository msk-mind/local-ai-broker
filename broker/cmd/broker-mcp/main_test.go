package main

import (
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
)

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
