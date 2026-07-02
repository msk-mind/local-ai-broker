package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseEnvFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "broker.env")
	content := `
# comment
BROKER_BACKEND=slurm
export BROKER_LISTEN_ADDR="127.0.0.1:18081"
BROKER_RUNTIME_LLAMACPP_BASE_URL='http://127.0.0.1:8080'
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	values, err := parseEnvFile(path)
	if err != nil {
		t.Fatalf("parse env file: %v", err)
	}
	if values["BROKER_BACKEND"] != "slurm" {
		t.Fatalf("unexpected backend: %#v", values)
	}
	if values["BROKER_LISTEN_ADDR"] != "127.0.0.1:18081" {
		t.Fatalf("unexpected listen addr: %#v", values)
	}
	if values["BROKER_RUNTIME_LLAMACPP_BASE_URL"] != "http://127.0.0.1:8080" {
		t.Fatalf("unexpected runtime URL: %#v", values)
	}
}

func TestLooksLikeRepoRoot(t *testing.T) {
	dir := t.TempDir()
	mustMkdirAll(t, filepath.Join(dir, "broker", "cmd", "broker-server"))
	mustMkdirAll(t, filepath.Join(dir, "examples", "mcp-clients", "codex-profiles"))
	mustWriteFile(t, filepath.Join(dir, "go.mod"), "module github.com/msk-mind/local-ai-broker\n")
	mustWriteFile(t, filepath.Join(dir, "broker", "cmd", "broker-server", "main.go"), "package main\n")
	mustWriteFile(t, filepath.Join(dir, "examples", "mcp-clients", "codex-profiles", "local-broker.config.toml.template"), "x\n")
	if !looksLikeRepoRoot(dir) {
		t.Fatal("expected repo root to be recognized")
	}
}

func TestInstallCodexProfile(t *testing.T) {
	repo := t.TempDir()
	codexHome := t.TempDir()
	mustMkdirAll(t, filepath.Join(repo, "examples", "mcp-clients", "codex-profiles"))
	mustWriteFile(t, filepath.Join(repo, "examples", "mcp-clients", "codex-profiles", "local-broker.config.toml.template"), "path=__REPO_ROOT__\n")
	if err := installCodexProfile(repo, codexHome, "local-broker.config.toml.template", "local-broker.config.toml"); err != nil {
		t.Fatalf("install profile: %v", err)
	}
	content, err := os.ReadFile(filepath.Join(codexHome, "local-broker.config.toml"))
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "path="+repo+"\n" {
		t.Fatalf("unexpected rendered content: %q", string(content))
	}
}

func mustMkdirAll(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
