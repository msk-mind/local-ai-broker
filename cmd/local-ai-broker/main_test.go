package main

import (
	"net"
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
	mustWriteFile(t, filepath.Join(repo, "examples", "mcp-clients", "codex-profiles", "local-broker.config.toml.template"), "developer_instructions = \"\"\"\nuse __REPO_ROOT__\n\"\"\"\npath=__REPO_ROOT__\n")
	if err := installCodexProfile(repo, codexHome, "local-broker.config.toml.template", "local-broker.config.toml"); err != nil {
		t.Fatalf("install profile: %v", err)
	}
	content, err := os.ReadFile(filepath.Join(codexHome, "local-broker.config.toml"))
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "developer_instructions = \"\"\"\nuse "+repo+"\n\"\"\"\npath="+repo+"\n" {
		t.Fatalf("unexpected rendered content: %q", string(content))
	}
}

func TestLoadBootstrapConfig(t *testing.T) {
	repo := t.TempDir()
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "local.json")
	content := `{
  "listen_addr": "127.0.0.1:18081",
  "job_store_path": "__REPO_ROOT__/.broker/jobs.json",
  "backend": "local",
  "local": {
    "script_path": "__REPO_ROOT__/deploy/local/broker_worker.sh"
  },
  "runtime": {
    "llama_cpp_timeout_seconds": 7
  }
}`
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	values, err := loadBootstrapConfig(repo, configPath)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if values["BROKER_BACKEND"] != "local" {
		t.Fatalf("unexpected backend: %#v", values)
	}
	if values["BROKER_JOB_STORE_PATH"] != filepath.Join(repo, ".broker", "jobs.json") {
		t.Fatalf("unexpected job store path: %#v", values)
	}
	if values["BROKER_LOCAL_SCRIPT_PATH"] != filepath.Join(repo, "deploy", "local", "broker_worker.sh") {
		t.Fatalf("unexpected local script path: %#v", values)
	}
	if values["BROKER_RUNTIME_LLAMACPP_TIMEOUT_SECONDS"] != "7" {
		t.Fatalf("unexpected timeout value: %#v", values)
	}
}

func TestLoadSlurmBootstrapConfigWithGPUTypeDefaults(t *testing.T) {
	repo := t.TempDir()
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "slurm.json")
	content := `{
  "backend": "slurm",
  "slurm": {
    "partition_gpu": "hpc",
    "gpu_request_mode": "gres",
    "gpu_type_p40": "p40",
    "gpu_type_a100": "a100"
  }
}`
	if err := os.WriteFile(configPath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	values, err := loadBootstrapConfig(repo, configPath)
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if values["BROKER_SLURM_PARTITION_GPU"] != "hpc" {
		t.Fatalf("unexpected GPU partition: %#v", values)
	}
	if values["BROKER_SLURM_GPU_REQUEST_MODE"] != "gres" {
		t.Fatalf("unexpected GPU request mode: %#v", values)
	}
	if values["BROKER_SLURM_GPU_TYPE_P40"] != "p40" || values["BROKER_SLURM_GPU_TYPE_A100"] != "a100" {
		t.Fatalf("unexpected GPU type defaults: %#v", values)
	}
}

func TestWriteBootstrapConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "generated.local.json")
	cfg := bootstrapConfig{
		ListenAddr:   "127.0.0.1:8081",
		Backend:      "local",
		JobStorePath: "__REPO_ROOT__/.broker/jobs.json",
		Local: localBootstrapConfig{
			Mode:       "command",
			ScriptPath: "__REPO_ROOT__/deploy/local/broker_worker.sh",
		},
	}
	if err := writeBootstrapConfig(path, cfg); err != nil {
		t.Fatalf("write config: %v", err)
	}
	values, err := loadBootstrapConfig("/repo", path)
	if err != nil {
		t.Fatalf("load written config: %v", err)
	}
	if values["BROKER_BACKEND"] != "local" {
		t.Fatalf("unexpected backend: %#v", values)
	}
	if values["BROKER_LOCAL_SCRIPT_PATH"] != "/repo/deploy/local/broker_worker.sh" {
		t.Fatalf("unexpected local script path: %#v", values)
	}
}

func TestRunInstallRequiresTarget(t *testing.T) {
	if err := runInstall(nil); err == nil {
		t.Fatal("expected install usage error")
	}
}

func TestPickFreeLoopbackAddr(t *testing.T) {
	addr, err := pickFreeLoopbackAddr()
	if err != nil {
		t.Fatalf("pick free addr: %v", err)
	}
	if _, err := net.ResolveTCPAddr("tcp", addr); err != nil {
		t.Fatalf("resolve addr %q: %v", addr, err)
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
