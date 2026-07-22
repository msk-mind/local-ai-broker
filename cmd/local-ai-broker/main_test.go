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
	mustMkdirAll(t, filepath.Join(dir, "examples", "mcp-clients", "copilot-profiles"))
	mustWriteFile(t, filepath.Join(dir, "go.mod"), "module github.com/msk-mind/local-ai-broker\n")
	mustWriteFile(t, filepath.Join(dir, "broker", "cmd", "broker-server", "main.go"), "package main\n")
	mustWriteFile(t, filepath.Join(dir, "examples", "mcp-clients", "codex-profiles", "local-broker.config.toml.template"), "x\n")
	mustWriteFile(t, filepath.Join(dir, "examples", "mcp-clients", "copilot-profiles", "local-broker.mcp-config.json.template"), "x\n")
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

func TestRunInstallRejectsUnknownTarget(t *testing.T) {
	if err := runInstall([]string{"bogus"}); err == nil {
		t.Fatal("expected unknown install target error")
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

func TestResolveConfigPathAndDefaults(t *testing.T) {
	repoRoot := "/repo"
	configDir := "/repo/configs/broker"

	if got := resolveConfigPath(repoRoot, configDir, "__REPO_ROOT__/deploy/local/broker_worker.sh"); got != "/repo/deploy/local/broker_worker.sh" {
		t.Fatalf("unexpected repo-root substitution: %q", got)
	}
	if got := resolveConfigPath(repoRoot, configDir, "generated.local.json"); got != "/repo/configs/broker/generated.local.json" {
		t.Fatalf("unexpected relative config path: %q", got)
	}
	if got := resolveConfigPath(repoRoot, configDir, "/tmp/demo.json"); got != "/tmp/demo.json" {
		t.Fatalf("unexpected absolute config path rewrite: %q", got)
	}

	tempRepo := t.TempDir()
	mustMkdirAll(t, filepath.Join(tempRepo, "configs", "broker"))
	mustWriteFile(t, filepath.Join(tempRepo, "configs", "broker", "generated.local.json"), "{}\n")
	mustWriteFile(t, filepath.Join(tempRepo, "configs", "broker", "generated.cdsi-slurm.json"), "{}\n")

	if got := defaultBootstrapConfigPath(tempRepo, "local"); got != filepath.Join(tempRepo, "configs", "broker", "generated.local.json") {
		t.Fatalf("unexpected local default config path: %q", got)
	}
	if got := defaultBootstrapConfigPath(tempRepo, "slurm"); got != filepath.Join(tempRepo, "configs", "broker", "generated.cdsi-slurm.json") {
		t.Fatalf("unexpected slurm default config path: %q", got)
	}
}

func TestSelectModeAndEnvHelpers(t *testing.T) {
	if got := selectMode(true, false, "slurm"); got != "local" {
		t.Fatalf("unexpected local selection: %q", got)
	}
	if got := selectMode(false, true, "local"); got != "slurm" {
		t.Fatalf("unexpected slurm selection: %q", got)
	}
	if got := selectMode(false, false, "local"); got != "local" {
		t.Fatalf("unexpected fallback selection: %q", got)
	}

	t.Setenv("BROKER_TEST_ENV_OR", "set")
	if got := envOr("BROKER_TEST_ENV_OR", "fallback"); got != "set" {
		t.Fatalf("unexpected envOr value: %q", got)
	}
	if got := envOr("BROKER_TEST_ENV_OR_MISSING", "fallback"); got != "fallback" {
		t.Fatalf("unexpected envOr fallback: %q", got)
	}
}

func TestDefaultBrokerEnv(t *testing.T) {
	envMap := defaultBrokerEnv("/repo", "local")
	if envMap["BROKER_BACKEND"] != "local" {
		t.Fatalf("unexpected backend: %#v", envMap)
	}
	if envMap["BROKER_LOCAL_SCRIPT_PATH"] != "/repo/deploy/local/broker_worker.sh" {
		t.Fatalf("unexpected local script path: %#v", envMap)
	}
	if envMap["BROKER_REPO_ROOT_PATH"] != "/repo" {
		t.Fatalf("unexpected repo root path: %#v", envMap)
	}

	slurmEnv := defaultBrokerEnv("/repo", "slurm")
	if slurmEnv["BROKER_BACKEND"] != "slurm" {
		t.Fatalf("unexpected slurm backend: %#v", slurmEnv)
	}
	if slurmEnv["BROKER_SLURM_SCRIPT_PATH"] != "/repo/deploy/slurm/broker_worker.slurm" {
		t.Fatalf("unexpected slurm script path: %#v", slurmEnv)
	}
}

func TestMergeEnvAndPathHelpers(t *testing.T) {
	t.Setenv("LOCAL_AI_BROKER_TEST_KEEP", "present")
	merged := mergeEnv(map[string]string{
		"LOCAL_AI_BROKER_TEST_KEEP": "override",
		"LOCAL_AI_BROKER_TEST_NEW":  "new",
	})

	foundKeep := false
	foundNew := false
	for _, item := range merged {
		if item == "LOCAL_AI_BROKER_TEST_KEEP=override" {
			foundKeep = true
		}
		if item == "LOCAL_AI_BROKER_TEST_NEW=new" {
			foundNew = true
		}
	}
	if !foundKeep || !foundNew {
		t.Fatalf("expected merged env overrides, got %#v", merged)
	}

	path := filepath.Join(t.TempDir(), "profile.toml")
	if profileExists(path) {
		t.Fatal("expected missing profile to return false")
	}
	mustWriteFile(t, path, "demo\n")
	if !profileExists(path) {
		t.Fatal("expected existing profile to return true")
	}
}

func TestBuildBrokerServerBinaryCreatesExecutableAndCleanupRemovesIt(t *testing.T) {
	repoRoot, err := findRepoRoot()
	if err != nil {
		t.Fatalf("find repo root: %v", err)
	}
	binaryPath, cleanup, err := buildBrokerServerBinary(repoRoot)
	if err != nil {
		t.Fatalf("build broker server binary: %v", err)
	}
	if _, err := os.Stat(binaryPath); err != nil {
		t.Fatalf("expected built binary to exist: %v", err)
	}
	binaryDir := filepath.Dir(binaryPath)
	cleanup()
	if _, err := os.Stat(binaryDir); !os.IsNotExist(err) {
		t.Fatalf("expected cleanup to remove temp binary dir, stat err=%v", err)
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
