package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type commandError struct {
	message string
	code    int
}

func (e commandError) Error() string {
	return e.message
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		var cmdErr commandError
		if errors.As(err, &cmdErr) {
			fmt.Fprintln(os.Stderr, cmdErr.message)
			os.Exit(cmdErr.code)
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		printRootUsage()
		return nil
	}
	switch args[0] {
	case "init":
		return runInit(args[1:])
	case "doctor":
		return runDoctor(args[1:])
	case "install":
		return runInstall(args[1:])
	case "up":
		return runUp(args[1:])
	case "help", "-h", "--help":
		printRootUsage()
		return nil
	default:
		return commandError{message: "unknown subcommand: " + args[0], code: 2}
	}
}

func printRootUsage() {
	fmt.Print(`local-ai-broker

Usage:
  local-ai-broker init [--local|--slurm] [--output PATH] [flags]
  local-ai-broker doctor [--local|--slurm] [--config PATH]
  local-ai-broker install codex [--local|--slurm|--all] [--codex-home PATH]
  local-ai-broker up [--local|--slurm] [--listen-addr ADDR] [--config PATH] [--env-file PATH]
`)
}

func runInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	localMode := fs.Bool("local", false, "generate a local backend config")
	slurmMode := fs.Bool("slurm", false, "generate a Slurm backend config")
	outputPath := fs.String("output", "", "config output path")
	listenAddr := fs.String("listen-addr", "127.0.0.1:8081", "broker listen address")
	authMode := fs.String("auth-mode", "header", "broker auth mode")
	localScript := fs.String("local-script", "", "local worker script path")
	slurmScript := fs.String("slurm-script", "", "Slurm worker script path")
	partitionCPU := fs.String("partition-cpu", "cpu", "Slurm CPU partition")
	partitionP40 := fs.String("partition-p40", "hpc", "Slurm P40 partition")
	partitionA100 := fs.String("partition-a100", "gpu-a100", "Slurm A100 partition")
	nodelistP40 := fs.String("nodelist-p40", "pllimsksparky[1-4]", "Slurm P40 nodelist")
	constraintP40 := fs.String("constraint-p40", "", "Slurm P40 constraint")
	modelP40 := fs.String("model-p40", "gpt-oss-20b.p40", "default P40 model profile")
	modelA100 := fs.String("model-a100", "qwen3-coder-30b.a100", "default A100 model profile")
	if err := fs.Parse(args); err != nil {
		return commandError{message: err.Error(), code: 2}
	}
	repoRoot, err := findRepoRoot()
	if err != nil {
		return err
	}
	mode := selectMode(*localMode, *slurmMode, "local")
	if *outputPath == "" {
		if mode == "slurm" {
			*outputPath = filepath.Join(repoRoot, "configs", "broker", "generated.slurm.json")
		} else {
			*outputPath = filepath.Join(repoRoot, "configs", "broker", "generated.local.json")
		}
	}
	cfg := bootstrapConfig{
		ListenAddr:      *listenAddr,
		JobStorePath:    "__REPO_ROOT__/.broker/jobs.json",
		RunRootPath:     "__REPO_ROOT__/.broker/runs",
		RepoRootPath:    "__REPO_ROOT__",
		AuditLogPath:    "__REPO_ROOT__/.broker/audit.jsonl",
		AuditVerifyMode: "warn",
		AuthMode:        *authMode,
		Backend:         mode,
	}
	if mode == "slurm" {
		slurmScriptPath := *slurmScript
		if slurmScriptPath == "" {
			slurmScriptPath = "__REPO_ROOT__/deploy/slurm/broker_worker.slurm"
		}
		cfg.Slurm = slurmBootstrapConfig{
			Mode:          "command",
			SubmitCmd:     "sbatch",
			StatusCmd:     "sacct",
			CancelCmd:     "scancel",
			ScriptPath:    slurmScriptPath,
			PartitionCPU:  *partitionCPU,
			PartitionP40:  *partitionP40,
			PartitionA100: *partitionA100,
			NodeListP40:   *nodelistP40,
			ConstraintP40: *constraintP40,
			ModelP40:      *modelP40,
			ModelA100:     *modelA100,
		}
		cfg.Runtime = runtimeBootstrapConfig{
			LlamaCPPTimeoutSeconds: 20,
		}
		cfg.Parallel = parallelBootstrapConfig{
			MaxBatchSize:         32,
			MaxActiveBatches:     2,
			MaxAdditionalBatches: 1,
			MaxRetriedShards:     4,
		}
	} else {
		localScriptPath := *localScript
		if localScriptPath == "" {
			localScriptPath = "__REPO_ROOT__/deploy/local/broker_worker.sh"
		}
		cfg.Local = localBootstrapConfig{
			Mode:       "command",
			ScriptPath: localScriptPath,
		}
	}
	if err := writeBootstrapConfig(*outputPath, cfg); err != nil {
		return err
	}
	fmt.Printf("wrote %s\n", *outputPath)
	fmt.Println("Next:")
	fmt.Printf("  go run ./cmd/local-ai-broker doctor --config %s\n", *outputPath)
	fmt.Printf("  go run ./cmd/local-ai-broker up --config %s\n", *outputPath)
	return nil
}

func runDoctor(args []string) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	localMode := fs.Bool("local", false, "check local backend requirements")
	slurmMode := fs.Bool("slurm", false, "check Slurm backend requirements")
	configPath := fs.String("config", "", "optional broker config JSON file")
	if err := fs.Parse(args); err != nil {
		return commandError{message: err.Error(), code: 2}
	}
	repoRoot, err := findRepoRoot()
	if err != nil {
		return err
	}
	mode := selectMode(*localMode, *slurmMode, "local")
	envMap := defaultBrokerEnv(repoRoot, mode)
	if *configPath != "" {
		loaded, err := loadBootstrapConfig(repoRoot, *configPath)
		if err != nil {
			return err
		}
		for k, v := range loaded {
			envMap[k] = v
		}
		if backend := envMap["BROKER_BACKEND"]; backend == "slurm" {
			mode = "slurm"
		} else if backend == "local" {
			mode = "local"
		}
		reportCheck("config", *configPath, true)
	}
	failures := 0
	warnings := 0

	reportCheck("repo-root", repoRoot, true)
	if !checkExecutable("go", true) {
		failures++
	}
	if !checkPath(filepath.Join(repoRoot, "broker", "cmd", "broker-server", "main.go"), true) {
		failures++
	}
	if !checkPath(filepath.Join(repoRoot, "broker", "cmd", "broker-mcp", "main.go"), true) {
		failures++
	}
	if !checkWritableBrokerPaths(repoRoot) {
		failures++
	}
	if mode == "local" {
		if !checkExecutable("python3", true) {
			failures++
		}
		localWorker := envMap["BROKER_LOCAL_SCRIPT_PATH"]
		if localWorker == "" {
			localWorker = filepath.Join(repoRoot, "deploy", "local", "broker_worker.sh")
		}
		if !checkPath(localWorker, true) {
			failures++
		}
	}
	if mode == "slurm" {
		for _, name := range []string{"sbatch", "sacct", "scancel"} {
			if !checkExecutable(name, true) {
				failures++
			}
		}
		slurmScript := envMap["BROKER_SLURM_SCRIPT_PATH"]
		if slurmScript == "" {
			slurmScript = filepath.Join(repoRoot, "deploy", "slurm", "broker_worker.slurm")
		}
		if !checkPath(slurmScript, true) {
			failures++
		}
	}
	codexHome := defaultCodexHome()
	if profileExists(filepath.Join(codexHome, "local-broker.config.toml")) || profileExists(filepath.Join(codexHome, "slurm-broker.config.toml")) {
		reportCheck("codex-profiles", codexHome, true)
	} else {
		reportCheck("codex-profiles", "not installed under "+codexHome, false)
		warnings++
	}

	if failures > 0 {
		return commandError{message: fmt.Sprintf("doctor found %d failure(s) and %d warning(s)", failures, warnings), code: 1}
	}
	fmt.Printf("doctor completed with %d warning(s)\n", warnings)
	return nil
}

func runInstall(args []string) error {
	if len(args) == 0 {
		return commandError{message: "usage: local-ai-broker install codex [--local|--slurm|--all] [--codex-home PATH]", code: 2}
	}
	switch args[0] {
	case "codex":
		return runInstallCodex(args[1:])
	default:
		return commandError{message: "unknown install target: " + args[0], code: 2}
	}
}

func runInstallCodex(args []string) error {
	fs := flag.NewFlagSet("install codex", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	localMode := fs.Bool("local", false, "install only the local profile")
	slurmMode := fs.Bool("slurm", false, "install only the Slurm profile")
	allMode := fs.Bool("all", false, "install both profiles")
	codexHome := fs.String("codex-home", defaultCodexHome(), "target Codex config directory")
	if err := fs.Parse(args); err != nil {
		return commandError{message: err.Error(), code: 2}
	}
	repoRoot, err := findRepoRoot()
	if err != nil {
		return err
	}
	mode := "all"
	if *localMode || *slurmMode || *allMode {
		switch {
		case *localMode:
			mode = "local"
		case *slurmMode:
			mode = "slurm"
		default:
			mode = "all"
		}
	}
	if err := os.MkdirAll(*codexHome, 0o755); err != nil {
		return err
	}
	if mode == "local" || mode == "all" {
		if err := installCodexProfile(repoRoot, *codexHome, "local-broker.config.toml.template", "local-broker.config.toml"); err != nil {
			return err
		}
	}
	if mode == "slurm" || mode == "all" {
		if err := installCodexProfile(repoRoot, *codexHome, "slurm-broker.config.toml.template", "slurm-broker.config.toml"); err != nil {
			return err
		}
	}
	fmt.Println("Installed Codex profile(s).")
	fmt.Println("Use:")
	if mode == "local" || mode == "all" {
		fmt.Println("  codex -p local-broker")
	}
	if mode == "slurm" || mode == "all" {
		fmt.Println("  codex -p slurm-broker")
	}
	return nil
}

func runUp(args []string) error {
	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	localMode := fs.Bool("local", false, "start broker server in local backend mode")
	slurmMode := fs.Bool("slurm", false, "start broker server in Slurm backend mode")
	listenAddr := fs.String("listen-addr", "", "override BROKER_LISTEN_ADDR")
	configPath := fs.String("config", "", "optional broker config JSON file")
	envFile := fs.String("env-file", "", "optional env file with KEY=VALUE lines")
	if err := fs.Parse(args); err != nil {
		return commandError{message: err.Error(), code: 2}
	}
	mode := selectMode(*localMode, *slurmMode, "local")
	repoRoot, err := findRepoRoot()
	if err != nil {
		return err
	}
	envMap := defaultBrokerEnv(repoRoot, mode)
	if *configPath != "" {
		loaded, err := loadBootstrapConfig(repoRoot, *configPath)
		if err != nil {
			return err
		}
		for k, v := range loaded {
			envMap[k] = v
		}
		if backend := envMap["BROKER_BACKEND"]; backend == "slurm" {
			mode = "slurm"
		} else if backend == "local" {
			mode = "local"
		}
	}
	if *envFile != "" {
		loaded, err := parseEnvFile(*envFile)
		if err != nil {
			return err
		}
		for k, v := range loaded {
			envMap[k] = v
		}
	}
	if *listenAddr != "" {
		envMap["BROKER_LISTEN_ADDR"] = *listenAddr
	}
	if err := os.MkdirAll(filepath.Join(repoRoot, ".broker", "runs"), 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(envMap["BROKER_AUDIT_LOG_PATH"]), 0o755); err != nil {
		return err
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		return commandError{message: "missing required executable: go", code: 1}
	}
	cmd := exec.Command(goBin, "run", "./broker/cmd/broker-server")
	cmd.Dir = repoRoot
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = mergeEnv(envMap)
	fmt.Printf("starting broker-server in %s mode from %s\n", mode, repoRoot)
	return cmd.Run()
}

type bootstrapConfig struct {
	ListenAddr      string                 `json:"listen_addr"`
	JobStorePath    string                 `json:"job_store_path"`
	RunRootPath     string                 `json:"run_root_path"`
	RepoRootPath    string                 `json:"repo_root_path"`
	AuditLogPath    string                 `json:"audit_log_path"`
	AuditVerifyMode string                 `json:"audit_verify_mode"`
	AuthMode        string                 `json:"auth_mode"`
	MCPActor        string                 `json:"mcp_actor"`
	MCPRole         string                 `json:"mcp_role"`
	Backend         string                 `json:"backend"`
	Local           localBootstrapConfig   `json:"local"`
	Slurm           slurmBootstrapConfig   `json:"slurm"`
	Runtime         runtimeBootstrapConfig `json:"runtime"`
	Parallel        parallelBootstrapConfig `json:"parallel"`
}

type localBootstrapConfig struct {
	Mode       string `json:"mode"`
	ScriptPath string `json:"script_path"`
}

type slurmBootstrapConfig struct {
	Mode           string `json:"mode"`
	SubmitCmd      string `json:"submit_cmd"`
	StatusCmd      string `json:"status_cmd"`
	CancelCmd      string `json:"cancel_cmd"`
	ScriptPath     string `json:"script_path"`
	PartitionCPU   string `json:"partition_cpu"`
	PartitionP40   string `json:"partition_p40"`
	PartitionA100  string `json:"partition_a100"`
	NodeListCPU    string `json:"nodelist_cpu"`
	NodeListP40    string `json:"nodelist_p40"`
	NodeListA100   string `json:"nodelist_a100"`
	ConstraintCPU  string `json:"constraint_cpu"`
	ConstraintP40  string `json:"constraint_p40"`
	ConstraintA100 string `json:"constraint_a100"`
	ModelCPU       string `json:"model_profile_cpu"`
	ModelP40       string `json:"model_profile_p40"`
	ModelA100      string `json:"model_profile_a100"`
}

type runtimeBootstrapConfig struct {
	LlamaCPPBaseURL        string `json:"llama_cpp_base_url"`
	LlamaCPPTimeoutSeconds int    `json:"llama_cpp_timeout_seconds"`
	VLLMBaseURL            string `json:"vllm_base_url"`
	VLLMTimeoutSeconds     int    `json:"vllm_timeout_seconds"`
	SGLangBaseURL          string `json:"sglang_base_url"`
	SGLangTimeoutSeconds   int    `json:"sglang_timeout_seconds"`
}

type parallelBootstrapConfig struct {
	MaxBatchSize           int `json:"max_batch_size"`
	MaxActiveBatches       int `json:"max_active_batches"`
	MaxAdditionalBatches   int `json:"max_additional_batches"`
	MaxRetriedShards       int `json:"max_retried_shards"`
}

func findRepoRoot() (string, error) {
	candidates := []string{}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, wd)
	}
	if exePath, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Dir(exePath))
	}
	for _, start := range candidates {
		if root, ok := walkUpForRepoRoot(start); ok {
			return root, nil
		}
	}
	return "", commandError{message: "could not locate local-ai-broker repo root", code: 1}
}

func walkUpForRepoRoot(start string) (string, bool) {
	current := filepath.Clean(start)
	for {
		if looksLikeRepoRoot(current) {
			return current, true
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", false
		}
		current = parent
	}
}

func looksLikeRepoRoot(path string) bool {
	for _, rel := range []string{
		"go.mod",
		filepath.Join("broker", "cmd", "broker-server", "main.go"),
		filepath.Join("examples", "mcp-clients", "codex-profiles", "local-broker.config.toml.template"),
	} {
		if _, err := os.Stat(filepath.Join(path, rel)); err != nil {
			return false
		}
	}
	return true
}

func loadBootstrapConfig(repoRoot, path string) (map[string]string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg bootstrapConfig
	if err := json.Unmarshal(content, &cfg); err != nil {
		return nil, fmt.Errorf("parse config JSON: %w", err)
	}
	configDir := filepath.Dir(path)
	envMap := map[string]string{}
	set := func(key, value string, pathLike bool) {
		if value == "" {
			return
		}
		if pathLike {
			value = resolveConfigPath(repoRoot, configDir, value)
		}
		envMap[key] = value
	}
	set("BROKER_LISTEN_ADDR", cfg.ListenAddr, false)
	set("BROKER_JOB_STORE_PATH", cfg.JobStorePath, true)
	set("BROKER_RUN_ROOT_PATH", cfg.RunRootPath, true)
	set("BROKER_REPO_ROOT_PATH", cfg.RepoRootPath, true)
	set("BROKER_AUDIT_LOG_PATH", cfg.AuditLogPath, true)
	set("BROKER_AUDIT_VERIFY_MODE", cfg.AuditVerifyMode, false)
	set("BROKER_AUTH_MODE", cfg.AuthMode, false)
	set("BROKER_MCP_ACTOR", cfg.MCPActor, false)
	set("BROKER_MCP_ROLE", cfg.MCPRole, false)
	set("BROKER_BACKEND", cfg.Backend, false)
	set("BROKER_LOCAL_MODE", cfg.Local.Mode, false)
	set("BROKER_LOCAL_SCRIPT_PATH", cfg.Local.ScriptPath, true)
	set("BROKER_SLURM_MODE", cfg.Slurm.Mode, false)
	set("BROKER_SLURM_SUBMIT_CMD", cfg.Slurm.SubmitCmd, false)
	set("BROKER_SLURM_STATUS_CMD", cfg.Slurm.StatusCmd, false)
	set("BROKER_SLURM_CANCEL_CMD", cfg.Slurm.CancelCmd, false)
	set("BROKER_SLURM_SCRIPT_PATH", cfg.Slurm.ScriptPath, true)
	set("BROKER_SLURM_PARTITION_CPU", cfg.Slurm.PartitionCPU, false)
	set("BROKER_SLURM_PARTITION_P40", cfg.Slurm.PartitionP40, false)
	set("BROKER_SLURM_PARTITION_A100", cfg.Slurm.PartitionA100, false)
	set("BROKER_SLURM_NODELIST_CPU", cfg.Slurm.NodeListCPU, false)
	set("BROKER_SLURM_NODELIST_P40", cfg.Slurm.NodeListP40, false)
	set("BROKER_SLURM_NODELIST_A100", cfg.Slurm.NodeListA100, false)
	set("BROKER_SLURM_CONSTRAINT_CPU", cfg.Slurm.ConstraintCPU, false)
	set("BROKER_SLURM_CONSTRAINT_P40", cfg.Slurm.ConstraintP40, false)
	set("BROKER_SLURM_CONSTRAINT_A100", cfg.Slurm.ConstraintA100, false)
	set("BROKER_MODEL_PROFILE_CPU", cfg.Slurm.ModelCPU, false)
	set("BROKER_MODEL_PROFILE_P40", cfg.Slurm.ModelP40, false)
	set("BROKER_MODEL_PROFILE_A100", cfg.Slurm.ModelA100, false)
	set("BROKER_RUNTIME_LLAMACPP_BASE_URL", cfg.Runtime.LlamaCPPBaseURL, false)
	set("BROKER_RUNTIME_VLLM_BASE_URL", cfg.Runtime.VLLMBaseURL, false)
	set("BROKER_RUNTIME_SGLANG_BASE_URL", cfg.Runtime.SGLangBaseURL, false)
	if cfg.Runtime.LlamaCPPTimeoutSeconds > 0 {
		envMap["BROKER_RUNTIME_LLAMACPP_TIMEOUT_SECONDS"] = fmt.Sprintf("%d", cfg.Runtime.LlamaCPPTimeoutSeconds)
	}
	if cfg.Runtime.VLLMTimeoutSeconds > 0 {
		envMap["BROKER_RUNTIME_VLLM_TIMEOUT_SECONDS"] = fmt.Sprintf("%d", cfg.Runtime.VLLMTimeoutSeconds)
	}
	if cfg.Runtime.SGLangTimeoutSeconds > 0 {
		envMap["BROKER_RUNTIME_SGLANG_TIMEOUT_SECONDS"] = fmt.Sprintf("%d", cfg.Runtime.SGLangTimeoutSeconds)
	}
	if cfg.Parallel.MaxBatchSize > 0 {
		envMap["BROKER_PARALLEL_MAX_BATCH_SIZE"] = fmt.Sprintf("%d", cfg.Parallel.MaxBatchSize)
	}
	if cfg.Parallel.MaxActiveBatches > 0 {
		envMap["BROKER_PARALLEL_MAX_ACTIVE_BATCHES"] = fmt.Sprintf("%d", cfg.Parallel.MaxActiveBatches)
	}
	if cfg.Parallel.MaxAdditionalBatches > 0 {
		envMap["BROKER_ROOT_ACTION_MAX_ADDITIONAL_BATCHES"] = fmt.Sprintf("%d", cfg.Parallel.MaxAdditionalBatches)
	}
	if cfg.Parallel.MaxRetriedShards > 0 {
		envMap["BROKER_ROOT_ACTION_MAX_RETRIED_SHARDS"] = fmt.Sprintf("%d", cfg.Parallel.MaxRetriedShards)
	}
	return envMap, nil
}

func writeBootstrapConfig(path string, cfg bootstrapConfig) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	content, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	content = append(content, '\n')
	return os.WriteFile(path, content, 0o644)
}

func resolveConfigPath(repoRoot, configDir, value string) string {
	value = strings.ReplaceAll(value, "__REPO_ROOT__", repoRoot)
	if filepath.IsAbs(value) {
		return value
	}
	return filepath.Clean(filepath.Join(configDir, value))
}

func defaultBrokerEnv(repoRoot, mode string) map[string]string {
	envMap := map[string]string{
		"GOENV":                  "off",
		"GOCACHE":                envOr("GOCACHE", "/tmp/local-ai-broker-gocache"),
		"GOPATH":                 envOr("GOPATH", "/tmp/local-ai-broker-gopath"),
		"BROKER_LISTEN_ADDR":     envOr("BROKER_LISTEN_ADDR", "127.0.0.1:8081"),
		"BROKER_JOB_STORE_PATH":  envOr("BROKER_JOB_STORE_PATH", filepath.Join(repoRoot, ".broker", "jobs.json")),
		"BROKER_RUN_ROOT_PATH":   envOr("BROKER_RUN_ROOT_PATH", filepath.Join(repoRoot, ".broker", "runs")),
		"BROKER_REPO_ROOT_PATH":  envOr("BROKER_REPO_ROOT_PATH", repoRoot),
		"BROKER_AUDIT_LOG_PATH":  envOr("BROKER_AUDIT_LOG_PATH", filepath.Join(repoRoot, ".broker", "audit.jsonl")),
		"BROKER_AUDIT_VERIFY_MODE": envOr("BROKER_AUDIT_VERIFY_MODE", "warn"),
	}
	if mode == "local" {
		envMap["BROKER_BACKEND"] = envOr("BROKER_BACKEND", "local")
		envMap["BROKER_LOCAL_MODE"] = envOr("BROKER_LOCAL_MODE", "command")
		envMap["BROKER_LOCAL_SCRIPT_PATH"] = envOr("BROKER_LOCAL_SCRIPT_PATH", filepath.Join(repoRoot, "deploy", "local", "broker_worker.sh"))
	} else {
		envMap["BROKER_BACKEND"] = envOr("BROKER_BACKEND", "slurm")
		envMap["BROKER_SLURM_MODE"] = envOr("BROKER_SLURM_MODE", "command")
		envMap["BROKER_SLURM_SUBMIT_CMD"] = envOr("BROKER_SLURM_SUBMIT_CMD", "sbatch")
		envMap["BROKER_SLURM_STATUS_CMD"] = envOr("BROKER_SLURM_STATUS_CMD", "sacct")
		envMap["BROKER_SLURM_CANCEL_CMD"] = envOr("BROKER_SLURM_CANCEL_CMD", "scancel")
		envMap["BROKER_SLURM_SCRIPT_PATH"] = envOr("BROKER_SLURM_SCRIPT_PATH", filepath.Join(repoRoot, "deploy", "slurm", "broker_worker.slurm"))
	}
	return envMap
}

func installCodexProfile(repoRoot, codexHome, templateName, outputName string) error {
	templatePath := filepath.Join(repoRoot, "examples", "mcp-clients", "codex-profiles", templateName)
	content, err := os.ReadFile(templatePath)
	if err != nil {
		return err
	}
	rendered := strings.ReplaceAll(string(content), "__REPO_ROOT__", repoRoot)
	outputPath := filepath.Join(codexHome, outputName)
	if err := os.WriteFile(outputPath, []byte(rendered), 0o644); err != nil {
		return err
	}
	fmt.Printf("installed %s\n", outputPath)
	return nil
}

func parseEnvFile(path string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	values := map[string]string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("invalid env line: %s", line)
		}
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		values[strings.TrimSpace(key)] = value
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return values, nil
}

func mergeEnv(overrides map[string]string) []string {
	base := map[string]string{}
	for _, entry := range os.Environ() {
		key, value, ok := strings.Cut(entry, "=")
		if ok {
			base[key] = value
		}
	}
	for _, unsetKey := range []string{"GOROOT"} {
		delete(base, unsetKey)
	}
	for k, v := range overrides {
		base[k] = v
	}
	env := make([]string, 0, len(base))
	for k, v := range base {
		env = append(env, k+"="+v)
	}
	return env
}

func checkExecutable(name string, required bool) bool {
	path, err := exec.LookPath(name)
	if err != nil {
		reportCheck(name, "missing", false)
		return !required
	}
	reportCheck(name, path, true)
	return true
}

func checkPath(path string, required bool) bool {
	if _, err := os.Stat(path); err != nil {
		reportCheck(filepath.Base(path), "missing", false)
		return !required
	}
	reportCheck(filepath.Base(path), path, true)
	return true
}

func checkWritableBrokerPaths(repoRoot string) bool {
	brokerRoot := filepath.Join(repoRoot, ".broker")
	if err := os.MkdirAll(filepath.Join(brokerRoot, "runs"), 0o755); err != nil {
		reportCheck(".broker", err.Error(), false)
		return false
	}
	testPath := filepath.Join(brokerRoot, ".doctor-write-test")
	if err := os.WriteFile(testPath, []byte("ok"), 0o644); err != nil {
		reportCheck(".broker", err.Error(), false)
		return false
	}
	_ = os.Remove(testPath)
	reportCheck(".broker", brokerRoot, true)
	return true
}

func reportCheck(name, detail string, ok bool) {
	status := "OK"
	if !ok {
		status = "WARN"
	}
	fmt.Printf("[%s] %s: %s\n", status, name, detail)
}

func selectMode(localMode, slurmMode bool, fallback string) string {
	switch {
	case localMode:
		return "local"
	case slurmMode:
		return "slurm"
	default:
		return fallback
	}
}

func defaultCodexHome() string {
	if value := os.Getenv("CODEX_HOME"); value != "" {
		return value
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ".codex"
	}
	return filepath.Join(home, ".codex")
}

func profileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
