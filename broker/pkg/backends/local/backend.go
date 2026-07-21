package local

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/jobenv"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type Backend struct {
	counter     atomic.Uint64
	mode        string
	cfg         config.Config
	completions struct {
		once     sync.Once
		listener net.PacketConn
		mu       sync.Mutex
		waiters  map[string]chan struct{}
	}
}

type heartbeat struct {
	State string `json:"state"`
	Phase string `json:"phase"`
}

func NewBackend(cfg config.Config) *Backend {
	return &Backend{
		mode: cfg.LocalMode,
		cfg:  cfg,
	}
}

func (b *Backend) Name() string {
	return "local"
}

func (b *Backend) StartInspectRepoWarmDaemon() (int, bool, error) {
	if !b.cfg.LocalInspectRepoWarmEnabled || !b.commandMode() {
		return 0, false, nil
	}
	repoRoot := strings.TrimSpace(b.cfg.RepoRootPath)
	if repoRoot == "" {
		repoRoot = "."
	}
	workerPath, ok := directWorkerPath(repoRoot, b.cfg.LocalScriptPath, "inspect_repo")
	if !ok {
		return 0, false, nil
	}
	pid, err := ensureInspectRepoWarmDaemon(repoRoot, filepath.Join(b.cfg.RunRootPath, ".inspect-repo-warm"), workerPath)
	if err != nil {
		return 0, true, err
	}
	return pid, true, nil
}

func (b *Backend) commandMode() bool {
	return b.mode == "command"
}

func (b *Backend) nextStubRunID() string {
	return fmt.Sprintf("local-%06d", b.counter.Add(1))
}

func localRunStatus(backendRunID string, state types.JobState, rawState string) backends.RunStatus {
	return backends.RunStatus{
		BackendRunID: backendRunID,
		State:        state,
		RawState:     rawState,
	}
}

func (b *Backend) SubmitRun(_ context.Context, job types.Job) (backends.SubmitResponse, error) {
	if !b.commandMode() {
		return backends.StubSubmitResponse(b.Name(), b.nextStubRunID()), nil
	}

	runRoot := jobenv.RunRoot(job)
	repoRoot := jobenv.RepoRoot(job)
	runID := job.ID
	outputDir := filepath.Join(runRoot, runID)
	if info, err := os.Stat(outputDir); err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(outputDir, 0o755); err != nil {
				return backends.SubmitResponse{}, fmt.Errorf("create output dir: %w", err)
			}
		} else {
			return backends.SubmitResponse{}, fmt.Errorf("stat output dir: %w", err)
		}
	} else if !info.IsDir() {
		return backends.SubmitResponse{}, fmt.Errorf("output dir path is not a directory: %s", outputDir)
	}

	cmd, err := b.commandForJob(repoRoot, outputDir, job)
	if err != nil {
		return backends.SubmitResponse{}, err
	}
	completionSocketPath := b.registerCompletionWaiter(runID)
	if requestPath, daemonPID, ok, err := b.enqueueInspectRepoWarmRequest(repoRoot, outputDir, job, backends.InlineExecutionBundle{}); err != nil {
		return backends.SubmitResponse{}, err
	} else if ok {
		if err := os.WriteFile(filepath.Join(outputDir, "local.pid"), []byte(strconv.Itoa(daemonPID)), 0o644); err != nil {
			return backends.SubmitResponse{}, fmt.Errorf("write warm daemon pid file: %w", err)
		}
		if err := os.WriteFile(filepath.Join(outputDir, "warm-request.marker"), []byte(filepath.Base(requestPath)), 0o644); err != nil {
			return backends.SubmitResponse{}, fmt.Errorf("write warm request marker: %w", err)
		}
		return backends.SubmitResponse{
			BackendKind:  b.Name(),
			BackendRunID: runID,
			InitialState: types.JobStateDispatching,
		}, nil
	}

	stdoutLog, err := os.OpenFile(filepath.Join(outputDir, "stdout.log"), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return backends.SubmitResponse{}, fmt.Errorf("open local stdout log: %w", err)
	}
	stderrLog, err := os.OpenFile(filepath.Join(outputDir, "stderr.log"), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		_ = stdoutLog.Close()
		return backends.SubmitResponse{}, fmt.Errorf("open local stderr log: %w", err)
	}
	cmd.Stdout = stdoutLog
	cmd.Stderr = stderrLog

	if completionSocketPath != "" {
		cmd.Args = append(cmd.Args, "--completion-socket-path", completionSocketPath)
	}
	if err := cmd.Start(); err != nil {
		_ = stdoutLog.Close()
		_ = stderrLog.Close()
		return backends.SubmitResponse{}, fmt.Errorf("start local worker: %w", err)
	}

	if err := os.WriteFile(filepath.Join(outputDir, "local.pid"), []byte(strconv.Itoa(cmd.Process.Pid)), 0o644); err != nil {
		_ = cmd.Process.Kill()
		_ = stdoutLog.Close()
		_ = stderrLog.Close()
		return backends.SubmitResponse{}, fmt.Errorf("write pid file: %w", err)
	}

	go func(stdoutHandle, stderrHandle *os.File) {
		_ = cmd.Wait()
		_ = stdoutHandle.Close()
		_ = stderrHandle.Close()
	}(stdoutLog, stderrLog)

	return backends.SubmitResponse{
		BackendKind:  b.Name(),
		BackendRunID: runID,
		InitialState: types.JobStateDispatching,
	}, nil
}

func (b *Backend) SubmitWarmInspectRepoRun(_ context.Context, job types.Job, bundle backends.InlineExecutionBundle) (backends.SubmitResponse, bool, error) {
	if !b.commandMode() || !b.warmInspectRepoEnabled(job) {
		return backends.SubmitResponse{}, false, nil
	}
	runRoot := jobenv.RunRoot(job)
	repoRoot := jobenv.RepoRoot(job)
	runID := job.ID
	outputDir := filepath.Join(runRoot, runID)
	if info, err := os.Stat(outputDir); err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(outputDir, 0o755); err != nil {
				return backends.SubmitResponse{}, false, fmt.Errorf("create output dir: %w", err)
			}
		} else {
			return backends.SubmitResponse{}, false, fmt.Errorf("stat output dir: %w", err)
		}
	} else if !info.IsDir() {
		return backends.SubmitResponse{}, false, fmt.Errorf("output dir path is not a directory: %s", outputDir)
	}
	b.registerCompletionWaiter(runID)
	requestPath, daemonPID, ok, err := b.enqueueInspectRepoWarmRequest(repoRoot, outputDir, job, bundle)
	if err != nil {
		return backends.SubmitResponse{}, false, err
	}
	if !ok {
		return backends.SubmitResponse{}, false, nil
	}
	if err := os.WriteFile(filepath.Join(outputDir, "local.pid"), []byte(strconv.Itoa(daemonPID)), 0o644); err != nil {
		return backends.SubmitResponse{}, false, fmt.Errorf("write warm daemon pid file: %w", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "warm-request.marker"), []byte(filepath.Base(requestPath)), 0o644); err != nil {
		return backends.SubmitResponse{}, false, fmt.Errorf("write warm request marker: %w", err)
	}
	return backends.SubmitResponse{
		BackendKind:  b.Name(),
		BackendRunID: runID,
		InitialState: types.JobStateDispatching,
	}, true, nil
}

func (b *Backend) commandForJob(repoRoot, outputDir string, job types.Job) (*exec.Cmd, error) {
	jobSpecPath := filepath.Join(outputDir, "job_spec.json")
	executionPlanPath := filepath.Join(outputDir, "execution_plan.json")
	inputManifestPath := filepath.Join(outputDir, "input_manifest.json")
	heartbeatPath := filepath.Join(outputDir, "heartbeat.json")

	if workerPath, ok := directWorkerPath(repoRoot, b.cfg.LocalScriptPath, job.TaskType); ok {
		args := append(pythonLauncherArgsForTask(job.TaskType, workerPath), "--job-spec", jobSpecPath)
		if workerAcceptsExecutionPlan(job.TaskType) {
			args = append(args, "--execution-plan", executionPlanPath)
		}
		args = append(args,
			"--input-manifest", inputManifestPath,
			"--output-dir", outputDir,
			"--heartbeat-path", heartbeatPath)
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Dir = repoRoot
		cmd.Env = append(os.Environ(), "BROKER_REPO_ROOT="+repoRoot)
		return cmd, nil
	}

	scriptPath := resolvePath(repoRoot, b.cfg.LocalScriptPath)
	cmd := exec.Command("bash", scriptPath)
	cmd.Dir = repoRoot
	cmd.Env = append(os.Environ(),
		"BROKER_JOB_ID="+job.ID,
		"BROKER_TASK_TYPE="+job.TaskType,
		"BROKER_REPO_ROOT="+repoRoot,
		"BROKER_OUTPUT_DIR="+outputDir,
		"BROKER_OUTPUT_SCHEMA="+job.Request.OutputSchema.Name,
	)
	return cmd, nil
}

func workerAcceptsExecutionPlan(taskType string) bool {
	switch strings.TrimSpace(taskType) {
	case "inspect_repo", "rag_compress", "debug_with_local_context", "summarize_logs", "propose_patch":
		return true
	default:
		return false
	}
}

func pythonLauncherArgsForTask(taskType, workerPath string) []string {
	args := []string{"python3"}
	switch strings.TrimSpace(taskType) {
	case "inspect_repo":
		args = append(args, "-S")
	}
	args = append(args, workerPath)
	return args
}

func directWorkerPath(repoRoot, localScriptPath, taskType string) (string, bool) {
	base := filepath.Base(strings.TrimSpace(localScriptPath))
	if base != "broker_worker.sh" && base != "broker_worker.slurm" {
		return "", false
	}
	var relative string
	switch strings.TrimSpace(taskType) {
	case "document_summary":
		relative = filepath.Join("workers", "document-summary", "main.py")
	case "log_analysis":
		relative = filepath.Join("workers", "log-analysis", "main.py")
	case "repo_summary":
		relative = filepath.Join("workers", "repo-summary", "main.py")
	case "inspect_repo":
		relative = filepath.Join("workers", "rag-compression", "inspect_repo_worker.py")
	case "rag_compress", "debug_with_local_context", "summarize_logs", "propose_patch":
		relative = filepath.Join("workers", "rag-compression", "main.py")
	default:
		return "", false
	}
	path := filepath.Join(repoRoot, relative)
	if _, err := os.Stat(path); err != nil {
		return "", false
	}
	return path, true
}

func (b *Backend) GetRun(_ context.Context, backendRunID string) (backends.RunStatus, error) {
	if !b.commandMode() {
		return backends.StubRunStatus(backendRunID), nil
	}

	runDir := filepath.Join(b.cfg.RunRootPath, backendRunID)
	if _, err := os.Stat(filepath.Join(runDir, "result.json")); err == nil {
		return localRunStatus(backendRunID, types.JobStateSucceeded, "COMPLETED"), nil
	}
	if hb, err := readHeartbeat(filepath.Join(runDir, "heartbeat.json")); err == nil {
		if state, raw := mapHeartbeatState(hb.State); state != "" {
			if state == types.JobStateRunning {
				pid, pidErr := readPID(filepath.Join(runDir, "local.pid"))
				if pidErr == nil && !processAlive(pid) {
					return failedRunStatus(backendRunID, runDir, "worker_exited", "EXITED"), nil
				}
			}
			if state == types.JobStateFailed {
				return failedRunStatus(backendRunID, runDir, "worker_failed", raw), nil
			}
			return localRunStatus(backendRunID, state, raw), nil
		}
	}

	pid, err := readPID(filepath.Join(runDir, "local.pid"))
	if err == nil && processAlive(pid) {
		return localRunStatus(backendRunID, types.JobStateRunning, "RUNNING"), nil
	}

	if err == nil {
		return failedRunStatus(backendRunID, runDir, "worker_exited", "EXITED"), nil
	}

	return localRunStatus(backendRunID, types.JobStateQueued, "PENDING"), nil
}

func failedRunStatus(backendRunID, runDir, category, rawState string) backends.RunStatus {
	return backends.RunStatus{
		BackendRunID: backendRunID,
		State:        types.JobStateFailed,
		RawState:     rawState,
		Diagnostics: map[string]any{
			"backend_failure_category": category,
			"worker_output_dir":        runDir,
			"stdout_log":               filepath.Join(runDir, "stdout.log"),
			"stderr_log":               filepath.Join(runDir, "stderr.log"),
		},
	}
}

func (b *Backend) CancelRun(_ context.Context, backendRunID string) error {
	if !b.commandMode() {
		return nil
	}
	runDir := filepath.Join(b.cfg.RunRootPath, backendRunID)
	if _, err := os.Stat(filepath.Join(runDir, "warm-request.marker")); err == nil {
		if err := os.WriteFile(filepath.Join(runDir, "cancel.request"), []byte("cancel\n"), 0o644); err != nil {
			return fmt.Errorf("write cancel request: %w", err)
		}
		return nil
	}

	pid, err := readPID(filepath.Join(runDir, "local.pid"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("read pid file: %w", err)
	}

	if err := syscall.Kill(pid, syscall.SIGTERM); err != nil && !errors.Is(err, syscall.ESRCH) {
		return fmt.Errorf("terminate local worker pid %d: %w", pid, err)
	}
	return nil
}

func (b *Backend) warmInspectRepoEnabled(job types.Job) bool {
	return b.cfg.LocalInspectRepoWarmEnabled && b.commandMode() && strings.TrimSpace(job.TaskType) == "inspect_repo"
}

func (b *Backend) enqueueInspectRepoWarmRequest(repoRoot, outputDir string, job types.Job, bundle backends.InlineExecutionBundle) (string, int, bool, error) {
	if !b.warmInspectRepoEnabled(job) {
		return "", 0, false, nil
	}
	workerPath, ok := directWorkerPath(repoRoot, b.cfg.LocalScriptPath, job.TaskType)
	if !ok {
		return "", 0, false, nil
	}
	spoolDir := filepath.Join(b.cfg.RunRootPath, ".inspect-repo-warm")
	daemonPID, err := ensureInspectRepoWarmDaemon(repoRoot, spoolDir, workerPath)
	if err != nil {
		return "", 0, false, err
	}
	requestPath := filepath.Join(spoolDir, "requests", job.ID+".json")
	enqueuedNS := time.Now().UnixNano()
	requestPayload := map[string]any{
		"job_id":                     job.ID,
		"output_dir":                 outputDir,
		"heartbeat_path":             filepath.Join(outputDir, "heartbeat.json"),
		"completion_socket_path":     b.completionSocketPath(),
		"broker_request_enqueued_ns": enqueuedNS,
		"broker_request_written_ns":  time.Now().UnixNano(),
	}
	if len(bundle.JobSpec) != 0 {
		requestPayload["job_spec"] = bundle.JobSpec
	} else {
		requestPayload["job_spec_path"] = filepath.Join(outputDir, "job_spec.json")
	}
	if len(bundle.ExecutionPlan) != 0 {
		requestPayload["execution_plan"] = bundle.ExecutionPlan
	} else {
		requestPayload["execution_plan_path"] = filepath.Join(outputDir, "execution_plan.json")
	}
	if len(bundle.InputManifest) != 0 {
		requestPayload["input_manifest"] = bundle.InputManifest
	} else {
		requestPayload["input_manifest_path"] = filepath.Join(outputDir, "input_manifest.json")
	}
	if err := atomicWriteJSONFile(requestPath, requestPayload, 0o644); err != nil {
		return "", 0, false, fmt.Errorf("write warm request: %w", err)
	}
	notifyInspectRepoWarmDaemon(spoolDir, filepath.Base(requestPath))
	return requestPath, daemonPID, true, nil
}

func warmDaemonSocketPath(spoolDir string) string {
	return filepath.Join(spoolDir, "daemon.sock")
}

func warmDaemonBusyMarkerPath(spoolDir string) string {
	return filepath.Join(spoolDir, "busy.marker")
}

func warmDaemonBusy(spoolDir string) bool {
	if _, err := os.Stat(warmDaemonBusyMarkerPath(spoolDir)); err == nil {
		return true
	}
	return false
}

func waitForInspectRepoWarmDaemonReady(spoolDir string, pid int, waitWindow time.Duration) bool {
	if pid <= 0 || waitWindow <= 0 {
		return false
	}
	heartbeatPath := filepath.Join(spoolDir, "daemon-heartbeat.json")
	socketPath := warmDaemonSocketPath(spoolDir)
	deadline := time.Now().Add(waitWindow)
	for time.Now().Before(deadline) {
		if !processAlive(pid) {
			return false
		}
		heartbeatReady := false
		if _, err := os.Stat(heartbeatPath); err == nil {
			heartbeatReady = true
		}
		socketReady := false
		if _, err := os.Stat(socketPath); err == nil {
			socketReady = true
		}
		if heartbeatReady && socketReady {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return processAlive(pid)
}

func ensureInspectRepoWarmDaemon(repoRoot, spoolDir, workerPath string) (int, error) {
	if err := os.MkdirAll(filepath.Join(spoolDir, "requests"), 0o755); err != nil {
		return 0, fmt.Errorf("create warm daemon spool: %w", err)
	}
	pidPath := filepath.Join(spoolDir, "daemon.pid")
	if pid, err := readPID(pidPath); err == nil && processAlive(pid) {
		_ = waitForInspectRepoWarmDaemonReady(spoolDir, pid, 100*time.Millisecond)
		return pid, nil
	}
	stdoutLog, err := os.OpenFile(filepath.Join(spoolDir, "daemon.stdout.log"), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		return 0, fmt.Errorf("open warm daemon stdout log: %w", err)
	}
	stderrLog, err := os.OpenFile(filepath.Join(spoolDir, "daemon.stderr.log"), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o644)
	if err != nil {
		_ = stdoutLog.Close()
		return 0, fmt.Errorf("open warm daemon stderr log: %w", err)
	}
	cmd := exec.Command("python3", "-S", workerPath, "--daemon-spool-dir", spoolDir, "--repo-root", repoRoot)
	cmd.Dir = repoRoot
	cmd.Env = append(os.Environ(), "BROKER_REPO_ROOT="+repoRoot)
	cmd.Stdout = stdoutLog
	cmd.Stderr = stderrLog
	if err := cmd.Start(); err != nil {
		_ = stdoutLog.Close()
		_ = stderrLog.Close()
		return 0, fmt.Errorf("start warm daemon: %w", err)
	}
	pid := cmd.Process.Pid
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(pid)), 0o644); err != nil {
		_ = cmd.Process.Kill()
		_ = stdoutLog.Close()
		_ = stderrLog.Close()
		return 0, fmt.Errorf("write warm daemon pid: %w", err)
	}
	go func(stdoutHandle, stderrHandle *os.File) {
		_ = cmd.Wait()
		_ = stdoutHandle.Close()
		_ = stderrHandle.Close()
	}(stdoutLog, stderrLog)
	_ = waitForInspectRepoWarmDaemonReady(spoolDir, pid, 2*time.Second)
	return pid, nil
}

func atomicWriteJSONFile(path string, payload any, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := fmt.Sprintf("%s.tmp-%d", path, os.Getpid())
	_ = os.Remove(tmp)
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func notifyInspectRepoWarmDaemon(spoolDir, requestName string) {
	socketPath := warmDaemonSocketPath(spoolDir)
	requestName = strings.TrimSpace(requestName)
	if requestName == "" {
		return
	}
	conn, err := net.Dial("unixgram", socketPath)
	if err != nil {
		return
	}
	defer conn.Close()
	_, _ = conn.Write([]byte(requestName))
}

func (b *Backend) AwaitLocalInspectRepoResult(ctx context.Context, backendRunID string, waitWindow time.Duration) bool {
	if !b.commandMode() || strings.TrimSpace(backendRunID) == "" || waitWindow <= 0 {
		return false
	}
	waiter := b.completionWaiter(backendRunID)
	if waiter == nil {
		return false
	}
	timer := time.NewTimer(waitWindow)
	defer timer.Stop()
	select {
	case <-waiter:
		return true
	case <-timer.C:
		return false
	case <-ctx.Done():
		return false
	}
}

func (b *Backend) completionSocketPath() string {
	runRoot := strings.TrimSpace(b.cfg.RunRootPath)
	if runRoot == "" {
		return ""
	}
	return filepath.Join(runRoot, ".local-inspect-repo-complete.sock")
}

func (b *Backend) registerCompletionWaiter(runID string) string {
	runID = strings.TrimSpace(runID)
	if runID == "" || !b.commandMode() {
		return ""
	}
	if err := b.ensureCompletionListener(); err != nil {
		return ""
	}
	b.completions.mu.Lock()
	defer b.completions.mu.Unlock()
	if b.completions.waiters == nil {
		b.completions.waiters = make(map[string]chan struct{})
	}
	if existing := b.completions.waiters[runID]; existing != nil {
		return b.completionSocketPath()
	}
	b.completions.waiters[runID] = make(chan struct{})
	return b.completionSocketPath()
}

func (b *Backend) completionWaiter(runID string) chan struct{} {
	if err := b.ensureCompletionListener(); err != nil {
		return nil
	}
	b.completions.mu.Lock()
	defer b.completions.mu.Unlock()
	if b.completions.waiters == nil {
		return nil
	}
	return b.completions.waiters[strings.TrimSpace(runID)]
}

func (b *Backend) ensureCompletionListener() error {
	var setupErr error
	b.completions.once.Do(func() {
		socketPath := b.completionSocketPath()
		if socketPath == "" {
			setupErr = fmt.Errorf("empty completion socket path")
			return
		}
		if err := os.MkdirAll(filepath.Dir(socketPath), 0o755); err != nil {
			setupErr = err
			return
		}
		_ = os.Remove(socketPath)
		listener, err := net.ListenPacket("unixgram", socketPath)
		if err != nil {
			setupErr = err
			return
		}
		b.completions.listener = listener
		go b.serveCompletionSignals(listener)
	})
	if setupErr != nil {
		return setupErr
	}
	if b.completions.listener == nil {
		return fmt.Errorf("completion listener unavailable")
	}
	return nil
}

func (b *Backend) serveCompletionSignals(listener net.PacketConn) {
	buffer := make([]byte, 512)
	for {
		n, _, err := listener.ReadFrom(buffer)
		if err != nil {
			return
		}
		runID := strings.TrimSpace(string(buffer[:n]))
		if runID == "" {
			continue
		}
		b.completions.mu.Lock()
		waiter := b.completions.waiters[runID]
		if waiter != nil {
			close(waiter)
			delete(b.completions.waiters, runID)
		}
		b.completions.mu.Unlock()
	}
}

func resolvePath(repoRoot, candidate string) string {
	if filepath.IsAbs(candidate) {
		return candidate
	}
	return filepath.Join(repoRoot, candidate)
}

func readPID(path string) (int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return 0, fmt.Errorf("parse pid: %w", err)
	}
	return pid, nil
}

func processAlive(pid int) bool {
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

func readHeartbeat(path string) (heartbeat, error) {
	var hb heartbeat
	data, err := os.ReadFile(path)
	if err != nil {
		return hb, err
	}
	if err := json.Unmarshal(data, &hb); err != nil {
		return hb, err
	}
	return hb, nil
}

func mapHeartbeatState(state string) (types.JobState, string) {
	switch strings.ToLower(strings.TrimSpace(state)) {
	case "running":
		return types.JobStateRunning, "RUNNING"
	case "completed":
		return types.JobStateSucceeded, "COMPLETED"
	case "failed":
		return types.JobStateFailed, "FAILED"
	case "cancelled":
		return types.JobStateCancelled, "CANCELLED"
	default:
		return "", ""
	}
}
