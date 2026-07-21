package local

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestGetRunFromHeartbeatCompleted(t *testing.T) {
	runRoot := t.TempDir()
	runDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "heartbeat.json"), []byte(`{"state":"completed","phase":"completed"}`), 0o644); err != nil {
		t.Fatalf("write heartbeat: %v", err)
	}

	backend := NewBackend(config.Config{
		LocalMode:       "command",
		RunRootPath:     runRoot,
		LocalScriptPath: "deploy/local/broker_worker.sh",
	})
	status, err := backend.GetRun(context.Background(), "job_123")
	if err != nil {
		t.Fatalf("get run: %v", err)
	}
	if status.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", status.State)
	}
}

func TestGetRunRunningFromPID(t *testing.T) {
	runRoot := t.TempDir()
	runDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "local.pid"), []byte(strconv.Itoa(os.Getpid())), 0o644); err != nil {
		t.Fatalf("write pid: %v", err)
	}

	backend := NewBackend(config.Config{
		LocalMode:       "command",
		RunRootPath:     runRoot,
		LocalScriptPath: "deploy/local/broker_worker.sh",
	})
	status, err := backend.GetRun(context.Background(), "job_123")
	if err != nil {
		t.Fatalf("get run: %v", err)
	}
	if status.State != types.JobStateRunning {
		t.Fatalf("expected running, got %q", status.State)
	}
}

func TestSubmitRunStubMode(t *testing.T) {
	backend := NewBackend(config.Config{LocalMode: "stub"})
	resp, err := backend.SubmitRun(context.Background(), types.Job{TaskType: "document_summary"})
	if err != nil {
		t.Fatalf("submit run: %v", err)
	}
	if resp.BackendKind != "local" {
		t.Fatalf("expected local backend kind, got %q", resp.BackendKind)
	}
	if resp.InitialState != types.JobStateQueued {
		t.Fatalf("expected queued, got %q", resp.InitialState)
	}
}

func TestGetRunSucceededFromResultFile(t *testing.T) {
	runRoot := t.TempDir()
	runDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "result.json"), []byte(`{"schema_name":"document_summary_v1"}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}

	backend := NewBackend(config.Config{LocalMode: "command", RunRootPath: runRoot})
	status, err := backend.GetRun(context.Background(), "job_123")
	if err != nil {
		t.Fatalf("get run: %v", err)
	}
	if status.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", status.State)
	}
}

func TestGetRunPrefersResultFileOverLivePID(t *testing.T) {
	runRoot := t.TempDir()
	runDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "result.json"), []byte(`{"schema_name":"document_summary_v1"}`), 0o644); err != nil {
		t.Fatalf("write result: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "local.pid"), []byte(strconv.Itoa(os.Getpid())), 0o644); err != nil {
		t.Fatalf("write pid: %v", err)
	}

	backend := NewBackend(config.Config{LocalMode: "command", RunRootPath: runRoot})
	status, err := backend.GetRun(context.Background(), "job_123")
	if err != nil {
		t.Fatalf("get run: %v", err)
	}
	if status.State != types.JobStateSucceeded {
		t.Fatalf("expected succeeded, got %q", status.State)
	}
}

func TestGetRunFailedWhenPIDExitedWithoutResult(t *testing.T) {
	runRoot := t.TempDir()
	runDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "local.pid"), []byte("999999"), 0o644); err != nil {
		t.Fatalf("write pid: %v", err)
	}

	backend := NewBackend(config.Config{LocalMode: "command", RunRootPath: runRoot})
	status, err := backend.GetRun(context.Background(), "job_123")
	if err != nil {
		t.Fatalf("get run: %v", err)
	}
	if status.State != types.JobStateFailed {
		t.Fatalf("expected failed, got %q", status.State)
	}
}

func TestGetRunFailedWhenRunningHeartbeatHasExitedPID(t *testing.T) {
	runRoot := t.TempDir()
	runDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "heartbeat.json"), []byte(`{"state":"running","phase":"gpu_first_retrieval"}`), 0o644); err != nil {
		t.Fatalf("write heartbeat: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "local.pid"), []byte("999999"), 0o644); err != nil {
		t.Fatalf("write pid: %v", err)
	}

	backend := NewBackend(config.Config{LocalMode: "command", RunRootPath: runRoot})
	status, err := backend.GetRun(context.Background(), "job_123")
	if err != nil {
		t.Fatalf("get run: %v", err)
	}
	if status.State != types.JobStateFailed {
		t.Fatalf("expected failed, got %q", status.State)
	}
	if status.RawState != "EXITED" {
		t.Fatalf("expected EXITED raw state, got %q", status.RawState)
	}
	if status.Diagnostics["backend_failure_category"] != "worker_exited" {
		t.Fatalf("expected worker_exited diagnostic, got %#v", status.Diagnostics)
	}
	if status.Diagnostics["stdout_log"] != filepath.Join(runDir, "stdout.log") {
		t.Fatalf("expected stdout log diagnostic, got %#v", status.Diagnostics["stdout_log"])
	}
}

func TestCancelRunMissingPIDIsNoop(t *testing.T) {
	runRoot := t.TempDir()
	backend := NewBackend(config.Config{LocalMode: "command", RunRootPath: runRoot})
	if err := backend.CancelRun(context.Background(), "job_123"); err != nil {
		t.Fatalf("cancel run should ignore missing pid file: %v", err)
	}
}

func TestDirectWorkerPathUsesInspectRepoWorkerForInspectRepo(t *testing.T) {
	repoRoot := t.TempDir()
	workerPath := filepath.Join(repoRoot, "workers", "rag-compression", "inspect_repo_worker.py")
	if err := os.MkdirAll(filepath.Dir(workerPath), 0o755); err != nil {
		t.Fatalf("mkdir worker dir: %v", err)
	}
	if err := os.WriteFile(workerPath, []byte("#!/usr/bin/env python3\n"), 0o644); err != nil {
		t.Fatalf("write worker: %v", err)
	}

	path, ok := directWorkerPath(repoRoot, "deploy/local/broker_worker.sh", "inspect_repo")
	if !ok {
		t.Fatalf("expected direct worker path")
	}
	if path != workerPath {
		t.Fatalf("expected %q, got %q", workerPath, path)
	}
}

func TestCommandForJobUsesDirectPythonWorkerForDefaultWrapper(t *testing.T) {
	repoRoot := t.TempDir()
	workerPath := filepath.Join(repoRoot, "workers", "rag-compression", "inspect_repo_worker.py")
	if err := os.MkdirAll(filepath.Dir(workerPath), 0o755); err != nil {
		t.Fatalf("mkdir worker dir: %v", err)
	}
	if err := os.WriteFile(workerPath, []byte("#!/usr/bin/env python3\n"), 0o644); err != nil {
		t.Fatalf("write worker: %v", err)
	}
	outputDir := filepath.Join(t.TempDir(), "job_123")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output dir: %v", err)
	}

	backend := NewBackend(config.Config{
		LocalMode:       "command",
		RunRootPath:     filepath.Dir(outputDir),
		LocalScriptPath: "deploy/local/broker_worker.sh",
	})
	cmd, err := backend.commandForJob(repoRoot, outputDir, types.Job{
		ID:       "job_123",
		TaskType: "inspect_repo",
	})
	if err != nil {
		t.Fatalf("command for job: %v", err)
	}
	if got := strings.Join(cmd.Args[:3], " "); got != "python3 -S "+workerPath {
		t.Fatalf("expected direct python -S worker launch, got %q", got)
	}
}

func TestSubmitRunAcceptsPrecreatedOutputDir(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	outputDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(outputDir, 0o700); err != nil {
		t.Fatalf("mkdir output dir: %v", err)
	}
	workerPath := filepath.Join(repoRoot, "workers", "rag-compression", "inspect_repo_worker.py")
	if err := os.MkdirAll(filepath.Dir(workerPath), 0o755); err != nil {
		t.Fatalf("mkdir worker dir: %v", err)
	}
	if err := os.WriteFile(workerPath, []byte("#!/usr/bin/env python3\n"), 0o644); err != nil {
		t.Fatalf("write worker: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "job_spec.json"), []byte(`{"job_id":"job_123","task_params":{"query":"x","mode":"evidence"},"constraints":{}}`), 0o600); err != nil {
		t.Fatalf("write job spec: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "input_manifest.json"), []byte(`{"input_refs":[{"type":"repo","uri":"file:///tmp"}]}`), 0o600); err != nil {
		t.Fatalf("write input manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "execution_plan.json"), []byte(`{"repo_inspection_cache_path":"/tmp/cache"}`), 0o600); err != nil {
		t.Fatalf("write execution plan: %v", err)
	}

	backend := NewBackend(config.Config{
		LocalMode:       "command",
		RunRootPath:     runRoot,
		LocalScriptPath: "deploy/local/broker_worker.sh",
	})
	_, err := backend.SubmitRun(context.Background(), types.Job{
		ID:       "job_123",
		TaskType: "inspect_repo",
		Request: types.SubmitJobRequest{
			TaskParams: map[string]any{
				"_broker_run_root":  runRoot,
				"_broker_repo_root": repoRoot,
			},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		},
	})
	if err != nil {
		t.Fatalf("submit run with precreated output dir: %v", err)
	}
}

func TestSubmitRunWarmInspectRepoDaemonEnqueuesRequest(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	outputDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(outputDir, 0o700); err != nil {
		t.Fatalf("mkdir output dir: %v", err)
	}
	workerPath := filepath.Join(repoRoot, "workers", "rag-compression", "inspect_repo_worker.py")
	if err := os.MkdirAll(filepath.Dir(workerPath), 0o755); err != nil {
		t.Fatalf("mkdir worker dir: %v", err)
	}
	script := "#!/usr/bin/env python3\nimport json, os, socket, sys, time\nspool=sys.argv[sys.argv.index('--daemon-spool-dir')+1]\nos.makedirs(spool, exist_ok=True)\nsock=socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)\nsock.bind(os.path.join(spool,'daemon.sock'))\nwith open(os.path.join(spool,'daemon-heartbeat.json'),'w',encoding='utf-8') as h: json.dump({'state':'running','pid':os.getpid()}, h)\ntime.sleep(30)\n"
	if err := os.WriteFile(workerPath, []byte(script), 0o755); err != nil {
		t.Fatalf("write worker: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "job_spec.json"), []byte(`{"job_id":"job_123","task_params":{"query":"x","mode":"evidence"},"constraints":{}}`), 0o600); err != nil {
		t.Fatalf("write job spec: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "input_manifest.json"), []byte(`{"input_refs":[{"type":"repo","uri":"file:///tmp"}]}`), 0o600); err != nil {
		t.Fatalf("write input manifest: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "execution_plan.json"), []byte(`{"repo_inspection_cache_path":"/tmp/cache"}`), 0o600); err != nil {
		t.Fatalf("write execution plan: %v", err)
	}

	backend := NewBackend(config.Config{
		LocalMode:                   "command",
		RunRootPath:                 runRoot,
		LocalScriptPath:             "deploy/local/broker_worker.sh",
		LocalInspectRepoWarmEnabled: true,
	})
	_, err := backend.SubmitRun(context.Background(), types.Job{
		ID:       "job_123",
		TaskType: "inspect_repo",
		Request: types.SubmitJobRequest{
			TaskParams: map[string]any{
				"_broker_run_root":  runRoot,
				"_broker_repo_root": repoRoot,
			},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		},
	})
	if err != nil {
		t.Fatalf("submit run with warm daemon: %v", err)
	}
	spoolDir := filepath.Join(runRoot, ".inspect-repo-warm")
	requestPath := filepath.Join(spoolDir, "requests", "job_123.json")
	if _, err := os.Stat(requestPath); err != nil {
		t.Fatalf("expected request file: %v", err)
	}
	payloadBytes, err := os.ReadFile(requestPath)
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		t.Fatalf("parse request: %v", err)
	}
	if got := payload["output_dir"]; got != outputDir {
		t.Fatalf("expected output_dir %q, got %#v", outputDir, got)
	}
	pid, err := readPID(filepath.Join(spoolDir, "daemon.pid"))
	if err != nil {
		t.Fatalf("read daemon pid: %v", err)
	}
	defer syscall.Kill(pid, syscall.SIGTERM)
	if !processAlive(pid) {
		t.Fatalf("expected warm daemon pid %d to be alive", pid)
	}
	if _, err := os.Stat(filepath.Join(outputDir, "warm-request.marker")); err != nil {
		t.Fatalf("expected warm request marker: %v", err)
	}
}

func TestSubmitWarmInspectRepoRunEnqueuesInlineBundle(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	outputDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(outputDir, 0o700); err != nil {
		t.Fatalf("mkdir output dir: %v", err)
	}
	workerPath := filepath.Join(repoRoot, "workers", "rag-compression", "inspect_repo_worker.py")
	if err := os.MkdirAll(filepath.Dir(workerPath), 0o755); err != nil {
		t.Fatalf("mkdir worker dir: %v", err)
	}
	script := "#!/usr/bin/env python3\nimport json, os, socket, sys, time\nspool=sys.argv[sys.argv.index('--daemon-spool-dir')+1]\nos.makedirs(spool, exist_ok=True)\nsock=socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)\nsock.bind(os.path.join(spool,'daemon.sock'))\nwith open(os.path.join(spool,'daemon-heartbeat.json'),'w',encoding='utf-8') as h: json.dump({'state':'running','pid':os.getpid()}, h)\ntime.sleep(30)\n"
	if err := os.WriteFile(workerPath, []byte(script), 0o755); err != nil {
		t.Fatalf("write worker: %v", err)
	}

	backend := NewBackend(config.Config{
		LocalMode:                   "command",
		RunRootPath:                 runRoot,
		LocalScriptPath:             "deploy/local/broker_worker.sh",
		LocalInspectRepoWarmEnabled: true,
	})
	resp, ok, err := backend.SubmitWarmInspectRepoRun(context.Background(), types.Job{
		ID:       "job_123",
		TaskType: "inspect_repo",
		Request: types.SubmitJobRequest{
			TaskParams: map[string]any{
				"_broker_run_root":  runRoot,
				"_broker_repo_root": repoRoot,
			},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		},
	}, backends.InlineExecutionBundle{
		JobSpec: map[string]any{
			"job_id":      "job_123",
			"task_type":   "inspect_repo",
			"task_params": map[string]any{"query": "x", "mode": "evidence"},
			"constraints": map[string]any{},
		},
		ExecutionPlan: map[string]any{"repo_inspection_cache_path": "/tmp/cache"},
		InputManifest: map[string]any{"input_refs": []map[string]any{{"type": "repo", "uri": "file:///tmp"}}},
	})
	if err != nil {
		t.Fatalf("submit warm inspect_repo run: %v", err)
	}
	if !ok {
		t.Fatal("expected warm submit path to be accepted")
	}
	if resp.BackendRunID != "job_123" || resp.InitialState != types.JobStateDispatching {
		t.Fatalf("unexpected submit response: %#v", resp)
	}
	spoolDir := filepath.Join(runRoot, ".inspect-repo-warm")
	payloadBytes, err := os.ReadFile(filepath.Join(spoolDir, "requests", "job_123.json"))
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		t.Fatalf("parse request: %v", err)
	}
	if _, exists := payload["job_spec_path"]; exists {
		t.Fatalf("expected inline request to omit job_spec_path, got %#v", payload)
	}
	if _, ok := payload["job_spec"].(map[string]any); !ok {
		t.Fatalf("expected inline job_spec payload, got %#v", payload["job_spec"])
	}
	pid, err := readPID(filepath.Join(spoolDir, "daemon.pid"))
	if err != nil {
		t.Fatalf("read daemon pid: %v", err)
	}
	defer syscall.Kill(pid, syscall.SIGTERM)
}

func TestCancelRunWarmRequestWritesCancelMarkerWithoutKillingDaemon(t *testing.T) {
	runRoot := t.TempDir()
	runDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "warm-request.marker"), []byte("job_123.json"), 0o644); err != nil {
		t.Fatalf("write warm marker: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "local.pid"), []byte(strconv.Itoa(os.Getpid())), 0o644); err != nil {
		t.Fatalf("write pid: %v", err)
	}

	backend := NewBackend(config.Config{LocalMode: "command", RunRootPath: runRoot, LocalInspectRepoWarmEnabled: true})
	if err := backend.CancelRun(context.Background(), "job_123"); err != nil {
		t.Fatalf("cancel run: %v", err)
	}
	cancelPath := filepath.Join(runDir, "cancel.request")
	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		if _, err := os.Stat(cancelPath); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("expected cancel request marker")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestStartInspectRepoWarmDaemonStartsWhenEnabled(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	workerPath := filepath.Join(repoRoot, "workers", "rag-compression", "inspect_repo_worker.py")
	if err := os.MkdirAll(filepath.Dir(workerPath), 0o755); err != nil {
		t.Fatalf("mkdir worker dir: %v", err)
	}
	script := "#!/usr/bin/env python3\nimport json, os, socket, sys, time\nspool=sys.argv[sys.argv.index('--daemon-spool-dir')+1]\nos.makedirs(spool, exist_ok=True)\nsock=socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)\nsock.bind(os.path.join(spool,'daemon.sock'))\nwith open(os.path.join(spool,'daemon-heartbeat.json'),'w',encoding='utf-8') as h: json.dump({'state':'running','pid':os.getpid()}, h)\ntime.sleep(30)\n"
	if err := os.WriteFile(workerPath, []byte(script), 0o755); err != nil {
		t.Fatalf("write worker: %v", err)
	}
	backend := NewBackend(config.Config{
		LocalMode:                   "command",
		RunRootPath:                 runRoot,
		RepoRootPath:                repoRoot,
		LocalScriptPath:             "deploy/local/broker_worker.sh",
		LocalInspectRepoWarmEnabled: true,
	})

	pid, started, err := backend.StartInspectRepoWarmDaemon()
	if err != nil {
		t.Fatalf("start warm daemon: %v", err)
	}
	if !started {
		t.Fatal("expected warm daemon startup")
	}
	defer syscall.Kill(pid, syscall.SIGTERM)
	if !processAlive(pid) {
		t.Fatalf("expected daemon pid %d to be alive", pid)
	}
	if _, err := os.Stat(filepath.Join(runRoot, ".inspect-repo-warm", "daemon.pid")); err != nil {
		t.Fatalf("expected daemon pid file: %v", err)
	}
}

func TestNotifyInspectRepoWarmDaemonSendsWakeupDatagram(t *testing.T) {
	spoolDir := t.TempDir()
	socketPath := filepath.Join(spoolDir, "daemon.sock")
	listener, err := net.ListenPacket("unixgram", socketPath)
	if err != nil {
		t.Fatalf("listen unixgram: %v", err)
	}
	defer listener.Close()
	defer os.Remove(socketPath)

	received := make(chan string, 1)
	go func() {
		buffer := make([]byte, 256)
		_ = listener.SetDeadline(time.Now().Add(2 * time.Second))
		n, _, err := listener.ReadFrom(buffer)
		if err != nil {
			received <- "ERR:" + err.Error()
			return
		}
		received <- string(buffer[:n])
	}()

	notifyInspectRepoWarmDaemon(spoolDir, "job_123.json")

	select {
	case got := <-received:
		if got != "job_123.json" {
			t.Fatalf("received wakeup = %q, want %q", got, "job_123.json")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for wakeup datagram")
	}
}

func TestNotifyInspectRepoWarmDaemonReturnsQuicklyWhenSocketMissing(t *testing.T) {
	spoolDir := t.TempDir()
	started := time.Now()
	notifyInspectRepoWarmDaemon(spoolDir, "job_123.json")
	if elapsed := time.Since(started); elapsed > 20*time.Millisecond {
		t.Fatalf("expected missing-socket notify to return quickly, took %s", elapsed)
	}
}

func TestEnqueueInspectRepoWarmRequestQueuesWhenDaemonBusy(t *testing.T) {
	runRoot := t.TempDir()
	repoRoot := t.TempDir()
	outputDir := filepath.Join(runRoot, "job_123")
	if err := os.MkdirAll(outputDir, 0o700); err != nil {
		t.Fatalf("mkdir output dir: %v", err)
	}
	workerPath := filepath.Join(repoRoot, "workers", "rag-compression", "inspect_repo_worker.py")
	if err := os.MkdirAll(filepath.Dir(workerPath), 0o755); err != nil {
		t.Fatalf("mkdir worker dir: %v", err)
	}
	script := "#!/usr/bin/env python3\nimport json, os, socket, sys, time\nspool=sys.argv[sys.argv.index('--daemon-spool-dir')+1]\nos.makedirs(spool, exist_ok=True)\nsock=socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)\nsock.bind(os.path.join(spool,'daemon.sock'))\nwith open(os.path.join(spool,'daemon-heartbeat.json'),'w',encoding='utf-8') as h: json.dump({'state':'running','pid':os.getpid()}, h)\ntime.sleep(30)\n"
	if err := os.WriteFile(workerPath, []byte(script), 0o755); err != nil {
		t.Fatalf("write worker: %v", err)
	}
	backend := NewBackend(config.Config{
		LocalMode:                   "command",
		RunRootPath:                 runRoot,
		LocalScriptPath:             "deploy/local/broker_worker.sh",
		LocalInspectRepoWarmEnabled: true,
	})
	spoolDir := filepath.Join(runRoot, ".inspect-repo-warm")
	pid, err := ensureInspectRepoWarmDaemon(repoRoot, spoolDir, workerPath)
	if err != nil {
		t.Fatalf("ensure warm daemon: %v", err)
	}
	defer syscall.Kill(pid, syscall.SIGTERM)
	if err := os.WriteFile(warmDaemonBusyMarkerPath(spoolDir), []byte("busy\n"), 0o644); err != nil {
		t.Fatalf("write busy marker: %v", err)
	}
	requestPath, daemonPID, ok, err := backend.enqueueInspectRepoWarmRequest(repoRoot, outputDir, types.Job{
		ID:       "job_123",
		TaskType: "inspect_repo",
		Request: types.SubmitJobRequest{
			TaskParams: map[string]any{
				"_broker_run_root":  runRoot,
				"_broker_repo_root": repoRoot,
			},
			OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
		},
	}, backends.InlineExecutionBundle{})
	if err != nil {
		t.Fatalf("enqueue warm request: %v", err)
	}
	if !ok {
		t.Fatalf("expected busy daemon queueing, got fallback")
	}
	if daemonPID != pid {
		t.Fatalf("daemon pid = %d, want %d", daemonPID, pid)
	}
	if filepath.Base(requestPath) != "job_123.json" {
		t.Fatalf("unexpected request path %q", requestPath)
	}
	if _, err := os.Stat(requestPath); err != nil {
		t.Fatalf("expected queued request file %q: %v", requestPath, err)
	}
}
