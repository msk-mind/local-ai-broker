package slurm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/jobenv"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

var slurmPassthroughEnvVars = []string{
	"PATH",
	"LD_LIBRARY_PATH",
	"LIBRARY_PATH",
	"CUDA_HOME",
	"CUDA_PATH",
	"HOME",
	"USER",
	"LOGNAME",
	"LANG",
	"LC_ALL",
	"LC_CTYPE",
	"TMPDIR",
	"TMP",
	"TEMP",
	"PYTHONPATH",
	"VIRTUAL_ENV",
	"CONDA_PREFIX",
	"CONDA_DEFAULT_ENV",
}

type Backend struct {
	counter atomic.Uint64
	mode    string
	runner  commandRunner
	cfg     config.Config
}

type commandRunner interface {
	Run(context.Context, string, ...string) ([]byte, error)
}

type execRunner struct{}

func (execRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	return cmd.CombinedOutput()
}

func NewBackend(cfg config.Config) *Backend {
	return &Backend{
		mode:   cfg.SlurmMode,
		runner: execRunner{},
		cfg:    cfg,
	}
}

func NewBackendWithRunner(cfg config.Config, runner commandRunner) *Backend {
	return &Backend{
		mode:   cfg.SlurmMode,
		runner: runner,
		cfg:    cfg,
	}
}

func (b *Backend) Name() string {
	return "slurm"
}

func (b *Backend) commandMode() bool {
	return b.mode == "command"
}

func (b *Backend) nextStubRunID() string {
	return fmt.Sprintf("slurm-%06d", b.counter.Add(1))
}

func (b *Backend) ResolveExecutionProfile(ctx context.Context, req types.SubmitJobRequest) (types.ExecutionProfile, error) {
	profile := req.ExecutionProfile
	if !b.commandMode() || !b.cfg.SlurmEnableDynamicPlacement {
		return profile, nil
	}
	if !allowsDynamicTierSelection(profile) {
		return profile, nil
	}

	availability, err := b.queryAvailability(ctx)
	if err != nil {
		return profile, nil
	}
	queue, _ := b.queryQueuePressure(ctx)

	bestProfile := profile
	bestScore := -1 << 30
	for _, candidateTier := range candidateTiers(profile.Tier) {
		candidate := profile
		candidate.Tier = candidateTier
		candidate.Accelerator = ""
		candidate.NodeList = ""
		candidate.Constraint = ""
		score, ok := tierAvailabilityScore(candidate, availability, queue, b.cfg)
		if !ok {
			continue
		}
		if score > bestScore {
			bestScore = score
			bestProfile = candidate
		}
	}
	return bestProfile, nil
}

func (b *Backend) SubmitRun(ctx context.Context, job types.Job) (backends.SubmitResponse, error) {
	if !b.commandMode() {
		return backends.StubSubmitResponse(b.Name(), b.nextStubRunID()), nil
	}

	args := b.baseSubmitArgs(job.Request.ExecutionProfile, "broker-"+job.TaskType)
	if dependencyArg := buildDependencyArg(job); dependencyArg != "" {
		args = append(args, "--dependency", dependencyArg)
	}
	args = append(args,
		"--export", buildExport(job),
		b.cfg.SlurmScriptPath,
	)
	output, err := b.runner.Run(ctx, b.cfg.SlurmSubmitCmd, args...)
	if err != nil {
		return backends.SubmitResponse{}, fmt.Errorf("submit slurm job: %w: %s", err, strings.TrimSpace(string(output)))
	}
	jobID := strings.TrimSpace(string(output))
	if jobID == "" {
		return backends.SubmitResponse{}, errors.New("empty slurm job id from submit command")
	}

	return backends.SubmitResponse{
		BackendKind:  b.Name(),
		BackendRunID: jobID,
		InitialState: types.JobStateQueued,
	}, nil
}

func (b *Backend) SubmitRunBatch(ctx context.Context, jobs []types.Job) ([]backends.SubmitResponse, error) {
	if len(jobs) == 0 {
		return nil, nil
	}
	if !b.commandMode() {
		return b.stubBatchResponses(len(jobs)), nil
	}

	manifestPath, err := writeArrayManifest(jobs)
	if err != nil {
		return nil, fmt.Errorf("write array manifest: %w", err)
	}

	args := b.baseSubmitArgs(jobs[0].Request.ExecutionProfile, "broker-"+jobs[0].TaskType+"-batch")
	args = append(args, "--array", fmt.Sprintf("0-%d", len(jobs)-1))
	args = append(args,
		"--export", buildBatchExport(jobs[0], manifestPath),
		b.cfg.SlurmScriptPath,
	)
	output, err := b.runner.Run(ctx, b.cfg.SlurmSubmitCmd, args...)
	if err != nil {
		return nil, fmt.Errorf("submit slurm job array: %w: %s", err, strings.TrimSpace(string(output)))
	}
	arrayJobID := strings.TrimSpace(string(output))
	if arrayJobID == "" {
		return nil, errors.New("empty slurm array job id from submit command")
	}

	responses := make([]backends.SubmitResponse, 0, len(jobs))
	for i := range jobs {
		responses = append(responses, backends.SubmitResponse{
			BackendKind:  b.Name(),
			BackendRunID: fmt.Sprintf("%s_%d", arrayJobID, i),
			InitialState: types.JobStateQueued,
		})
	}
	return responses, nil
}

func (b *Backend) stubBatchResponses(jobCount int) []backends.SubmitResponse {
	return backends.IndexedStubResponses(b.Name(), "slurm", jobCount, func() uint64 {
		return b.counter.Add(1)
	})
}

func (b *Backend) baseSubmitArgs(profile types.ExecutionProfile, jobName string) []string {
	args := []string{
		"--parsable",
		"--job-name", jobName,
	}
	if partition := strings.TrimSpace(selectPartition(profile.Tier, b.cfg)); partition != "" {
		args = append(args, "--partition", partition)
	}
	if gpuFlag, gpuValue := selectGPURequest(profile, b.cfg); gpuFlag != "" && gpuValue != "" {
		args = append(args, gpuFlag, gpuValue)
	}
	if qos := strings.TrimSpace(profile.QOS); qos != "" {
		args = append(args, "--qos", qos)
	}
	if nodelist := strings.TrimSpace(selectNodeList(profile, b.cfg)); nodelist != "" {
		args = append(args, "--nodelist", nodelist)
	}
	args = append(args, singleWorkerSchedulingArgs()...)
	if constraint := strings.TrimSpace(selectConstraint(profile, b.cfg)); constraint != "" {
		args = append(args, "--constraint", constraint)
	}
	return args
}

func (b *Backend) GetRun(ctx context.Context, backendRunID string) (backends.RunStatus, error) {
	if !b.commandMode() {
		return backends.StubRunStatus(backendRunID), nil
	}

	runRef := parseRunRef(backendRunID)

	output, err := b.runner.Run(
		ctx,
		b.cfg.SlurmStatusCmd,
		"--jobs", runRef.queryID,
		"--noheader",
		"--parsable2",
		"--format", "JobIDRaw,State,ExitCode",
	)
	if err != nil {
		return b.getRunFromSqueue(ctx, runRef, err, output)
	}

	state, rawState, exitCode := parseSlurmStatus(output, runRef)
	return backends.RunStatus{
		BackendRunID: backendRunID,
		State:        state,
		RawState:     rawState,
		ExitCode:     exitCode,
	}, nil
}

func (b *Backend) CancelRun(ctx context.Context, backendRunID string) error {
	if !b.commandMode() {
		return nil
	}
	runRef := parseRunRef(backendRunID)
	output, err := b.runner.Run(ctx, b.cfg.SlurmCancelCmd, runRef.queryID)
	if err != nil {
		return fmt.Errorf("cancel slurm job: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

func parseSlurmState(output []byte) types.JobState {
	state := strings.ToUpper(strings.TrimSpace(string(output)))
	switch {
	case strings.Contains(state, "PENDING"):
		return types.JobStateQueued
	case strings.Contains(state, "RUNNING"):
		return types.JobStateRunning
	case strings.Contains(state, "COMPLETED"):
		return types.JobStateSucceeded
	case strings.Contains(state, "CANCELLED"):
		return types.JobStateCancelled
	case strings.Contains(state, "TIMEOUT"):
		return types.JobStateTimedOut
	case strings.Contains(state, "PREEMPTED"):
		return types.JobStatePreempted
	case strings.Contains(state, "FAILED"), strings.Contains(state, "OUT_OF_MEMORY"):
		return types.JobStateFailed
	default:
		return types.JobStateQueued
	}
}

func parseSlurmStatus(output []byte, runRef slurmRunRef) (types.JobState, string, string) {
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, "|")
		if len(fields) == 2 {
			rawState := strings.TrimSpace(fields[0])
			exitCode := strings.TrimSpace(fields[1])
			return parseSlurmState([]byte(rawState)), rawState, exitCode
		}
		if len(fields) < 3 {
			continue
		}
		jobIDRaw := strings.TrimSpace(fields[0])
		rawState := strings.TrimSpace(fields[1])
		exitCode := strings.TrimSpace(fields[2])
		if runRef.matches(jobIDRaw) {
			return parseSlurmState([]byte(rawState)), rawState, exitCode
		}
	}
	return types.JobStateQueued, "", ""
}

type nodeAvailability struct {
	Partition string
	NodeName  string
	Features  string
	GRES      string
	State     string
}

type queueJob struct {
	Partition string
	State     string
	Nodes     string
}

func (b *Backend) queryAvailability(ctx context.Context) ([]nodeAvailability, error) {
	output, err := b.runner.Run(
		ctx,
		b.cfg.SlurmInfoCmd,
		"--noheader",
		"--Node",
		"--format",
		"%P|%N|%f|%G|%t",
	)
	if err != nil {
		return nil, err
	}
	return parseNodeAvailability(output), nil
}

func parseNodeAvailability(output []byte) []nodeAvailability {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	nodes := make([]nodeAvailability, 0, len(lines))
	for _, line := range lines {
		fields := strings.Split(strings.TrimSpace(line), "|")
		if len(fields) != 5 {
			continue
		}
		nodes = append(nodes, nodeAvailability{
			Partition: strings.TrimSuffix(strings.TrimSpace(fields[0]), "*"),
			NodeName:  strings.TrimSpace(fields[1]),
			Features:  strings.TrimSpace(fields[2]),
			GRES:      strings.TrimSpace(fields[3]),
			State:     strings.ToLower(strings.TrimSpace(fields[4])),
		})
	}
	return nodes
}

func (b *Backend) queryQueuePressure(ctx context.Context) ([]queueJob, error) {
	output, err := b.runner.Run(
		ctx,
		"squeue",
		"--noheader",
		"--format",
		"%P|%T|%N",
	)
	if err != nil {
		return nil, err
	}
	return parseQueueJobs(output), nil
}

func parseQueueJobs(output []byte) []queueJob {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	jobs := make([]queueJob, 0, len(lines))
	for _, line := range lines {
		fields := strings.Split(strings.TrimSpace(line), "|")
		if len(fields) != 3 {
			continue
		}
		jobs = append(jobs, queueJob{
			Partition: strings.TrimSuffix(strings.TrimSpace(fields[0]), "*"),
			State:     strings.ToLower(strings.TrimSpace(fields[1])),
			Nodes:     strings.TrimSpace(fields[2]),
		})
	}
	return jobs
}

func allowsDynamicTierSelection(profile types.ExecutionProfile) bool {
	if strings.TrimSpace(profile.Tier) != "p40-rag-compression" {
		return false
	}
	if strings.TrimSpace(profile.NodeList) != "" || strings.TrimSpace(profile.Constraint) != "" || strings.TrimSpace(profile.Accelerator) != "" {
		return false
	}
	return true
}

func candidateTiers(current string) []string {
	switch strings.TrimSpace(current) {
	case "p40-rag-compression":
		return []string{"p40-rag-compression", "a100-reasoning"}
	default:
		return []string{strings.TrimSpace(current)}
	}
}

func tierAvailabilityScore(profile types.ExecutionProfile, nodes []nodeAvailability, queue []queueJob, cfg config.Config) (int, bool) {
	partition := strings.TrimSpace(selectPartition(profile.Tier, cfg))
	constraint := strings.TrimSpace(selectConstraint(profile, cfg))
	nodelist := strings.TrimSpace(selectNodeList(profile, cfg))
	accelerator := effectiveAccelerator(profile, cfg)
	matchingNodes := make([]string, 0, len(nodes))
	idleCount := 0
	mixCount := 0

	for _, node := range nodes {
		if partition != "" && node.Partition != partition {
			continue
		}
		if !isAvailableState(node.State) {
			continue
		}
		if nodelist != "" && !hostListMatches(nodelist, node.NodeName) {
			continue
		}
		if constraint != "" && !constraintMatches(constraint, node.Features) {
			continue
		}
		if accelerator != "" && !gresMatchesAccelerator(node.GRES, accelerator) {
			continue
		}
		if accelerator == "" && requiresGPU(profile.Tier) && !strings.Contains(strings.ToLower(node.GRES), "gpu") {
			continue
		}
		matchingNodes = append(matchingNodes, node.NodeName)
		if strings.Contains(node.State, "idle") {
			idleCount++
		} else if strings.Contains(node.State, "mix") {
			mixCount++
		}
	}
	if len(matchingNodes) == 0 {
		return 0, false
	}
	nodeSet := make(map[string]struct{}, len(matchingNodes))
	for _, node := range matchingNodes {
		nodeSet[node] = struct{}{}
	}
	pendingCount := 0
	runningCount := 0
	for _, job := range queue {
		if partition != "" && job.Partition != partition {
			continue
		}
		switch {
		case strings.Contains(job.State, "pending"):
			pendingCount++
		case strings.Contains(job.State, "running"):
			if jobTouchesNodeSet(job.Nodes, nodeSet) {
				runningCount++
			}
		}
	}
	score := idleCount*100 + mixCount*25 - pendingCount*10 - runningCount*5
	return score, true
}

func effectiveAccelerator(profile types.ExecutionProfile, cfg config.Config) string {
	if value := strings.TrimSpace(profile.Accelerator); value != "" {
		return value
	}
	switch strings.TrimSpace(profile.Tier) {
	case "p40-rag-compression", "p40-retrieval", "p40-synthesis":
		return strings.TrimSpace(cfg.SlurmGPUTypeP40)
	case "v100-reasoning":
		return strings.TrimSpace(cfg.SlurmGPUTypeV100)
	case "a100-reasoning", "a100-single", "a100-multigpu":
		return strings.TrimSpace(cfg.SlurmGPUTypeA100)
	default:
		return ""
	}
}

func isAvailableState(state string) bool {
	state = strings.ToLower(strings.TrimSpace(state))
	return strings.Contains(state, "idle") || strings.Contains(state, "mix")
}

func requiresGPU(tier string) bool {
	switch strings.TrimSpace(tier) {
	case "p40-rag-compression", "p40-retrieval", "p40-synthesis", "v100-reasoning", "a100-reasoning", "a100-single", "a100-multigpu":
		return true
	default:
		return false
	}
}

func gresMatchesAccelerator(gres, accelerator string) bool {
	gres = strings.ToLower(strings.TrimSpace(gres))
	accelerator = strings.ToLower(strings.TrimSpace(accelerator))
	if gres == "" || accelerator == "" {
		return false
	}
	return strings.Contains(gres, "gpu:"+accelerator+":") || strings.Contains(gres, "gpu:"+accelerator+"(") || strings.Contains(gres, "gpu:"+accelerator+",") || strings.HasSuffix(gres, "gpu:"+accelerator)
}

func constraintMatches(constraint, features string) bool {
	for _, token := range strings.FieldsFunc(strings.ToLower(constraint), func(r rune) bool {
		switch r {
		case '&', '|', ',', '(', ')':
			return true
		default:
			return false
		}
	}) {
		token = strings.TrimSpace(token)
		if token != "" && strings.Contains(strings.ToLower(features), token) {
			return true
		}
	}
	return strings.TrimSpace(constraint) == ""
}

func hostListMatches(pattern, node string) bool {
	pattern = strings.TrimSpace(pattern)
	node = strings.TrimSpace(node)
	if pattern == "" || node == "" {
		return false
	}
	for _, part := range splitHostList(pattern) {
		if singleHostPatternMatches(part, node) {
			return true
		}
	}
	return false
}

func jobTouchesNodeSet(nodes string, nodeSet map[string]struct{}) bool {
	nodes = strings.TrimSpace(nodes)
	if nodes == "" || nodes == "n/a" || nodes == "(null)" {
		return false
	}
	for node := range nodeSet {
		if hostListMatches(nodes, node) {
			return true
		}
	}
	return false
}

func splitHostList(pattern string) []string {
	parts := []string{}
	start := 0
	depth := 0
	for i, r := range pattern {
		switch r {
		case '[':
			depth++
		case ']':
			if depth > 0 {
				depth--
			}
		case ',':
			if depth == 0 {
				parts = append(parts, strings.TrimSpace(pattern[start:i]))
				start = i + 1
			}
		}
	}
	parts = append(parts, strings.TrimSpace(pattern[start:]))
	return filterNonEmpty(parts)
}

func singleHostPatternMatches(pattern, node string) bool {
	if !strings.Contains(pattern, "[") || !strings.Contains(pattern, "]") {
		return pattern == node
	}
	open := strings.Index(pattern, "[")
	close := strings.Index(pattern, "]")
	if open < 0 || close <= open {
		return pattern == node
	}
	prefix := pattern[:open]
	suffix := pattern[close+1:]
	if !strings.HasPrefix(node, prefix) || !strings.HasSuffix(node, suffix) {
		return false
	}
	middle := strings.TrimSuffix(strings.TrimPrefix(node, prefix), suffix)
	value, err := strconv.Atoi(middle)
	if err != nil {
		return false
	}
	width := len(middle)
	for _, item := range strings.Split(pattern[open+1:close], ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if strings.Contains(item, "-") {
			rangeParts := strings.SplitN(item, "-", 2)
			start, errStart := strconv.Atoi(rangeParts[0])
			end, errEnd := strconv.Atoi(rangeParts[1])
			if errStart == nil && errEnd == nil && value >= start && value <= end {
				if len(rangeParts[0]) == width || len(rangeParts[1]) == width || width == len(middle) {
					return true
				}
			}
			continue
		}
		single, err := strconv.Atoi(item)
		if err == nil && single == value {
			return true
		}
	}
	return false
}

func (b *Backend) getRunFromSqueue(ctx context.Context, runRef slurmRunRef, originalErr error, originalOutput []byte) (backends.RunStatus, error) {
	output, err := b.runner.Run(
		ctx,
		"squeue",
		"--jobs", runRef.queryID,
		"--noheader",
		"--format", "%i|%T",
	)
	if err != nil {
		return backends.RunStatus{}, fmt.Errorf(
			"get slurm status: %w: %s; fallback squeue failed: %v: %s",
			originalErr,
			strings.TrimSpace(string(originalOutput)),
			err,
			strings.TrimSpace(string(output)),
		)
	}

	rawState := parseSqueueState(output, runRef)
	return backends.RunStatus{
		BackendRunID: runRef.originalID,
		State:        parseSlurmState([]byte(rawState)),
		RawState:     rawState,
	}, nil
}

func parseSqueueState(output []byte, runRef slurmRunRef) string {
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.SplitN(line, "|", 2)
		if len(fields) != 2 {
			continue
		}
		jobIDRaw := strings.TrimSpace(fields[0])
		rawState := strings.TrimSpace(fields[1])
		if runRef.matches(jobIDRaw) {
			return rawState
		}
	}
	return strings.TrimSpace(string(output))
}

func buildExport(job types.Job) string {
	return strings.Join(buildExportParts(job), ",")
}

func buildBatchExport(job types.Job, manifestPath string) string {
	parts := append(exportPassthroughEnvParts(),
		"BROKER_ARRAY_MANIFEST="+manifestPath,
		"BROKER_REPO_ROOT="+jobenv.RepoRoot(job),
	)
	return strings.Join(parts, ",")
}

func buildExportParts(job types.Job) []string {
	runRoot := strings.TrimRight(jobenv.RunRoot(job), "/")
	repoRoot := jobenv.RepoRoot(job)
	outputDir := fmt.Sprintf("%s/%s", runRoot, job.ID)
	parts := append(exportPassthroughEnvParts(),
		"BROKER_JOB_ID="+job.ID,
		"BROKER_TASK_TYPE="+job.TaskType,
		"BROKER_REPO_ROOT="+repoRoot,
		"BROKER_OUTPUT_DIR="+outputDir,
	)
	if job.Request.OutputSchema.Name != "" {
		parts = append(parts, "BROKER_OUTPUT_SCHEMA="+job.Request.OutputSchema.Name)
	}
	return parts
}

func exportPassthroughEnvParts() []string {
	parts := make([]string, 0, len(slurmPassthroughEnvVars))
	for _, key := range slurmPassthroughEnvVars {
		if value, ok := os.LookupEnv(key); ok && strings.TrimSpace(value) != "" {
			parts = append(parts, key+"="+value)
		}
	}
	return parts
}

func writeArrayManifest(jobs []types.Job) (string, error) {
	runRoot := jobenv.RunRoot(jobs[0])
	manifestDir := filepath.Join(runRoot, "_slurm_arrays")
	if err := os.MkdirAll(manifestDir, 0o755); err != nil {
		return "", err
	}

	manifestEntries := make([]map[string]string, 0, len(jobs))
	for _, job := range jobs {
		entry := map[string]string{
			"broker_job_id":    job.ID,
			"broker_task_type": job.TaskType,
			"broker_output_dir": filepath.Join(
				strings.TrimRight(jobenv.RunRoot(job), "/"),
				job.ID,
			),
		}
		if schema := strings.TrimSpace(job.Request.OutputSchema.Name); schema != "" {
			entry["broker_output_schema"] = schema
		}
		manifestEntries = append(manifestEntries, entry)
	}

	manifestBytes, err := json.MarshalIndent(manifestEntries, "", "  ")
	if err != nil {
		return "", err
	}
	manifestPath := filepath.Join(manifestDir, fmt.Sprintf("%s.json", jobs[0].RootJobID))
	if jobs[0].RootJobID == "" {
		manifestPath = filepath.Join(manifestDir, fmt.Sprintf("%s.json", jobs[0].ID))
	}
	if err := os.WriteFile(manifestPath, manifestBytes, 0o644); err != nil {
		return "", err
	}
	return manifestPath, nil
}

func buildDependencyArg(job types.Job) string {
	ids := dependencyBackendRunIDs(job)
	if len(ids) == 0 {
		return ""
	}
	return "afterany:" + strings.Join(ids, ":")
}

func singleWorkerSchedulingArgs() []string {
	return []string{
		"--nodes", "1",
		"--ntasks", "1",
	}
}

func dependencyBackendRunIDs(job types.Job) []string {
	if job.Request.TaskParams == nil {
		return nil
	}
	raw, ok := job.Request.TaskParams["_dependency_backend_run_ids"]
	if !ok {
		return nil
	}
	items, ok := raw.([]string)
	if ok {
		return filterNonEmpty(items)
	}
	generic, ok := raw.([]any)
	if !ok {
		return nil
	}
	ids := make([]string, 0, len(generic))
	for _, item := range generic {
		if text, ok := item.(string); ok && strings.TrimSpace(text) != "" {
			ids = append(ids, strings.TrimSpace(text))
		}
	}
	return ids
}

func filterNonEmpty(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			out = append(out, strings.TrimSpace(value))
		}
	}
	return out
}

func selectPartition(tier string, cfg config.Config) string {
	switch strings.TrimSpace(tier) {
	case "cpu-rag-indexing":
		return cfg.SlurmPartitionCPU
	case "p40-rag-compression", "p40-retrieval", "p40-synthesis":
		if value := strings.TrimSpace(cfg.SlurmPartitionP40); value != "" {
			return value
		}
		return cfg.SlurmPartitionGPU
	case "v100-reasoning":
		if value := strings.TrimSpace(cfg.SlurmPartitionV100); value != "" {
			return value
		}
		return cfg.SlurmPartitionGPU
	case "a100-reasoning", "a100-single", "a100-multigpu":
		if value := strings.TrimSpace(cfg.SlurmPartitionA100); value != "" {
			return value
		}
		return cfg.SlurmPartitionGPU
	default:
		return ""
	}
}

func selectGPURequest(profile types.ExecutionProfile, cfg config.Config) (string, string) {
	tier := strings.TrimSpace(profile.Tier)
	count := gpuCountForTier(tier)
	if count == 0 {
		return "", ""
	}
	accelerator := effectiveAccelerator(profile, cfg)
	countValue := strconv.Itoa(count)
	switch strings.ToLower(strings.TrimSpace(cfg.SlurmGPURequestMode)) {
	case "gpus":
		if accelerator != "" {
			return "--gpus", accelerator + ":" + countValue
		}
		return "--gpus", countValue
	default:
		if accelerator != "" {
			return "--gres", "gpu:" + accelerator + ":" + countValue
		}
		return "--gres", "gpu:" + countValue
	}
}

func gpuCountForTier(tier string) int {
	switch strings.TrimSpace(tier) {
	case "v100-reasoning", "a100-multigpu":
		return 4
	case "p40-rag-compression", "p40-retrieval", "p40-synthesis", "a100-reasoning", "a100-single":
		return 1
	default:
		return 0
	}
}

func selectNodeList(profile types.ExecutionProfile, cfg config.Config) string {
	if value := strings.TrimSpace(profile.NodeList); value != "" {
		return value
	}
	switch strings.TrimSpace(profile.Tier) {
	case "cpu-rag-indexing":
		return cfg.SlurmNodeListCPU
	case "p40-rag-compression", "p40-retrieval", "p40-synthesis":
		return cfg.SlurmNodeListP40
	case "v100-reasoning":
		return cfg.SlurmNodeListV100
	case "a100-reasoning", "a100-single", "a100-multigpu":
		return cfg.SlurmNodeListA100
	default:
		return ""
	}
}

func selectConstraint(profile types.ExecutionProfile, cfg config.Config) string {
	if value := strings.TrimSpace(profile.Constraint); value != "" {
		return value
	}
	switch strings.TrimSpace(profile.Tier) {
	case "cpu-rag-indexing":
		return cfg.SlurmConstraintCPU
	case "p40-rag-compression", "p40-retrieval", "p40-synthesis":
		return cfg.SlurmConstraintP40
	case "v100-reasoning":
		return cfg.SlurmConstraintV100
	case "a100-reasoning", "a100-single", "a100-multigpu":
		return cfg.SlurmConstraintA100
	default:
		return ""
	}
}

type slurmRunRef struct {
	originalID string
	queryID    string
	arrayJobID string
	taskID     string
}

func parseRunRef(backendRunID string) slurmRunRef {
	trimmed := strings.TrimSpace(backendRunID)
	parts := strings.SplitN(trimmed, "_", 2)
	ref := slurmRunRef{
		originalID: trimmed,
		queryID:    trimmed,
		arrayJobID: trimmed,
	}
	if len(parts) == 2 && parts[0] != "" && parts[1] != "" {
		ref.arrayJobID = parts[0]
		ref.taskID = parts[1]
		ref.queryID = trimmed
	}
	return ref
}

func (r slurmRunRef) matches(jobIDRaw string) bool {
	candidate := strings.TrimSpace(jobIDRaw)
	if candidate == "" {
		return false
	}
	if candidate == r.queryID || candidate == r.originalID {
		return true
	}
	if r.taskID == "" {
		return candidate == r.arrayJobID
	}
	if candidate == r.arrayJobID+"_"+r.taskID || candidate == r.arrayJobID+"."+r.taskID {
		return true
	}
	return false
}
