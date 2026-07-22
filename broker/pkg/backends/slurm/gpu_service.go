package slurm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/gpuservice"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

var _ gpuservice.Scheduler = (*Backend)(nil)

// SubmitGPUService starts a dedicated long-lived service job. The private
// launch spec is intentionally separate from broker_worker.slurm so inspection
// request workers never initialize a model.
func (b *Backend) SubmitGPUService(ctx context.Context, request gpuservice.LaunchRequest) (string, error) {
	if !b.commandMode() {
		return b.nextStubRunID(), nil
	}
	if err := b.validateGPUServiceLaunch(request); err != nil {
		return "", err
	}
	specPath, err := b.writeGPUServiceLaunchSpec(request)
	if err != nil {
		return "", err
	}

	profile := types.ExecutionProfile{
		Tier:        string(request.Tier),
		Accelerator: request.Placement.GPU.Type,
		QOS:         request.Placement.QOS,
		NodeList:    request.Placement.NodeList,
		Constraint:  request.Placement.Constraint,
	}
	args := []string{
		"--parsable",
		"--job-name", "broker-gpu-" + string(request.Tier),
	}
	partition := strings.TrimSpace(request.Placement.Partition)
	if partition == "" {
		partition = strings.TrimSpace(selectPartition(profile.Tier, b.cfg))
	}
	if partition != "" {
		args = append(args, "--partition", partition)
	}
	if gpuFlag, gpuValue := selectGPURequest(profile, b.cfg); gpuFlag != "" {
		args = append(args, gpuFlag, gpuValue)
	}
	if profile.QOS != "" {
		args = append(args, "--qos", profile.QOS)
	}
	if profile.NodeList != "" {
		args = append(args, "--nodelist", profile.NodeList)
	}
	args = append(args, singleWorkerSchedulingArgs()...)
	if profile.Constraint != "" {
		args = append(args, "--constraint", profile.Constraint)
	}
	export := append(exportPassthroughEnvParts(), "BROKER_GPU_SERVICE_SPEC_PATH="+specPath)
	args = append(args,
		"--export", strings.Join(export, ","),
		b.cfg.GPUServiceScriptPath,
	)
	output, err := b.runner.Run(ctx, b.cfg.SlurmSubmitCmd, args...)
	if err != nil {
		_ = os.Remove(specPath)
		return "", fmt.Errorf("submit GPU service job: %w: %s", err, strings.TrimSpace(string(output)))
	}
	jobID := strings.TrimSpace(string(output))
	if jobID == "" {
		_ = os.Remove(specPath)
		return "", errors.New("empty Slurm job id from GPU service submit command")
	}
	return jobID, nil
}

func (b *Backend) GPUServiceStatus(ctx context.Context, jobID string) (gpuservice.ServiceJobStatus, error) {
	status, err := b.GetRun(ctx, jobID)
	if err != nil {
		return gpuservice.ServiceJobStatus{State: gpuservice.JobStateUnknown}, err
	}
	result := gpuservice.ServiceJobStatus{RawState: strings.TrimSpace(status.RawState)}
	switch status.State {
	case types.JobStateQueued:
		result.State = gpuservice.JobStateQueued
	case types.JobStateRunning:
		result.State = gpuservice.JobStateRunning
	case types.JobStateSucceeded, types.JobStateCancelled:
		result.State = gpuservice.JobStateStopped
		result.FailureCategory = gpuservice.FailureService
	case types.JobStateTimedOut:
		result.State = gpuservice.JobStateFailed
		result.FailureCategory = gpuservice.FailureTimeout
	case types.JobStatePreempted:
		result.State = gpuservice.JobStateFailed
		result.FailureCategory = gpuservice.FailureUnavailable
	case types.JobStateFailed:
		result.State = gpuservice.JobStateFailed
		if strings.Contains(strings.ToUpper(result.RawState), "OUT_OF_MEMORY") {
			result.FailureCategory = gpuservice.FailureOOM
		} else {
			result.FailureCategory = gpuservice.FailureService
		}
	default:
		result.State = gpuservice.JobStateUnknown
	}
	if result.RawState == "" {
		result.RawState = string(status.State)
	}
	return result, nil
}

func (b *Backend) CancelGPUService(ctx context.Context, jobID string) error {
	return b.CancelRun(ctx, jobID)
}

func (b *Backend) validateGPUServiceLaunch(request gpuservice.LaunchRequest) error {
	if strings.TrimSpace(b.cfg.GPUServiceScriptPath) == "" {
		return errors.New("BROKER_GPU_SERVICE_SCRIPT_PATH is required")
	}
	if !request.Tier.Valid() || request.ServiceID == "" || request.RegistryPath == "" || request.RegistrationToken == "" {
		return errors.New("GPU service launch requires tier, service id, registry, and registration token")
	}
	if request.HeartbeatIntervalSeconds <= 0 || request.LeaseDurationSeconds <= 0 || len(request.Capabilities) == 0 {
		return errors.New("GPU service launch requires lease timing and capabilities")
	}
	wantCount := gpuCountForTier(string(request.Tier))
	if wantCount == 0 || request.Placement.GPU.Count != wantCount || strings.TrimSpace(request.Placement.GPU.Type) == "" {
		return fmt.Errorf("tier %s requires a typed %d-GPU placement", request.Tier, wantCount)
	}
	if strings.TrimSpace(request.Deployment.Name) == "" || strings.TrimSpace(request.Deployment.Model) == "" ||
		strings.TrimSpace(request.Deployment.Runtime) == "" || len(request.Deployment.RuntimeArgs) == 0 {
		return errors.New("GPU service launch requires an exact deployment profile")
	}
	if !safeServiceID(request.ServiceID) {
		return errors.New("GPU service id contains unsafe characters")
	}
	return nil
}

func (b *Backend) writeGPUServiceLaunchSpec(request gpuservice.LaunchRequest) (string, error) {
	specDir := filepath.Join(filepath.Dir(request.RegistryPath), "gpu-service-launches")
	if err := os.MkdirAll(specDir, 0o700); err != nil {
		return "", fmt.Errorf("create GPU service launch directory: %w", err)
	}
	content, err := json.MarshalIndent(request, "", "  ")
	if err != nil {
		return "", fmt.Errorf("encode GPU service launch spec: %w", err)
	}
	specPath := filepath.Join(specDir, request.ServiceID+".json")
	file, err := os.OpenFile(specPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return "", fmt.Errorf("create GPU service launch spec: %w", err)
	}
	if _, err := file.Write(content); err != nil {
		file.Close()
		_ = os.Remove(specPath)
		return "", err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		_ = os.Remove(specPath)
		return "", err
	}
	if err := file.Close(); err != nil {
		_ = os.Remove(specPath)
		return "", err
	}
	return specPath, nil
}

func safeServiceID(value string) bool {
	if value == "" {
		return false
	}
	for _, char := range value {
		if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') ||
			(char >= '0' && char <= '9') || char == '-' || char == '_' {
			continue
		}
		return false
	}
	return true
}

// Keep the backend interface assertion nearby: GPU service methods are an
// additive scheduler control surface, not a replacement for request jobs.
var _ backends.Backend = (*Backend)(nil)
