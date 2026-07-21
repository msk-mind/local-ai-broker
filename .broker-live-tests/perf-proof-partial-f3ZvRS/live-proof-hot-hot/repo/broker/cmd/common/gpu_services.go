package common

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/gpuservice"
)

// StartGPUServiceControlPlane starts the scheduler owner independently from
// request workers. A transient scheduler or health failure is retried while
// the broker remains available for lexical evidence fallback.
func StartGPUServiceControlPlane(
	ctx context.Context,
	cfg config.Config,
	backend backends.Backend,
	logger *log.Logger,
) (*gpuservice.Manager, error) {
	if !cfg.GPUServiceEnabled {
		return nil, nil
	}
	scheduler, ok := backend.(gpuservice.Scheduler)
	if !ok {
		return nil, fmt.Errorf("GPU service control plane requires a scheduler-backed backend")
	}
	manager, err := gpuservice.NewManagerFromConfig(cfg, scheduler)
	if err != nil {
		if errors.Is(err, gpuservice.ErrControlPlaneDisabled) {
			return nil, nil
		}
		return nil, err
	}
	if logger == nil {
		logger = log.Default()
	}
	retryInterval := time.Duration(cfg.GPUServiceHealthIntervalSeconds) * time.Second
	if retryInterval <= 0 {
		retryInterval = 15 * time.Second
	}
	go runGPUServiceControlPlane(ctx, manager, logger, retryInterval)
	return manager, nil
}

func runGPUServiceControlPlane(ctx context.Context, manager *gpuservice.Manager, logger *log.Logger, retryInterval time.Duration) {
	for ctx.Err() == nil {
		err := manager.Run(ctx)
		if ctx.Err() != nil {
			return
		}
		logger.Printf("GPU service reconciler stopped: %v; retrying in %s", err, retryInterval)
		timer := time.NewTimer(retryInterval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}
