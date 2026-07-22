package main

import (
	"context"
	"log"
	"os"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/cmd/common"
	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends/local"
	"github.com/msk-mind/local-ai-broker/broker/pkg/backends/slurm"
	"github.com/msk-mind/local-ai-broker/broker/pkg/config"
	"github.com/msk-mind/local-ai-broker/broker/pkg/mcp"
	"github.com/msk-mind/local-ai-broker/broker/pkg/service"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
)

func main() {
	cfg := config.Load()
	logger := log.New(os.Stderr, "broker-mcp ", log.LstdFlags|log.LUTC)
	if err := common.VerifyAuditStartup(logger, cfg.AuditLogPath, cfg.AuditVerifyMode); err != nil {
		logger.Fatalf("audit startup verification failed: %v", err)
	}

	jobStore, err := store.NewFileJobStore(cfg.JobStorePath)
	if err != nil {
		logger.Fatalf("initialize job store: %v", err)
	}

	backend, err := buildBackend(cfg)
	if err != nil {
		logger.Fatalf("initialize backend: %v", err)
	}
	if localBackend, ok := backend.(*local.Backend); ok {
		if pid, started, err := localBackend.StartInspectRepoWarmDaemon(); err != nil {
			logger.Fatalf("initialize local inspect_repo warm daemon: %v", err)
		} else if started {
			logger.Printf("local inspect_repo warm daemon ready pid=%d", pid)
		}
	}

	svc := service.NewWithAuditAndOptionsAndConfig(
		jobStore,
		backend,
		logger,
		audit.NewFileLogger(cfg.AuditLogPath),
		cfg.RunRootPath,
		cfg.RepoRootPath,
		service.Options{
			ParallelMaxBatchSize:           cfg.ParallelMaxBatchSize,
			ParallelMaxActiveBatches:       cfg.ParallelMaxActiveBatches,
			RootActionMaxAdditionalBatches: cfg.RootActionMaxAdditionalBatches,
			RootActionMaxRetriedShards:     cfg.RootActionMaxRetriedShards,
		},
		&cfg,
	)
	gpuContext, stopGPUControlPlane := context.WithCancel(context.Background())
	defer stopGPUControlPlane()
	gpuManager, err := common.StartGPUServiceControlPlane(gpuContext, cfg, backend, logger)
	if err != nil {
		logger.Fatalf("initialize GPU service control plane: %v", err)
	}
	if cfg.InspectRepoPrewarmEnabled {
		svc.StartInspectRepoPrewarm(context.Background(), logger, cfg.InspectRepoPrewarmURI, cfg.InspectRepoPrewarmQuery)
	}
	defer func() {
		stopGPUControlPlane()
		if gpuManager == nil {
			return
		}
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cleanupCancel()
		if err := gpuManager.Shutdown(cleanupCtx); err != nil {
			logger.Printf("GPU service shutdown cleanup failed: %v", err)
		}
	}()

	var gpuCapabilities func(context.Context) (any, error)
	if gpuManager != nil {
		gpuCapabilities = func(ctx context.Context) (any, error) {
			return gpuManager.Capabilities(ctx)
		}
	}
	server := mcp.NewServerWithGPUCapabilities(svc, auth.Principal{
		Actor: cfg.MCPActor,
		Role:  cfg.MCPRole,
	}, gpuCapabilities)
	if err := server.ServeStdio(context.Background(), os.Stdin, os.Stdout); err != nil {
		logger.Fatalf("serve mcp: %v", err)
	}
}

func buildBackend(cfg config.Config) (backends.Backend, error) {
	switch cfg.BackendKind {
	case "", "slurm":
		return slurm.NewBackend(cfg), nil
	case "local":
		return local.NewBackend(cfg), nil
	default:
		return nil, errUnsupportedBackend(cfg.BackendKind)
	}
}

type unsupportedBackendError string

func (e unsupportedBackendError) Error() string {
	return "unsupported backend: " + string(e)
}

func errUnsupportedBackend(name string) error {
	return unsupportedBackendError(name)
}
