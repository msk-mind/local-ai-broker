package service

import (
	"context"
	"log"
	"net/url"
	"strings"

	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

const brokerInspectRepoPrewarmActor = "broker-system"

func (s *Service) StartInspectRepoPrewarm(ctx context.Context, logger *log.Logger, inputURI, query string) bool {
	req, ok := s.inspectRepoPrewarmRequest(inputURI, query)
	if !ok {
		return false
	}
	go func() {
		prewarmCtx := auth.WithPrincipal(ctx, auth.Principal{
			Actor: brokerInspectRepoPrewarmActor,
			Role:  "admin",
		})
		resp, err := s.SubmitJob(prewarmCtx, req)
		if err != nil {
			if logger != nil {
				logger.Printf("inspect_repo prewarm submit failed: %v", err)
			}
			return
		}
		if logger != nil {
			logger.Printf("inspect_repo prewarm submitted job_id=%s cache=%s", resp.JobID, strings.TrimSpace(resp.Cache.Status))
		}
	}()
	return true
}

func (s *Service) inspectRepoPrewarmRequest(inputURI, query string) (types.SubmitJobRequest, bool) {
	query = strings.TrimSpace(query)
	if query == "" {
		return types.SubmitJobRequest{}, false
	}
	inputURI = strings.TrimSpace(inputURI)
	if inputURI == "" {
		inputURI = strings.TrimSpace(s.repoRoot)
	}
	if inputURI == "" {
		return types.SubmitJobRequest{}, false
	}
	if normalized, ok := normalizeLocalInputURI(inputURI); ok {
		inputURI = normalized
	} else if !strings.Contains(inputURI, "://") {
		inputURI = (&url.URL{Scheme: "file", Path: inputURI}).String()
	}
	return types.SubmitJobRequest{
		TaskType: "inspect_repo",
		InputRefs: []types.InputRef{
			{
				Type:           "repo",
				URI:            inputURI,
				Classification: "internal",
			},
		},
		TaskParams: map[string]any{
			"query":                 query,
			"mode":                  "evidence",
			"_broker_index_prewarm": true,
		},
		OutputSchema: types.OutputSchemaRef{Name: "repo_inspection_v2"},
	}, true
}
