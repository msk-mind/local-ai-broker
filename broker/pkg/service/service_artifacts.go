package service

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/policy"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func (s *Service) resolveRequestInputRefs(ctx context.Context, req types.SubmitJobRequest) (types.SubmitJobRequest, error) {
	requestingJob := types.Job{
		SubmittedBy: auth.PrincipalFromContext(ctx).Actor,
		Request:     req,
	}
	resolved, err := s.resolveInputRefs(ctx, requestingJob)
	if err != nil {
		return types.SubmitJobRequest{}, err
	}
	req.InputRefs = resolved
	return req, nil
}

func (s *Service) resolveInputRefs(ctx context.Context, job types.Job) ([]types.InputRef, error) {
	if len(job.Request.InputRefs) == 0 {
		return nil, nil
	}
	principal := auth.PrincipalFromContext(ctx)
	resolved := make([]types.InputRef, 0, len(job.Request.InputRefs))
	for _, input := range job.Request.InputRefs {
		cloned := input
		if !isArtifactInputRef(input) {
			resolved = append(resolved, cloned)
			continue
		}
		artifactID := strings.TrimSpace(strings.TrimPrefix(input.URI, "artifact://"))
		if artifactID == "" {
			return nil, fmt.Errorf("artifact input uri %q is missing an artifact id", input.URI)
		}
		sourceJobID := ""
		if input.Metadata != nil {
			sourceJobID, _ = input.Metadata["source_job_id"].(string)
		}
		meta, err := s.resolveArtifactRef(ctx, principal, job, artifactID, strings.TrimSpace(sourceJobID))
		if err != nil {
			return nil, err
		}
		cloned.Metadata = mergeMetadata(cloned.Metadata, meta)
		cloned.Classification = higherClassification(
			cloned.Classification,
			stringValue(meta["classification"]),
		)
		resolved = append(resolved, cloned)
	}
	return resolved, nil
}

func (s *Service) resolveArtifactRef(ctx context.Context, principal auth.Principal, requestingJob types.Job, artifactID, sourceJobID string) (map[string]any, error) {
	jobs, err := s.store.ListJobs(ctx)
	if err != nil {
		return nil, fmt.Errorf("list jobs for artifact resolution: %w", err)
	}
	sort.SliceStable(jobs, func(i, j int) bool {
		return jobs[i].SubmittedAt.After(jobs[j].SubmittedAt)
	})
	for _, candidate := range jobs {
		if candidate.ID == requestingJob.ID {
			continue
		}
		if sourceJobID != "" && candidate.ID != sourceJobID {
			continue
		}
		if !artifactJobAccessible(principal, requestingJob, candidate) {
			continue
		}
		eligible, err := releasedArtifacts(candidate)
		if err != nil {
			return nil, fmt.Errorf("evaluate artifact release for job %s: %w", candidate.ID, err)
		}
		eligibleIDs := make(map[string]struct{}, len(eligible))
		for _, artifact := range eligible {
			eligibleIDs[artifact.ArtifactID] = struct{}{}
		}
		for _, artifact := range candidate.Artifacts {
			if artifact.ArtifactID != artifactID {
				continue
			}
			if _, ok := eligibleIDs[artifact.ArtifactID]; !ok {
				continue
			}
			resolvedPath := resolveArtifactPath(s.runRoot, candidate.ID, artifact.Path)
			if resolvedPath != "" {
				if _, statErr := os.Stat(resolvedPath); statErr != nil {
					return nil, fmt.Errorf("artifact %s path %q is unavailable: %w", artifactID, resolvedPath, statErr)
				}
			}
			return map[string]any{
				"artifact_id":        artifact.ArtifactID,
				"artifact_type":      artifact.ArtifactType,
				"source_job_id":      candidate.ID,
				"resolved_path":      resolvedPath,
				"classification":     artifactSourceClassification(candidate, artifact),
				"source_result_name": resultSchemaName(candidate.Result),
			}, nil
		}
	}
	return nil, fmt.Errorf("artifact %s not found in accessible broker jobs", artifactID)
}

func releasedArtifacts(job types.Job) ([]types.Artifact, error) {
	_, artifacts, err := policy.FilterJobResult(job)
	if err != nil {
		return nil, err
	}
	return filterJobArtifactsForRelease(job, artifacts), nil
}

func filterJobArtifactsForRelease(job types.Job, artifacts []types.Artifact) []types.Artifact {
	if job.TaskType == "inspect_repo" && !boolValue(job.Request.TaskParams["include_full_trace"]) {
		return filterArtifactsByType(artifacts, "evidence_pack")
	}
	return artifacts
}

func artifactSourceClassification(job types.Job, artifact types.Artifact) string {
	classification := artifact.Classification
	for _, input := range job.Request.InputRefs {
		classification = higherClassification(classification, input.Classification)
	}
	confidentiality := strings.ToLower(strings.TrimSpace(job.Request.Constraints.Confidentiality))
	if confidentiality == "local_only" {
		classification = higherClassification(classification, "restricted")
	} else {
		classification = higherClassification(classification, confidentiality)
	}
	return classification
}

func higherClassification(values ...string) string {
	ranks := map[string]int{
		"public": 0, "unknown": 1, "internal": 2, "restricted": 3, "phi": 4, "secret_adjacent": 5,
	}
	best := ""
	bestRank := -1
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value == "" {
			continue
		}
		rank, known := ranks[value]
		if !known {
			value, rank = "restricted", ranks["restricted"]
		}
		if rank > bestRank {
			best, bestRank = value, rank
		}
	}
	return best
}

func artifactJobAccessible(principal auth.Principal, requestingJob, candidate types.Job) bool {
	if auth.IsAdmin(principal) {
		return true
	}
	if requestingJob.SubmittedBy != "" && candidate.SubmittedBy != "" {
		return requestingJob.SubmittedBy == candidate.SubmittedBy
	}
	if principal.Actor != "" {
		if candidate.SubmittedBy == "" {
			return true
		}
		return principal.Actor == candidate.SubmittedBy
	}
	return candidate.SubmittedBy == ""
}

func resolveArtifactPath(runRoot, jobID, artifactPath string) string {
	if strings.TrimSpace(artifactPath) == "" {
		return ""
	}
	if filepath.IsAbs(artifactPath) {
		return artifactPath
	}
	return filepath.Join(runRoot, jobID, artifactPath)
}

func isArtifactInputRef(input types.InputRef) bool {
	return input.Type == "artifact" || strings.HasPrefix(input.URI, "artifact://")
}
