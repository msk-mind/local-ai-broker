package service

import (
	"context"
	"fmt"
	"strings"

	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/authz"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func (s *Service) listStoredJobs(ctx context.Context) ([]types.Job, error) {
	return s.store.ListJobs(ctx)
}

func filterAuthorizedJobs(principal auth.Principal, jobs []types.Job) []types.Job {
	if auth.IsAdmin(principal) {
		return append([]types.Job(nil), jobs...)
	}
	filtered := make([]types.Job, 0, len(jobs))
	for _, job := range jobs {
		if err := authz.AuthorizeJobAccess(principal, job); err == nil {
			filtered = append(filtered, job)
		}
	}
	return filtered
}

func filterRootJobs(rootJobID string, jobs []types.Job) []types.Job {
	filtered := make([]types.Job, 0, len(jobs))
	for _, job := range jobs {
		if job.RootJobID == rootJobID {
			filtered = append(filtered, job)
		}
	}
	return filtered
}

func ensureAuthorizedRootJobs(principal auth.Principal, rootJobID string, jobs []types.Job) ([]types.Job, error) {
	rootJobID = strings.TrimSpace(rootJobID)
	if rootJobID == "" {
		return nil, store.ErrNotFound
	}
	filtered := filterRootJobs(rootJobID, jobs)
	if len(filtered) == 0 {
		return nil, store.ErrNotFound
	}
	if auth.IsAdmin(principal) {
		return filtered, nil
	}
	for _, job := range filtered {
		if err := authz.AuthorizeJobAccess(principal, job); err != nil {
			return nil, fmt.Errorf("%w: root %q includes inaccessible jobs", authz.ErrForbidden, rootJobID)
		}
	}
	return filtered, nil
}

func authorizeCumulativeNonAdminAction(ctx context.Context, requested, existing, limit int, requestedLabel, cumulativeLabel string) error {
	if requested <= 0 {
		return nil
	}
	principal := auth.PrincipalFromContext(ctx)
	if auth.IsAdmin(principal) {
		return nil
	}
	if requested > limit {
		return fmt.Errorf(
			"%w: requested %s=%d exceeds non-admin limit %d",
			authz.ErrForbidden,
			requestedLabel,
			requested,
			limit,
		)
	}
	if existing+requested > limit {
		return fmt.Errorf(
			"%w: cumulative %s=%d would exceed non-admin limit %d",
			authz.ErrForbidden,
			cumulativeLabel,
			existing+requested,
			limit,
		)
	}
	return nil
}
