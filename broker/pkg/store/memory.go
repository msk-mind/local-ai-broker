package store

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

var ErrNotFound = errors.New("job not found")

type JobStore interface {
	CreateJob(context.Context, types.Job) error
	GetJob(context.Context, string) (types.Job, error)
	UpdateJob(context.Context, types.Job) error
	ListJobs(context.Context) ([]types.Job, error)
}

type CompletedCacheKeyLookup interface {
	FindCompletedJobByCacheKey(context.Context, string) (types.Job, error)
}

type JobsByCacheKeyLookup interface {
	ListJobsByCacheKey(context.Context, string) ([]types.Job, error)
}

type MemoryJobStore struct {
	mu                  sync.RWMutex
	jobs                map[string]types.Job
	jobsByCacheKey      map[string]map[string]struct{}
	completedByCacheKey map[string]string
}

func NewMemoryJobStore() *MemoryJobStore {
	return &MemoryJobStore{
		jobs:                make(map[string]types.Job),
		jobsByCacheKey:      make(map[string]map[string]struct{}),
		completedByCacheKey: make(map[string]string),
	}
}

func (s *MemoryJobStore) CreateJob(_ context.Context, job types.Job) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.jobs[job.ID]; exists {
		return errors.New("job already exists")
	}
	s.jobs[job.ID] = job
	s.indexJobByCacheKeyLocked(job)
	s.indexCacheKeyLocked(job)
	return nil
}

func (s *MemoryJobStore) GetJob(_ context.Context, id string) (types.Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.jobs[id]
	if !ok {
		return types.Job{}, ErrNotFound
	}
	return job, nil
}

func (s *MemoryJobStore) UpdateJob(_ context.Context, job types.Job) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	previous, exists := s.jobs[job.ID]
	if !exists {
		return ErrNotFound
	}
	job.UpdatedAt = time.Now().UTC()
	s.jobs[job.ID] = job
	if previous.CacheKey != job.CacheKey {
		s.refreshJobsByCacheKeyLocked(previous.CacheKey)
		s.refreshJobsByCacheKeyLocked(job.CacheKey)
	} else {
		s.indexJobByCacheKeyLocked(job)
	}
	s.refreshCacheKeyLocked(previous.CacheKey)
	if previous.CacheKey != job.CacheKey {
		s.refreshCacheKeyLocked(job.CacheKey)
	} else {
		s.indexCacheKeyLocked(job)
	}
	return nil
}

func (s *MemoryJobStore) ListJobs(_ context.Context) ([]types.Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	jobs := make([]types.Job, 0, len(s.jobs))
	for _, job := range s.jobs {
		jobs = append(jobs, job)
	}
	return jobs, nil
}

func (s *MemoryJobStore) FindCompletedJobByCacheKey(_ context.Context, cacheKey string) (types.Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	jobID := s.completedByCacheKey[cacheKey]
	if jobID == "" {
		return types.Job{}, ErrNotFound
	}
	job, ok := s.jobs[jobID]
	if !ok || !cacheEligible(job) {
		return types.Job{}, ErrNotFound
	}
	return job, nil
}

func (s *MemoryJobStore) ListJobsByCacheKey(_ context.Context, cacheKey string) ([]types.Job, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := s.jobsByCacheKey[cacheKey]
	if len(ids) == 0 {
		return nil, nil
	}
	jobs := make([]types.Job, 0, len(ids))
	for jobID := range ids {
		job, ok := s.jobs[jobID]
		if !ok {
			continue
		}
		jobs = append(jobs, job)
	}
	return jobs, nil
}

func cacheEligible(job types.Job) bool {
	return job.CacheKey != "" && job.State == types.JobStateSucceeded && job.Result != nil
}

func preferCacheCandidate(current, candidate types.Job) bool {
	if current.ID == "" {
		return true
	}
	if candidate.SubmittedAt.After(current.SubmittedAt) {
		return true
	}
	if candidate.SubmittedAt.Equal(current.SubmittedAt) {
		if candidate.UpdatedAt.After(current.UpdatedAt) {
			return true
		}
		if candidate.UpdatedAt.Equal(current.UpdatedAt) && candidate.ID > current.ID {
			return true
		}
	}
	return false
}

func (s *MemoryJobStore) indexCacheKeyLocked(job types.Job) {
	if !cacheEligible(job) {
		return
	}
	currentID := s.completedByCacheKey[job.CacheKey]
	if currentID == "" {
		s.completedByCacheKey[job.CacheKey] = job.ID
		return
	}
	current, ok := s.jobs[currentID]
	if !ok || preferCacheCandidate(current, job) {
		s.completedByCacheKey[job.CacheKey] = job.ID
	}
}

func (s *MemoryJobStore) refreshCacheKeyLocked(cacheKey string) {
	if cacheKey == "" {
		return
	}
	best := types.Job{}
	for _, candidate := range s.jobs {
		if candidate.CacheKey != cacheKey || !cacheEligible(candidate) {
			continue
		}
		if preferCacheCandidate(best, candidate) {
			best = candidate
		}
	}
	if best.ID == "" {
		delete(s.completedByCacheKey, cacheKey)
		return
	}
	s.completedByCacheKey[cacheKey] = best.ID
}

func (s *MemoryJobStore) indexJobByCacheKeyLocked(job types.Job) {
	if job.CacheKey == "" {
		return
	}
	if s.jobsByCacheKey == nil {
		s.jobsByCacheKey = make(map[string]map[string]struct{})
	}
	ids := s.jobsByCacheKey[job.CacheKey]
	if ids == nil {
		ids = make(map[string]struct{})
		s.jobsByCacheKey[job.CacheKey] = ids
	}
	ids[job.ID] = struct{}{}
}

func (s *MemoryJobStore) refreshJobsByCacheKeyLocked(cacheKey string) {
	if cacheKey == "" {
		return
	}
	ids := make(map[string]struct{})
	for _, candidate := range s.jobs {
		if candidate.CacheKey != cacheKey {
			continue
		}
		ids[candidate.ID] = struct{}{}
	}
	if len(ids) == 0 {
		delete(s.jobsByCacheKey, cacheKey)
		return
	}
	s.jobsByCacheKey[cacheKey] = ids
}
