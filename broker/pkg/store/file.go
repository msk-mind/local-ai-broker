package store

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

type FileJobStore struct {
	mu                  sync.RWMutex
	path                string
	lockPath            string
	cacheIndexPath      string
	jobs                map[string]types.Job
	jobsByCacheKey      map[string]map[string]struct{}
	jobsMTimeNs         int64
	completedByCacheKey map[string]string
	cacheIndexEntries   map[string]fileCacheIndexEntry
	cacheIndexMTimeNs   int64
}

type fileCacheIndexEntry struct {
	JobID         string `json:"job_id"`
	SubmittedBy   string `json:"submitted_by,omitempty"`
	ArtifactCount int    `json:"artifact_count,omitempty"`
	SchemaName    string `json:"schema_name,omitempty"`
	QualityResult string `json:"quality_result,omitempty"`
}

func NewFileJobStore(path string) (*FileJobStore, error) {
	store := &FileJobStore{
		path:                path,
		lockPath:            path + ".lock",
		cacheIndexPath:      path + ".cache_index",
		jobs:                make(map[string]types.Job),
		jobsByCacheKey:      make(map[string]map[string]struct{}),
		completedByCacheKey: make(map[string]string),
		cacheIndexEntries:   make(map[string]fileCacheIndexEntry),
	}
	if err := store.withFileLock(func() error {
		return store.loadLocked()
	}); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *FileJobStore) CreateJob(ctx context.Context, job types.Job) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.withFileLock(func() error {
		if err := s.loadLocked(); err != nil {
			return err
		}
		if _, exists := s.jobs[job.ID]; exists {
			return errors.New("job already exists")
		}
		s.jobs[job.ID] = job
		s.indexJobByCacheKeyLocked(job)
		cacheIndexDirty := s.indexCacheKeyLocked(job)
		return s.persistLocked(cacheIndexDirty, !nonDurableWriteRequested(ctx))
	})
}

func (s *FileJobStore) GetJob(_ context.Context, id string) (types.Job, error) {
	if job, ok, knownFresh := s.cachedJobFastPath(id); knownFresh {
		if ok {
			return job, nil
		}
		return types.Job{}, ErrNotFound
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.withFileLock(func() error {
		return s.loadLocked()
	}); err != nil {
		return types.Job{}, err
	}
	job, ok := s.jobs[id]
	if !ok {
		return types.Job{}, ErrNotFound
	}
	return job, nil
}

func (s *FileJobStore) UpdateJob(ctx context.Context, job types.Job) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.withFileLock(func() error {
		if err := s.loadLocked(); err != nil {
			return err
		}
		if _, exists := s.jobs[job.ID]; !exists {
			return ErrNotFound
		}
		previous := s.jobs[job.ID]
		s.jobs[job.ID] = job
		if previous.CacheKey != job.CacheKey {
			s.refreshJobsByCacheKeyLocked(previous.CacheKey)
			s.refreshJobsByCacheKeyLocked(job.CacheKey)
		} else {
			s.indexJobByCacheKeyLocked(job)
		}
		cacheIndexDirty := false
		if previous.CacheKey != "" {
			cacheIndexDirty = s.refreshCacheKeyLocked(previous.CacheKey) || cacheIndexDirty
		}
		if previous.CacheKey != job.CacheKey && job.CacheKey != "" {
			cacheIndexDirty = s.refreshCacheKeyLocked(job.CacheKey) || cacheIndexDirty
		} else if job.CacheKey != "" {
			cacheIndexDirty = s.indexCacheKeyLocked(job) || cacheIndexDirty
		}
		return s.persistLocked(cacheIndexDirty, !nonDurableWriteRequested(ctx))
	})
}

func (s *FileJobStore) ListJobs(_ context.Context) ([]types.Job, error) {
	if jobs, ok := s.cachedJobsFastPath(); ok {
		return jobs, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.withFileLock(func() error {
		return s.loadLocked()
	}); err != nil {
		return nil, err
	}
	jobs := make([]types.Job, 0, len(s.jobs))
	for _, job := range s.jobs {
		jobs = append(jobs, job)
	}
	sort.Slice(jobs, func(i, j int) bool {
		if jobs[i].SubmittedAt.Equal(jobs[j].SubmittedAt) {
			return jobs[i].ID < jobs[j].ID
		}
		return jobs[i].SubmittedAt.Before(jobs[j].SubmittedAt)
	})
	return jobs, nil
}

func (s *FileJobStore) FindCompletedJobByCacheKey(_ context.Context, cacheKey string) (types.Job, error) {
	if job, ok, knownFresh := s.cachedCompletedJobFastPath(cacheKey); knownFresh {
		if ok {
			return job, nil
		}
		return types.Job{}, ErrNotFound
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var found types.Job
	if err := s.withFileLock(func() error {
		entry, err := s.loadCacheIndexEntryLocked(cacheKey)
		if err == nil {
			if job, ok := s.jobs[entry.JobID]; ok && cacheEligible(job) {
				found = job
				return nil
			}
			found = types.Job{
				ID:          entry.JobID,
				State:       types.JobStateSucceeded,
				SubmittedBy: entry.SubmittedBy,
				Artifacts:   make([]types.Artifact, entry.ArtifactCount),
			}
			if entry.SchemaName != "" {
				found.Result = &types.Result{
					SchemaName: entry.SchemaName,
					Payload:    map[string]any{"quality": map[string]any{"result": entry.QualityResult}},
				}
			}
			return nil
		}
		if !errors.Is(err, ErrNotFound) {
			return err
		}
		if err := s.loadLocked(); err != nil {
			return err
		}
		if err := s.persistCacheIndexLocked(); err != nil {
			return err
		}
		jobID := s.completedByCacheKey[cacheKey]
		if jobID == "" {
			return ErrNotFound
		}
		job, ok := s.jobs[jobID]
		if !ok || !cacheEligible(job) {
			return ErrNotFound
		}
		found = job
		return nil
	}); err != nil {
		return types.Job{}, err
	}
	if found.ID != "" {
		return found, nil
	}
	return types.Job{}, ErrNotFound
}

func (s *FileJobStore) ListJobsByCacheKey(_ context.Context, cacheKey string) ([]types.Job, error) {
	if jobs, ok, knownFresh := s.cachedJobsByCacheKeyFastPath(cacheKey); knownFresh {
		if ok {
			return jobs, nil
		}
		return nil, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var jobs []types.Job
	if err := s.withFileLock(func() error {
		if err := s.loadLocked(); err != nil {
			return err
		}
		jobs = s.collectJobsByCacheKeyLocked(cacheKey)
		return nil
	}); err != nil {
		return nil, err
	}
	return jobs, nil
}

func (s *FileJobStore) cachedFileStateFresh() bool {
	s.mu.RLock()
	jobsLoaded := s.jobs != nil
	expectedMTime := s.jobsMTimeNs
	s.mu.RUnlock()
	if !jobsLoaded {
		return false
	}
	info, err := os.Stat(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return expectedMTime == 0
	}
	if err != nil {
		return false
	}
	mtimeNs := info.ModTime().UnixNano()
	return mtimeNs != 0 && expectedMTime == mtimeNs
}

func (s *FileJobStore) cachedJobFastPath(id string) (types.Job, bool, bool) {
	if !s.cachedFileStateFresh() {
		return types.Job{}, false, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.jobs[id]
	return job, ok, true
}

func (s *FileJobStore) cachedJobsFastPath() ([]types.Job, bool) {
	if !s.cachedFileStateFresh() {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	jobs := make([]types.Job, 0, len(s.jobs))
	for _, job := range s.jobs {
		jobs = append(jobs, job)
	}
	sort.Slice(jobs, func(i, j int) bool {
		if jobs[i].SubmittedAt.Equal(jobs[j].SubmittedAt) {
			return jobs[i].ID < jobs[j].ID
		}
		return jobs[i].SubmittedAt.Before(jobs[j].SubmittedAt)
	})
	return jobs, true
}

func (s *FileJobStore) cachedCompletedJobFastPath(cacheKey string) (types.Job, bool, bool) {
	if !s.cachedFileStateFresh() {
		return types.Job{}, false, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	jobID := s.completedByCacheKey[cacheKey]
	if jobID == "" {
		return types.Job{}, false, true
	}
	job, ok := s.jobs[jobID]
	if !ok || !cacheEligible(job) {
		return types.Job{}, false, true
	}
	return job, true, true
}

func (s *FileJobStore) cachedJobsByCacheKeyFastPath(cacheKey string) ([]types.Job, bool, bool) {
	if !s.cachedFileStateFresh() {
		return nil, false, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := s.jobsByCacheKey[cacheKey]
	if len(ids) == 0 {
		return nil, false, true
	}
	jobs := make([]types.Job, 0, len(ids))
	for jobID := range ids {
		job, ok := s.jobs[jobID]
		if !ok {
			continue
		}
		jobs = append(jobs, job)
	}
	return jobs, len(jobs) > 0, true
}

func (s *FileJobStore) loadCacheIndexEntryLocked(cacheKey string) (fileCacheIndexEntry, error) {
	info, err := os.Stat(s.cacheIndexPath)
	if errors.Is(err, os.ErrNotExist) {
		s.cacheIndexEntries = make(map[string]fileCacheIndexEntry)
		s.cacheIndexMTimeNs = 0
		return fileCacheIndexEntry{}, ErrNotFound
	}
	if err != nil {
		return fileCacheIndexEntry{}, err
	}
	if info.ModTime().UnixNano() != s.cacheIndexMTimeNs {
		data, err := os.ReadFile(s.cacheIndexPath)
		if err != nil {
			return fileCacheIndexEntry{}, err
		}
		index := map[string]fileCacheIndexEntry{}
		if len(data) > 0 {
			if err := json.Unmarshal(data, &index); err != nil {
				return fileCacheIndexEntry{}, err
			}
		}
		s.cacheIndexEntries = index
		s.cacheIndexMTimeNs = info.ModTime().UnixNano()
	}
	entry, ok := s.cacheIndexEntries[cacheKey]
	if !ok || strings.TrimSpace(entry.JobID) == "" {
		return fileCacheIndexEntry{}, ErrNotFound
	}
	return entry, nil
}

func (s *FileJobStore) loadLocked() error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	info, err := os.Stat(s.path)
	if err == nil {
		mtimeNs := info.ModTime().UnixNano()
		if s.jobs != nil && mtimeNs != 0 && s.jobsMTimeNs == mtimeNs {
			return nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	data, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		s.jobs = make(map[string]types.Job)
		s.jobsByCacheKey = make(map[string]map[string]struct{})
		s.completedByCacheKey = make(map[string]string)
		s.jobsMTimeNs = 0
		return nil
	}
	if err != nil {
		return err
	}
	if len(data) == 0 {
		s.jobs = make(map[string]types.Job)
		s.jobsByCacheKey = make(map[string]map[string]struct{})
		s.completedByCacheKey = make(map[string]string)
		if info, err := os.Stat(s.path); err == nil {
			s.jobsMTimeNs = info.ModTime().UnixNano()
		} else {
			s.jobsMTimeNs = 0
		}
		return nil
	}
	loaded := make(map[string]types.Job)
	if err := json.Unmarshal(data, &loaded); err != nil {
		return err
	}
	s.jobs = loaded
	s.rebuildCacheIndexLocked()
	if info, err := os.Stat(s.path); err == nil {
		s.jobsMTimeNs = info.ModTime().UnixNano()
	} else {
		s.jobsMTimeNs = 0
	}
	return nil
}

func (s *FileJobStore) persistLocked(cacheIndexDirty bool, durable bool) error {
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(s.jobs)
	if err != nil {
		return err
	}

	tmpFile, err := os.CreateTemp(filepath.Dir(s.path), "jobs-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		return err
	}
	if durable {
		if err := tmpFile.Sync(); err != nil {
			tmpFile.Close()
			return err
		}
	}
	if err := tmpFile.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, s.path); err != nil {
		return err
	}
	if info, err := os.Stat(s.path); err == nil {
		s.jobsMTimeNs = info.ModTime().UnixNano()
	} else {
		s.jobsMTimeNs = 0
	}
	if !cacheIndexDirty {
		return nil
	}
	return s.persistCacheIndexLocked()
}

func (s *FileJobStore) withFileLock(fn func() error) error {
	if err := os.MkdirAll(filepath.Dir(s.lockPath), 0o755); err != nil {
		return err
	}
	lockFile, err := os.OpenFile(s.lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer lockFile.Close()

	if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	defer syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN)

	return fn()
}

func (s *FileJobStore) rebuildCacheIndexLocked() {
	index := make(map[string]string)
	jobsByCacheKey := make(map[string]map[string]struct{})
	best := make(map[string]types.Job)
	for _, job := range s.jobs {
		if job.CacheKey != "" {
			ids := jobsByCacheKey[job.CacheKey]
			if ids == nil {
				ids = make(map[string]struct{})
				jobsByCacheKey[job.CacheKey] = ids
			}
			ids[job.ID] = struct{}{}
		}
		if !cacheEligible(job) {
			continue
		}
		current := best[job.CacheKey]
		if preferCacheCandidate(current, job) {
			best[job.CacheKey] = job
			index[job.CacheKey] = job.ID
		}
	}
	s.jobsByCacheKey = jobsByCacheKey
	s.completedByCacheKey = index
}

func (s *FileJobStore) indexCacheKeyLocked(job types.Job) bool {
	if !cacheEligible(job) {
		return false
	}
	currentID := s.completedByCacheKey[job.CacheKey]
	if currentID == "" {
		s.completedByCacheKey[job.CacheKey] = job.ID
		return true
	}
	current, ok := s.jobs[currentID]
	if !ok || !cacheEligible(current) || preferCacheCandidate(current, job) {
		s.completedByCacheKey[job.CacheKey] = job.ID
		return true
	}
	return false
}

func (s *FileJobStore) refreshCacheKeyLocked(cacheKey string) bool {
	if cacheKey == "" {
		return false
	}
	currentID := s.completedByCacheKey[cacheKey]
	best := types.Job{}
	bestFound := false
	for _, candidate := range s.jobs {
		if candidate.CacheKey != cacheKey || !cacheEligible(candidate) {
			continue
		}
		if !bestFound || preferCacheCandidate(best, candidate) {
			best = candidate
			bestFound = true
		}
	}
	if !bestFound {
		if currentID != "" {
			delete(s.completedByCacheKey, cacheKey)
			return true
		}
		return false
	}
	if currentID != best.ID {
		s.completedByCacheKey[cacheKey] = best.ID
		return true
	}
	return false
}

func (s *FileJobStore) indexJobByCacheKeyLocked(job types.Job) {
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

func (s *FileJobStore) refreshJobsByCacheKeyLocked(cacheKey string) {
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

func (s *FileJobStore) collectJobsByCacheKeyLocked(cacheKey string) []types.Job {
	ids := s.jobsByCacheKey[cacheKey]
	if len(ids) == 0 {
		return nil
	}
	jobs := make([]types.Job, 0, len(ids))
	for jobID := range ids {
		job, ok := s.jobs[jobID]
		if !ok {
			continue
		}
		jobs = append(jobs, job)
	}
	return jobs
}

func cacheIndexEntriesEqual(left, right map[string]fileCacheIndexEntry) bool {
	if len(left) != len(right) {
		return false
	}
	for key, leftEntry := range left {
		rightEntry, ok := right[key]
		if !ok || rightEntry != leftEntry {
			return false
		}
	}
	return true
}

func (s *FileJobStore) persistCacheIndexLocked() error {
	index := make(map[string]fileCacheIndexEntry, len(s.completedByCacheKey))
	for cacheKey, jobID := range s.completedByCacheKey {
		job, ok := s.jobs[jobID]
		if !ok || !cacheEligible(job) {
			continue
		}
		schemaName := ""
		qualityResult := ""
		if job.Result != nil {
			schemaName = job.Result.SchemaName
			if quality, ok := job.Result.Payload["quality"].(map[string]any); ok {
				if value, ok := quality["result"].(string); ok {
					qualityResult = value
				}
			}
		}
		index[cacheKey] = fileCacheIndexEntry{
			JobID:         job.ID,
			SubmittedBy:   job.SubmittedBy,
			ArtifactCount: len(job.Artifacts),
			SchemaName:    schemaName,
			QualityResult: qualityResult,
		}
	}
	if cacheIndexEntriesEqual(s.cacheIndexEntries, index) {
		return nil
	}
	data, err := json.Marshal(index)
	if err != nil {
		return err
	}
	tmpFile, err := os.CreateTemp(filepath.Dir(s.cacheIndexPath), "jobs-cache-index-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)
	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		return err
	}
	if err := tmpFile.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, s.cacheIndexPath); err != nil {
		return err
	}
	s.cacheIndexEntries = index
	if info, err := os.Stat(s.cacheIndexPath); err == nil {
		s.cacheIndexMTimeNs = info.ModTime().UnixNano()
	}
	return nil
}
