package cache

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/tasks"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

var (
	errFingerprintBudgetExceeded        = errors.New("directory fingerprint budget exceeded")
	metadataFingerprintMaxEntries       = 20000
	metadataFingerprintMaxDuration      = 2 * time.Second
	gitFingerprintFastpathFileThreshold = 512
	runGitFunc                          = runGit
	runGitExitCodeFunc                  = runGitExitCode
	userCacheDirFunc                    = os.UserCacheDir
	runRGFilesFunc                      = rgFileList
	gitFingerprintMemoTTL               = 5 * time.Second
	gitEphemeralExcludePathspecs        = []string{":(exclude).broker-live-tests", ":(exclude)slurm-*.out"}
	ephemeralIgnoreFileGlobs            = []string{"slurm-*.out"}
	gitExecPathOnce                     sync.Once
	gitExecPathValue                    string
	gitExecPathErr                      error
	gitCommandEnvOnce                   sync.Once
	gitCommandEnvValue                  []string
)

type gitFingerprintMemoEntry struct {
	Fingerprint  string
	StatusDigest string
	ExpiresAt    time.Time
}

type gitCleanFastpathMemoEntry struct {
	TopLevel    string
	RelPath     string
	Fingerprint string
	HeadSig     string
	IndexSig    string
	CleanFiles  map[string]string
	ExpiresAt   time.Time
}

type worktreeSignatureMemoEntry struct {
	StateSignature   string
	ContentSignature string
	GitBlobOID       string
}

type gitDirtyFastpathMemoEntry struct {
	TopLevel           string
	RelPath            string
	Fingerprint        string
	Head               string
	StatusDigest       string
	StagedStatus       string
	IndexSig           string
	DirtyPaths         []string
	ScopeFiles         map[string]string
	StagedEntries      []map[string]string
	UnstagedEntries    []map[string]string
	UntrackedPaths     []string
	WorktreeSignatures map[string]worktreeSignatureMemoEntry
	ExpiresAt          time.Time
}

var gitFingerprintMemo struct {
	sync.Mutex
	entries map[string]gitFingerprintMemoEntry
}

var gitCleanFastpathMemo struct {
	sync.Mutex
	entries map[string]gitCleanFastpathMemoEntry
}

var gitDirtyFastpathMemo struct {
	sync.Mutex
	entries map[string]gitDirtyFastpathMemoEntry
}

type manifestFileStamp struct {
	Size    int64
	MTimeNS int64
}

var gitScopeManifestCache struct {
	sync.Mutex
	entries map[string]cachedGitScopeManifest
}

type cachedGitScopeManifest struct {
	Stamp    manifestFileStamp
	TopLevel string
	RelPath  string
}

var gitFingerprintManifestCache struct {
	sync.Mutex
	entries map[string]cachedGitFingerprintManifest
}

type cachedGitFingerprintManifest struct {
	Stamp    manifestFileStamp
	Manifest gitFingerprintManifest
}

var gitDirPathCache struct {
	sync.Mutex
	entries map[string]string
}

var gitIndexStateSignatureCache struct {
	sync.Mutex
	entries map[string]cachedStateSignature
}

var gitHeadStateSignatureCache struct {
	sync.Mutex
	entries map[string]cachedHeadSignature
}

type cachedStateSignature struct {
	Stamp     manifestFileStamp
	Signature string
}

type cachedHeadSignature struct {
	HeadStamp manifestFileStamp
	RefPath   string
	RefStamp  manifestFileStamp
	Signature string
}

type gitFingerprintManifest struct {
	TopLevel     string            `json:"top_level"`
	Head         string            `json:"head"`
	RelPath      string            `json:"rel_path"`
	StatusDigest string            `json:"status_digest"`
	Fingerprint  string            `json:"fingerprint"`
	DirtyPaths   []string          `json:"dirty_paths,omitempty"`
	ScopeFiles   map[string]string `json:"scope_files,omitempty"`
	CleanFiles   map[string]string `json:"clean_files,omitempty"`
	IndexSig     string            `json:"index_sig,omitempty"`
}

type gitStatusPayload struct {
	Staged    string `json:"staged"`
	Unstaged  string `json:"unstaged"`
	Untracked string `json:"untracked"`
}

type gitPorcelainEntry struct {
	x          byte
	y          byte
	path       string
	sourcePath string
}

type gitScopeManifest struct {
	TopLevel string `json:"top_level"`
	RelPath  string `json:"rel_path"`
}

type metadataFingerprintManifest struct {
	Root        string                               `json:"root"`
	Fingerprint string                               `json:"fingerprint"`
	Files       map[string]metadataFingerprintRecord `json:"files"`
}

type metadataFingerprintRecord struct {
	Size    int64 `json:"size"`
	MTimeNS int64 `json:"mtime_ns"`
}

type RequestKeyDetails struct {
	Key                string
	Cacheable          bool
	ContentHash        string
	DirtyPaths         []string
	CleanWorktreeFiles []string
	TimingsMS          map[string]int64
}

func gitFingerprintMemoKey(topLevel, relPath string) string {
	return strings.TrimSpace(topLevel) + "\x00" + strings.TrimSpace(relPath)
}

func loadGitFingerprintMemo(topLevel, relPath, statusDigest string, now time.Time) (string, bool) {
	key := gitFingerprintMemoKey(topLevel, relPath)
	gitFingerprintMemo.Lock()
	defer gitFingerprintMemo.Unlock()
	entry, ok := gitFingerprintMemo.entries[key]
	if !ok {
		return "", false
	}
	if now.After(entry.ExpiresAt) {
		delete(gitFingerprintMemo.entries, key)
		return "", false
	}
	if entry.StatusDigest != strings.TrimSpace(statusDigest) || strings.TrimSpace(entry.Fingerprint) == "" {
		return "", false
	}
	return entry.Fingerprint, true
}

func storeGitFingerprintMemo(topLevel, relPath, statusDigest, fingerprint string, now time.Time) {
	if strings.TrimSpace(fingerprint) == "" {
		return
	}
	key := gitFingerprintMemoKey(topLevel, relPath)
	gitFingerprintMemo.Lock()
	defer gitFingerprintMemo.Unlock()
	if gitFingerprintMemo.entries == nil {
		gitFingerprintMemo.entries = make(map[string]gitFingerprintMemoEntry)
	}
	gitFingerprintMemo.entries[key] = gitFingerprintMemoEntry{
		Fingerprint:  strings.TrimSpace(fingerprint),
		StatusDigest: strings.TrimSpace(statusDigest),
		ExpiresAt:    now.Add(gitFingerprintMemoTTL),
	}
}

func loadGitCleanFastpathMemo(topLevel, relPath string, now time.Time) (gitCleanFastpathMemoEntry, bool) {
	key := gitFingerprintMemoKey(topLevel, relPath)
	gitCleanFastpathMemo.Lock()
	defer gitCleanFastpathMemo.Unlock()
	entry, ok := gitCleanFastpathMemo.entries[key]
	if !ok {
		return gitCleanFastpathMemoEntry{}, false
	}
	if now.After(entry.ExpiresAt) {
		delete(gitCleanFastpathMemo.entries, key)
		return gitCleanFastpathMemoEntry{}, false
	}
	if entry.TopLevel != strings.TrimSpace(topLevel) || entry.RelPath != strings.TrimSpace(relPath) || strings.TrimSpace(entry.Fingerprint) == "" {
		return gitCleanFastpathMemoEntry{}, false
	}
	cloned := entry
	if len(entry.CleanFiles) > 0 {
		cloned.CleanFiles = make(map[string]string, len(entry.CleanFiles))
		for key, value := range entry.CleanFiles {
			cloned.CleanFiles[key] = value
		}
	}
	return cloned, true
}

func storeGitCleanFastpathMemo(topLevel, relPath, fingerprint, headSig, indexSig string, cleanFiles map[string]string, now time.Time) {
	if strings.TrimSpace(fingerprint) == "" || strings.TrimSpace(headSig) == "" || strings.TrimSpace(indexSig) == "" || len(cleanFiles) == 0 {
		return
	}
	key := gitFingerprintMemoKey(topLevel, relPath)
	clonedFiles := make(map[string]string, len(cleanFiles))
	for key, value := range cleanFiles {
		clonedFiles[key] = value
	}
	gitCleanFastpathMemo.Lock()
	defer gitCleanFastpathMemo.Unlock()
	if gitCleanFastpathMemo.entries == nil {
		gitCleanFastpathMemo.entries = make(map[string]gitCleanFastpathMemoEntry)
	}
	gitCleanFastpathMemo.entries[key] = gitCleanFastpathMemoEntry{
		TopLevel:    strings.TrimSpace(topLevel),
		RelPath:     strings.TrimSpace(relPath),
		Fingerprint: strings.TrimSpace(fingerprint),
		HeadSig:     strings.TrimSpace(headSig),
		IndexSig:    strings.TrimSpace(indexSig),
		CleanFiles:  clonedFiles,
		ExpiresAt:   now.Add(gitFingerprintMemoTTL),
	}
}

func loadGitDirtyFastpathMemoClone(topLevel, relPath string, now time.Time, cloneStagedEntries bool, cloneWorktreeSignatures bool, cloneDirtyPaths bool, cloneScopeFiles bool) (gitDirtyFastpathMemoEntry, bool) {
	key := gitFingerprintMemoKey(topLevel, relPath)
	gitDirtyFastpathMemo.Lock()
	defer gitDirtyFastpathMemo.Unlock()
	entry, ok := gitDirtyFastpathMemo.entries[key]
	if !ok {
		return gitDirtyFastpathMemoEntry{}, false
	}
	if now.After(entry.ExpiresAt) {
		delete(gitDirtyFastpathMemo.entries, key)
		return gitDirtyFastpathMemoEntry{}, false
	}
	if entry.TopLevel != strings.TrimSpace(topLevel) || entry.RelPath != strings.TrimSpace(relPath) || strings.TrimSpace(entry.Fingerprint) == "" {
		return gitDirtyFastpathMemoEntry{}, false
	}
	cloned := entry
	if cloneStagedEntries && len(entry.StagedEntries) > 0 {
		cloned.StagedEntries = make([]map[string]string, 0, len(entry.StagedEntries))
		for _, raw := range entry.StagedEntries {
			clonedEntry := make(map[string]string, len(raw))
			for key, value := range raw {
				clonedEntry[key] = value
			}
			cloned.StagedEntries = append(cloned.StagedEntries, clonedEntry)
		}
	} else {
		cloned.StagedEntries = nil
	}
	if cloneStagedEntries && len(entry.UnstagedEntries) > 0 {
		cloned.UnstagedEntries = make([]map[string]string, 0, len(entry.UnstagedEntries))
		for _, raw := range entry.UnstagedEntries {
			clonedEntry := make(map[string]string, len(raw))
			for key, value := range raw {
				clonedEntry[key] = value
			}
			cloned.UnstagedEntries = append(cloned.UnstagedEntries, clonedEntry)
		}
	} else {
		cloned.UnstagedEntries = nil
	}
	if cloneDirtyPaths && len(entry.UntrackedPaths) > 0 {
		cloned.UntrackedPaths = append([]string(nil), entry.UntrackedPaths...)
	} else {
		cloned.UntrackedPaths = nil
	}
	if cloneWorktreeSignatures && len(entry.WorktreeSignatures) > 0 {
		cloned.WorktreeSignatures = make(map[string]worktreeSignatureMemoEntry, len(entry.WorktreeSignatures))
		for key, value := range entry.WorktreeSignatures {
			cloned.WorktreeSignatures[key] = value
		}
	} else {
		cloned.WorktreeSignatures = nil
	}
	if cloneDirtyPaths && len(entry.DirtyPaths) > 0 {
		cloned.DirtyPaths = append([]string(nil), entry.DirtyPaths...)
	} else {
		cloned.DirtyPaths = nil
	}
	if cloneScopeFiles && len(entry.ScopeFiles) > 0 {
		cloned.ScopeFiles = make(map[string]string, len(entry.ScopeFiles))
		for key, value := range entry.ScopeFiles {
			cloned.ScopeFiles[key] = value
		}
	} else {
		cloned.ScopeFiles = nil
	}
	return cloned, true
}

func loadGitDirtyFastpathMemo(topLevel, relPath string, now time.Time) (gitDirtyFastpathMemoEntry, bool) {
	return loadGitDirtyFastpathMemoClone(topLevel, relPath, now,
		true,
		true,
		true,
		true,
	)
}

func storeGitDirtyFastpathMemo(topLevel, relPath, fingerprint, head, statusDigest, stagedStatus, indexSig string, dirtyPaths []string, scopeFiles map[string]string, stagedEntries []map[string]string, unstagedEntries []map[string]string, untrackedPaths []string, worktreeSignatures map[string]worktreeSignatureMemoEntry, now time.Time) {
	if strings.TrimSpace(fingerprint) == "" || strings.TrimSpace(head) == "" || strings.TrimSpace(statusDigest) == "" {
		return
	}
	key := gitFingerprintMemoKey(topLevel, relPath)
	clonedStagedEntries := make([]map[string]string, 0, len(stagedEntries))
	for _, raw := range stagedEntries {
		clonedEntry := make(map[string]string, len(raw))
		for key, value := range raw {
			clonedEntry[key] = value
		}
		clonedStagedEntries = append(clonedStagedEntries, clonedEntry)
	}
	clonedUnstagedEntries := make([]map[string]string, 0, len(unstagedEntries))
	for _, raw := range unstagedEntries {
		clonedEntry := make(map[string]string, len(raw))
		for key, value := range raw {
			clonedEntry[key] = value
		}
		clonedUnstagedEntries = append(clonedUnstagedEntries, clonedEntry)
	}
	clonedWorktreeSignatures := make(map[string]worktreeSignatureMemoEntry, len(worktreeSignatures))
	for key, value := range worktreeSignatures {
		clonedWorktreeSignatures[key] = value
	}
	clonedDirtyPaths := append([]string(nil), dirtyPaths...)
	clonedUntrackedPaths := append([]string(nil), untrackedPaths...)
	clonedScopeFiles := make(map[string]string, len(scopeFiles))
	for key, value := range scopeFiles {
		clonedScopeFiles[key] = value
	}
	gitDirtyFastpathMemo.Lock()
	defer gitDirtyFastpathMemo.Unlock()
	if gitDirtyFastpathMemo.entries == nil {
		gitDirtyFastpathMemo.entries = make(map[string]gitDirtyFastpathMemoEntry)
	}
	gitDirtyFastpathMemo.entries[key] = gitDirtyFastpathMemoEntry{
		TopLevel:           strings.TrimSpace(topLevel),
		RelPath:            strings.TrimSpace(relPath),
		Fingerprint:        strings.TrimSpace(fingerprint),
		Head:               strings.TrimSpace(head),
		StatusDigest:       strings.TrimSpace(statusDigest),
		StagedStatus:       strings.TrimSpace(stagedStatus),
		IndexSig:           strings.TrimSpace(indexSig),
		DirtyPaths:         clonedDirtyPaths,
		ScopeFiles:         clonedScopeFiles,
		StagedEntries:      clonedStagedEntries,
		UnstagedEntries:    clonedUnstagedEntries,
		UntrackedPaths:     clonedUntrackedPaths,
		WorktreeSignatures: clonedWorktreeSignatures,
		ExpiresAt:          now.Add(gitFingerprintMemoTTL),
	}
}

func KeyForRequest(req types.SubmitJobRequest) (string, bool, error) {
	details, err := KeyDetailsForRequest(req)
	if err != nil {
		return "", false, err
	}
	return details.Key, details.Cacheable, nil
}

func KeyDetailsForRequest(req types.SubmitJobRequest) (RequestKeyDetails, error) {
	if !tasks.IsCacheableTask(req.TaskType) {
		return RequestKeyDetails{}, nil
	}
	if len(req.InputRefs) != 1 {
		return RequestKeyDetails{}, nil
	}
	input := req.InputRefs[0]
	if input.Type != "file" && input.Type != "directory" && input.Type != "repo" {
		return RequestKeyDetails{}, nil
	}

	contentHash := strings.TrimSpace(input.ContentHash)
	dirtyPaths := []string(nil)
	cleanWorktreeFiles := []string(nil)
	timingsMS := map[string]int64{}
	if contentHash == "" {
		pathResolveStartedAt := time.Now()
		path, err := filePathFromURI(input.URI)
		timingsMS["resolve_input_path_ms"] = time.Since(pathResolveStartedAt).Milliseconds()
		if err != nil {
			return RequestKeyDetails{}, err
		}
		switch input.Type {
		case "file":
			fileHashStartedAt := time.Now()
			data, err := os.ReadFile(path)
			if err != nil {
				return RequestKeyDetails{}, nil
			}
			contentHash = sumBytes(data)
			timingsMS["hash_input_file_ms"] = time.Since(fileHashStartedAt).Milliseconds()
		case "directory", "repo":
			var details pathFingerprintDetails
			pathHashStartedAt := time.Now()
			details, err = hashPathWithoutRequestManifest(path, input.Type)
			timingsMS["hash_input_path_ms"] = time.Since(pathHashStartedAt).Milliseconds()
			if err != nil {
				if errors.Is(err, errFingerprintBudgetExceeded) {
					return RequestKeyDetails{}, nil
				}
				return RequestKeyDetails{}, nil
			}
			contentHash = details.Fingerprint
			dirtyPaths = append([]string(nil), details.DirtyPaths...)
			cleanWorktreeFiles = append([]string(nil), details.CleanWorktreeFiles...)
		}
	}

	payload := map[string]any{
		"task_type":      req.TaskType,
		"schema_name":    req.OutputSchema.Name,
		"input_type":     input.Type,
		"input_uri":      input.URI,
		"content_hash":   contentHash,
		"task_params":    stableTaskParams(req.TaskParams),
		"constraints":    req.Constraints,
		"execution":      stableExecutionProfile(req.ExecutionProfile),
		"classification": input.Classification,
	}

	serializeStartedAt := time.Now()
	serialized, err := json.Marshal(payload)
	timingsMS["serialize_cache_key_payload_ms"] = time.Since(serializeStartedAt).Milliseconds()
	if err != nil {
		return RequestKeyDetails{}, err
	}
	hashKeyStartedAt := time.Now()
	key := "sha256:" + sumBytes(serialized)
	timingsMS["hash_cache_key_payload_ms"] = time.Since(hashKeyStartedAt).Milliseconds()
	return RequestKeyDetails{
		Key:                key,
		Cacheable:          true,
		ContentHash:        contentHash,
		DirtyPaths:         dirtyPaths,
		CleanWorktreeFiles: cleanWorktreeFiles,
		TimingsMS:          timingsMS,
	}, nil
}

func stableExecutionProfile(profile types.ExecutionProfile) map[string]any {
	return map[string]any{
		"tier":        strings.TrimSpace(profile.Tier),
		"model":       strings.TrimSpace(profile.Model),
		"runtime":     strings.TrimSpace(profile.Runtime),
		"accelerator": strings.TrimSpace(profile.Accelerator),
	}
}

func FindCompletedJobByCacheKey(ctx context.Context, jobStore store.JobStore, cacheKey string) (*types.Job, error) {
	if cacheKey == "" {
		return nil, nil
	}
	if finder, ok := jobStore.(store.CompletedCacheKeyLookup); ok {
		job, err := finder.FindCompletedJobByCacheKey(ctx, cacheKey)
		if errors.Is(err, store.ErrNotFound) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		candidate := job
		return &candidate, nil
	}
	jobs, err := jobStore.ListJobs(ctx)
	if err != nil {
		return nil, err
	}
	for _, job := range jobs {
		if job.CacheKey == cacheKey && job.State == types.JobStateSucceeded && job.Result != nil {
			candidate := job
			return &candidate, nil
		}
	}
	return nil, nil
}

func filePathFromURI(uri string) (string, error) {
	parsed, err := url.Parse(uri)
	if err != nil {
		return "", err
	}
	if parsed.Scheme != "file" {
		return "", fmt.Errorf("unsupported cacheable input uri: %s", uri)
	}
	return parsed.Path, nil
}

func sumBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func stableTaskParams(taskParams map[string]any) map[string]any {
	if len(taskParams) == 0 {
		return map[string]any{}
	}
	keys := make([]string, 0, len(taskParams))
	for k := range taskParams {
		if len(k) > 0 && k[0] == '_' {
			continue
		}
		if isNonSemanticTaskParam(k) {
			continue
		}
		keys = append(keys, k)
	}
	slices.Sort(keys)
	out := make(map[string]any, len(keys))
	for _, k := range keys {
		out[k] = taskParams[k]
	}
	return out
}

func isNonSemanticTaskParam(key string) bool {
	switch strings.TrimSpace(key) {
	case "client_nonce":
		return true
	default:
		return false
	}
}

func hashPath(path, inputType string) (string, error) {
	details, err := hashPathWithoutRequestManifest(path, inputType)
	if err != nil {
		return "", err
	}
	return details.Fingerprint, nil
}

type pathFingerprintDetails struct {
	Fingerprint        string
	DirtyPaths         []string
	CleanWorktreeFiles []string
}

func hashPathWithoutRequestManifest(path, inputType string) (pathFingerprintDetails, error) {
	if details, ok := gitFingerprint(path); ok {
		return details, nil
	}
	fingerprint, err := fingerprintDirectoryMetadata(path)
	if err != nil {
		return pathFingerprintDetails{}, err
	}
	return pathFingerprintDetails{Fingerprint: fingerprint}, nil
}

func cleanWorktreeFilesFromManifestFiles(files map[string]string) []string {
	if len(files) == 0 {
		return nil
	}
	paths := make([]string, 0, len(files))
	for rel := range files {
		rel = filepath.ToSlash(strings.TrimSpace(rel))
		if rel == "" || rel == "." {
			continue
		}
		paths = append(paths, rel)
	}
	slices.Sort(paths)
	return paths
}

func fingerprintDirectoryMetadata(root string) (string, error) {
	manifestPath := metadataFingerprintManifestPath(root)
	previousManifest, _ := loadMetadataFingerprintManifest(manifestPath)
	hasher := sha256.New()
	startedAt := time.Now()
	entryCount := 0
	nextManifest := metadataFingerprintManifest{
		Root:  filepath.Clean(root),
		Files: map[string]metadataFingerprintRecord{},
	}
	entries, rgErr := runRGFilesFunc(root)
	if rgErr == nil {
		for _, path := range entries {
			rel, err := filepath.Rel(root, path)
			if err != nil {
				return "", err
			}
			rel = filepath.ToSlash(rel)
			if shouldIgnoreDir(rel) {
				continue
			}
			entryCount++
			if entryCount > metadataFingerprintMaxEntries || time.Since(startedAt) > metadataFingerprintMaxDuration {
				return "", errFingerprintBudgetExceeded
			}
			info, err := os.Stat(path)
			if err != nil {
				return "", err
			}
			nextManifest.Files[rel] = metadataFingerprintRecord{
				Size:    info.Size(),
				MTimeNS: info.ModTime().UTC().UnixNano(),
			}
			if _, err := io.WriteString(hasher, rel); err != nil {
				return "", err
			}
			if _, err := io.WriteString(hasher, "\x00"); err != nil {
				return "", err
			}
			if _, err := io.WriteString(hasher, fmt.Sprintf("%d", info.Size())); err != nil {
				return "", err
			}
			if _, err := io.WriteString(hasher, "\x00"); err != nil {
				return "", err
			}
			if _, err := io.WriteString(hasher, fmt.Sprintf("%d", info.ModTime().UTC().UnixNano())); err != nil {
				return "", err
			}
			if _, err := io.WriteString(hasher, "\x00"); err != nil {
				return "", err
			}
		}
		fingerprint := "meta:" + hex.EncodeToString(hasher.Sum(nil))
		if metadataFingerprintManifestEqual(previousManifest, nextManifest) && strings.TrimSpace(previousManifest.Fingerprint) != "" {
			return previousManifest.Fingerprint, nil
		}
		nextManifest.Fingerprint = fingerprint
		_ = writeMetadataFingerprintManifest(manifestPath, nextManifest)
		return fingerprint, nil
	}

	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == root {
			return nil
		}

		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)

		if d.IsDir() {
			if shouldIgnoreDir(rel) {
				return filepath.SkipDir
			}
			return nil
		}
		entryCount++
		if entryCount > metadataFingerprintMaxEntries || time.Since(startedAt) > metadataFingerprintMaxDuration {
			return errFingerprintBudgetExceeded
		}

		info, err := d.Info()
		if err != nil {
			return err
		}
		nextManifest.Files[rel] = metadataFingerprintRecord{
			Size:    info.Size(),
			MTimeNS: info.ModTime().UTC().UnixNano(),
		}

		if _, err := io.WriteString(hasher, rel); err != nil {
			return err
		}
		if _, err := io.WriteString(hasher, "\x00"); err != nil {
			return err
		}
		if _, err := io.WriteString(hasher, fmt.Sprintf("%d", info.Size())); err != nil {
			return err
		}
		if _, err := io.WriteString(hasher, "\x00"); err != nil {
			return err
		}
		if _, err := io.WriteString(hasher, fmt.Sprintf("%d", info.ModTime().UTC().UnixNano())); err != nil {
			return err
		}
		if _, err := io.WriteString(hasher, "\x00"); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	fingerprint := "meta:" + hex.EncodeToString(hasher.Sum(nil))
	if metadataFingerprintManifestEqual(previousManifest, nextManifest) && strings.TrimSpace(previousManifest.Fingerprint) != "" {
		return previousManifest.Fingerprint, nil
	}
	nextManifest.Fingerprint = fingerprint
	_ = writeMetadataFingerprintManifest(manifestPath, nextManifest)
	return fingerprint, nil
}

func rgFileList(root string) ([]string, error) {
	rgPath, err := exec.LookPath("rg")
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(rgPath, "--files", "--hidden", "--no-ignore", root)
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.ReplaceAll(string(output), "\r\n", "\n"), "\n")
	entries := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		path := line
		if !filepath.IsAbs(path) {
			path = filepath.Join(root, path)
		}
		entries = append(entries, filepath.Clean(path))
	}
	slices.Sort(entries)
	return entries, nil
}

func gitExecutablePath() (string, error) {
	gitExecPathOnce.Do(func() {
		gitExecPathValue, gitExecPathErr = exec.LookPath("git")
	})
	if gitExecPathErr != nil {
		return "", gitExecPathErr
	}
	return gitExecPathValue, nil
}

func gitFingerprint(path string) (pathFingerprintDetails, bool) {
	gitPath, err := gitExecutablePath()
	if err != nil {
		return pathFingerprintDetails{}, false
	}

	topLevel, relPath, ok := loadGitScopeManifest(path)
	if !ok {
		topLevel, relPath, ok = gitScopeFromFilesystem(path)
		if !ok {
			topLevelRaw, err := runGitFunc(gitPath, path, "rev-parse", "--show-toplevel")
			if err != nil {
				return pathFingerprintDetails{}, false
			}
			topLevel = strings.TrimSpace(topLevelRaw)
			if topLevel == "" {
				return pathFingerprintDetails{}, false
			}
			relPath = "."
			if rel, err := filepath.Rel(topLevel, path); err == nil && rel != "" {
				relPath = filepath.ToSlash(rel)
			}
		}
		_ = writeGitScopeManifest(path, gitScopeManifest{TopLevel: topLevel, RelPath: relPath})
	}

	manifestPath := gitFingerprintManifestPath(topLevel, relPath)
	now := time.Now()
	if memo, ok := loadGitCleanFastpathMemo(topLevel, relPath, now); ok {
		if cleanFingerprint, ok := gitCleanFastpathFingerprintFromState(topLevel, relPath, memo.Fingerprint, memo.HeadSig, memo.IndexSig, memo.CleanFiles); ok {
			emptyStatus := `{"staged":"","unstaged":"","untracked":""}`
			statusDigest := sumBytes([]byte(emptyStatus))
			storeGitFingerprintMemo(topLevel, relPath, statusDigest, cleanFingerprint, now)
			return pathFingerprintDetails{
				Fingerprint:        cleanFingerprint,
				CleanWorktreeFiles: cleanWorktreeFilesFromManifestFiles(memo.CleanFiles),
			}, true
		}
	}
	if manifest, ok := loadGitFingerprintManifest(manifestPath); ok {
		if cleanFingerprint, ok := gitCleanFastpathFingerprint(gitPath, topLevel, relPath, manifest); ok {
			emptyStatus := `{"staged":"","unstaged":"","untracked":""}`
			statusDigest := sumBytes([]byte(emptyStatus))
			storeGitFingerprintMemo(topLevel, relPath, statusDigest, cleanFingerprint, now)
			return pathFingerprintDetails{
				Fingerprint:        cleanFingerprint,
				CleanWorktreeFiles: cleanWorktreeFilesFromManifestFiles(manifest.CleanFiles),
			}, true
		}
	}
	if memo, ok := loadGitDirtyFastpathMemoClone(topLevel, relPath, now, false, true, true, true); ok {
		if dirtyFingerprint, dirtyPaths, ok := gitDirtyFastpathFingerprintFromState(gitPath, topLevel, relPath, memo.Fingerprint, memo.Head, memo.IndexSig, memo.ScopeFiles, memo.DirtyPaths, memo.WorktreeSignatures); ok {
			return pathFingerprintDetails{
				Fingerprint: dirtyFingerprint,
				DirtyPaths:  dirtyPaths,
			}, true
		}
	}
	if manifest, ok := loadGitFingerprintManifest(manifestPath); ok {
		if dirtyFingerprint, dirtyPaths, ok := gitDirtyFastpathFingerprintFromState(gitPath, topLevel, relPath, manifest.Fingerprint, manifest.Head, manifest.IndexSig, manifest.ScopeFiles, manifest.DirtyPaths, nil); ok {
			return pathFingerprintDetails{
				Fingerprint: dirtyFingerprint,
				DirtyPaths:  dirtyPaths,
			}, true
		}
	}

	if status, ok := gitDirtySubsetStatusSummary(gitPath, topLevel, relPath, now); ok {
		return gitFingerprintFromDirtyStatus(path, gitPath, topLevel, relPath, manifestPath, now, status)
	}

	status, err := gitStatusSummary(gitPath, topLevel, relPath)
	if err != nil {
		topLevelRaw, topErr := runGitFunc(gitPath, path, "rev-parse", "--show-toplevel")
		if topErr != nil {
			status = ""
		} else {
			topLevel = strings.TrimSpace(topLevelRaw)
			if topLevel == "" {
				return pathFingerprintDetails{}, false
			}
			relPath = "."
			if rel, relErr := filepath.Rel(topLevel, path); relErr == nil && rel != "" {
				relPath = filepath.ToSlash(rel)
			}
			_ = writeGitScopeManifest(path, gitScopeManifest{TopLevel: topLevel, RelPath: relPath})
			status, err = gitStatusSummary(gitPath, topLevel, relPath)
			if err != nil {
				status = ""
			}
		}
	}
	status = strings.TrimSpace(status)
	return gitFingerprintFromDirtyStatus(path, gitPath, topLevel, relPath, manifestPath, now, status)
}

func gitFingerprintFromDirtyStatus(path, gitPath, topLevel, relPath, manifestPath string, now time.Time, status string) (pathFingerprintDetails, bool) {
	status = strings.TrimSpace(status)
	statusDigest := sumBytes([]byte(status))
	clean := statusDigest == sumBytes([]byte(`{"staged":"","unstaged":"","untracked":""}`))

	if manifest, ok := loadGitFingerprintManifest(manifestPath); ok {
		if clean && manifest.TopLevel == topLevel && manifest.RelPath == relPath && manifest.StatusDigest == statusDigest && manifest.Fingerprint != "" {
			storeGitFingerprintMemo(topLevel, relPath, statusDigest, manifest.Fingerprint, now)
			return pathFingerprintDetails{
				Fingerprint:        manifest.Fingerprint,
				CleanWorktreeFiles: cleanWorktreeFilesFromManifestFiles(manifest.CleanFiles),
			}, true
		}
	}
	if clean {
		if fingerprint, ok := loadGitFingerprintMemo(topLevel, relPath, statusDigest, now); ok {
			return pathFingerprintDetails{Fingerprint: fingerprint}, true
		}
	}

	head := currentGitFingerprintHead(gitPath, path, topLevel, relPath)

	payload := map[string]any{
		"repo_root": topLevel,
		"repo_head": head,
		"scope":     relPath,
	}
	if clean {
		payload["status"] = status
	} else {
		var previousDirtyMemo *gitDirtyFastpathMemoEntry
		if memo, ok := loadGitDirtyFastpathMemoClone(topLevel, relPath, now, true, true, false, false); ok &&
			strings.TrimSpace(memo.Head) == strings.TrimSpace(head) &&
			(strings.TrimSpace(memo.StatusDigest) == strings.TrimSpace(statusDigest) ||
				strings.TrimSpace(memo.StagedStatus) == strings.TrimSpace(extractGitStatusStage(status))) {
			previousDirtyMemo = &memo
		}
		dirtyState, worktreeSignatures, err := gitDirtyStateWithCache(gitPath, topLevel, relPath, status, previousDirtyMemo)
		if err != nil {
			return pathFingerprintDetails{}, false
		}
		payload["dirty_state"] = dirtyState
		serialized, err := json.Marshal(payload)
		if err != nil {
			return pathFingerprintDetails{}, false
		}
		fingerprint := "git:" + sumBytes(serialized)
		stagedEntries, _ := payload["dirty_state"].(map[string]any)["staged"].([]map[string]string)
		if stagedEntries == nil {
			stagedEntries = []map[string]string{}
			if rawStaged, ok := dirtyState["staged"].([]map[string]string); ok {
				stagedEntries = rawStaged
			}
		}
		unstagedEntries, _ := dirtyState["unstaged"].([]map[string]string)
		if unstagedEntries == nil {
			unstagedEntries = []map[string]string{}
		}
		untrackedRaw, _ := dirtyState["untracked"].([][]string)
		untrackedPaths := make([]string, 0, len(untrackedRaw))
		for _, entry := range untrackedRaw {
			if len(entry) == 0 {
				continue
			}
			rel := filepath.ToSlash(strings.TrimSpace(entry[0]))
			if rel != "" {
				untrackedPaths = append(untrackedPaths, rel)
			}
		}
		dirtyPaths, _ := gitDirtyPathsFromStatusSummary(relPath, status)
		scopeFiles, _ := gitDirtyFastpathCaptureFiles(topLevel, relPath)
		storeGitDirtyFastpathMemo(
			topLevel,
			relPath,
			fingerprint,
			head,
			statusDigest,
			extractGitStatusStage(status),
			gitIndexStateSignature(topLevel),
			dirtyPaths,
			scopeFiles,
			stagedEntries,
			unstagedEntries,
			untrackedPaths,
			worktreeSignatures,
			now,
		)
		_ = writeGitFingerprintManifest(manifestPath, gitFingerprintManifest{
			TopLevel:     topLevel,
			Head:         head,
			RelPath:      relPath,
			StatusDigest: statusDigest,
			Fingerprint:  fingerprint,
			DirtyPaths:   dirtyPaths,
			ScopeFiles:   scopeFiles,
			CleanFiles:   nil,
			IndexSig:     gitIndexStateSignature(topLevel),
		})
		return pathFingerprintDetails{
			Fingerprint: fingerprint,
			DirtyPaths:  dirtyPaths,
		}, true
	}
	serialized, err := json.Marshal(payload)
	if err != nil {
		return pathFingerprintDetails{}, false
	}
	fingerprint := "git:" + sumBytes(serialized)
	_ = writeGitFingerprintManifest(manifestPath, gitFingerprintManifest{
		TopLevel:     topLevel,
		Head:         head,
		RelPath:      relPath,
		StatusDigest: statusDigest,
		Fingerprint:  fingerprint,
		CleanFiles:   gitCleanFastpathManifestFiles(topLevel, relPath, clean),
		IndexSig:     gitCleanFastpathManifestIndexSignature(topLevel, clean),
	})
	if clean {
		storeGitFingerprintMemo(topLevel, relPath, statusDigest, fingerprint, now)
		storeGitCleanFastpathMemo(
			topLevel,
			relPath,
			fingerprint,
			gitHeadStateSignature(topLevel),
			gitCleanFastpathManifestIndexSignature(topLevel, true),
			gitCleanFastpathManifestFiles(topLevel, relPath, true),
			now,
		)
		cleanFiles := gitCleanFastpathManifestFiles(topLevel, relPath, true)
		return pathFingerprintDetails{
			Fingerprint:        fingerprint,
			CleanWorktreeFiles: cleanWorktreeFilesFromManifestFiles(cleanFiles),
		}, true
	}
	dirtyPaths, _ := gitDirtyPathsFromStatusSummary(relPath, status)
	return pathFingerprintDetails{
		Fingerprint: fingerprint,
		DirtyPaths:  dirtyPaths,
	}, true
}

func gitDirtySubsetStatusSummary(gitPath, topLevel, relPath string, now time.Time) (string, bool) {
	candidate, ok := loadGitDirtyFastpathMemoClone(topLevel, relPath, now, true, true, true, true)
	if !ok {
		manifestPath := gitFingerprintManifestPath(topLevel, relPath)
		manifest, manifestOK := loadGitFingerprintManifest(manifestPath)
		if !manifestOK || strings.TrimSpace(manifest.Fingerprint) == "" || strings.TrimSpace(manifest.Head) == "" || strings.TrimSpace(manifest.IndexSig) == "" || len(manifest.DirtyPaths) == 0 || len(manifest.ScopeFiles) == 0 {
			return "", false
		}
		candidate = gitDirtyFastpathMemoEntry{
			TopLevel:    manifest.TopLevel,
			RelPath:     manifest.RelPath,
			Fingerprint: manifest.Fingerprint,
			Head:        manifest.Head,
			IndexSig:    manifest.IndexSig,
			DirtyPaths:  append([]string(nil), manifest.DirtyPaths...),
			ScopeFiles:  manifest.ScopeFiles,
			ExpiresAt:   now.Add(gitFingerprintMemoTTL),
		}
	}
	if strings.TrimSpace(candidate.Head) == "" || strings.TrimSpace(candidate.IndexSig) == "" || len(candidate.DirtyPaths) == 0 || len(candidate.ScopeFiles) == 0 {
		return "", false
	}
	if currentGitFingerprintHead(gitPath, topLevel, topLevel, relPath) != strings.TrimSpace(candidate.Head) {
		return "", false
	}
	if gitIndexStateSignature(topLevel) != strings.TrimSpace(candidate.IndexSig) {
		return "", false
	}
	currentFiles, ok := gitDirtyFastpathCaptureFiles(topLevel, relPath)
	if !ok {
		return "", false
	}
	changedPaths := changedScopePaths(candidate.ScopeFiles, currentFiles)
	if len(changedPaths) == 0 || !allPathsInSet(changedPaths, candidate.DirtyPaths) {
		return "", false
	}
	if status, ok := reconstructedDirtySubsetStatus(topLevel, candidate, currentFiles); ok {
		return status, true
	}
	status, err := gitStatusSummaryForDirtyPaths(gitPath, topLevel, relPath, candidate.DirtyPaths)
	if err != nil {
		return "", false
	}
	return status, true
}

func reconstructedDirtySubsetStatus(topLevel string, candidate gitDirtyFastpathMemoEntry, currentFiles map[string]string) (string, bool) {
	if len(candidate.StagedEntries) == 0 && len(candidate.UnstagedEntries) == 0 && len(candidate.UntrackedPaths) == 0 {
		return "", false
	}
	stagedEntries := make([]gitNameStatusEntry, 0, len(candidate.StagedEntries))
	stagedWorktreeMismatches := []string{}
	for _, entry := range candidate.StagedEntries {
		if entry == nil {
			continue
		}
		code := strings.TrimSpace(entry["code"])
		rel := filepath.ToSlash(strings.TrimSpace(entry["path"]))
		if code == "" || rel == "" {
			continue
		}
		stagedEntries = append(stagedEntries, gitNameStatusEntry{
			code:       code,
			path:       rel,
			sourcePath: filepath.ToSlash(strings.TrimSpace(entry["source_path"])),
		})
		if _, present := currentFiles[rel]; present {
			_, gitBlobOID := worktreeContentSignatureAndBlobOID(
				filepath.Join(topLevel, filepath.FromSlash(rel)),
			)
			if blobOID := strings.TrimSpace(entry["blob_oid"]); blobOID != "" && strings.TrimSpace(gitBlobOID) != blobOID {
				stagedWorktreeMismatches = append(stagedWorktreeMismatches, rel)
			}
		}
	}
	unstagedEntries := make([]gitNameStatusEntry, 0, len(candidate.UnstagedEntries))
	unstagedSeen := map[string]struct{}{}
	for _, entry := range candidate.UnstagedEntries {
		if entry == nil {
			continue
		}
		rel := filepath.ToSlash(strings.TrimSpace(entry["path"]))
		if rel == "" {
			continue
		}
		contentSignature, gitBlobOID, _ := worktreeContentSignatureWithCache(
			filepath.Join(topLevel, filepath.FromSlash(rel)),
			candidate.WorktreeSignatures[rel],
		)
		_, present := currentFiles[rel]
		if contentSignature == "missing" || !present {
			unstagedEntries = append(unstagedEntries, gitNameStatusEntry{code: "D", path: rel})
			continue
		}
		baseBlobOID := strings.TrimSpace(entry["base_blob_oid"])
		if baseBlobOID != "" && strings.TrimSpace(gitBlobOID) == baseBlobOID {
			continue
		}
		code := strings.TrimSpace(entry["code"])
		if code == "" {
			code = "M"
		}
		unstagedEntries = append(unstagedEntries, gitNameStatusEntry{code: code, path: rel})
		unstagedSeen[rel] = struct{}{}
	}
	for _, rel := range stagedWorktreeMismatches {
		if _, seen := unstagedSeen[rel]; seen {
			continue
		}
		unstagedEntries = append(unstagedEntries, gitNameStatusEntry{code: "M", path: rel})
	}
	untrackedPaths := make([]string, 0, len(candidate.UntrackedPaths))
	for _, rel := range candidate.UntrackedPaths {
		rel = filepath.ToSlash(strings.TrimSpace(rel))
		if rel == "" {
			continue
		}
		if _, ok := currentFiles[rel]; !ok {
			continue
		}
		contentSignature, _, _ := worktreeContentSignatureWithCache(
			filepath.Join(topLevel, filepath.FromSlash(rel)),
			candidate.WorktreeSignatures[rel],
		)
		if contentSignature == "missing" || contentSignature == "unreadable" || contentSignature == "symlink" || contentSignature == "nonfile" {
			continue
		}
		untrackedPaths = append(untrackedPaths, rel)
	}
	return encodedGitStatusSummary(stagedEntries, unstagedEntries, untrackedPaths)
}

func changedScopePaths(previous, current map[string]string) []string {
	if len(previous) == 0 || len(current) == 0 {
		return nil
	}
	seen := map[string]struct{}{}
	for rel, prev := range previous {
		if current[rel] != prev {
			seen[rel] = struct{}{}
		}
	}
	for rel, cur := range current {
		if previous[rel] != cur {
			seen[rel] = struct{}{}
		}
	}
	if len(seen) == 0 {
		return nil
	}
	paths := make([]string, 0, len(seen))
	for rel := range seen {
		paths = append(paths, rel)
	}
	slices.Sort(paths)
	return paths
}

func allPathsInSet(paths, setPaths []string) bool {
	if len(paths) == 0 || len(setPaths) == 0 {
		return false
	}
	set := make(map[string]struct{}, len(setPaths))
	for _, rel := range setPaths {
		rel = filepath.ToSlash(strings.TrimSpace(rel))
		if rel == "" {
			continue
		}
		set[rel] = struct{}{}
	}
	for _, rel := range paths {
		rel = filepath.ToSlash(strings.TrimSpace(rel))
		if rel == "" {
			return false
		}
		if _, ok := set[rel]; !ok {
			return false
		}
	}
	return true
}

func gitStatusSummaryForDirtyPaths(gitPath, topLevel, relPath string, dirtyPaths []string) (string, error) {
	if len(dirtyPaths) == 0 {
		return "", fmt.Errorf("empty dirty path set")
	}
	pathspec := make([]string, 0, len(dirtyPaths))
	scopePrefix := filepath.ToSlash(strings.TrimSpace(relPath))
	for _, rel := range dirtyPaths {
		rel = filepath.ToSlash(strings.TrimSpace(rel))
		if rel == "" || rel == "." {
			continue
		}
		if scopePrefix != "" && scopePrefix != "." {
			rel = filepath.ToSlash(filepath.Join(scopePrefix, filepath.FromSlash(rel)))
		}
		pathspec = append(pathspec, rel)
	}
	if len(pathspec) == 0 {
		return "", fmt.Errorf("empty dirty pathspec")
	}
	args := []string{"--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=no", "--"}
	args = append(args, pathspec...)
	output, err := runGitFunc(gitPath, topLevel, args...)
	if err != nil {
		return "", err
	}
	staged, unstaged, untracked, err := gitStatusPayloadFromPorcelain(output)
	if err != nil {
		return "", err
	}
	statusPayload := map[string]string{
		"staged":    staged,
		"unstaged":  unstaged,
		"untracked": untracked,
	}
	encoded, err := json.Marshal(statusPayload)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func currentGitFingerprintHead(gitPath, path, topLevel, relPath string) string {
	if relPath == "." || relPath == "" {
		if headSig := gitHeadStateSignature(topLevel); headSig != "" {
			return strings.TrimSpace(headSig)
		}
	}
	headRef := "HEAD"
	if relPath == "." || relPath == "" {
		headRef = "HEAD^{tree}"
	}
	head, err := runGitFunc(gitPath, path, "rev-parse", headRef)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(head)
}

func gitScopeFromFilesystem(path string) (string, string, bool) {
	if strings.TrimSpace(path) == "" {
		return "", "", false
	}
	scopePath := filepath.Clean(path)
	info, err := os.Stat(scopePath)
	if err == nil && !info.IsDir() {
		scopePath = filepath.Dir(scopePath)
	}
	if abs, err := filepath.Abs(scopePath); err == nil {
		scopePath = abs
	}
	start := scopePath
	for {
		gitMarker := filepath.Join(scopePath, ".git")
		if _, err := os.Stat(gitMarker); err == nil {
			relPath := "."
			if rel, err := filepath.Rel(scopePath, start); err == nil && rel != "" && rel != "." {
				relPath = filepath.ToSlash(rel)
			}
			return scopePath, relPath, true
		}
		parent := filepath.Dir(scopePath)
		if parent == scopePath {
			return "", "", false
		}
		scopePath = parent
	}
}

func gitDirtyPathsFromStatusSummary(relPath, encodedStatus string) ([]string, error) {
	var status gitStatusPayload
	if err := json.Unmarshal([]byte(encodedStatus), &status); err != nil {
		return nil, err
	}
	normalize := func(path string) string {
		path = filepath.ToSlash(strings.TrimSpace(path))
		if path == "" {
			return ""
		}
		if relPath == "" || relPath == "." {
			return path
		}
		prefix := filepath.ToSlash(strings.TrimSpace(relPath))
		if path == prefix {
			return "."
		}
		if strings.HasPrefix(path, prefix+"/") {
			return strings.TrimPrefix(path, prefix+"/")
		}
		return ""
	}
	seen := map[string]struct{}{}
	addPath := func(path string) {
		normalized := normalize(path)
		if normalized == "" {
			return
		}
		seen[normalized] = struct{}{}
	}
	stagedEntries, _ := parseGitNameStatus(status.Staged)
	unstagedEntries, _ := parseGitNameStatus(status.Unstaged)
	for _, entry := range stagedEntries {
		addPath(entry.path)
		if entry.sourcePath != "" {
			addPath(entry.sourcePath)
		}
	}
	for _, entry := range unstagedEntries {
		addPath(entry.path)
		if entry.sourcePath != "" {
			addPath(entry.sourcePath)
		}
	}
	for _, rel := range parseGitUntracked(status.Untracked) {
		addPath(rel)
	}
	if len(seen) == 0 {
		return nil, nil
	}
	paths := make([]string, 0, len(seen))
	for rel := range seen {
		paths = append(paths, rel)
	}
	slices.Sort(paths)
	return paths, nil
}

func extractGitStatusStage(encodedStatus string) string {
	var status gitStatusPayload
	if err := json.Unmarshal([]byte(encodedStatus), &status); err != nil {
		return ""
	}
	return strings.TrimSpace(status.Staged)
}

func gitDirtyState(gitPath, topLevel, relPath, encodedStatus string) (map[string]any, error) {
	state, _, err := gitDirtyStateWithCache(gitPath, topLevel, relPath, encodedStatus, nil)
	return state, err
}

func gitDirtyStateWithCache(gitPath, topLevel, relPath, encodedStatus string, previous *gitDirtyFastpathMemoEntry) (map[string]any, map[string]worktreeSignatureMemoEntry, error) {
	var status gitStatusPayload
	if err := json.Unmarshal([]byte(encodedStatus), &status); err != nil {
		return nil, nil, err
	}
	stagedEntries, trackedIndexPaths := parseGitNameStatus(status.Staged)
	unstagedEntries, _ := parseGitNameStatus(status.Unstaged)
	untrackedPaths := parseGitUntracked(status.Untracked)
	trackedIndexPathSet := make(map[string]struct{}, len(trackedIndexPaths)+len(unstagedEntries))
	for _, rel := range trackedIndexPaths {
		rel = filepath.ToSlash(strings.TrimSpace(rel))
		if rel != "" {
			trackedIndexPathSet[rel] = struct{}{}
		}
	}
	for _, entry := range unstagedEntries {
		rel := filepath.ToSlash(strings.TrimSpace(entry.path))
		if rel != "" {
			trackedIndexPathSet[rel] = struct{}{}
		}
	}
	trackedIndexPaths = trackedIndexPaths[:0]
	for rel := range trackedIndexPathSet {
		trackedIndexPaths = append(trackedIndexPaths, rel)
	}
	slices.Sort(trackedIndexPaths)
	indexRecords := map[string]map[string]string{}
	previousIndexSig := ""
	previousStagedByPath := map[string]map[string]string{}
	previousWorktree := map[string]worktreeSignatureMemoEntry{}
	if previous != nil {
		previousIndexSig = strings.TrimSpace(previous.IndexSig)
		for _, entry := range previous.StagedEntries {
			if entry == nil {
				continue
			}
			rel := filepath.ToSlash(strings.TrimSpace(entry["path"]))
			if rel == "" {
				continue
			}
			cloned := make(map[string]string, len(entry))
			for key, value := range entry {
				cloned[key] = value
			}
			previousStagedByPath[rel] = cloned
		}
		for key, value := range previous.WorktreeSignatures {
			previousWorktree[key] = value
		}
	}
	currentIndexSig := gitIndexStateSignature(topLevel)
	reuseStaged := previous != nil &&
		previousIndexSig != "" &&
		currentIndexSig != "" &&
		previousIndexSig == currentIndexSig
	if !reuseStaged {
		var err error
		indexRecords, err = gitIndexBlobOIDs(gitPath, topLevel, trackedIndexPaths)
		if err != nil {
			return nil, nil, err
		}
	}
	staged := make([]map[string]string, 0, len(stagedEntries))
	worktreeSignatures := map[string]worktreeSignatureMemoEntry{}
	for _, entry := range stagedEntries {
		record := map[string]string{
			"code": entry.code,
			"path": entry.path,
		}
		_, _, cachedSignature := worktreeContentSignatureWithCache(
			filepath.Join(topLevel, filepath.FromSlash(entry.path)),
			previousWorktree[entry.path],
		)
		worktreeSignatures[entry.path] = cachedSignature
		if entry.sourcePath != "" {
			record["source_path"] = entry.sourcePath
		}
		if reuseStaged {
			if cachedRecord := previousStagedByPath[entry.path]; cachedRecord != nil {
				if mode := strings.TrimSpace(cachedRecord["mode"]); mode != "" {
					record["mode"] = mode
				}
				if blobOID := strings.TrimSpace(cachedRecord["blob_oid"]); blobOID != "" {
					record["blob_oid"] = blobOID
				}
				if stage := strings.TrimSpace(cachedRecord["stage"]); stage != "" {
					record["stage"] = stage
				}
			}
		} else if index := indexRecords[entry.path]; index != nil {
			record["mode"] = index["mode"]
			record["blob_oid"] = index["blob_oid"]
			record["stage"] = index["stage"]
		}
		staged = append(staged, record)
	}
	unstaged := make([]map[string]string, 0, len(unstagedEntries))
	for _, entry := range unstagedEntries {
		contentSignature, gitBlobOID, cachedSignature := worktreeContentSignatureWithCache(
			filepath.Join(topLevel, filepath.FromSlash(entry.path)),
			previousWorktree[entry.path],
		)
		worktreeSignatures[entry.path] = cachedSignature
		record := map[string]string{
			"code":      entry.code,
			"path":      entry.path,
			"signature": contentSignature,
		}
		if strings.TrimSpace(gitBlobOID) != "" {
			record["git_blob_oid"] = strings.TrimSpace(gitBlobOID)
		}
		if index := indexRecords[entry.path]; index != nil {
			if blobOID := strings.TrimSpace(index["blob_oid"]); blobOID != "" {
				record["base_blob_oid"] = blobOID
			}
		}
		unstaged = append(unstaged, record)
	}
	untracked := make([][]string, 0, len(untrackedPaths))
	for _, rel := range untrackedPaths {
		contentSignature, _, cachedSignature := worktreeContentSignatureWithCache(
			filepath.Join(topLevel, filepath.FromSlash(rel)),
			previousWorktree[rel],
		)
		worktreeSignatures[rel] = cachedSignature
		untracked = append(untracked, []string{
			rel,
			contentSignature,
		})
	}
	return map[string]any{
		"staged":    staged,
		"unstaged":  unstaged,
		"untracked": untracked,
	}, worktreeSignatures, nil
}

type gitNameStatusEntry struct {
	code       string
	path       string
	sourcePath string
}

func parseGitNameStatus(encoded string) ([]gitNameStatusEntry, []string) {
	entries := []gitNameStatusEntry{}
	tracked := []string{}
	if strings.TrimSpace(encoded) == "" {
		return entries, tracked
	}
	parts := strings.Split(encoded, "\x00")
	for i := 0; i < len(parts); i++ {
		part := strings.TrimSpace(parts[i])
		if part == "" {
			continue
		}
		var entry gitNameStatusEntry
		if strings.Contains(part, "\t") {
			fields := strings.SplitN(part, "\t", 2)
			if len(fields) != 2 {
				continue
			}
			entry.code = strings.TrimSpace(fields[0])
			entry.path = filepath.ToSlash(strings.TrimSpace(fields[1]))
			if strings.HasPrefix(entry.code, "R") || strings.HasPrefix(entry.code, "C") {
				if i+1 < len(parts) {
					entry.sourcePath = entry.path
					i++
					entry.path = filepath.ToSlash(strings.TrimSpace(parts[i]))
				}
			}
		} else {
			entry.code = part
			if i+1 >= len(parts) {
				continue
			}
			i++
			entry.path = filepath.ToSlash(strings.TrimSpace(parts[i]))
			if strings.HasPrefix(entry.code, "R") || strings.HasPrefix(entry.code, "C") {
				entry.sourcePath = entry.path
				if i+1 < len(parts) {
					i++
					entry.path = filepath.ToSlash(strings.TrimSpace(parts[i]))
				}
			}
		}
		if entry.code == "" || entry.path == "" {
			continue
		}
		entries = append(entries, entry)
		if entry.path != "" {
			tracked = append(tracked, entry.path)
		}
	}
	return entries, tracked
}

func parseGitUntracked(encoded string) []string {
	values := []string{}
	if strings.TrimSpace(encoded) == "" {
		return values
	}
	for _, part := range strings.Split(encoded, "\x00") {
		part = filepath.ToSlash(strings.TrimSpace(part))
		if part == "" {
			continue
		}
		values = append(values, part)
	}
	return values
}

func gitIndexBlobOIDs(gitPath, topLevel string, relativePaths []string) (map[string]map[string]string, error) {
	records := map[string]map[string]string{}
	if len(relativePaths) == 0 {
		return records, nil
	}
	args := []string{"--no-optional-locks", "ls-files", "-s", "-z", "--"}
	args = append(args, relativePaths...)
	output, err := runGitFunc(gitPath, topLevel, args...)
	if err != nil {
		return nil, err
	}
	for _, rawEntry := range strings.Split(output, "\x00") {
		rawEntry = strings.TrimSpace(rawEntry)
		if rawEntry == "" {
			continue
		}
		meta, rel, ok := strings.Cut(rawEntry, "\t")
		if !ok {
			continue
		}
		fields := strings.Fields(meta)
		if len(fields) != 3 {
			continue
		}
		records[filepath.ToSlash(strings.TrimSpace(rel))] = map[string]string{
			"mode":     fields[0],
			"blob_oid": fields[1],
			"stage":    fields[2],
		}
	}
	return records, nil
}

func worktreeContentSignature(path string) string {
	info, err := os.Lstat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "missing"
		}
		return "unreadable"
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return "symlink"
	}
	if !info.Mode().IsRegular() {
		return "nonfile"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "unreadable"
	}
	return "sha256:" + sumBytes(data)
}

func gitBlobOIDForBytes(data []byte) string {
	header := fmt.Sprintf("blob %d\x00", len(data))
	sum := sha1SumBytes(append([]byte(header), data...))
	return sum
}

func sha1SumBytes(data []byte) string {
	h := sha1.New()
	_, _ = h.Write(data)
	return hex.EncodeToString(h.Sum(nil))
}

func worktreeContentSignatureAndBlobOID(path string) (string, string) {
	info, err := os.Lstat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "missing", "missing"
		}
		return "unreadable", "unreadable"
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return "symlink", "symlink"
	}
	if !info.Mode().IsRegular() {
		return "nonfile", "nonfile"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "unreadable", "unreadable"
	}
	return "sha256:" + sumBytes(data), gitBlobOIDForBytes(data)
}

func worktreeContentSignatureWithCache(path string, cached worktreeSignatureMemoEntry) (string, string, worktreeSignatureMemoEntry) {
	stateSignature := fileStateSignature(path)
	if strings.TrimSpace(cached.StateSignature) != "" &&
		strings.TrimSpace(cached.ContentSignature) != "" &&
		strings.TrimSpace(cached.StateSignature) == strings.TrimSpace(stateSignature) {
		return strings.TrimSpace(cached.ContentSignature), strings.TrimSpace(cached.GitBlobOID), worktreeSignatureMemoEntry{
			StateSignature:   strings.TrimSpace(cached.StateSignature),
			ContentSignature: strings.TrimSpace(cached.ContentSignature),
			GitBlobOID:       strings.TrimSpace(cached.GitBlobOID),
		}
	}
	contentSignature, gitBlobOID := worktreeContentSignatureAndBlobOID(path)
	return contentSignature, gitBlobOID, worktreeSignatureMemoEntry{
		StateSignature:   strings.TrimSpace(stateSignature),
		ContentSignature: strings.TrimSpace(contentSignature),
		GitBlobOID:       strings.TrimSpace(gitBlobOID),
	}
}

func gitCleanFastpathManifestFiles(topLevel, relPath string, clean bool) map[string]string {
	if !clean {
		return nil
	}
	files, ok := gitDirtyFastpathCaptureFiles(topLevel, relPath)
	if !ok {
		return nil
	}
	return files
}

func gitCleanFastpathManifestIndexSignature(topLevel string, clean bool) string {
	if !clean {
		return ""
	}
	return gitIndexStateSignature(topLevel)
}

func gitCleanFastpathFingerprint(gitPath, topLevel, relPath string, manifest gitFingerprintManifest) (string, bool) {
	if strings.TrimSpace(manifest.Fingerprint) == "" || strings.TrimSpace(manifest.Head) == "" || strings.TrimSpace(manifest.IndexSig) == "" || len(manifest.CleanFiles) == 0 {
		return "", false
	}
	if manifest.TopLevel != topLevel || manifest.RelPath != relPath {
		return "", false
	}
	headSig := gitHeadStateSignature(topLevel)
	if headSig == "" {
		return "", false
	}
	if headSig == strings.TrimSpace(manifest.Head) {
		return gitCleanFastpathFingerprintFromState(topLevel, relPath, manifest.Fingerprint, headSig, manifest.IndexSig, manifest.CleanFiles)
	}
	if gitIndexStateSignature(topLevel) != strings.TrimSpace(manifest.IndexSig) {
		return "", false
	}
	currentFiles, ok := gitDirtyFastpathCaptureFiles(topLevel, relPath)
	if !ok {
		return "", false
	}
	if !mapsEqualStringString(currentFiles, manifest.CleanFiles) {
		return "", false
	}
	head := currentGitFingerprintHead(gitPath, topLevel, topLevel, relPath)
	if head == "" {
		return "", false
	}
	if strings.TrimSpace(head) != strings.TrimSpace(manifest.Head) {
		return "", false
	}
	storeGitCleanFastpathMemo(topLevel, relPath, manifest.Fingerprint, headSig, manifest.IndexSig, manifest.CleanFiles, time.Now())
	return strings.TrimSpace(manifest.Fingerprint), true
}

func gitCleanFastpathFingerprintFromState(topLevel, relPath, fingerprint, headSig, indexSig string, cleanFiles map[string]string) (string, bool) {
	if strings.TrimSpace(fingerprint) == "" || strings.TrimSpace(headSig) == "" || strings.TrimSpace(indexSig) == "" || len(cleanFiles) == 0 {
		return "", false
	}
	if gitHeadStateSignature(topLevel) != strings.TrimSpace(headSig) {
		return "", false
	}
	if gitIndexStateSignature(topLevel) != strings.TrimSpace(indexSig) {
		return "", false
	}
	currentFiles, ok := gitDirtyFastpathCaptureFiles(topLevel, relPath)
	if !ok {
		return "", false
	}
	if !mapsEqualStringString(currentFiles, cleanFiles) {
		return "", false
	}
	return strings.TrimSpace(fingerprint), true
}

func gitDirtyFastpathCaptureFiles(topLevel, relPath string) (map[string]string, bool) {
	scopeRoot := topLevel
	if relPath != "" && relPath != "." {
		scopeRoot = filepath.Join(topLevel, filepath.FromSlash(relPath))
	}
	info, err := os.Lstat(scopeRoot)
	if err != nil || info.Mode()&os.ModeSymlink != 0 {
		return nil, false
	}
	current := map[string]string{}
	if info.Mode().IsRegular() {
		if shouldIgnoreDir(filepath.ToSlash(strings.TrimSpace(relPath))) || matchesEphemeralIgnore(filepath.Base(scopeRoot)) {
			return current, true
		}
		rel := filepath.ToSlash(strings.TrimSpace(relPath))
		if rel == "" {
			rel = filepath.Base(scopeRoot)
		}
		current[rel] = fileStateSignatureFromInfo(info)
		return current, true
	}
	if !info.IsDir() {
		return nil, false
	}
	topPrefix := filepath.Clean(topLevel)
	if !strings.HasSuffix(topPrefix, string(filepath.Separator)) {
		topPrefix += string(filepath.Separator)
	}
	err = filepath.WalkDir(scopeRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == scopeRoot {
			return nil
		}
		relToTop := ""
		cleanPath := filepath.Clean(path)
		if strings.HasPrefix(cleanPath, topPrefix) {
			relToTop = filepath.ToSlash(cleanPath[len(topPrefix):])
		} else {
			relToTopRaw, relErr := filepath.Rel(topLevel, path)
			if relErr != nil {
				return relErr
			}
			relToTop = filepath.ToSlash(relToTopRaw)
		}
		if d.IsDir() {
			if shouldIgnoreDir(relToTop) {
				return filepath.SkipDir
			}
			return nil
		}
		if matchesEphemeralIgnore(relToTop) {
			return nil
		}
		mode := d.Type()
		if mode&os.ModeSymlink != 0 {
			return nil
		}
		if mode.IsRegular() {
			info, statErr := d.Info()
			if statErr != nil {
				return statErr
			}
			current[relToTop] = fileStateSignatureFromInfo(info)
		} else {
			info, statErr := d.Info()
			if statErr != nil {
				return statErr
			}
			if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
				return nil
			}
			current[relToTop] = fileStateSignatureFromInfo(info)
		}
		if len(current) > gitFingerprintFastpathFileThreshold {
			return errFingerprintBudgetExceeded
		}
		return nil
	})
	if errors.Is(err, errFingerprintBudgetExceeded) {
		return nil, false
	}
	if err != nil {
		return nil, false
	}
	return current, true
}

func dirtyPathStateSignaturesMatch(topLevel string, dirtyPaths []string, worktreeSignatures map[string]worktreeSignatureMemoEntry) bool {
	if len(dirtyPaths) == 0 {
		return true
	}
	if len(worktreeSignatures) == 0 {
		return false
	}
	for _, rel := range dirtyPaths {
		rel = filepath.ToSlash(strings.TrimSpace(rel))
		if rel == "" || rel == "." {
			continue
		}
		cached, ok := worktreeSignatures[rel]
		cachedContentSignature := strings.TrimSpace(cached.ContentSignature)
		if !ok {
			continue
		}
		path := filepath.Join(topLevel, filepath.FromSlash(rel))
		if cachedContentSignature != "" {
			current, _ := worktreeContentSignatureAndBlobOID(path)
			if strings.TrimSpace(current) != cachedContentSignature {
				return false
			}
			continue
		}
		if cachedStateSignature := strings.TrimSpace(cached.StateSignature); cachedStateSignature != "" {
			current := fileStateSignature(path)
			if strings.TrimSpace(current) != cachedStateSignature {
				return false
			}
		}
	}
	return true
}

func gitDirtyFastpathFingerprintFromState(gitPath, topLevel, relPath, fingerprint, head, indexSig string, scopeFiles map[string]string, dirtyPaths []string, worktreeSignatures map[string]worktreeSignatureMemoEntry) (string, []string, bool) {
	if strings.TrimSpace(fingerprint) == "" || strings.TrimSpace(head) == "" || strings.TrimSpace(indexSig) == "" || len(scopeFiles) == 0 || len(dirtyPaths) == 0 {
		return "", nil, false
	}
	if currentGitFingerprintHead(gitPath, topLevel, topLevel, relPath) != strings.TrimSpace(head) {
		return "", nil, false
	}
	if gitIndexStateSignature(topLevel) != strings.TrimSpace(indexSig) {
		return "", nil, false
	}
	if !dirtyPathStateSignaturesMatch(topLevel, dirtyPaths, worktreeSignatures) {
		return "", nil, false
	}
	currentFiles, ok := gitDirtyFastpathCaptureFiles(topLevel, relPath)
	if !ok {
		return "", nil, false
	}
	if !mapsEqualStringString(currentFiles, scopeFiles) {
		return "", nil, false
	}
	return strings.TrimSpace(fingerprint), append([]string(nil), dirtyPaths...), true
}

func gitIndexStateSignature(topLevel string) string {
	gitDir, err := gitDirPath(topLevel)
	if err != nil {
		return ""
	}
	indexPath := filepath.Join(gitDir, "index")
	stamp, ok := manifestPathStamp(indexPath)
	if !ok {
		return ""
	}
	gitIndexStateSignatureCache.Lock()
	cached, cachedOK := gitIndexStateSignatureCache.entries[indexPath]
	gitIndexStateSignatureCache.Unlock()
	if cachedOK && cached.Stamp == stamp {
		return cached.Signature
	}
	signature := fmt.Sprintf("meta:%d:%d", stamp.Size, stamp.MTimeNS)
	gitIndexStateSignatureCache.Lock()
	if gitIndexStateSignatureCache.entries == nil {
		gitIndexStateSignatureCache.entries = make(map[string]cachedStateSignature)
	}
	gitIndexStateSignatureCache.entries[indexPath] = cachedStateSignature{Stamp: stamp, Signature: signature}
	gitIndexStateSignatureCache.Unlock()
	return signature
}

func gitHeadStateSignature(topLevel string) string {
	gitDir, err := gitDirPath(topLevel)
	if err != nil {
		return ""
	}
	headPath := filepath.Join(gitDir, "HEAD")
	headStamp, ok := manifestPathStamp(headPath)
	if !ok {
		return ""
	}
	gitHeadStateSignatureCache.Lock()
	cached, cachedOK := gitHeadStateSignatureCache.entries[headPath]
	gitHeadStateSignatureCache.Unlock()
	if cachedOK && cached.HeadStamp == headStamp && cached.RefPath == "" {
		return cached.Signature
	}
	headBytes, err := os.ReadFile(headPath)
	if err != nil {
		return ""
	}
	headText := strings.TrimSpace(string(headBytes))
	if headText == "" {
		return ""
	}
	if strings.HasPrefix(headText, "ref: ") {
		refPath := strings.TrimSpace(strings.TrimPrefix(headText, "ref: "))
		resolvedRefPath := filepath.Join(gitDir, filepath.FromSlash(refPath))
		refStamp, refOK := manifestPathStamp(resolvedRefPath)
		if cachedOK && cached.HeadStamp == headStamp && cached.RefPath == resolvedRefPath && refOK && cached.RefStamp == refStamp {
			return cached.Signature
		}
		refBytes, err := os.ReadFile(resolvedRefPath)
		if err != nil {
			return "headref:" + headText + ":missing"
		}
		signature := "headref:" + headText + ":" + strings.TrimSpace(string(refBytes))
		if refOK {
			gitHeadStateSignatureCache.Lock()
			if gitHeadStateSignatureCache.entries == nil {
				gitHeadStateSignatureCache.entries = make(map[string]cachedHeadSignature)
			}
			gitHeadStateSignatureCache.entries[headPath] = cachedHeadSignature{
				HeadStamp: headStamp,
				RefPath:   resolvedRefPath,
				RefStamp:  refStamp,
				Signature: signature,
			}
			gitHeadStateSignatureCache.Unlock()
		}
		return signature
	}
	signature := "head:" + headText
	gitHeadStateSignatureCache.Lock()
	if gitHeadStateSignatureCache.entries == nil {
		gitHeadStateSignatureCache.entries = make(map[string]cachedHeadSignature)
	}
	gitHeadStateSignatureCache.entries[headPath] = cachedHeadSignature{
		HeadStamp: headStamp,
		Signature: signature,
	}
	gitHeadStateSignatureCache.Unlock()
	return signature
}

func gitDirPath(topLevel string) (string, error) {
	gitDirPathCache.Lock()
	cached, ok := gitDirPathCache.entries[topLevel]
	gitDirPathCache.Unlock()
	if ok && strings.TrimSpace(cached) != "" {
		return cached, nil
	}
	gitPath := filepath.Join(topLevel, ".git")
	info, err := os.Stat(gitPath)
	if err == nil && info.IsDir() {
		gitDirPathCache.Lock()
		if gitDirPathCache.entries == nil {
			gitDirPathCache.entries = make(map[string]string)
		}
		gitDirPathCache.entries[topLevel] = gitPath
		gitDirPathCache.Unlock()
		return gitPath, nil
	}
	data, readErr := os.ReadFile(gitPath)
	if readErr != nil {
		if err != nil {
			return "", err
		}
		return "", readErr
	}
	text := strings.TrimSpace(string(data))
	const prefix = "gitdir:"
	if !strings.HasPrefix(strings.ToLower(text), prefix) {
		return "", fmt.Errorf("unsupported gitdir file format")
	}
	target := strings.TrimSpace(text[len(prefix):])
	if target == "" {
		return "", fmt.Errorf("empty gitdir target")
	}
	if !filepath.IsAbs(target) {
		target = filepath.Join(topLevel, target)
	}
	resolved := filepath.Clean(target)
	gitDirPathCache.Lock()
	if gitDirPathCache.entries == nil {
		gitDirPathCache.entries = make(map[string]string)
	}
	gitDirPathCache.entries[topLevel] = resolved
	gitDirPathCache.Unlock()
	return resolved, nil
}

func fileStateSignature(path string) string {
	info, err := os.Lstat(path)
	if err != nil {
		return "missing"
	}
	return fileStateSignatureFromInfo(info)
}

func fileStateSignatureFromInfo(info os.FileInfo) string {
	if info == nil {
		return "missing"
	}
	return fmt.Sprintf("meta:%d:%d", info.Size(), info.ModTime().UTC().UnixNano())
}

func mapsEqualStringString(left, right map[string]string) bool {
	if len(left) != len(right) {
		return false
	}
	for key, leftValue := range left {
		if right[key] != leftValue {
			return false
		}
	}
	return true
}

func gitStatusSummary(gitPath, topLevel, relPath string) (string, error) {
	pathspec := gitStatusPathspec(relPath)
	statusOutput, err := runGitFunc(
		gitPath,
		topLevel,
		append([]string{"--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=no"}, pathspec...)...,
	)
	if err != nil {
		return "", err
	}
	staged, unstaged, untracked, err := gitStatusPayloadFromPorcelain(statusOutput)
	if err != nil {
		return "", err
	}
	statusPayload := map[string]string{
		"staged":    staged,
		"unstaged":  unstaged,
		"untracked": untracked,
	}
	encoded, err := json.Marshal(statusPayload)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func gitStatusPayloadFromPorcelain(output string) (string, string, string, error) {
	entries, err := parseGitPorcelainV1Z(output)
	if err != nil {
		return "", "", "", err
	}
	var staged strings.Builder
	var unstaged strings.Builder
	var untracked strings.Builder
	for _, entry := range entries {
		if entry.path == "" {
			continue
		}
		switch {
		case entry.x == '?' && entry.y == '?':
			untracked.WriteString(entry.path)
			untracked.WriteByte(0)
		default:
			if code := porcelainNameStatusCode(entry.x); code != "" {
				appendGitNameStatusEntry(&staged, code, entry)
			}
			if code := porcelainNameStatusCode(entry.y); code != "" {
				appendGitNameStatusEntry(&unstaged, code, entry)
			}
		}
	}
	return staged.String(), unstaged.String(), untracked.String(), nil
}

func encodedGitStatusSummary(stagedEntries, unstagedEntries []gitNameStatusEntry, untrackedPaths []string) (string, bool) {
	var staged strings.Builder
	var unstaged strings.Builder
	var untracked strings.Builder
	for _, entry := range stagedEntries {
		if entry.code == "" || entry.path == "" {
			continue
		}
		appendGitNameStatusEntry(&staged, entry.code, gitPorcelainEntry{
			path:       entry.path,
			sourcePath: entry.sourcePath,
		})
	}
	for _, entry := range unstagedEntries {
		if entry.code == "" || entry.path == "" {
			continue
		}
		appendGitNameStatusEntry(&unstaged, entry.code, gitPorcelainEntry{
			path:       entry.path,
			sourcePath: entry.sourcePath,
		})
	}
	for _, rel := range untrackedPaths {
		rel = filepath.ToSlash(strings.TrimSpace(rel))
		if rel == "" {
			continue
		}
		untracked.WriteString(rel)
		untracked.WriteByte(0)
	}
	statusPayload := map[string]string{
		"staged":    staged.String(),
		"unstaged":  unstaged.String(),
		"untracked": untracked.String(),
	}
	encoded, err := json.Marshal(statusPayload)
	if err != nil {
		return "", false
	}
	return string(encoded), true
}

func appendGitNameStatusEntry(builder *strings.Builder, code string, entry gitPorcelainEntry) {
	builder.WriteString(code)
	builder.WriteByte(0)
	if entry.sourcePath != "" && (strings.HasPrefix(code, "R") || strings.HasPrefix(code, "C")) {
		builder.WriteString(entry.sourcePath)
		builder.WriteByte(0)
	}
	builder.WriteString(entry.path)
	builder.WriteByte(0)
}

func porcelainNameStatusCode(code byte) string {
	switch code {
	case ' ', '?', '!':
		return ""
	default:
		return string(code)
	}
}

func parseGitPorcelainV1Z(output string) ([]gitPorcelainEntry, error) {
	if output == "" {
		return nil, nil
	}
	parts := strings.Split(output, "\x00")
	entries := make([]gitPorcelainEntry, 0, len(parts))
	for i := 0; i < len(parts); i++ {
		record := parts[i]
		if record == "" {
			continue
		}
		if len(record) < 4 {
			return nil, fmt.Errorf("malformed git porcelain entry %q", record)
		}
		entry := gitPorcelainEntry{
			x:    record[0],
			y:    record[1],
			path: filepath.ToSlash(strings.TrimSpace(record[3:])),
		}
		if entry.path == "" {
			return nil, fmt.Errorf("malformed git porcelain path in %q", record)
		}
		if entry.x == 'R' || entry.x == 'C' || entry.y == 'R' || entry.y == 'C' {
			if i+1 >= len(parts) {
				return nil, fmt.Errorf("malformed git porcelain rename/copy entry %q", record)
			}
			i++
			entry.sourcePath = entry.path
			entry.path = filepath.ToSlash(strings.TrimSpace(parts[i]))
			if entry.path == "" {
				return nil, fmt.Errorf("malformed git porcelain target path in %q", record)
			}
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func gitStatusPathspec(relPath string) []string {
	specs := make([]string, 0, 2+len(gitEphemeralExcludePathspecs))
	if relPath != "." {
		specs = append(specs, relPath)
	}
	specs = append(specs, gitEphemeralExcludePathspecs...)
	if len(specs) == 0 {
		return nil
	}
	return append([]string{"--"}, specs...)
}

func runGit(gitPath, dir string, args ...string) (string, error) {
	cmd := exec.Command(gitPath, args...)
	cmd.Dir = dir
	cmd.Env = gitCommandEnv()
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(output)))
	}
	return string(output), nil
}

func runGitExitCode(gitPath, dir string, args ...string) (int, string, error) {
	cmd := exec.Command(gitPath, args...)
	cmd.Dir = dir
	cmd.Env = gitCommandEnv()
	output, err := cmd.CombinedOutput()
	if err == nil {
		return 0, string(output), nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), string(output), nil
	}
	return -1, string(output), fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(output)))
}

func gitCommandEnv() []string {
	gitCommandEnvOnce.Do(func() {
		keys := []string{
			"HOME",
			"LANG",
			"LC_ALL",
			"LC_CTYPE",
			"PATH",
			"SSH_AUTH_SOCK",
			"TMPDIR",
			"USER",
			"LOGNAME",
			"XDG_CACHE_HOME",
			"XDG_CONFIG_HOME",
			"GIT_CONFIG_GLOBAL",
			"GIT_CONFIG_NOSYSTEM",
		}
		env := make([]string, 0, len(keys))
		for _, key := range keys {
			if value, ok := os.LookupEnv(key); ok {
				env = append(env, key+"="+value)
			}
		}
		gitCommandEnvValue = env
	})
	return gitCommandEnvValue
}

func shouldIgnoreDir(rel string) bool {
	rel = filepath.ToSlash(strings.TrimSpace(rel))
	if rel != "" && rel != "." && matchesEphemeralIgnore(rel) {
		return true
	}
	parts := strings.Split(rel, "/")
	for _, part := range parts {
		switch part {
		case ".git", ".broker", ".broker-live-tests", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".venv", "venv", "env", "node_modules", "site-packages", "build", "dist":
			return true
		}
	}
	return false
}

func matchesEphemeralIgnore(rel string) bool {
	base := path.Base(rel)
	for _, pattern := range ephemeralIgnoreFileGlobs {
		if ok, _ := path.Match(pattern, base); ok {
			return true
		}
	}
	return false
}

func gitFingerprintManifestPath(topLevel, relPath string) string {
	cacheRoot, err := userCacheDirFunc()
	if err != nil || strings.TrimSpace(cacheRoot) == "" {
		cacheRoot = os.TempDir()
	}
	scopeKey := sumBytes([]byte(topLevel + "\x00" + relPath))
	return filepath.Join(cacheRoot, "local-ai-broker", "git-fingerprint-cache", scopeKey+".json")
}

func metadataFingerprintManifestPath(root string) string {
	cacheRoot, err := userCacheDirFunc()
	if err != nil || strings.TrimSpace(cacheRoot) == "" {
		cacheRoot = os.TempDir()
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		absolute = filepath.Clean(root)
	}
	return filepath.Join(cacheRoot, "local-ai-broker", "metadata-fingerprint-cache", sumBytes([]byte(filepath.Clean(absolute)))+".json")
}

func gitScopeManifestPath(path string) string {
	cacheRoot, err := userCacheDirFunc()
	if err != nil || strings.TrimSpace(cacheRoot) == "" {
		cacheRoot = os.TempDir()
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		absolute = filepath.Clean(path)
	}
	return filepath.Join(cacheRoot, "local-ai-broker", "git-scope-cache", sumBytes([]byte(filepath.Clean(absolute)))+".json")
}

func loadGitScopeManifest(path string) (string, string, bool) {
	manifestPath := gitScopeManifestPath(path)
	if stamp, ok := manifestPathStamp(manifestPath); ok {
		gitScopeManifestCache.Lock()
		cached, cachedOK := gitScopeManifestCache.entries[manifestPath]
		gitScopeManifestCache.Unlock()
		if cachedOK && cached.Stamp == stamp && strings.TrimSpace(cached.TopLevel) != "" {
			relPath := strings.TrimSpace(cached.RelPath)
			if relPath == "" {
				relPath = "."
			}
			return strings.TrimSpace(cached.TopLevel), relPath, true
		}
	} else {
		gitScopeManifestCache.Lock()
		delete(gitScopeManifestCache.entries, manifestPath)
		gitScopeManifestCache.Unlock()
	}
	data, err := os.ReadFile(manifestPath)
	if err != nil || len(data) == 0 {
		return "", "", false
	}
	var manifest gitScopeManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return "", "", false
	}
	topLevel := strings.TrimSpace(manifest.TopLevel)
	relPath := strings.TrimSpace(manifest.RelPath)
	if topLevel == "" {
		return "", "", false
	}
	if relPath == "" {
		relPath = "."
	}
	if stamp, ok := manifestPathStamp(manifestPath); ok {
		gitScopeManifestCache.Lock()
		if gitScopeManifestCache.entries == nil {
			gitScopeManifestCache.entries = make(map[string]cachedGitScopeManifest)
		}
		gitScopeManifestCache.entries[manifestPath] = cachedGitScopeManifest{
			Stamp:    stamp,
			TopLevel: topLevel,
			RelPath:  relPath,
		}
		gitScopeManifestCache.Unlock()
	}
	return topLevel, relPath, true
}

func writeGitScopeManifest(path string, manifest gitScopeManifest) error {
	manifestPath := gitScopeManifestPath(path)
	if existingTopLevel, existingRelPath, ok := loadGitScopeManifest(path); ok {
		if existingTopLevel == strings.TrimSpace(manifest.TopLevel) {
			relPath := strings.TrimSpace(manifest.RelPath)
			if relPath == "" {
				relPath = "."
			}
			if existingRelPath == relPath {
				return nil
			}
		}
	}
	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	tmpFile, err := os.CreateTemp(filepath.Dir(manifestPath), "git-scope-*.tmp")
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
	if err := os.Rename(tmpPath, manifestPath); err != nil {
		return err
	}
	if stamp, ok := manifestPathStamp(manifestPath); ok {
		relPath := strings.TrimSpace(manifest.RelPath)
		if relPath == "" {
			relPath = "."
		}
		gitScopeManifestCache.Lock()
		if gitScopeManifestCache.entries == nil {
			gitScopeManifestCache.entries = make(map[string]cachedGitScopeManifest)
		}
		gitScopeManifestCache.entries[manifestPath] = cachedGitScopeManifest{
			Stamp:    stamp,
			TopLevel: strings.TrimSpace(manifest.TopLevel),
			RelPath:  relPath,
		}
		gitScopeManifestCache.Unlock()
	}
	return nil
}

func loadMetadataFingerprintManifest(path string) (metadataFingerprintManifest, bool) {
	data, err := os.ReadFile(path)
	if err != nil || len(data) == 0 {
		return metadataFingerprintManifest{}, false
	}
	var manifest metadataFingerprintManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return metadataFingerprintManifest{}, false
	}
	if strings.TrimSpace(manifest.Fingerprint) == "" {
		return metadataFingerprintManifest{}, false
	}
	if manifest.Files == nil {
		manifest.Files = map[string]metadataFingerprintRecord{}
	}
	return manifest, true
}

func metadataFingerprintManifestEqual(previous, next metadataFingerprintManifest) bool {
	if len(previous.Files) != len(next.Files) {
		return false
	}
	for rel, nextRecord := range next.Files {
		prevRecord, ok := previous.Files[rel]
		if !ok {
			return false
		}
		if prevRecord != nextRecord {
			return false
		}
	}
	return true
}

func writeMetadataFingerprintManifest(path string, manifest metadataFingerprintManifest) error {
	if existing, ok := loadMetadataFingerprintManifest(path); ok {
		if existing.Root == manifest.Root &&
			existing.Fingerprint == manifest.Fingerprint &&
			metadataFingerprintManifestEqual(existing, manifest) {
			return nil
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	tmpFile, err := os.CreateTemp(filepath.Dir(path), "metadata-fingerprint-*.tmp")
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
	return os.Rename(tmpPath, path)
}

func manifestPathStamp(path string) (manifestFileStamp, bool) {
	info, err := os.Stat(path)
	if err != nil {
		return manifestFileStamp{}, false
	}
	return manifestFileStamp{
		Size:    info.Size(),
		MTimeNS: info.ModTime().UTC().UnixNano(),
	}, true
}

func loadGitFingerprintManifest(path string) (gitFingerprintManifest, bool) {
	if stamp, ok := manifestPathStamp(path); ok {
		gitFingerprintManifestCache.Lock()
		cached, cachedOK := gitFingerprintManifestCache.entries[path]
		gitFingerprintManifestCache.Unlock()
		if cachedOK && cached.Stamp == stamp && strings.TrimSpace(cached.Manifest.Fingerprint) != "" {
			return cached.Manifest, true
		}
	} else {
		gitFingerprintManifestCache.Lock()
		delete(gitFingerprintManifestCache.entries, path)
		gitFingerprintManifestCache.Unlock()
	}
	data, err := os.ReadFile(path)
	if err != nil || len(data) == 0 {
		return gitFingerprintManifest{}, false
	}
	var manifest gitFingerprintManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return gitFingerprintManifest{}, false
	}
	if strings.TrimSpace(manifest.Fingerprint) == "" {
		return gitFingerprintManifest{}, false
	}
	if stamp, ok := manifestPathStamp(path); ok {
		gitFingerprintManifestCache.Lock()
		if gitFingerprintManifestCache.entries == nil {
			gitFingerprintManifestCache.entries = make(map[string]cachedGitFingerprintManifest)
		}
		gitFingerprintManifestCache.entries[path] = cachedGitFingerprintManifest{Stamp: stamp, Manifest: manifest}
		gitFingerprintManifestCache.Unlock()
	}
	return manifest, true
}

func writeGitFingerprintManifest(path string, manifest gitFingerprintManifest) error {
	if existing, ok := loadGitFingerprintManifest(path); ok {
		if existing.TopLevel == manifest.TopLevel &&
			existing.Head == manifest.Head &&
			existing.RelPath == manifest.RelPath &&
			existing.StatusDigest == manifest.StatusDigest &&
			existing.Fingerprint == manifest.Fingerprint &&
			existing.IndexSig == manifest.IndexSig &&
			slices.Equal(existing.DirtyPaths, manifest.DirtyPaths) &&
			mapsEqualStringString(existing.ScopeFiles, manifest.ScopeFiles) &&
			mapsEqualStringString(existing.CleanFiles, manifest.CleanFiles) {
			return nil
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	tmpFile, err := os.CreateTemp(filepath.Dir(path), "git-fingerprint-*.tmp")
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
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	if stamp, ok := manifestPathStamp(path); ok {
		gitFingerprintManifestCache.Lock()
		if gitFingerprintManifestCache.entries == nil {
			gitFingerprintManifestCache.entries = make(map[string]cachedGitFingerprintManifest)
		}
		gitFingerprintManifestCache.entries[path] = cachedGitFingerprintManifest{Stamp: stamp, Manifest: manifest}
		gitFingerprintManifestCache.Unlock()
	}
	return nil
}
