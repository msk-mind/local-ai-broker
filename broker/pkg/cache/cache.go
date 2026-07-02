package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

var (
	errFingerprintBudgetExceeded   = errors.New("directory fingerprint budget exceeded")
	metadataFingerprintMaxEntries  = 20000
	metadataFingerprintMaxDuration = 2 * time.Second
)

func KeyForRequest(req types.SubmitJobRequest) (string, bool, error) {
	if !isCacheableTask(req.TaskType) {
		return "", false, nil
	}
	if len(req.InputRefs) != 1 {
		return "", false, nil
	}
	input := req.InputRefs[0]
	if input.Type != "file" && input.Type != "directory" && input.Type != "repo" {
		return "", false, nil
	}

	path, err := filePathFromURI(input.URI)
	if err != nil {
		return "", false, err
	}

	contentHash := ""
	switch input.Type {
	case "file":
		data, err := os.ReadFile(path)
		if err != nil {
			return "", false, nil
		}
		contentHash = sumBytes(data)
	case "directory", "repo":
		contentHash, err = hashPath(path, input.Type)
		if err != nil {
			if errors.Is(err, errFingerprintBudgetExceeded) {
				return "", false, nil
			}
			return "", false, nil
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

	serialized, err := json.Marshal(payload)
	if err != nil {
		return "", false, err
	}
	return "sha256:" + sumBytes(serialized), true, nil
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

func isCacheableTask(taskType string) bool {
	switch taskType {
	case "document_summary", "log_analysis", "repo_summary", "rag_compress", "summarize_logs":
		return true
	default:
		return false
	}
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
		keys = append(keys, k)
	}
	slices.Sort(keys)
	out := make(map[string]any, len(keys))
	for _, k := range keys {
		out[k] = taskParams[k]
	}
	return out
}

func hashPath(path, inputType string) (string, error) {
	if fingerprint, ok := gitFingerprint(path); ok {
		return fingerprint, nil
	}
	return fingerprintDirectoryMetadata(path)
}

func fingerprintDirectoryMetadata(root string) (string, error) {
	hasher := sha256.New()
	startedAt := time.Now()
	entryCount := 0
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
	return "meta:" + hex.EncodeToString(hasher.Sum(nil)), nil
}

func gitFingerprint(path string) (string, bool) {
	gitPath, err := exec.LookPath("git")
	if err != nil {
		return "", false
	}

	topLevelRaw, err := runGit(gitPath, path, "rev-parse", "--show-toplevel")
	if err != nil {
		return "", false
	}
	topLevel := strings.TrimSpace(topLevelRaw)
	if topLevel == "" {
		return "", false
	}

	head, err := runGit(gitPath, path, "rev-parse", "HEAD")
	if err != nil {
		head = ""
	}
	head = strings.TrimSpace(head)

	relPath := "."
	if rel, err := filepath.Rel(topLevel, path); err == nil && rel != "" {
		relPath = filepath.ToSlash(rel)
	}

	statusArgs := []string{
		"status",
		"--porcelain=v1",
		"--untracked-files=normal",
	}
	if relPath != "." {
		statusArgs = append(statusArgs, "--", relPath)
	}
	status, err := runGit(gitPath, topLevel, statusArgs...)
	if err != nil {
		status = ""
	}

	payload := map[string]any{
		"repo_root": topLevel,
		"repo_head": head,
		"scope":     relPath,
		"status":    strings.TrimSpace(status),
	}
	serialized, err := json.Marshal(payload)
	if err != nil {
		return "", false
	}
	return "git:" + sumBytes(serialized), true
}

func runGit(gitPath, dir string, args ...string) (string, error) {
	cmd := exec.Command(gitPath, args...)
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(output)))
	}
	return string(output), nil
}

func shouldIgnoreDir(rel string) bool {
	parts := strings.Split(rel, "/")
	for _, part := range parts {
		switch part {
		case ".git", ".broker", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".venv", "venv", "env", "node_modules", "site-packages", "build", "dist":
			return true
		}
	}
	return false
}
