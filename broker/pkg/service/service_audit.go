package service

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/msk-mind/local-ai-broker/broker/pkg/audit"
	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/authz"
	"github.com/msk-mind/local-ai-broker/broker/pkg/store"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func (s *Service) audit(ctx context.Context, action, outcome string, job *types.Job, fields map[string]any) {
	principal := auth.PrincipalFromContext(ctx)
	event := audit.Event{
		Actor:   principal.Actor,
		Role:    principal.Role,
		Action:  action,
		Outcome: outcome,
		Fields:  fields,
	}
	if job != nil {
		event.JobID = job.ID
		event.TaskType = job.TaskType
	}
	_ = s.auditLogger.Log(ctx, event)
}

func (s *Service) auditDeniedLookup(ctx context.Context, action, jobID string, err error) {
	outcome := "error"
	if errors.Is(err, authz.ErrForbidden) {
		outcome = "forbidden"
	} else if errors.Is(err, store.ErrNotFound) {
		outcome = "not_found"
	}
	s.audit(ctx, action, outcome, &types.Job{ID: jobID}, map[string]any{
		"error": err.Error(),
	})
}

func (s *Service) auditAndReturnLogs(ctx context.Context, job types.Job, stream string, maxBytes int) error {
	s.audit(ctx, "job.fetch_logs", "success", &job, map[string]any{
		"stream":    stream,
		"max_bytes": maxBytes,
	})
	return nil
}

func readLogFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", err
	}
	return string(data), nil
}

func combineLogs(stdoutText, stderrText string) (string, []string) {
	parts := make([]string, 0, 2)
	sourceRefs := make([]string, 0, 2)
	if stdoutText != "" {
		parts = append(parts, "== stdout ==\n"+strings.TrimRight(stdoutText, "\n"))
		sourceRefs = append(sourceRefs, "stdout.log")
	}
	if stderrText != "" {
		parts = append(parts, "== stderr ==\n"+strings.TrimRight(stderrText, "\n"))
		sourceRefs = append(sourceRefs, "stderr.log")
	}
	return strings.Join(parts, "\n\n"), sourceRefs
}

var secretPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(bearer\s+)([A-Za-z0-9._-]+)`),
	regexp.MustCompile(`(?i)(token=)([^&\s]+)`),
	regexp.MustCompile(`(?i)(api[_-]?key=)([^&\s]+)`),
}

func redactLogContent(content string) string {
	redacted := content
	for _, pattern := range secretPatterns {
		redacted = pattern.ReplaceAllString(redacted, `${1}[REDACTED]`)
	}
	return redacted
}

func truncateLogContent(content string, maxBytes int) (string, bool) {
	if maxBytes <= 0 || len(content) <= maxBytes {
		return content, false
	}
	const suffix = "\n[TRUNCATED]\n"
	if maxBytes <= len(suffix) {
		return suffix[:maxBytes], true
	}
	return content[:maxBytes-len(suffix)] + suffix, true
}

func progressEquals(a, b *types.ProgressInfo) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a.State != b.State || a.Phase != b.Phase || a.Percent != b.Percent || a.Message != b.Message {
		return false
	}
	if !timePtrEqual(a.Timestamp, b.Timestamp) {
		return false
	}
	aMetrics, _ := json.Marshal(a.Metrics)
	bMetrics, _ := json.Marshal(b.Metrics)
	return string(aMetrics) == string(bMetrics)
}

func progressNewer(current, previous *types.ProgressInfo) bool {
	if current == nil {
		return false
	}
	if previous == nil {
		return true
	}
	if current.LastUpdated != nil && previous.LastUpdated != nil {
		return current.LastUpdated.After(*previous.LastUpdated)
	}
	if current.Timestamp != nil && previous.Timestamp != nil {
		return current.Timestamp.After(*previous.Timestamp)
	}
	return false
}

func timePtrEqual(a, b *time.Time) bool {
	if a == nil || b == nil {
		return a == b
	}
	return a.Equal(*b)
}
