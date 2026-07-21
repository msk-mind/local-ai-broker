package jobenv

import (
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestRunRootAndRepoRoot(t *testing.T) {
	job := types.Job{}
	if got := RunRoot(job); got != ".broker/runs" {
		t.Fatalf("unexpected default run root: %q", got)
	}
	if got := RepoRoot(job); got != "." {
		t.Fatalf("unexpected default repo root: %q", got)
	}

	job.Request.TaskParams = map[string]any{
		TaskParamRunRoot:  "/tmp/runs",
		TaskParamRepoRoot: "/tmp/repo",
	}
	if got := RunRoot(job); got != "/tmp/runs" {
		t.Fatalf("unexpected overridden run root: %q", got)
	}
	if got := RepoRoot(job); got != "/tmp/repo" {
		t.Fatalf("unexpected overridden repo root: %q", got)
	}
}
