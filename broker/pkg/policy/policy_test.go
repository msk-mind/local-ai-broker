package policy

import (
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestAuthorizeJobLogs(t *testing.T) {
	restrictedJob := types.Job{
		Request: types.SubmitJobRequest{
			Constraints: types.Constraints{Confidentiality: "local_only"},
			InputRefs:   []types.InputRef{{Classification: "restricted"}},
		},
	}
	if err := AuthorizeJobLogs(restrictedJob); err == nil {
		t.Fatal("expected restricted job logs to be denied")
	}

	overrideJob := restrictedJob
	overrideJob.Request.TaskParams = map[string]any{"allow_log_release": true}
	if err := AuthorizeJobLogs(overrideJob); err != nil {
		t.Fatalf("expected override to allow log release: %v", err)
	}
}

func TestFilterJobResultRedactsSensitiveFields(t *testing.T) {
	job := types.Job{
		Request: types.SubmitJobRequest{
			Constraints: types.Constraints{Confidentiality: "local_only"},
			InputRefs:   []types.InputRef{{Classification: "restricted"}},
		},
		Result: &types.Result{
			SchemaName:    "demo",
			SchemaVersion: "1.0.0",
			Payload: map[string]any{
				"path":          "/secret/file.txt",
				"excerpt":       "patient-linked source text",
				"paths":         []any{"/secret/one", "/secret/two"},
				"related_paths": []any{"/secret/three"},
				"nested": map[string]any{
					"path":    "/secret/nested.txt",
					"content": "restricted repository code",
				},
				"warnings": []any{"existing"},
			},
		},
		Artifacts: []types.Artifact{{ArtifactID: "artifact-1", Path: "/secret/one"}},
	}

	result, artifacts, err := FilterJobResult(job)
	if err != nil {
		t.Fatalf("filter job result: %v", err)
	}
	if result == nil {
		t.Fatal("expected filtered result")
	}
	if got := result.Payload["path"]; got != "[REDACTED]" {
		t.Fatalf("unexpected path redaction: %#v", got)
	}
	if got := result.Payload["excerpt"]; got != "[REDACTED]" {
		t.Fatalf("unexpected excerpt redaction: %#v", got)
	}
	nested := result.Payload["nested"].(map[string]any)
	if nested["content"] != "[REDACTED]" {
		t.Fatalf("unexpected nested content redaction: %#v", nested)
	}
	if got := result.Payload["warnings"].([]any); !containsString(got, "broker_redacted_sensitive_fields") || !containsString(got, "broker_withheld_artifacts") {
		t.Fatalf("expected sensitivity warnings, got %#v", got)
	}
	if artifacts != nil {
		t.Fatalf("expected artifacts to be withheld, got %#v", artifacts)
	}

	override := job
	override.Request.TaskParams = map[string]any{"allow_artifact_release": true}
	result, artifacts, err = FilterJobResult(override)
	if err != nil {
		t.Fatalf("filter job result with artifact override: %v", err)
	}
	if artifacts == nil || len(artifacts) != 1 {
		t.Fatalf("expected artifacts to be retained with path redacted, got %#v", artifacts)
	}
	if artifacts[0].Path != "" {
		t.Fatalf("expected artifact path to be redacted, got %#v", artifacts[0])
	}
}

func TestFilterJobResultHonorsArtifactClassification(t *testing.T) {
	job := types.Job{
		Request: types.SubmitJobRequest{},
		Result: &types.Result{SchemaName: "demo", SchemaVersion: "1.0.0", Payload: map[string]any{
			"excerpt": "restricted artifact content",
		}},
		Artifacts: []types.Artifact{{ArtifactID: "restricted", Classification: "restricted", Path: "/private/path"}},
	}
	result, artifacts, err := FilterJobResult(job)
	if err != nil {
		t.Fatalf("filter classified artifact result: %v", err)
	}
	if result.Payload["excerpt"] != "[REDACTED]" || artifacts != nil {
		t.Fatalf("artifact classification did not trigger release filtering: result=%#v artifacts=%#v", result, artifacts)
	}
}

func containsString(values []any, needle string) bool {
	for _, value := range values {
		if text, ok := value.(string); ok && text == needle {
			return true
		}
	}
	return false
}
