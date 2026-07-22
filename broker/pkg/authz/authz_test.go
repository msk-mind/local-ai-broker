package authz

import (
	"testing"

	"github.com/msk-mind/local-ai-broker/broker/pkg/auth"
	"github.com/msk-mind/local-ai-broker/broker/pkg/types"
)

func TestAuthorizeJobAccess(t *testing.T) {
	job := types.Job{SubmittedBy: "alice"}

	if err := AuthorizeJobAccess(auth.Principal{Actor: "alice", Role: "user"}, job); err != nil {
		t.Fatalf("expected matching actor to be allowed: %v", err)
	}
	if err := AuthorizeJobAccess(auth.Principal{Actor: "bob", Role: "user"}, job); err == nil {
		t.Fatal("expected mismatched actor to be denied")
	}
	if err := AuthorizeJobAccess(auth.Principal{Actor: "", Role: "user"}, job); err == nil {
		t.Fatal("expected missing actor to be denied")
	}
	if err := AuthorizeJobAccess(auth.Principal{Actor: "root", Role: "admin"}, job); err != nil {
		t.Fatalf("expected admin to be allowed: %v", err)
	}
	if err := AuthorizeJobAccess(auth.Principal{Actor: "", Role: "admin"}, types.Job{}); err != nil {
		t.Fatalf("expected unowned job to be accessible: %v", err)
	}
}
