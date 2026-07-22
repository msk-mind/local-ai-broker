package auth

import (
	"net/http"
	"testing"
)

func TestParseStaticTokens(t *testing.T) {
	tokens, err := ParseStaticTokens("alpha=alice:admin, beta = bob : user")
	if err != nil {
		t.Fatalf("parse static tokens: %v", err)
	}
	if got := tokens["alpha"]; got.Actor != "alice" || got.Role != "admin" {
		t.Fatalf("unexpected alpha principal: %#v", got)
	}
	if got := tokens["beta"]; got.Actor != "bob" || got.Role != "user" {
		t.Fatalf("unexpected beta principal: %#v", got)
	}
}

func TestAuthenticateHeaderIdentity(t *testing.T) {
	authenticator := NewHeaderAuthenticator()
	req, err := http.NewRequest(http.MethodGet, "/", nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("X-Broker-Actor", "alice")
	req.Header.Set("X-Broker-Role", "admin")

	principal, err := authenticator.Authenticate(req)
	if err != nil {
		t.Fatalf("authenticate: %v", err)
	}
	if principal.Actor != "alice" || principal.Role != "admin" {
		t.Fatalf("unexpected principal: %#v", principal)
	}
}

func TestAuthenticateStaticToken(t *testing.T) {
	authenticator := NewStaticTokenAuthenticator(map[string]Principal{
		"secret": {Actor: "alice", Role: "admin"},
	})
	req, err := http.NewRequest(http.MethodGet, "/", nil)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer secret")

	principal, err := authenticator.Authenticate(req)
	if err != nil {
		t.Fatalf("authenticate: %v", err)
	}
	if principal.Actor != "alice" || principal.Role != "admin" {
		t.Fatalf("unexpected principal: %#v", principal)
	}
}

func TestIsAdmin(t *testing.T) {
	if !IsAdmin(Principal{Role: "ADMIN"}) {
		t.Fatal("expected admin role to be recognized case-insensitively")
	}
	if IsAdmin(Principal{Role: "user"}) {
		t.Fatal("expected non-admin role to be rejected")
	}
}
