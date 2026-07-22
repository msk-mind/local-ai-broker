package gpuservice

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type HTTPHealthChecker struct {
	client *http.Client
}

func NewHTTPHealthChecker(timeout time.Duration) *HTTPHealthChecker {
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	return &HTTPHealthChecker{client: &http.Client{Timeout: timeout}}
}

func NewHTTPHealthCheckerWithClient(client *http.Client) *HTTPHealthChecker {
	return &HTTPHealthChecker{client: client}
}

func (h *HTTPHealthChecker) Check(ctx context.Context, record Record) error {
	if h == nil || h.client == nil {
		return fmt.Errorf("health check client is not configured")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(record.Endpoint, "/")+"/health", nil)
	if err != nil {
		return err
	}
	if strings.EqualFold(record.EndpointAuth.Type, "bearer") {
		request.Header.Set("Authorization", "Bearer "+record.EndpointAuth.BearerToken)
	}
	response, err := h.client.Do(request)
	if err != nil {
		return fmt.Errorf("GPU service health check: %w", err)
	}
	defer response.Body.Close()
	_, _ = io.CopyN(io.Discard, response.Body, 4096)
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("GPU service health check returned HTTP %d", response.StatusCode)
	}
	return nil
}
