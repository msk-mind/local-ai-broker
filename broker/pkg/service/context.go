package service

import "context"

type inlineLocalReleaseContextKey struct{}
type skipInspectRepoProbeContextKey struct{}

func WithPreferInlineLocalRelease(ctx context.Context) context.Context {
	return context.WithValue(ctx, inlineLocalReleaseContextKey{}, true)
}

func preferInlineLocalRelease(ctx context.Context) bool {
	return ctx != nil && ctx.Value(inlineLocalReleaseContextKey{}) == true
}

func WithSkipInspectRepoResultProbe(ctx context.Context) context.Context {
	return context.WithValue(ctx, skipInspectRepoProbeContextKey{}, true)
}

func skipInspectRepoResultProbe(ctx context.Context) bool {
	return ctx != nil && ctx.Value(skipInspectRepoProbeContextKey{}) == true
}
