package store

import "context"

type nonDurableWriteContextKey struct{}

func WithNonDurableWrite(ctx context.Context) context.Context {
	return context.WithValue(ctx, nonDurableWriteContextKey{}, true)
}

func nonDurableWriteRequested(ctx context.Context) bool {
	return ctx != nil && ctx.Value(nonDurableWriteContextKey{}) == true
}
