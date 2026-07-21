package service

import "github.com/msk-mind/local-ai-broker/broker/pkg/types"

func (s *Service) buildParallelReducerRequest(req types.SubmitParallelJobsRequest, rootJobID string, childJobIDs, childBackendRunIDs []string) types.SubmitJobRequest {
	orderedChildJobIDs := compactNonEmptyStrings(childJobIDs)
	reducerTaskType := firstNonEmpty(req.Reducer.TaskType, req.TaskType)
	reducerSchema := req.Reducer.OutputSchema
	if reducerSchema.Name == "" {
		reducerSchema = req.OutputSchema
	}
	reducerProfile := req.ExecutionProfile
	if req.Reducer.ExecutionProfile != (types.ExecutionProfile{}) {
		reducerProfile = req.Reducer.ExecutionProfile
	}
	reducerConstraints := req.Constraints
	if req.Reducer.Constraints != (types.Constraints{}) {
		reducerConstraints = req.Reducer.Constraints
	}
	reducerTaskParams := cloneTaskParams(req.TaskParams)
	for k, v := range req.Reducer.TaskParams {
		reducerTaskParams[k] = v
	}
	reducerTaskParams[taskParamChildJobIDs] = append([]string(nil), orderedChildJobIDs...)
	reducerTaskParams[taskParamRootJobID] = rootJobID
	reducerTaskParams[taskParamDependencyBackendRunIDs] = append([]string(nil), childBackendRunIDs...)
	return types.SubmitJobRequest{
		TaskType:         reducerTaskType,
		InputRefs:        append([]types.InputRef(nil), req.Reducer.InputRefs...),
		TaskParams:       reducerTaskParams,
		Constraints:      reducerConstraints,
		ExecutionProfile: s.applyExecutionProfileDefaults(reducerProfile),
		OutputSchema:     reducerSchema,
		Orchestration: types.OrchestrationRequest{
			ParentJobID:     req.ParentJobID,
			RootJobID:       rootJobID,
			Strategy:        "aggregator",
			AggregationKey:  firstNonEmpty(req.Reducer.AggregationKey, "aggregate"),
			DependsOnJobIDs: append([]string(nil), orderedChildJobIDs...),
		},
	}
}
