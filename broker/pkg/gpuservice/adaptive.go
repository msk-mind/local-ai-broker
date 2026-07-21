package gpuservice

type TierAttempt struct {
	Tier             Tier            `json:"tier"`
	SlurmJobID       string          `json:"slurm_job_id,omitempty"`
	GPUCount         int             `json:"gpu_count"`
	ModelProfile     string          `json:"model_profile,omitempty"`
	FailureCategory  FailureCategory `json:"failure_category,omitempty"`
	EscalationReason string          `json:"escalation_reason,omitempty"`
}

// AdaptiveSynthesisPlan always places P40 and four-GPU V100 before A100.
// Only model-capacity failures select the four-A100 profile.
func AdaptiveSynthesisPlan(v100Failure FailureCategory) []Tier {
	a100 := TierA100Single
	switch v100Failure {
	case FailureOOM, FailureContextOverflow, FailureRepeatedInvalidOutput, FailureModelLimit:
		a100 = TierA100Multigpu
	}
	return []Tier{TierP40Synthesis, TierV100Reasoning, a100}
}

// NextSynthesisTier is safe for retry histories assembled across processes: it
// will not return either A100 profile until both P40 and V100 were recorded.
func NextSynthesisTier(history []TierAttempt, latestFailure FailureCategory) (Tier, bool) {
	attempted := make(map[Tier]bool, len(history))
	for _, attempt := range history {
		attempted[attempt.Tier] = true
	}
	if !attempted[TierP40Synthesis] {
		return TierP40Synthesis, true
	}
	if !attempted[TierV100Reasoning] {
		return TierV100Reasoning, true
	}
	plan := AdaptiveSynthesisPlan(latestFailure)
	a100 := plan[len(plan)-1]
	if attempted[a100] || attempted[TierA100Single] || attempted[TierA100Multigpu] {
		return "", false
	}
	return a100, true
}
