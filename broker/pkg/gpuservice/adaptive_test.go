package gpuservice

import "testing"

func TestAdaptiveSynthesisPlanUsesStrictOrderAndAdaptiveA100Size(t *testing.T) {
	for _, test := range []struct {
		failure FailureCategory
		want    Tier
	}{
		{failure: FailureUnavailable, want: TierA100Single},
		{failure: FailureQueueDelay, want: TierA100Single},
		{failure: FailureTimeout, want: TierA100Single},
		{failure: FailureService, want: TierA100Single},
		{failure: FailureOOM, want: TierA100Multigpu},
		{failure: FailureContextOverflow, want: TierA100Multigpu},
		{failure: FailureModelLimit, want: TierA100Multigpu},
		{failure: FailureRepeatedInvalidOutput, want: TierA100Multigpu},
	} {
		plan := AdaptiveSynthesisPlan(test.failure)
		if len(plan) != 3 || plan[0] != TierP40Synthesis || plan[1] != TierV100Reasoning || plan[2] != test.want {
			t.Fatalf("failure %s produced plan %#v", test.failure, plan)
		}
	}
}

func TestNextSynthesisTierNeverReturnsA100BeforeP40AndV100(t *testing.T) {
	if tier, ok := NextSynthesisTier(nil, FailureOOM); !ok || tier != TierP40Synthesis {
		t.Fatalf("first tier = %s, %t", tier, ok)
	}
	if tier, ok := NextSynthesisTier([]TierAttempt{{Tier: TierP40Synthesis}}, FailureOOM); !ok || tier != TierV100Reasoning {
		t.Fatalf("second tier = %s, %t", tier, ok)
	}
	history := []TierAttempt{{Tier: TierP40Synthesis}, {Tier: TierV100Reasoning}}
	if tier, ok := NextSynthesisTier(history, FailureOOM); !ok || tier != TierA100Multigpu {
		t.Fatalf("adaptive A100 tier = %s, %t", tier, ok)
	}
}
