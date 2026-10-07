import pytest


def calculation_retention_action(churn_prob: float, ltv: float) -> str:
	if churn_prob > 0.7:
		if ltv >= 500:
			return "PERSONAL_CALL_AND_20_PERCENT_DISCOUNT"
		return "AUTOMATED_15_PERCENT_DISCOUNT"
	elif churn_prob >= 0.4:
		return "PUSH_NOTIFICATION_FEATURE_HIGHLIGHT"
	return "NO_ACTION"

def test_retention_high_risk_high_ltv():
	action = calculation_retention_action(churn_prob = 0.85, ltv = 750.0)
	assert action == "PERSONAL_CALL_AND_20_PERCENT_DISCOUNT"

def test_retention_high_risk_low_ltv():
	action = calculation_retention_action(churn_prob = 0.75, ltv = 200.0)
	assert action == "AUTOMATED_15_PERCENT_DISCOUNT"

def test_retention_low_risk():
	action = calculation_retention_action(churn_prob = 0.2, ltv = 1000.0)
	assert action == "NO_ACTION"
