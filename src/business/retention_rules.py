def evaluate_retention_policy(churn_prob: float, total_charges: float) -> str:
    """Оценка матрицы удержания на основе риска оттока и ценности клиента (LTV)"""
    if churn_prob >= 0.7:
        if total_charges >= 500.0:
            return "EXECUTIVE_CALL_AND_20_PERCENT_RENEWAL_DISCOUNT"
        return "AUTOMATED_15_PERCENT_DISCOUNT_PROMOCODE"
    elif churn_prob >= 0.4:
        return "IN_APP_PUSH_FEATURE_TUTORIAL_AND_ENGAGEMENT_EMAIL"
    return "NO_INTERVENTION_STANDARD_MONITORING"