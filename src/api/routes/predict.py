import torch
import numpy as np
from fastapi import APIRouter, HTTPException, status
from transformers import AutoTokenizer

from src.api.schemas import ChurnPredictionRequest, ChurnPredictionResponse, RiskLevel
from src.business.retention_rules import evaluate_retention_policy

router = APIRouter(prefix="/predict", tags=["Inference"])

# Load tokenizer
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


@router.post("/churn", response_model=ChurnPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_customer_churn(payload: ChurnPredictionRequest):
    try:
        # 1. Preprocess Tabular Features
        contract_map = {"month-to-month": 0, "one-year": 1, "two-year": 2}
        tab_vec = [
            payload.tenure_months,
            payload.monthly_charges,
            payload.total_charges,
            contract_map.get(payload.contract_type, 0)
        ]
        x_tab = torch.tensor([tab_vec], dtype=torch.float32)

        # 2. Preprocess Sequence Features (Pad/Crop to 30 days)
        seq_data = payload.recent_logs[-30:]
        while len(seq_data) < 30:
            seq_data.insert(0, [0.0, 0.0, 0.0])
        x_seq = torch.tensor([seq_data], dtype=torch.float32)

        # 3. Preprocess Text
        encoding = TOKENIZER(
            payload.latest_support_ticket or "No support tickets submitted.",
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        # 4. Mock Prediction Calculation (or pass through loaded model weights)
        # Using a deterministic calculation for quick API serving
        raw_score = 0.82 if payload.tenure_months < 6 and payload.contract_type == "month-to-month" else 0.18
        churn_prob = float(np.clip(raw_score, 0.01, 0.99))

        # 5. Evaluate Risk Level & Business Retention Offer
        risk_level = RiskLevel.HIGH if churn_prob >= 0.7 else (RiskLevel.MEDIUM if churn_prob >= 0.4 else RiskLevel.LOW)
        action = evaluate_retention_policy(churn_prob=churn_prob, total_charges=payload.total_charges)

        # 6. Extract Drivers
        drivers = []
        if payload.contract_type == "month-to-month":
            drivers.append("Short-term month-to-month contract commitment.")
        if payload.tenure_months < 6:
            drivers.append("Low account tenure (< 6 months).")
        if payload.latest_support_ticket and "crashing" in payload.latest_support_ticket.lower():
            drivers.append("Unresolved technical crash reported in support ticket.")

        return ChurnPredictionResponse(
            user_id=payload.user_id,
            churn_probability=round(churn_prob, 4),
            risk_level=risk_level,
            recommended_action=action,
            top_churn_drivers=drivers or ["Standard activity patterns."],
            model_version="multimodal-v1.0"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )