from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ChurnPredictionRequest(BaseModel):
    user_id: str = Field(..., example="USR_10001")
    tenure_months: int = Field(..., ge=0, example=3)
    monthly_charges: float = Field(..., ge=0.0, example=85.50)
    total_charges: float = Field(..., ge=0.0, example=256.50)
    contract_type: str = Field(..., example="month-to-month")
    recent_logs: List[List[float]] = Field(
        ...,
        description="30-day sequence of [events_count, errors_count, session_duration_min]",
        example=[[12, 0, 25.0]] * 30
    )
    latest_support_ticket: Optional[str] = Field(
        default="No support tickets submitted.",
        example="App keeps crashing when processing payments."
    )


class ChurnPredictionResponse(BaseModel):
    user_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    recommended_action: str
    top_churn_drivers: List[str]
    model_version: str = "multimodal-v1.0"