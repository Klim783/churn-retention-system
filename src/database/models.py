from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base

class User(Base):
	__tablename__ = "users"

	id:Mapped[int] = mapped_column(Integer, primary_key = True, index = True)
	user_id:Mapped[int] = mapped_column(String(64), unique = True, index = True, nullable = False)
	created_at:Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

	tenure_months: Mapped[int] = mapped_column(Integer, default=0)
	monthly_charges: Mapped[float] = mapped_column(Float, default=0.0)
	total_charges: Mapped[float] = mapped_column(Float, default=0.0)
	contract_type: Mapped[str] = mapped_column(String(32), default="month-to-month")

	logs: Mapped[list["ActivityLog"]] = relationship("ActivityLog", back_populates="user", cascade="all, delete-orphan")
	support_tickets: Mapped[list["SupportTicket"]] = relationship("SupportTicket", back_populates="user",
																  cascade="all, delete-orphan")
	predictions: Mapped[list["PredictionHistory"]] = relationship("PredictionHistory", back_populates="user")


class ActivityLog(Base):
	__tablename__ = "activity_logs"

	id:Mapped[int] = mapped_column(Integer, primary_key = True, index = True)
	user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
	timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

	events_count: Mapped[int] = mapped_column(Integer, default=0)
	errors_count: Mapped[int] = mapped_column(Integer, default=0)
	session_duration_min: Mapped[float] = mapped_column(Float, default=0.0)

	user: Mapped["User"] = relationship("User", back_populates="logs")


class SupportTicket(Base):
	"""Текстовые обращения в техподдержку (для NLP Ветки)"""
	__tablename__ = "support_tickets"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
	user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
	created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

	ticket_text: Mapped[str] = mapped_column(Text, nullable=False)
	sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

	user: Mapped["User"] = relationship("User", back_populates="support_tickets")


class PredictionHistory(Base):
	"""История прогнозов оттока и выданных Retention-офферов"""
	__tablename__ = "prediction_history"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
	user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
	created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

	churn_probability: Mapped[float] = mapped_column(Float, nullable=False)
	risk_level: Mapped[str] = mapped_column(String(16), nullable=False)  # Low, Medium, High
	recommended_action: Mapped[str] = mapped_column(String(64), nullable=False)
	model_version: Mapped[str] = mapped_column(String(32), nullable=False)

	user: Mapped["User"] = relationship("User", back_populates="predictions")