import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from numba.np.npyfuncs import np_int_square_impl
from sqlalchemy.orm import Session

from src.database.connection import SessionLocal
from src.database.models import ActivityLog, SupportTicket, User
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureBuilder")

class FeatureBuilder:
	def __init__(self, db_session: Session):
		self.db = db_session
	def extract_multimodal_dataset(self) -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray]:
		logger.info("Fetching raw records from PostgreSQL...")

		users = self.db.query(User).all

		tabular_data = []
		sequence_data = []
		text_data = []
		labels = []

		for user in users:
			contract_map = {"month-to-month" : 0, "one-year":1, "two-year":2}
			tabular_data.append({
				"tenure_months": user.tenure_months,
				"monthly_charges": user.monthly_charges,
				"total_charges": user.total_charges,
				"contract_type": contract_map.get(user.contract_type, 0)
			})

			user_logs = (
				self.db.query(ActivityLog)
				.filter(ActivityLog.user_id == user.id)
				.order_by(ActivityLog.timestamp.asc())
				.all()
			)

			seq_matrix = []
			for log in user_logs[-30:]:  # Take last 30 days
				seq_matrix.append([log.events_count, log.errors_count, log.session_duration_min])

			while len(seq_matrix) < 30:
				seq_matrix.insert(0,[0,0,0.0])

			sequence_data.append(seq_matrix)

			latest_ticket = (
				self.db.query(SupportTicket)
				.filter(SupportTicket.user_id == user.id)
				.order_by(SupportTicket.created_at.desc())
				.first()
			)
			text_data.append(latest_ticket.ticket_text if latest_ticket else "No support tickets submitted")

			is_churn = 1 if (user.contract_type == "month-to-month" and user.tenure_months < 6) else 0
			labels.append(is_churn)

		df_tabular = pd.DataFrame(tabular_data)
		np_sequences = np.array(sequence_data, dtype = np.float32)
		np_labels = np.array(labels, dtype = np.int64)

		logger.info(f"Dataset extraction completed. Total samples {len(df_tabular)}")
		return df_tabular, np_sequences, text_data, np_labels

if __name__ == "__main__":
	db = SessionLocal()
	try:
		builder = FeatureBuilder(db)
		X_tab, X_seq, X_text, y = builder.extract_multimodal_dataset()
		logger.info(f"Tabular Shape: {X_tab.shape}, Sequence Shape: {X_seq.shape}, Text Count:{len(X_text)}")
	finally:
		db.close()
