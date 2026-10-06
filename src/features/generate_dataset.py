import logging
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker
from sqlalchemy.orm import Session

from src.database.connection import SessionLocal, engine, Base
from src.database.models import User, ActivityLog, SupportTicket


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataGenerator")


fake = Faker()
Faker.seed(1)
np.random.seed(1)
random.seed(1)

NEGATIVE_TICKETS = [
    "The app keeps crashing when I try to process payments. Terrible experience!",
    "Subscription fees are too high and increased without notice. I want to cancel.",
    "Support team hasn't responded in 3 days. My issue remains completely unresolved.",
    "Interface is confusing and login authentication errors occur constantly.",
    "Unable to withdraw funds, the withdrawal button is unresponsive. Fix this bug!"
]

NEUTRAL_TICKETS = [
    "How can I update my linked credit card in the account settings?",
    "Where can I view my billing receipts for last month?",
    "I would like to inquire about Premium plan features.",
    "Are there annual subscription discounts available?",
    "What are the customer support operating hours during weekends?"
]


def generate_synthetic_data(num_users: int = 1000):
	logger.info("Initializing database tables...")
	Base.metadata.create_all(bind = engine)
	db: Session = SessionLocal()

	try:
		logger.info(f"Generating synthetic records for {num_users} users...")
		for i in range(num_users):
			user_id = f"USER_{10000+i}"
			is_churn_risk = np.random.rand() < 0.25

			tenure = np.random.randint(1,48) if not is_churn_risk else np.random.randint(1,6)
			monthly = round(np.random.uniform(15.0, 120.0),2)
			total = round(monthly*tenure,2)
			contract = np.random.choice(
				["month-to-month", "one-year", "two-year"],
				p = [0.7, 0.2, 0.1] if is_churn_risk else [0.3, 0.4, 0.3]
			)

			user = User(
				user_id = user_id,
				tenure_months = tenure,
				monthly_charges = monthly,
				total_charges = total,
				contract_type = contract
			)
			db.add(user)
			db.flush()

			now = datetime.utcnow()
			for day in range(30, 0 , -1):
				log_date = now - timedelta(days = day)

				if is_churn_risk and day < 10:
					events = np.random.randint(0,3)
					errors = np.random.randint(1,5)
					duration = round(np.random.uniform(0.0, 5.0 ),1)
				else:
					events = np.random.randint(5,40)
					errors = np.random.randint(0,2)
					duration = round(np.random.uniform(10.0, 60.0),1)

				log = ActivityLog(
					user_id = user.id,
					timestamp = log_date,
					events_count = events,
					errors_count = errors,
					session_duration = errors,
					seession_duration_min = duration
				)
				db.add(log)

			if is_churn_risk or np.random.rand() < 0.4:
				ticker_text = random.choice(NEGATIVE_TICKETS if is_churn_risk else NEUTRAL_TICKETS)
				ticket = SupportTicket(
					user_id = user.id,
					created_at = now - timedelta(days = np.random.randint(1,15)),
					ticker_text = ticker_text
				)
				db.add(ticket)
		db.commit()
		logger.info("Dataset successfully generated and persisted to PostgreSQL.")
	except Exception as e:
		db.rollback()
		logger.error(f"Failed to generate dataset: {str(e)}")
		raise e
	finally:
		db.close()

if __name__ == "__main__":
	generate_synthetic_data(num_users=500)