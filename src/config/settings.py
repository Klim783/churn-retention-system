from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
	DEVELOPMENT = "development"
	STAGING = "staging"
	PRODUCTION = "production"

class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file = ".env",
		env_file_encoding = "utf-8",
		extra = "ignore"
	)
	PROJECT_NAME: str = "Customer churn predictor"
	Environment: Environment = Environment.DEVELOPMENT
	DEBUG: bool = True

	POSTGRES_USER: str = "ml_user"
	POSTGRES_PASSWORD: str = "ml_password"
	POSTGRES_HOST: str = "localhost"
	POSTGRES_PORT: int = 5331
	POSTGRES_DB: str = "churn_db"
	DATABASE_URL: str | None = None

	MLFLOW_TRACKING_URI: str = "http://localhost:5050"
	API_V1_STR: str = "/api/v1"

	def get_database_url(self) -> str:
		if self.DATABASE_URL:
			return self.DATABASE_URL
		return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()