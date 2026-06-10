from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    hf_token: str  # Pydantic lèvera une erreur explicite si HF_TOKEN est manquant

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings() #pyright: ignore