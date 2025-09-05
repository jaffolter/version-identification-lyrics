from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Bind environment variables here
    DATA_DIR: str = "/nfs/interns/jaffolter/data/"
    AUDIO_DIR: str = "/nfs/interns/jaffolter/data/audio/"
    EMBEDDINGS_DIR: str = "/nfs/interns/jaffolter/data/embeddings"
    WANDB_API_KEY: str = ""
    VAULT_TOKEN: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
