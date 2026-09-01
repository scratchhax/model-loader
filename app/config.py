from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    models_dir: Path = Path("/models")
    models_ini_path: Path = Path("/models/models.ini")
    data_dir: Path = Path("/data")
    llama_containers: str = ""  # empty = auto-discover any ghcr.io/ggml-org/llama.cpp:* container
    gpu_vram: str = "llama-7900xt:20,llama-5070:12"  # container_name:vram_gib — probed if not listed
    bind_port: int = 8090
    max_concurrent_downloads: int = 2

    @property
    def llama_container_names(self) -> list[str]:
        return [n.strip() for n in self.llama_containers.split(",") if n.strip()]

    @property
    def gpu_vram_map(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for pair in self.gpu_vram.split(","):
            pair = pair.strip()
            if ":" not in pair:
                continue
            name, val = pair.split(":", 1)
            try:
                result[name.strip()] = int(val.strip())
            except ValueError:
                continue
        return result


settings = Settings()
