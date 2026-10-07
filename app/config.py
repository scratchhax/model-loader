from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    models_dir: Path = Path("/models")
    models_ini_path: Path = Path("/models/models.ini")
    data_dir: Path = Path("/data")
    llama_containers: str = ""  # empty = auto-discover any ghcr.io/ggml-org/llama.cpp:* container
    gpu_vram: str = ""  # optional container_name:vram_gib overrides — auto-probed via nvidia-smi/rocm-smi if empty
    bind_port: int = 8090
    max_concurrent_downloads: int = 2
    # RAM the OS, page cache and everything else on the box need. Subtracted before deciding
    # whether a model fits in system memory, because total RAM is never all yours: sizing
    # against it produces plans that swap or get OOM-killed. Raise it on a busy host.
    host_ram_reserve_gb: float = 32.0

    # Keys this app ensures in Strata's run config, as a JSON object — e.g.
    # {"idle_unload_s": 300, "min_free_vram_mib": 4096}. Empty means the app never touches
    # the config. setup.py keeps keys it does not own (#629), so these survive a re-setup.
    strata_config_keys: str = ""

    # Where Strata's before_load hook asks for the cards back: a POST here stops the llama
    # backends so Strata's reload always wins. The default is the compose service name; the
    # hook only exists at all when time-share is switched on on the Strata card.
    strata_yield_url: str = "http://model-loader:8090/internal/yield"

    @property
    def strata_config_key_map(self) -> dict:
        try:
            v = json.loads(self.strata_config_keys) if self.strata_config_keys else {}
        except ValueError:
            return {}
        return v if isinstance(v, dict) else {}

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
