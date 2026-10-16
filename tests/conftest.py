from __future__ import annotations

import pytest

from app.config import settings


@pytest.fixture
def ini_env(tmp_path, monkeypatch):
    """Point models.ini and the models dir at a fresh tmp directory.

    `settings` is an import-time singleton, but ini.py reads its path attributes
    at CALL time, so redirecting the two paths per test is the whole isolation
    story - no env juggling before import, no docker, no network.
    """
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    return models
