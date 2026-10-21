"""Each engine's settings page opens with the same routing row the Router tab shows for it
(review items 14 and 28): Strata on the Strata tab, Maya on the Maya tab, llama on models.ini."""
from __future__ import annotations

import pytest

from tests.test_routes_smoke import web  # noqa: F401 - the fixture: real app, empty box


@pytest.mark.parametrize("url,engine", [("/strata/models", "strata"), ("/maya", "maya"),
                                        ("/config", "llama")])
def test_engine_page_has_its_own_row(web, url, engine):
    r = web.get(url)
    assert r.status_code == 200
    assert 'hx-post="/router/default"' in r.text
    assert f'aria-label="Auto-switch to {engine}"' in r.text
    others = {"strata", "maya", "llama"} - {engine}
    assert not any(f'aria-label="Auto-switch to {o}"' in r.text for o in others)


def test_router_tab_still_has_every_engine(web):
    r = web.get("/router")
    assert r.status_code == 200
    for e in ("strata", "llama", "maya"):
        assert f'aria-label="Auto-switch to {e}"' in r.text
    assert r.text.count("border-t border-slate-200 dark:border-slate-700") >= 2   # rows 2 and 3
