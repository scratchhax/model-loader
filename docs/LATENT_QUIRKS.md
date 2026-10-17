# Latent quirks

Things the characterization suites (PRs #83-#85) found that behave oddly but are NOT bugs
today: no caller hits them, or the effect is harmless. Each is pinned by a test so a future
change cannot turn it into a silent regression, and each says what fixing it would mean.
This is the report the repo rule asks for - surfaced, not silently fixed.

## `strata_models.read_selection` keeps inline comments

`app/strata_models.py` reads the selection file line by line and strips whitespace, but not
trailing comments. A line written

    MODEL="some-model"   # trying this one

yields the model name `some-model"   # trying this one` - quote and comment included. The
writer never emits comments, so today only a hand-edited file can hit it, and the result is
a clean "model not found" rather than a wrong install.

Pinned by `tests/test_strata_models.py`. A fix would strip from an unquoted `#` and re-pin
the new behavior; it changes what existing hand-edited files mean, so it is a decision, not
a cleanup.

## `vram_live._host_path("")` resolves to the current directory

`app/vram_live.py::_host_path` maps `/models/...` container paths onto the app's view of the
same directory. For the empty string it returns `Path("")`, which stringifies to `.` - the
process's working directory - not "no path". Every caller today passes a path that came from
a container's argv or config, so empty never occurs; the danger is a future caller that
treats the return value as "unset when falsy", because `Path("")` is truthy.

Pinned by `tests/test_vram_live.py`. A fix would return `None` for empty input and teach the
callers to branch on it.

## Strata cannot answer "how much of this context came from cache" - live

Not a code quirk but the one the hero had to learn: Strata's `/slots` has no cache field, and
its `/metrics` reports reuse (`reused`, `hit_rate`) only AFTER a request completes, with no
attribution to solo vs batch. `live.prompt_read` is read PROGRESS (it climbs in 16,384-token
chunks), not fresh-read-vs-cached. An earlier build divided the two and printed "0% from
cache" during every Strata generation - a number that was not measuring what it claimed.

Handled by `InferenceSpeed.cache_known` (False for Strata): the cache-ratio display is simply
absent on that engine instead of guessing. The conversation-cache tile under the slot strip
(PR #86) shows the engine's real cache state - parked conversations in host RAM - which is
the question the fake percentage was trying to answer.
