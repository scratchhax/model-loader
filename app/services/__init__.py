"""Services package: the docker SDK seam, backend probing, engine switching,
OpenWebUI reconciliation, and live speed math.

Split from the old single services.py along its section seams. This __init__ is a
pure re-export facade: every call site keeps spelling `services.foo`, and module
state (caches, locks, switch dicts) lives in exactly one submodule.
"""

from .hotpath import (  # noqa: F401
    _HTTP,
    _HTTP_LOOP,
    _http,
)

from .dockerbase import (  # noqa: F401
    _docker_client,
    strata_own_attrs,
)

from .modelsdir import (  # noqa: F401
    DiskInfo,
    GgufEntry,
    ModelShape,
    ModelsDirSnapshot,
    _QUANT_RE,
    _TIER_SHARE,
    canonical_families,
    delete_gguf,
    largest_card_gb,
    model_shape,
    read_ini_aliases,
    size_tier,
    snapshot_models_dir,
    split_quant,
)

from .backends import (  # noqa: F401
    LlamaBackend,
    _ENGINE_TTL,
    _LOADED_BY_BACKEND,
    _LOADED_LOCK,
    _LOADING_SINCE,
    _RE_SLEEP_MARK,
    _api_key_from_attrs,
    _effective_container_names,
    _engine_cache,
    _engine_label,
    _engine_lock,
    _extract_ports,
    _is_asleep,
    _load_failure,
    _note_loading,
    _parse_started_at,
    _probe_loaded_model,
    _restart_errors,
    _restart_errors_lock,
    _sleep_idle_seconds,
    _snapshot_backends_sync,
    browser_endpoints,
    discover_llama_containers,
    engine_for,
    auth_headers,
    api_key_of,
    engine_map,
    last_loaded_ids,
    snapshot_llama_backends,
)

from .gpu import (  # noqa: F401
    _gpu_device_container,
    gpu_engines,
    gpu_holders,
)

from .engines import (  # noqa: F401
    _switch_lock,
    _switch_state,
    _vram_by_container,
    _vram_held_gb,
    _wait_ready,
    activate_engine,
    engine_switch_state,
    restart_llama_backend,
    set_backend_running,
)

from .stratactl import (  # noqa: F401
    _CPU_CRAWL_GB,
    _cpu_backends,
    _fit_backends,
    _strata_control,
    _strata_errors,
    _strata_errors_lock,
    _strata_metrics_cache,
    declared_reserves_gb,
    strata_desired_keys,
    strata_error,
    strata_held_cards,
    strata_load,
    strata_metrics,
    strata_model_ids,
    strata_timeshare_keys,
    strata_unload,
    strata_yield,
    vram_fit_chips,
)

from .speed import (  # noqa: F401
    InferenceSpeed,
    SlotSpeed,
    _GEN_EMA_ALPHA,
    _PROPS_TTL,
    _RATES_TTL,
    _RE_LAUNCH,
    _RE_PP,
    _RE_RELEASE,
    _RE_ST_DONE,
    _RE_ST_GEN,
    _RE_ST_HIT,
    _RE_ST_PROMPT,
    _RE_TASK,
    _RE_TG,
    _SPEED_TTL,
    _TPS_HISTORY_MAX,
    _TPS_MIN_GAP_S,
    _ctx_history,
    _evict_other_models,
    _gen_ema,
    _int_c,
    _last_slots,
    _last_solo,
    _outside_slots,
    _pp_history,
    _props_cache,
    _props_state,
    _prune_slot_state,
    _rates_cache,
    _rates_from_log,
    _rates_from_log_strata,
    _record_ctx,
    _record_prefill_tps,
    _record_tps,
    _settled_task,
    _slot_key,
    _slot_speed,
    _slot_states,
    _slots_strata,
    _speed_cache,
    _speed_prev,
    _tps_history,
    ctx_history,
    ctx_series,
    inference_speed,
    prefill_history,
    slot_columns,
    slot_density,
    test_prompt,
    tps_history,
    tps_series,
)

from .logs import (  # noqa: F401
    _started_epoch,
    container_logs,
    diagnose_container,
)

from .modalities import (  # noqa: F401
    _PROJ_MODALITY_CACHE,
    projector_modalities,
    section_modalities,
    sections_with_speculative,
)

from .openwebui import (  # noqa: F401
    _OWUI_DB,
    _compose_service_of,
    _find_open_webui,
    _has_webui_db,
    _openwebui_api_configs,
    _openwebui_companions,
    _openwebui_persisted_urls,
    align_openwebui_capabilities,
    openwebui_capability_plan,
    openwebui_state,
    prune_openwebui_unknown_ids,
    set_openwebui_model_filter,
    sync_openwebui_capabilities,
    sync_openwebui_endpoints,
    toggle_openwebui_model,
)

from .capability import (  # noqa: F401
    _GPU_VENDORS,
    align_openwebui_capability_for,
    assign_new_model_to_gpu,
    openwebui_capability_state,
)
