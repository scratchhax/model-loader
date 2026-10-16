from __future__ import annotations

import pytest

from app import ini


class TestSectionNames:
    @pytest.mark.parametrize("name", ["a", "Model-Q4_K_M", "a.b-c_1+2", "voice-cpu"])
    def test_valid(self, name):
        assert ini.valid_section_name(name)

    @pytest.mark.parametrize("name", ["", "bad name", "a/b", "a\nb", "[x]", "ünïcode"])
    def test_invalid(self, name):
        assert not ini.valid_section_name(name)


class TestToCli:
    def test_default_model_path(self):
        assert ini.to_cli("sec", []) == "--alias sec --model /models/sec.gguf"

    def test_relative_model_joins_models_dir(self):
        out = ini.to_cli("sec", [("model", "Sub/Model.gguf")])
        assert "--model /models/Sub/Model.gguf" in out

    def test_absolute_model_passes_through(self):
        out = ini.to_cli("sec", [("model", "/models/x/y.gguf")])
        assert "--model /models/x/y.gguf" in out
        assert "//models" not in out

    def test_bool_true_is_bare_flag_false_dropped(self):
        assert "--jinja" in ini.to_cli("s", [("jinja", "true")])
        assert "--jinja" not in ini.to_cli("s", [("jinja", "false")])
        assert "--jinja" not in ini.to_cli("s", [("jinja", "off")])

    def test_bool_empty_still_emits_flag(self):
        # Characterization: the bool branch runs before the empty-value skip, so
        # a bool key present with an empty value renders as the bare flag.
        assert "--jinja" in ini.to_cli("s", [("jinja", "")])

    def test_numeric_that_reads_boolean_keeps_its_value(self):
        # Regression: `parallel = 1` once collapsed to a bare --parallel and
        # `main-gpu = 0` vanished entirely.
        out = ini.to_cli("s", [("parallel", "1"), ("main-gpu", "0")])
        assert "--parallel 1" in out
        assert "--main-gpu 0" in out

    def test_on_off_selects_emit_the_word(self):
        assert "--flash-attn on" in ini.to_cli("s", [("flash-attn", "on")])
        assert "--fit off" in ini.to_cli("s", [("fit", "off")])

    def test_empty_text_value_skipped(self):
        assert "--chat-template" not in ini.to_cli("s", [("chat-template", "")])

    def test_spaces_are_quoted(self):
        assert "--chat-template 'foo bar'" in ini.to_cli(
            "s", [("chat-template", "foo bar")])

    def test_unknown_key_with_on_value_keeps_word(self):
        # Unknown keys default to kind=text, so "on" is a value, not a flag.
        assert "--custom-flag on" in ini.to_cli("s", [("custom-flag", "on")])


class TestToClientConfig:
    def test_urls_and_payload(self):
        out = ini.to_client_config("voice-cpu", "http://box:8080/v1/")
        assert "curl http://box:8080/v1/chat/completions" in out["curl"]
        assert '"model": "voice-cpu"' in out["json"]
        assert 'base_url="http://box:8080/v1"' in out["openai_py"]

    def test_api_key_header_only_when_given(self):
        with_key = ini.to_client_config("s", "http://h/v1", "sk-1")
        assert 'Authorization: Bearer sk-1' in with_key["curl"]
        without = ini.to_client_config("s", "http://h/v1")
        assert "Bearer" not in without["curl"]
        assert 'api_key="not-needed"' in without["openai_py"]


class TestReadWrite:
    def test_read_missing_file_is_empty(self, ini_env):
        assert ini.read_ini().sections() == []
        assert ini.raw_text() == ""

    def test_upsert_creates_section_in_schema_order(self, ini_env):
        ini.upsert_section("sec", {
            "ctx-size": "8192", "model": "/models/a.gguf", "ngl": "999", "": "",
        }, "custom-key = 1\n# comment\nnot a pair\n; also skipped")
        cp = ini.read_ini()
        assert cp.sections() == ["sec"]
        keys = [k for k, _ in cp.items("sec")]
        assert keys == ["model", "ctx-size", "ngl", "custom-key"]
        assert cp.get("sec", "custom-key") == "1"

    def test_upsert_replaces_old_keys(self, ini_env):
        ini.upsert_section("sec", {"ctx-size": "1024"}, "")
        ini.upsert_section("sec", {"ngl": "999"}, "")
        assert dict(ini.read_ini().items("sec")) == {"ngl": "999"}

    def test_key_case_is_preserved(self, ini_env):
        ini.upsert_section("sec", {}, "MyKey = v")
        assert ini.read_ini().get("sec", "MyKey") == "v"

    def test_upsert_rejects_bad_name_without_touching_file(self, ini_env):
        ini.upsert_section("good", {"ngl": "1"}, "")
        with pytest.raises(ValueError):
            ini.upsert_section("bad name", {"ngl": "2"}, "")
        assert ini.section_names() == ["good"]

    def test_section_order_survives(self, ini_env):
        for n in ("b", "a", "c"):
            ini.upsert_section(n, {"ngl": "1"}, "")
        assert ini.section_names() == ["b", "a", "c"]

    def test_delete(self, ini_env):
        ini.upsert_section("sec", {"ngl": "1"}, "")
        assert ini.delete_section("sec") is True
        assert ini.delete_section("sec") is False
        assert ini.section_names() == []

    def test_rename_moves_items(self, ini_env):
        ini.upsert_section("old", {"model": "/models/x.gguf"}, "custom = 7")
        assert ini.rename_section("old", "new") is True
        assert ini.get_section("new") == {"model": "/models/x.gguf", "custom": "7"}
        assert ini.get_section("old") is None

    def test_rename_refuses_collision_and_bad_name(self, ini_env):
        ini.upsert_section("a", {"ngl": "1"}, "")
        ini.upsert_section("b", {"ngl": "2"}, "")
        assert ini.rename_section("a", "b") is False
        assert ini.rename_section("a", "bad name") is False
        assert ini.rename_section("nope", "fine") is False
        assert ini.get_section("a") == {"ngl": "1"}

    def test_raw_text_roundtrip(self, ini_env):
        ini.upsert_section("sec", {"ctx-size": "8192"}, "")
        text = ini.raw_text()
        assert "[sec]" in text and "ctx-size = 8192" in text


class TestBackups:
    def _write_many(self, n):
        for i in range(n):
            ini.upsert_section("sec", {"ngl": str(i)}, "")

    def test_backup_created_and_pruned_to_ten(self, ini_env):
        self._write_many(12)  # first write has no prior file, so 11 backups made
        path = ini.settings.models_ini_path
        backups = sorted(path.parent.glob("models.ini.bak-*"))
        assert len(backups) == ini.BACKUPS_TO_KEEP
        assert path.exists() and "[sec]" in path.read_text()

    def test_same_second_backups_do_not_collide(self, ini_env):
        self._write_many(3)  # all within one second: the -N suffix must kick in
        path = ini.settings.models_ini_path
        assert len(list(path.parent.glob("models.ini.bak-*"))) == 2

    def test_no_temp_files_left_behind(self, ini_env):
        self._write_many(2)
        path = ini.settings.models_ini_path
        assert list(path.parent.glob(".models.ini.*.tmp")) == []

    def test_list_backups_newest_first(self, ini_env):
        self._write_many(3)
        out = ini.list_backups()
        assert len(out) == 2
        assert all(isinstance(o[0], str) and o[2] > 0 for o in out)
        assert [o[1] for o in out] == sorted([o[1] for o in out], reverse=True)


class TestFileAssociation:
    def test_explicit_model_wins_over_name(self, ini_env):
        f = ini_env / "Sub" / "Model.gguf"
        f.parent.mkdir()
        f.write_bytes(b"x")
        # A renamed section (voice-cpu) still resolves through its model= value.
        assert ini.section_file_rel("voice-cpu", {"model": "/models/Sub/Model.gguf"}) == (
            "Sub/Model.gguf")

    def test_relative_model_resolves(self, ini_env):
        f = ini_env / "Sub" / "Model.gguf"
        f.parent.mkdir()
        f.write_bytes(b"x")
        assert ini.section_file_rel("whatever", {"model": "Sub/Model.gguf"}) == (
            "Sub/Model.gguf")

    def test_stem_fallback_when_no_model_key(self, ini_env):
        (ini_env / "Model.gguf").write_bytes(b"x")
        assert ini.section_file_rel("Model", {}) == "Model.gguf"

    def test_missing_file_gives_none(self, ini_env):
        assert ini.section_file_rel("gone", {"model": "/models/nope.gguf"}) is None

    def test_sections_by_file_lists_aliases(self, ini_env):
        (ini_env / "Model.gguf").write_bytes(b"x")
        ini.upsert_section("Model", {"model": "/models/Model.gguf"}, "")
        ini.upsert_section("voice-cpu", {"model": "/models/Model.gguf"}, "")
        assert ini.sections_by_file() == {"Model.gguf": ["Model", "voice-cpu"]}

    def test_companions_excluded_from_stems(self, ini_env):
        (ini_env / "Big.gguf").write_bytes(b"x")
        (ini_env / "Big-mmproj-f16.gguf").write_bytes(b"x")
        assert ini._stems_present() == {"Big": "Big.gguf"}

    def test_unregistered_skips_registered_and_strata_only(self, ini_env):
        (ini_env / "New.gguf").write_bytes(b"x")
        (ini_env / "Taken.gguf").write_bytes(b"x")
        (ini_env / "Pack-gsq-rco.gguf").write_bytes(b"x")
        ini.upsert_section("voice", {"model": "/models/Taken.gguf"}, "")
        assert ini.unregistered_gguf_stems() == ["New"]


class TestIsCompanion:
    def test_mmproj(self):
        assert ini._is_companion("Model-mmproj-f16.gguf")

    def test_mtp_token(self):
        assert ini._is_companion("Model-MTP.gguf")
        assert ini._is_companion("draft-Qwen3-1.7B.gguf")

    def test_negated_variants_are_models(self):
        assert not ini._is_companion("Qwen3.8-27B-noMTP-Q4_K_M.gguf")
        assert not ini._is_companion("Model-nodraft-Q5.gguf")

    def test_big_file_with_mtp_name_is_a_model(self):
        # -MTP- also appears in full quants that INCLUDE the head; size settles it.
        assert not ini._is_companion("Big-MTP-IQ2_M.gguf", 5 * 1024 ** 3)


class TestSuggestDefaults:
    def test_baseline_and_extras(self):
        summary = {
            "model": {"context_length": 8192, "rope_scaling_type": "yarn",
                      "rope_scaling_factor": 4.0,
                      "rope_scaling_original_context": 32768,
                      "expert_count": 64, "expert_used_count": 2},
            "chat_template": "jinja...", "arch": "gemma3",
        }
        fields, hints = ini.suggest_defaults(summary)
        assert fields["ctx-size"] == "8192"
        assert fields["ngl"] == "999"
        assert fields["flash-attn"] == "on"
        assert fields["cache-type-k"] == "q8_0"
        assert fields["cache-type-v"] == "q8_0"
        assert fields["parallel"] == "1"
        assert fields["split-mode"] == "layer"
        assert fields["jinja"] == "true"
        assert fields["rope-scaling"] == "yarn"
        assert fields["rope-scale"] == "4.0"
        assert any("32,768" in h for h in hints)
        assert any("64 experts" in h for h in hints)
        assert any("sliding-window" in h for h in hints)

    def test_minimal_summary(self):
        fields, hints = ini.suggest_defaults({})
        assert "ctx-size" not in fields
        assert fields["ngl"] == "999" and fields["parallel"] == "1"
        assert hints == []


class TestSplitSectionForForm:
    def test_known_vs_extras(self):
        form, extras = ini.split_section_for_form(
            {"ctx-size": "8192", "weird-thing": "x"})
        assert form == {"ctx-size": "8192"}
        assert extras == "weird-thing = x"
