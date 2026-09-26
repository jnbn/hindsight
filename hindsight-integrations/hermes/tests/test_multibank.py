"""Tests for per-project bank walk-up and multi-bank fan-out."""

import asyncio
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import SimpleNamespace

import pytest

import hindsight_hermes as plugin
from hindsight_hermes.settings import _discover_cwd_bank_id

HindsightMemoryProvider = plugin.HindsightMemoryProvider


def _fake_recall(provider, answers: dict, queried: list | None = None) -> None:
    """Route the provider's recall to *answers*: bank id -> list of texts, or an
    exception to raise for that bank. Runs the real async operation."""

    class _Client:
        async def arecall(self, bank_id, **kwargs):
            if queried is not None:
                queried.append(bank_id)
            answer = answers.get(bank_id, [])
            if isinstance(answer, Exception):
                raise answer
            return SimpleNamespace(results=[SimpleNamespace(text=t) for t in answer])

    provider._run_hindsight_operation = lambda op: asyncio.run(op(_Client()))


def test_discover_cwd_bank_id_finds_root(tmp_path: Path):
    project_root = tmp_path / "my_project"
    sub_dir = project_root / "src" / "deep"
    sub_dir.mkdir(parents=True)

    hindsight_dir = project_root / ".hindsight"
    hindsight_dir.mkdir()
    (hindsight_dir / "config.toml").write_text('bank_id = "project-bank-alpha"\n', encoding="utf-8")

    assert _discover_cwd_bank_id(str(sub_dir)) == "project-bank-alpha"


def test_discover_cwd_bank_id_returns_none_when_absent(tmp_path: Path):
    sub_dir = tmp_path / "no_hindsight" / "sub"
    sub_dir.mkdir(parents=True)
    assert _discover_cwd_bank_id(str(sub_dir)) is None


def test_provider_initializes_with_cwd_bank(tmp_path: Path):
    project_root = tmp_path / "project_x"
    project_root.mkdir()
    hindsight_dir = project_root / ".hindsight"
    hindsight_dir.mkdir()
    (hindsight_dir / "config.toml").write_text('bank_id = "project-x-bank"\n', encoding="utf-8")

    provider = HindsightMemoryProvider()
    provider.initialize(
        session_id="s1",
        cwd=str(project_root),
        bank_id_template="hermes-{workspace}",
    )
    assert provider._bank_id == "project-x-bank"


def test_provider_derives_workspace_from_cwd(tmp_path: Path):
    git_repo = tmp_path / "my-awesome-repo"
    (git_repo / ".git").mkdir(parents=True)
    sub = git_repo / "pkg" / "sub"
    sub.mkdir(parents=True)

    provider = HindsightMemoryProvider()
    with patch.object(plugin, "_load_config", return_value={"bank_id": "hermes", "bank_id_template": "{workspace}"}):
        provider.initialize(
            session_id="s2",
            cwd=str(sub),
            agent_workspace="hermes",  # upstream default fallback
        )
    assert provider._bank_id == "my-awesome-repo"

    # In user home / root with no project git repo, falls back to default bank
    provider_home = HindsightMemoryProvider()
    with patch.object(plugin, "_load_config", return_value={"bank_id": "hermes", "bank_id_template": "{workspace}"}):
        provider_home.initialize(
            session_id="s3",
            cwd=str(Path.home()),
            agent_workspace="hermes",
        )
    assert provider_home._bank_id == "hermes"


def test_provider_multibank_write_and_recall_order():
    provider = HindsightMemoryProvider()
    provider._config = {"bank_id": "default-bank"}
    provider._apply_connection_settings(
        {
            "bank_id": "default-bank",
            "mirror_to_own_bank": True,
            "additional_banks": ["shared-knowledge", "global-bank"],
        }
    )
    # Primary is default-bank, mirror is identical, so deduped:
    assert provider._write_bank_ids == ["default-bank", "shared-knowledge", "global-bank"]

    # When primary changes (e.g. from template/workspace):
    provider._bank_id = "project-bank"
    provider._write_bank_ids = provider._build_write_bank_ids()
    assert provider._write_bank_ids == ["project-bank", "default-bank", "shared-knowledge", "global-bank"]


def test_provider_recall_dedupes_across_banks():
    provider = HindsightMemoryProvider()
    provider._bank_id = "primary"
    provider._write_bank_ids = ["primary", "secondary"]
    _fake_recall(provider, {"primary": ["Fact 1", "Fact 2"], "secondary": ["Fact 2", "Fact 3"]})

    results = provider._recall("test query")
    texts = [r.text for r in results]
    assert texts == ["Fact 1", "Fact 2", "Fact 3"]


def test_provider_sync_turn_retains_to_all_write_banks():
    provider = HindsightMemoryProvider()
    provider._bank_id = "b1"
    provider._write_bank_ids = ["b1", "b2"]
    retained_banks = []

    def mock_retain_batch(item, bank_id, **kwargs):
        retained_banks.append(bank_id)
        return SimpleNamespace(operation_id=f"op-{bank_id}")

    provider._retain_batch = mock_retain_batch
    provider._ensure_writer = MagicMock()
    provider._register_atexit = MagicMock()

    # Enqueue a turn
    provider.sync_turn("User message", "Assistant reply", session_id="s1")

    # Drain queue
    job = provider._retain_queue.get_nowait()
    job()

    assert retained_banks == ["b1", "b2"]
    assert provider._pending_retain_ops == {("b1", "op-b1"), ("b2", "op-b2")}


def _provider_with(cfg: dict) -> HindsightMemoryProvider:
    provider = HindsightMemoryProvider()
    provider._config = {"bank_id": cfg.get("bank_id", "hermes")}
    provider._apply_connection_settings(cfg)
    return provider


def test_recall_only_banks_are_recalled_after_the_write_set_and_never_written():
    provider = _provider_with(
        {
            "bank_id": "primary",
            "additional_banks": ["shared"],
            "recall_additional_banks": ["vault"],
        }
    )
    assert provider._write_bank_ids == ["primary", "shared"]
    assert provider._build_recall_bank_ids() == ["primary", "shared", "vault"]


def test_recall_only_banks_accept_the_camel_case_alias():
    provider = _provider_with({"bank_id": "primary", "recallAdditionalBanks": ["vault"]})
    assert provider._write_bank_ids == ["primary"]
    assert provider._build_recall_bank_ids() == ["primary", "vault"]


def test_bank_in_both_lists_stays_writable_and_is_recalled_once():
    provider = _provider_with(
        {
            "bank_id": "primary",
            "additional_banks": ["shared"],
            "recall_additional_banks": ["shared", "vault", " "],
        }
    )
    assert provider._write_bank_ids == ["primary", "shared"]
    assert provider._build_recall_bank_ids() == ["primary", "shared", "vault"]


def test_recall_queries_recall_only_banks_last():
    provider = _provider_with({"bank_id": "primary", "recall_additional_banks": ["vault"]})
    queried = []
    _fake_recall(provider, {"primary": ["fact from primary"], "vault": ["fact from vault"]}, queried)
    texts = [r.text for r in provider._recall("query")]
    assert queried == ["primary", "vault"]
    assert texts == ["fact from primary", "fact from vault"]


def test_sync_turn_never_retains_to_recall_only_banks():
    provider = _provider_with(
        {
            "bank_id": "primary",
            "additional_banks": ["shared"],
            "recall_additional_banks": ["vault"],
        }
    )
    retained_banks = []

    def mock_retain_batch(item, bank_id, **kwargs):
        retained_banks.append(bank_id)
        return SimpleNamespace(operation_id=f"op-{bank_id}")

    provider._retain_batch = mock_retain_batch
    provider._ensure_writer = MagicMock()
    provider._register_atexit = MagicMock()

    provider.sync_turn("User message", "Assistant reply", session_id="s1")
    provider._retain_queue.get_nowait()()

    assert retained_banks == ["primary", "shared"]


def test_bank_lists_accept_comma_separated_text_from_the_settings_panel():
    provider = _provider_with(
        {
            "bank_id": "primary",
            "additional_banks": "shared, team",
            "recall_additional_banks": "vault,notes",
        }
    )
    assert provider._write_bank_ids == ["primary", "shared", "team"]
    assert provider._build_recall_bank_ids() == ["primary", "shared", "team", "vault", "notes"]


def test_bank_lists_accept_a_json_encoded_list():
    provider = _provider_with({"bank_id": "primary", "additional_banks": '["shared", "team"]'})
    assert provider._write_bank_ids == ["primary", "shared", "team"]


def test_empty_bank_list_text_means_no_extra_banks():
    provider = _provider_with({"bank_id": "primary", "additional_banks": "", "recall_additional_banks": " , "})
    assert provider._write_bank_ids == ["primary"]
    assert provider._build_recall_bank_ids() == ["primary"]


def test_recall_skips_a_failing_extra_bank_with_a_warning(caplog):
    provider = _provider_with({"bank_id": "primary", "recall_additional_banks": ["vault", "notes"]})
    _fake_recall(provider, {"primary": ["from primary"], "vault": RuntimeError("vault down"), "notes": ["from notes"]})

    with caplog.at_level(logging.WARNING):
        texts = [r.text for r in provider._recall("query")]

    assert texts == ["from primary", "from notes"]
    assert "skipping bank vault" in caplog.text


def test_recall_raises_when_the_primary_bank_fails():
    provider = _provider_with({"bank_id": "primary", "recall_additional_banks": ["vault"]})
    _fake_recall(provider, {"primary": RuntimeError("401 Unauthorized"), "vault": ["from vault"]})

    with pytest.raises(RuntimeError, match="401"):
        provider._recall("query")


def test_single_bank_recall_error_reaches_the_tool_as_a_failure():
    provider = _provider_with({"bank_id": "primary"})
    _fake_recall(provider, {"primary": RuntimeError("connection refused")})

    result = provider.handle_tool_call("hindsight_recall", {"query": "q"})
    assert "Failed to search memory: connection refused" in result
    assert "No relevant memories found" not in result


def _fake_retain(provider, failing: set) -> list:
    """Record retained banks; raise for banks in *failing*."""
    written = []

    def retain_batch(item, bank_id, **kwargs):
        written.append(bank_id)
        if bank_id in failing:
            raise RuntimeError(f"{bank_id} unavailable")
        return SimpleNamespace(operation_id=f"op-{bank_id}")

    provider._retain_batch = retain_batch
    provider._ensure_writer = MagicMock()
    provider._register_atexit = MagicMock()
    return written


def test_turn_retain_continues_past_a_failing_extra_bank(caplog):
    provider = _provider_with({"bank_id": "primary", "additional_banks": ["broken", "shared"]})
    written = _fake_retain(provider, failing={"broken"})

    provider.sync_turn("User message", "Assistant reply", session_id="s1")
    with caplog.at_level(logging.WARNING):
        provider._retain_queue.get_nowait()()

    assert written == ["primary", "broken", "shared"]
    assert "bank broken failed" in caplog.text


def test_turn_retain_still_raises_when_the_primary_fails_after_trying_the_rest():
    provider = _provider_with({"bank_id": "primary", "additional_banks": ["shared"]})
    written = _fake_retain(provider, failing={"primary"})

    provider.sync_turn("User message", "Assistant reply", session_id="s1")
    with pytest.raises(RuntimeError, match="primary unavailable"):
        provider._retain_queue.get_nowait()()
    assert written == ["primary", "shared"]


def test_retain_tool_reports_success_when_only_an_extra_bank_fails():
    provider = _provider_with({"bank_id": "primary", "additional_banks": ["broken"]})
    written = _fake_retain(provider, failing={"broken"})

    result = provider.handle_tool_call("hindsight_retain", {"content": "a fact"})

    assert written == ["primary", "broken"]
    assert "Memory stored successfully" in result


def test_retain_tool_reports_failure_when_the_primary_fails():
    provider = _provider_with({"bank_id": "primary", "additional_banks": ["shared"]})
    written = _fake_retain(provider, failing={"primary"})

    result = provider.handle_tool_call("hindsight_retain", {"content": "a fact"})

    assert written == ["primary", "shared"]
    assert "Failed to store memory: primary unavailable" in result
