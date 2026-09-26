"""Tests for per-project bank walk-up and multi-bank fan-out."""

import asyncio
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import SimpleNamespace

import pytest

import hindsight_hermes as plugin
from hindsight_hermes.settings import _discover_cwd_bank_id, _normalize_string_list, _repository_root

HindsightMemoryProvider = plugin.HindsightMemoryProvider


@pytest.fixture(autouse=True)
def _pin_update_mode_probe(monkeypatch):
    # The /version probe is a live HTTP call; pin the capability, as conftest's provider fixture does.
    monkeypatch.setattr(plugin, "_check_api_supports_update_mode_append", lambda *a, **k: True)


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


def _discover(start: Path, trusted: list) -> str | None:
    return _discover_cwd_bank_id(str(start), _repository_root(str(start)), trusted)


def _repo_with_config(root: Path, bank: str) -> Path:
    (root / ".git").mkdir(parents=True)
    (root / ".hindsight").mkdir()
    (root / ".hindsight" / "config.toml").write_text(f'bank_id = "{bank}"\n', encoding="utf-8")
    sub = root / "src" / "deep"
    sub.mkdir(parents=True)
    return sub


def test_trusted_repository_config_sets_the_bank(tmp_path: Path):
    sub = _repo_with_config(tmp_path / "work" / "my_project", "project-bank-alpha")
    assert _discover(sub, [str(tmp_path / "work")]) == "project-bank-alpha"


def test_repository_config_is_ignored_unless_its_folder_is_trusted(tmp_path: Path):
    sub = _repo_with_config(tmp_path / "downloads" / "cloned", "attacker-bank")
    assert _discover(sub, []) is None
    assert _discover(sub, [str(tmp_path / "work")]) is None


def test_repository_config_is_never_read_above_the_repository_root(tmp_path: Path):
    outer = tmp_path / "work"
    (outer / ".hindsight").mkdir(parents=True)
    (outer / ".hindsight" / "config.toml").write_text('bank_id = "outer"\n', encoding="utf-8")
    repo = outer / "repo"
    (repo / ".git").mkdir(parents=True)

    assert _discover(repo, [str(outer)]) is None


def test_repository_config_outside_a_repository_is_ignored(tmp_path: Path):
    folder = tmp_path / "work" / "notes"
    (folder / ".hindsight").mkdir(parents=True)
    (folder / ".hindsight" / "config.toml").write_text('bank_id = "notes"\n', encoding="utf-8")

    assert _discover(folder, [str(tmp_path / "work")]) is None


def test_provider_uses_trusted_repository_config_over_the_template(tmp_path: Path):
    sub = _repo_with_config(tmp_path / "work" / "project_x", "project-x-bank")
    config = {
        "bank_id": "hermes",
        "bank_id_template": "hermes-{project}",
        "trusted_project_dirs": str(tmp_path / "work"),
    }
    provider = HindsightMemoryProvider()
    with patch.object(plugin, "_load_config", return_value=config):
        provider.initialize(session_id="s1", cwd=str(sub))
    assert provider._bank_id == "project-x-bank"


def _bank_for(cwd: str, template: str, **session) -> str:
    provider = HindsightMemoryProvider()
    with patch.object(plugin, "_load_config", return_value={"bank_id": "hermes", "bank_id_template": template}):
        provider.initialize(session_id="s", cwd=cwd, **session)
    return provider._bank_id


def test_project_placeholder_is_the_git_repository_name(tmp_path: Path):
    repo = tmp_path / "my-awesome-repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "pkg" / "sub"
    sub.mkdir(parents=True)

    assert _bank_for(str(sub), "{project}") == "my-awesome-repo"
    assert _bank_for(str(sub), "hermes-{project}") == "hermes-my-awesome-repo"


def test_project_placeholder_resolves_a_linked_worktree_to_its_main_repository(tmp_path: Path):
    main = tmp_path / "main-repo"
    worktree_git = main / ".git" / "worktrees" / "feature-x"
    worktree_git.mkdir(parents=True)
    (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
    worktree = tmp_path / "main-repo-feature-x"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")

    assert _bank_for(str(worktree), "{project}") == "main-repo"


def test_project_placeholder_is_empty_outside_a_repository(tmp_path: Path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    assert _bank_for(str(scratch), "{project}") == "hermes"
    assert _bank_for(str(scratch), "hermes-{project}") == "hermes"
    assert _bank_for(str(Path.home()), "{project}") == "hermes"


def test_workspace_placeholder_keeps_its_upstream_meaning(tmp_path: Path):
    repo = tmp_path / "some-repo"
    (repo / ".git").mkdir(parents=True)

    assert _bank_for(str(repo), "hermes-{workspace}", agent_workspace="hermes") == "hermes-hermes"
    assert _bank_for(str(repo), "{workspace}", agent_workspace="team-a") == "team-a"


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
    assert "skipping bank broken" in caplog.text


def test_turn_retain_raises_when_the_primary_fails_before_touching_extra_banks():
    provider = _provider_with({"bank_id": "primary", "additional_banks": ["shared"]})
    written = _fake_retain(provider, failing={"primary"})

    provider.sync_turn("User message", "Assistant reply", session_id="s1")
    with pytest.raises(RuntimeError, match="primary unavailable"):
        provider._retain_queue.get_nowait()()
    assert written == ["primary"]


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

    assert written == ["primary"]
    assert "Failed to store memory: primary unavailable" in result


def test_mirror_uses_the_nested_bank_id_form_like_the_fallback(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    config = {"banks": {"hermes": {"bankId": "personal"}}, "bank_id_template": "{project}", "mirror_to_own_bank": True}
    provider = HindsightMemoryProvider()
    with patch.object(plugin, "_load_config", return_value=config):
        provider.initialize(session_id="s1", cwd=str(repo))

    assert provider._bank_id == "repo"
    assert provider._write_bank_ids == ["repo", "personal"]


def test_a_hung_extra_bank_cannot_cost_the_primary_its_results():
    provider = _provider_with({"bank_id": "primary", "recall_additional_banks": ["slow"]})
    provider._timeout = 0.5

    class _Client:
        async def arecall(self, bank_id, **kwargs):
            if bank_id == "slow":
                await asyncio.sleep(5)
            return SimpleNamespace(results=[SimpleNamespace(text=f"from {bank_id}")])

    provider._run_hindsight_operation = lambda op: asyncio.run(op(_Client()))
    assert [r.text for r in provider._recall("q")] == ["from primary"]


def test_worktree_trust_follows_the_main_repository(tmp_path: Path):
    main = tmp_path / "work" / "app"
    worktree_git = main / ".git" / "worktrees" / "wt"
    worktree_git.mkdir(parents=True)
    (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
    worktree = tmp_path / "elsewhere" / "app-wt"
    (worktree / ".hindsight").mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")
    (worktree / ".hindsight" / "config.toml").write_text('bank_id = "acme"\n', encoding="utf-8")

    assert _discover(worktree, [str(tmp_path / "work")]) == "acme"
    assert _discover(worktree, [str(tmp_path / "elsewhere")]) is None


def test_relative_trusted_folders_are_ignored(tmp_path: Path, monkeypatch):
    sub = _repo_with_config(tmp_path / "cloned", "attacker-bank")
    monkeypatch.chdir(tmp_path)
    assert _discover(sub, [".", "cloned"]) is None


def test_project_placeholder_is_found_with_a_format_spec(tmp_path: Path):
    repo = tmp_path / "long-repository-name"
    (repo / ".git").mkdir(parents=True)
    assert _bank_for(str(repo), "{project:.4}") == "long"


def test_empty_nested_bank_id_never_puts_an_empty_bank_in_the_write_set(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    config = {"banks": {"hermes": {"bankId": ""}}, "bank_id_template": "{project}", "mirror_to_own_bank": True}
    provider = HindsightMemoryProvider()
    with patch.object(plugin, "_load_config", return_value=config):
        provider.initialize(session_id="s1", cwd=str(repo))
    assert provider._write_bank_ids == ["repo", "hermes"]


def test_a_failing_extra_bank_warns_once(caplog):
    provider = _provider_with({"bank_id": "primary", "recall_additional_banks": ["vault"]})
    _fake_recall(provider, {"primary": ["from primary"], "vault": RuntimeError("vault down")})
    with caplog.at_level(logging.DEBUG):
        provider._recall("q")
        provider._recall("q")
    warnings = [r for r in caplog.records if "skipping bank vault" in r.getMessage()]
    assert [r.levelno for r in warnings] == [logging.WARNING, logging.DEBUG]


def test_bracketed_text_that_is_not_json_is_read_as_a_list():
    assert _normalize_string_list("[team, vault]") == ["team", "vault"]
    assert _normalize_string_list("['team', \"vault\"]") == ["team", "vault"]
