"""Tests for per-project bank walk-up and multi-bank fan-out."""

from unittest.mock import MagicMock
from pathlib import Path
from types import SimpleNamespace

import hindsight_hermes as plugin
from hindsight_hermes.settings import _discover_cwd_bank_id

HindsightMemoryProvider = plugin.HindsightMemoryProvider


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


def test_provider_multibank_write_and_recall_order():
    provider = HindsightMemoryProvider()
    provider._config = {"bank_id": "default-bank"}
    provider._apply_connection_settings({
        "bank_id": "default-bank",
        "mirror_to_own_bank": True,
        "additional_banks": ["shared-knowledge", "global-bank"],
    })
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

    # Mock _run_hindsight_operation
    def mock_run(op):
        mock_client = MagicMock()
        def arecall(bank_id, **kwargs):
            if bank_id == "primary":
                return SimpleNamespace(results=[SimpleNamespace(text="Fact 1"), SimpleNamespace(text="Fact 2")])
            elif bank_id == "secondary":
                return SimpleNamespace(results=[SimpleNamespace(text="Fact 2"), SimpleNamespace(text="Fact 3")])
            return SimpleNamespace(results=[])
        mock_client.arecall = arecall
        return op(mock_client)

    provider._run_hindsight_operation = mock_run

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
