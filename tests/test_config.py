"""Tests for the configuration system."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import tomli_w

from memsearch.config import (
    MemSearchConfig,
    MilvusConfig,
    EmbeddingConfig,
    deep_merge,
    get_config_value,
    load_config_file,
    load_env_overrides,
    resolve_config,
    save_config,
    set_config_value,
)


def test_default_config():
    """MemSearchConfig() should produce sensible defaults."""
    cfg = MemSearchConfig()
    assert cfg.milvus.uri == "~/.memsearch/milvus.db"
    assert cfg.milvus.collection == "memsearch_chunks"
    assert cfg.embedding.provider == "openai"
    assert cfg.chunking.max_chunk_size == 1500
    assert cfg.chunking.overlap_lines == 2
    assert cfg.watch.debounce_ms == 1500
    assert cfg.flush.llm_provider == "openai"


def test_load_toml_file(tmp_path: Path):
    """load_config_file should parse a TOML file into a nested dict."""
    cfg_file = tmp_path / "config.toml"
    data = {
        "milvus": {"uri": "http://localhost:19530", "collection": "test_col"},
        "embedding": {"provider": "google"},
    }
    with open(cfg_file, "wb") as f:
        tomli_w.dump(data, f)

    result = load_config_file(cfg_file)
    assert result["milvus"]["uri"] == "http://localhost:19530"
    assert result["milvus"]["collection"] == "test_col"
    assert result["embedding"]["provider"] == "google"


def test_load_missing_file(tmp_path: Path):
    """load_config_file should return {} for a missing file."""
    result = load_config_file(tmp_path / "nonexistent.toml")
    assert result == {}


def test_env_overrides(monkeypatch: pytest.MonkeyPatch):
    """MEMSEARCH_* env vars should produce correct nested dict."""
    monkeypatch.setenv("MEMSEARCH_MILVUS_URI", "http://envhost:19530")
    monkeypatch.setenv("MEMSEARCH_EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("MEMSEARCH_CHUNKING_MAX_CHUNK_SIZE", "2000")
    result = load_env_overrides()
    assert result["milvus"]["uri"] == "http://envhost:19530"
    assert result["embedding"]["provider"] == "voyage"
    assert result["chunking"]["max_chunk_size"] == 2000


def test_int_type_conversion(monkeypatch: pytest.MonkeyPatch):
    """Int fields should be auto-converted from env var strings."""
    monkeypatch.setenv("MEMSEARCH_WATCH_DEBOUNCE_MS", "3000")
    monkeypatch.setenv("MEMSEARCH_CHUNKING_OVERLAP_LINES", "5")
    result = load_env_overrides()
    assert result["watch"]["debounce_ms"] == 3000
    assert isinstance(result["watch"]["debounce_ms"], int)
    assert result["chunking"]["overlap_lines"] == 5


def test_unknown_keys_ignored(monkeypatch: pytest.MonkeyPatch):
    """Env vars with unknown sections or fields should be silently dropped."""
    monkeypatch.setenv("MEMSEARCH_BOGUS_FIELD", "nope")
    monkeypatch.setenv("MEMSEARCH_MILVUS_NONEXISTENT", "nope")
    result = load_env_overrides()
    assert "bogus" not in result
    assert result.get("milvus", {}).get("nonexistent") is None


def test_deep_merge_basic():
    """deep_merge should recursively merge nested dicts."""
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99}, "c": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 99}, "b": 3, "c": 4}


def test_deep_merge_none_skipped():
    """deep_merge should skip None values in override."""
    base = {"a": {"x": 1}}
    override = {"a": {"x": None}}
    merged = deep_merge(base, override)
    assert merged["a"]["x"] == 1


def test_resolve_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """resolve_config should layer: defaults < toml < env < cli."""
    # Write a "global" config
    global_cfg = tmp_path / "global.toml"
    save_config({"milvus": {"uri": "http://toml:19530"}}, global_cfg)

    # Patch the paths
    monkeypatch.setattr("memsearch.config.GLOBAL_CONFIG_PATH", global_cfg)
    monkeypatch.setattr("memsearch.config.PROJECT_CONFIG_PATH", tmp_path / "nope.toml")

    # Set env override
    monkeypatch.setenv("MEMSEARCH_EMBEDDING_PROVIDER", "google")

    # CLI override
    cli = {"milvus": {"collection": "cli_col"}}

    cfg = resolve_config(cli)
    # TOML wins over default
    assert cfg.milvus.uri == "http://toml:19530"
    # env wins over default
    assert cfg.embedding.provider == "google"
    # CLI wins over everything
    assert cfg.milvus.collection == "cli_col"
    # Untouched fields remain default
    assert cfg.chunking.max_chunk_size == 1500


def test_set_get_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """set_config_value + get_config_value should round-trip correctly."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr("memsearch.config.GLOBAL_CONFIG_PATH", cfg_path)
    monkeypatch.setattr("memsearch.config.PROJECT_CONFIG_PATH", tmp_path / "nope.toml")

    # Clear env to avoid interference
    for key in list(os.environ):
        if key.startswith("MEMSEARCH_"):
            monkeypatch.delenv(key, raising=False)

    set_config_value("milvus.uri", "http://roundtrip:19530")
    cfg = resolve_config()
    assert get_config_value("milvus.uri", cfg) == "http://roundtrip:19530"


def test_set_config_value_int_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """set_config_value should auto-convert int fields from strings."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr("memsearch.config.GLOBAL_CONFIG_PATH", cfg_path)

    set_config_value("chunking.max_chunk_size", "2000")
    data = load_config_file(cfg_path)
    assert data["chunking"]["max_chunk_size"] == 2000
    assert isinstance(data["chunking"]["max_chunk_size"], int)


def test_get_config_value_invalid_key():
    """get_config_value should raise KeyError for unknown keys."""
    cfg = MemSearchConfig()
    with pytest.raises(KeyError):
        get_config_value("nonexistent.key", cfg)


def test_save_and_load_roundtrip(tmp_path: Path):
    """save_config + load_config_file should round-trip a dict."""
    data = {"milvus": {"uri": "http://test:19530"}, "embedding": {"provider": "local"}}
    path = tmp_path / "test.toml"
    save_config(data, path)
    loaded = load_config_file(path)
    assert loaded == data
