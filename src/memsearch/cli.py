"""CLI interface for memsearch."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from .config import (
    MemSearchConfig,
    config_to_dict,
    get_config_value,
    load_config_file,
    resolve_config,
    save_config,
    set_config_value,
    GLOBAL_CONFIG_PATH,
    PROJECT_CONFIG_PATH,
    _SECTION_CLASSES,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# -- CLI param name → dotted config key mapping --
_PARAM_MAP = {
    "provider": "embedding.provider",
    "model": "embedding.model",
    "collection": "milvus.collection",
    "milvus_uri": "milvus.uri",
    "milvus_token": "milvus.token",
    "llm_provider": "flush.llm_provider",
    "llm_model": "flush.llm_model",
    "prompt_file": "flush.prompt_file",
    "max_chunk_size": "chunking.max_chunk_size",
    "overlap_lines": "chunking.overlap_lines",
    "debounce_ms": "watch.debounce_ms",
}


def _build_cli_overrides(**kwargs) -> dict:
    """Map flat CLI params to a nested config override dict.

    Only non-None values are included (None means "not set by user").
    """
    result: dict = {}
    for param, dotted_key in _PARAM_MAP.items():
        val = kwargs.get(param)
        if val is None:
            continue
        section, field = dotted_key.split(".")
        result.setdefault(section, {})[field] = val
    return result


def _cfg_to_memsearch_kwargs(cfg: MemSearchConfig) -> dict:
    """Extract MemSearch constructor kwargs from a resolved config."""
    return {
        "embedding_provider": cfg.embedding.provider,
        "embedding_model": cfg.embedding.model or None,
        "milvus_uri": cfg.milvus.uri,
        "milvus_token": cfg.milvus.token or None,
        "collection": cfg.milvus.collection,
        "max_chunk_size": cfg.chunking.max_chunk_size,
        "overlap_lines": cfg.chunking.overlap_lines,
    }


# -- Common CLI options --

def _common_options(f):
    """Shared options for commands that create a MemSearch instance."""
    f = click.option("--provider", "-p", default=None, help="Embedding provider.")(f)
    f = click.option("--model", "-m", default=None, help="Override embedding model.")(f)
    f = click.option("--collection", "-c", default=None, help="Milvus collection name.")(f)
    f = click.option("--milvus-uri", default=None, help="Milvus connection URI.")(f)
    f = click.option("--milvus-token", default=None, help="Milvus auth token.")(f)
    return f


@click.group()
@click.version_option(package_name="memsearch")
def cli() -> None:
    """memsearch — semantic memory search for markdown knowledge bases."""


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@_common_options
@click.option("--force", is_flag=True, help="Re-index all files.")
def index(
    paths: tuple[str, ...],
    provider: str | None,
    model: str | None,
    collection: str | None,
    milvus_uri: str | None,
    milvus_token: str | None,
    force: bool,
) -> None:
    """Index markdown files from PATHS."""
    from .core import MemSearch

    cfg = resolve_config(_build_cli_overrides(
        provider=provider, model=model, collection=collection,
        milvus_uri=milvus_uri, milvus_token=milvus_token,
    ))
    ms = MemSearch(list(paths), **_cfg_to_memsearch_kwargs(cfg))
    try:
        n = _run(ms.index(force=force))
        click.echo(f"Indexed {n} chunks.")
    finally:
        ms.close()


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=None, type=int, help="Number of results.")
@_common_options
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON.")
def search(
    query: str,
    top_k: int | None,
    provider: str | None,
    model: str | None,
    collection: str | None,
    milvus_uri: str | None,
    milvus_token: str | None,
    json_output: bool,
) -> None:
    """Search indexed memory for QUERY."""
    from .core import MemSearch

    cfg = resolve_config(_build_cli_overrides(
        provider=provider, model=model, collection=collection,
        milvus_uri=milvus_uri, milvus_token=milvus_token,
    ))
    ms = MemSearch(**_cfg_to_memsearch_kwargs(cfg))
    try:
        results = _run(ms.search(query, top_k=top_k or 5))
        if json_output:
            click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            if not results:
                click.echo("No results found.")
                return
            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                source = r.get("source", "?")
                heading = r.get("heading", "")
                content = r.get("content", "")
                click.echo(f"\n--- Result {i} (score: {score:.4f}) ---")
                click.echo(f"Source: {source}")
                if heading:
                    click.echo(f"Heading: {heading}")
                click.echo(content[:500])
    finally:
        ms.close()


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@_common_options
@click.option("--debounce-ms", default=None, type=int, help="Debounce delay in ms.")
def watch(
    paths: tuple[str, ...],
    provider: str | None,
    model: str | None,
    collection: str | None,
    milvus_uri: str | None,
    milvus_token: str | None,
    debounce_ms: int | None,
) -> None:
    """Watch PATHS for markdown changes and auto-index."""
    from .core import MemSearch

    cfg = resolve_config(_build_cli_overrides(
        provider=provider, model=model, collection=collection,
        milvus_uri=milvus_uri, milvus_token=milvus_token,
        debounce_ms=debounce_ms,
    ))
    ms = MemSearch(list(paths), **_cfg_to_memsearch_kwargs(cfg))

    def _on_event(event_type: str, summary: str, file_path) -> None:
        click.echo(summary)

    click.echo(f"Watching {len(paths)} path(s) for changes... (Ctrl+C to stop)")
    watcher = ms.watch(on_event=_on_event, debounce_ms=cfg.watch.debounce_ms)
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping watcher.")
    finally:
        watcher.stop()
        ms.close()


@cli.command()
@click.option("--source", "-s", default=None, help="Only flush chunks from this source.")
@click.option("--llm-provider", default=None, help="LLM for summarization.")
@click.option("--llm-model", default=None, help="Override LLM model.")
@click.option("--prompt", default=None, help="Custom prompt template (must contain {chunks}).")
@click.option("--prompt-file", default=None, type=click.Path(exists=True), help="Read prompt template from file.")
@_common_options
def flush(
    source: str | None,
    llm_provider: str | None,
    llm_model: str | None,
    prompt: str | None,
    prompt_file: str | None,
    provider: str | None,
    model: str | None,
    collection: str | None,
    milvus_uri: str | None,
    milvus_token: str | None,
) -> None:
    """Compress stored memories into a summary."""
    from .core import MemSearch

    cfg = resolve_config(_build_cli_overrides(
        provider=provider, model=model, collection=collection,
        milvus_uri=milvus_uri, milvus_token=milvus_token,
        llm_provider=llm_provider, llm_model=llm_model,
        prompt_file=prompt_file,
    ))

    prompt_template = prompt
    if cfg.flush.prompt_file and not prompt_template:
        prompt_template = Path(cfg.flush.prompt_file).read_text(encoding="utf-8")

    ms = MemSearch(**_cfg_to_memsearch_kwargs(cfg))
    try:
        summary = _run(ms.flush(
            source=source,
            llm_provider=cfg.flush.llm_provider,
            llm_model=cfg.flush.llm_model or None,
            prompt_template=prompt_template,
        ))
        if summary:
            click.echo("Flush complete. Summary:\n")
            click.echo(summary)
        else:
            click.echo("No chunks to flush.")
    finally:
        ms.close()


@cli.command()
@click.option("--collection", "-c", default=None, help="Milvus collection name.")
@click.option("--milvus-uri", default=None, help="Milvus connection URI.")
@click.option("--milvus-token", default=None, help="Milvus auth token.")
def stats(
    collection: str | None,
    milvus_uri: str | None,
    milvus_token: str | None,
) -> None:
    """Show statistics about the index."""
    from .store import MilvusStore

    cfg = resolve_config(_build_cli_overrides(
        collection=collection, milvus_uri=milvus_uri, milvus_token=milvus_token,
    ))
    store = MilvusStore(
        uri=cfg.milvus.uri,
        token=cfg.milvus.token or None,
        collection=cfg.milvus.collection,
    )
    try:
        count = store.count()
        click.echo(f"Total indexed chunks: {count}")
    finally:
        store.close()


@cli.command()
@click.option("--collection", "-c", default=None, help="Milvus collection name.")
@click.option("--milvus-uri", default=None, help="Milvus connection URI.")
@click.option("--milvus-token", default=None, help="Milvus auth token.")
@click.confirmation_option(prompt="This will delete all indexed data. Continue?")
def reset(
    collection: str | None,
    milvus_uri: str | None,
    milvus_token: str | None,
) -> None:
    """Drop all indexed data."""
    from .store import MilvusStore

    cfg = resolve_config(_build_cli_overrides(
        collection=collection, milvus_uri=milvus_uri, milvus_token=milvus_token,
    ))
    store = MilvusStore(
        uri=cfg.milvus.uri,
        token=cfg.milvus.token or None,
        collection=cfg.milvus.collection,
    )
    try:
        store.drop()
        click.echo("Dropped collection.")
    finally:
        store.close()


# ======================================================================
# Config command group
# ======================================================================

@cli.group("config")
def config_group() -> None:
    """Manage memsearch configuration."""


@config_group.command("init")
@click.option("--project", is_flag=True, help="Write to .memsearch.toml (project-level) instead of global.")
def config_init(project: bool) -> None:
    """Interactive configuration wizard."""
    from dataclasses import fields as dc_fields

    target = PROJECT_CONFIG_PATH if project else GLOBAL_CONFIG_PATH
    existing = load_config_file(target)
    current = resolve_config()

    result: dict = {}

    click.echo(f"memsearch configuration wizard")
    click.echo(f"Writing to: {target}\n")

    # Milvus
    click.echo("── Milvus ──")
    result["milvus"] = {}
    result["milvus"]["uri"] = click.prompt(
        "  Milvus URI", default=current.milvus.uri,
    )
    result["milvus"]["token"] = click.prompt(
        "  Milvus token (empty for none)", default=current.milvus.token,
    )
    result["milvus"]["collection"] = click.prompt(
        "  Collection name", default=current.milvus.collection,
    )

    # Embedding
    click.echo("\n── Embedding ──")
    result["embedding"] = {}
    result["embedding"]["provider"] = click.prompt(
        "  Provider (openai/google/voyage/ollama/local)",
        default=current.embedding.provider,
    )
    result["embedding"]["model"] = click.prompt(
        "  Model (empty for provider default)", default=current.embedding.model,
    )

    # Chunking
    click.echo("\n── Chunking ──")
    result["chunking"] = {}
    result["chunking"]["max_chunk_size"] = click.prompt(
        "  Max chunk size (chars)", default=current.chunking.max_chunk_size, type=int,
    )
    result["chunking"]["overlap_lines"] = click.prompt(
        "  Overlap lines", default=current.chunking.overlap_lines, type=int,
    )

    # Watch
    click.echo("\n── Watch ──")
    result["watch"] = {}
    result["watch"]["debounce_ms"] = click.prompt(
        "  Debounce (ms)", default=current.watch.debounce_ms, type=int,
    )

    # Flush
    click.echo("\n── Flush ──")
    result["flush"] = {}
    result["flush"]["llm_provider"] = click.prompt(
        "  LLM provider", default=current.flush.llm_provider,
    )
    result["flush"]["llm_model"] = click.prompt(
        "  LLM model (empty for default)", default=current.flush.llm_model,
    )
    result["flush"]["prompt_file"] = click.prompt(
        "  Prompt file path (empty for built-in)", default=current.flush.prompt_file,
    )

    save_config(result, target)
    click.echo(f"\nConfig saved to {target}")


@config_group.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--project", is_flag=True, help="Write to project config.")
def config_set(key: str, value: str, project: bool) -> None:
    """Set a config value (e.g. memsearch config set milvus.uri http://host:19530)."""
    try:
        set_config_value(key, value, project=project)
        target = PROJECT_CONFIG_PATH if project else GLOBAL_CONFIG_PATH
        click.echo(f"Set {key} = {value} in {target}")
    except (KeyError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@config_group.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a resolved config value (e.g. memsearch config get milvus.uri)."""
    try:
        val = get_config_value(key)
        click.echo(val)
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@config_group.command("list")
@click.option("--resolved", "mode", flag_value="resolved", default=True, help="Show fully resolved config (default).")
@click.option("--global", "mode", flag_value="global", help="Show global config file only.")
@click.option("--project", "mode", flag_value="project", help="Show project config file only.")
def config_list(mode: str) -> None:
    """Show configuration."""
    import tomli_w

    if mode == "global":
        data = load_config_file(GLOBAL_CONFIG_PATH)
        label = f"Global ({GLOBAL_CONFIG_PATH})"
    elif mode == "project":
        data = load_config_file(PROJECT_CONFIG_PATH)
        label = f"Project ({PROJECT_CONFIG_PATH})"
    else:
        cfg = resolve_config()
        data = config_to_dict(cfg)
        label = "Resolved (all sources merged)"

    click.echo(f"# {label}\n")
    if data:
        click.echo(tomli_w.dumps(data))
    else:
        click.echo("(empty)")
