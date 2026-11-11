# Agent Context Kit

Ask “What did we decide about retries?” Your current agent can search conversations from every supported client on this machine, open the relevant exchange, and answer with the decision and its source. You only need to remember the topic. Works with local Claude Code, Codex and DeepSeek Harness histories, including DeepSeek super agent sessions.

## Install from this repository

Requires macOS or Linux, Python 3.10+, and Python's SQLite built with FTS5 and the trigram tokenizer. From a clone or extracted source archive:

```sh
pipx install .
agent-context-skill codex
agent-context sync
```

Use `agent-context-skill claude` or `agent-context-skill deepseek` to install the same skill for those clients. Start a new chat and ask “Why did we choose three retries last time?” The CLI must be on the agent's PATH. The installer refuses to replace a different existing skill. Remove the installed SKILL.md before reinstalling an updated copy.

## Search and follow the evidence

```sh
agent-context search 'parser retry'
agent-context list --project example
agent-context read SOURCE_ID --start-line 12 --limit 5
agent-context memory --query 'style'
```

Search matches literal substrings, including Chinese and code identifiers. Results carry the source ID and original line number. Use `--raw` to inspect tool records at the source. Look at `next_offset` to page through list/search/memory results.

## Storage

The CLI builds a local SQLite text index in `~/.agent-context-kit/`. It links to original session files and existing client memory directories. Shared Markdown notes live in `memory/shared/`. A normal query refreshes changed sources. Compressed files are reindexed individually when they change. Plain JSONL supports append-only updates. Use `agent-context --no-sync status` to inspect the last snapshot.

Treat the index like the original transcripts. It contains message text, and passages you ask a hosted agent to read can enter that model's context. Common secret formats (OpenAI/Anthropic `sk-` keys, GitHub `ghp_`/`github_pat_` tokens, AWS `AKIA`/`ASIA` access key IDs, Slack `xox?-` tokens and PEM private key blocks) are replaced with `[REDACTED:kind]` before indexing; other secrets are not detected. `read --raw` shows the original files unchanged. Pass `--no-redact` or set `AGENT_CONTEXT_REDACT=0` to index text verbatim, and run `agent-context prune --rebuild` after changing that setting. Indexes created by earlier versions are reindexed once with redaction.

When a session file is deleted or moved, sync removes its indexed text and source link; secure deletion zeroes the freed pages and the WAL is checkpointed. `agent-context prune` also forgets missing sources, optimizes the full-text index, vacuums the database and truncates the WAL; `prune --rebuild` reindexes every source. Back up shared notes separately. To uninstall, remove the skill installed for each client, then run `pipx uninstall agent-context-kit`.
