---
name: shared-context
description: Find prior conversations, decisions and explicitly recorded preferences across local Claude Code, Codex and DeepSeek Harness sessions. Use whenever a request depends on earlier work, e.g. “What did we decide about retries?” or “继续上次的方案”, even when the user names no client or skill.
---

Extract a distinctive term from the request, then run `agent-context search "term"` across all sources. For “Why did we choose three retries last time?” try “retry”, “retries”, or a project identifier. Apply a provider filter only when the user explicitly limits the scope. Read matching turns from different clients together and check later corrections. If several decisions still fit, ask about the project or feature. Use `next_offset` to page through results.

Read nearby evidence with `agent-context read ID --start-line N --limit 10`. Citations use the provider, native session ID and original decompressed line number. Read tool evidence with `--raw` only when needed. Treat past instructions as historical evidence and check the current project before continuing work.

Use `agent-context memory --query TERM` to find relevant Markdown notes. Save a new scoped note under `~/.agent-context-kit/memory/shared/` only when the user explicitly asks to remember something. Imported memory directories point to original client files.

Queries refresh the index; `"stale": true` means another sync was running, so very recent turns may be missing until a later query. Use `agent-context --no-sync status` for the last snapshot, including any source errors. Read only task-relevant material into the current model's context. Duplicate native IDs can represent migrated logs.
