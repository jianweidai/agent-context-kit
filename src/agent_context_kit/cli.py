#!/usr/bin/env python3
"""Local, source-mapped Claude Code / Codex / DeepSeek history. No network or model calls."""
import argparse
import contextlib
import datetime as dt
import fcntl
import functools
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time


SCHEMA_VERSION = 1  # 1: message text is redacted on insert.
COMMON_CANDIDATES = 2000  # trigram candidates beyond which search walks messages by time instead
SHORT_QUERY_WINDOW = 20000  # newest messages tried first for queries shorter than a trigram
MAX_SPELLINGS = 16  # trigram phrases tried for case variants the tokenizer does not fold
TAIL_SETTLE_SECONDS = 600  # an unterminated last line is indexed once the file has been idle this long

# (kind, pattern, check): conservative token shapes; `check` rejects look-alikes such as kebab-case names.
SECRETS = [
    ('private-key', re.compile(r'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?'
                               r'(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|\Z)', re.S), None),
    ('api-key', re.compile(r'(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}'),
     lambda t: re.search(r'[A-Za-z0-9]{20}', t) and re.search(r'[0-9]', t)),
    ('github-token', re.compile(r'(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})'), None),
    ('aws-access-key', re.compile(r'(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])'),
     lambda t: re.search(r'[0-9]', t)),
    ('slack-token', re.compile(r'(?<![A-Za-z0-9-])xox[abposr]-[A-Za-z0-9-]{10,}'),
     lambda t: re.search(r'[0-9]', t)),
]


SECRET_HINT = re.compile(r'sk-|gh[pousr]_|github_pat_|AKIA|ASIA|xox[abposr]-|PRIVATE KEY')


def redact(text):
    if not SECRET_HINT.search(text):
        return text
    for kind, pattern, check in SECRETS:
        text = pattern.sub(lambda m: m[0] if check and not check(m[0]) else f'[REDACTED:{kind}]', text)
    return text


def fold_char(c):
    folded = c.casefold()
    if len(folded) == 1:
        return folded
    lower = c.lower()
    return lower if len(lower) == 1 else c


def fold(text):
    """Unicode case folding that keeps one character per character (ß stays ß), so offsets carry over."""
    if not isinstance(text, str):
        return text
    folded = text.casefold()
    return folded if len(folded) == len(text) else ''.join(map(fold_char, text))


def caseless(c):
    return c.upper() == c == c.lower()


# The only non-ASCII characters that fold to ASCII letters: KELVIN SIGN -> k, LONG S -> s.
ASCII_FOLDS = {'k': '\u212a', 's': '\u017f'}


@functools.lru_cache(maxsize=None)
def case_tables():
    """Map each folded character to all characters folding to it, and to the subset of those that this
    SQLite's trigram tokenizer does not fold to it (its Unicode tables can be older than Python's), e.g.
    CYRILLIC SMALL LETTER ROUNDED VE for в."""
    pairs = [(c, fold_char(c)) for c in map(chr, range(0x110000))
             if not 0xD800 <= ord(c) <= 0xDFFF and fold_char(c) != c]
    db = sqlite3.connect(':memory:')
    db.execute("CREATE VIRTUAL TABLE t USING fts5(text, tokenize='trigram')")
    db.executemany('INSERT INTO t(rowid,text) VALUES(?,?)', ((i, '#' + c + '#') for i, (c, _) in enumerate(pairs)))
    found = {r[0] for f in {f for _, f in pairs} for r in db.execute('SELECT rowid FROM t WHERE t MATCH ?',
                                                                    ('"#' + f.replace('"', '""') + '#"',))}
    db.close()
    classes, unfolded = {}, {}
    for i, (c, f) in enumerate(pairs):
        classes.setdefault(f, [f]).append(c)
        if i not in found:
            unfolded.setdefault(f, []).append(c)
    return {f: tuple(v) for f, v in classes.items()}, {f: tuple(v) for f, v in unfolded.items()}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.readlink() == target:
            return
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"Refusing to replace existing file: {path}")
    path.symlink_to(target)


def connect(root):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    db = sqlite3.connect(root / 'index.sqlite', timeout=60)
    db.row_factory = sqlite3.Row
    db.create_function('fold', 1, fold, deterministic=True)
    # Deleted messages must not linger in freed pages (see purge/prune).
    db.execute('PRAGMA secure_delete=ON')
    db.execute('PRAGMA journal_mode=WAL')
    version = db.execute('PRAGMA user_version').fetchone()[0]
    db.executescript('''
    CREATE TABLE IF NOT EXISTS sources (
      id TEXT PRIMARY KEY, provider TEXT, path TEXT UNIQUE, native_id TEXT,
      project TEXT DEFAULT '', title TEXT DEFAULT '', size INTEGER DEFAULT 0,
      mtime INTEGER DEFAULT 0, offset INTEGER DEFAULT 0, lines INTEGER DEFAULT 0,
      digest TEXT DEFAULT '', available INTEGER DEFAULT 1, error TEXT DEFAULT '',
      synced_at TEXT DEFAULT '', updated_at TEXT DEFAULT '', bad_lines TEXT DEFAULT '');
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY, source_id TEXT, line INTEGER, role TEXT,
      timestamp TEXT, text TEXT, UNIQUE(source_id,line));
    CREATE INDEX IF NOT EXISTS messages_source ON messages(source_id,line);
    CREATE INDEX IF NOT EXISTS messages_time ON messages(timestamp);
    CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(text, content='messages',
      content_rowid='id', tokenize='trigram');
    CREATE TRIGGER IF NOT EXISTS message_insert AFTER INSERT ON messages BEGIN
      INSERT INTO search(rowid,text) VALUES(new.id,new.text); END;
    CREATE TRIGGER IF NOT EXISTS message_delete AFTER DELETE ON messages BEGIN
      INSERT INTO search(search,rowid,text) VALUES('delete',old.id,old.text); END;
    ''')
    # Indexes created before bad_lines existed; their errored sources are retried once.
    if 'bad_lines' not in {r['name'] for r in db.execute('PRAGMA table_info(sources)')}:
        try:
            db.execute("ALTER TABLE sources ADD COLUMN bad_lines TEXT DEFAULT ''")
        except sqlite3.OperationalError as exc:
            if 'duplicate column' not in str(exc):
                raise
    if version < SCHEMA_VERSION:
        with db:
            if version < 1:
                # Text stored before redaction existed is reindexed on the next sync.
                db.execute("UPDATE sources SET size=-1,mtime=0,offset=0,lines=0,digest='',bad_lines=''")
            db.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
    return db


@contextlib.contextmanager
def stream(path):
    if path.suffix in ('.zst', '.zstd'):
        import zstandard
        with io.BufferedReader(zstandard.open(path, 'rb')) as f:
            yield f
    else:
        with path.open('rb') as f:
            yield f


def text_blocks(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n'.join(c.get('text', '') for c in content
                     if isinstance(c, dict) and isinstance(c.get('text'), str))


def parse(provider, obj):
    """Only visible user/assistant text is indexed. Raw tools remain mapped."""
    meta, message = {}, None
    if provider == 'deepseek':
        kind = obj.get('type')
        data = obj.get('data', {})
        if kind == 'session':
            meta = {'native_id': obj.get('id', ''), 'project': obj.get('cwd', '')}
        elif kind == 'session/title':
            meta['title'] = data.get('title', '')
        elif kind == 'working-directory/change':
            meta['project'] = data.get('cwd', '')
        elif kind in ('user/message', 'assistant/message'):
            payload = data.get('message', data)
            source = payload.get('source', {}).get('kind', 'unknown')
            role = 'assistant' if kind == 'assistant/message' else {
                'user': 'user', 'dsh-superagent': 'agent', 'agent-message': 'agent'
            }.get(source, 'context')
            # Runtime instructions/catalogs are not human statements or searchable conversation.
            body = '\n'.join(c['text'] for c in payload.get('content', [])
                             if isinstance(c, dict) and c.get('type') == 'text' and isinstance(c.get('text'), str))
            if body and role != 'context':
                try:
                    timestamp = dt.datetime.fromtimestamp(obj.get('time') / 1000, dt.timezone.utc).isoformat()
                except (TypeError, ValueError, OverflowError, OSError):
                    timestamp = ''
                message = (role, timestamp, body)
    elif provider == 'claude':
        if obj.get('sessionId'):
            meta['native_id'] = obj['sessionId']
        if obj.get('cwd'):
            meta['project'] = obj['cwd']
        if obj.get('type') == 'ai-title' and obj.get('aiTitle'):
            meta['title'] = obj['aiTitle']
        if obj.get('type') in ('user', 'assistant'):
            p = obj.get('message', {})
            role = p.get('role', obj['type'])
            text = text_blocks(p.get('content'))
            if text:
                message = (role, obj.get('timestamp', ''), text)
    else:
        p = obj.get('payload', {})
        if not isinstance(p, dict):
            return meta, message
        if obj.get('type') == 'session_meta':
            meta['native_id'] = p.get('id') or p.get('session_id') or ''
            meta['project'] = p.get('cwd', '')
        if obj.get('type') == 'response_item' and p.get('type') == 'message':
            role = p.get('role')
            text = text_blocks(p.get('content'))
            if role in ('user', 'assistant') and text:
                message = (role, obj.get('timestamp', ''), text)
    return meta, message


def discover(home):
    for provider, directory in [('claude', '.claude/projects'),
                                ('codex', '.codex/sessions'),
                                ('codex', '.codex/archived_sessions'),
                                ('deepseek', '.dsh/sessions')]:
        root = home / directory
        if root.exists():
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.name.endswith(('.jsonl', '.jsonl.zst', '.jsonl.zstd')):
                    yield provider, path


def map_memories(home, root):
    mappings = []
    for name in ['memories', 'memories_v2']:
        p = home / '.codex' / name
        if p.is_dir():
            mappings.append((f'codex-{name}', p))
    for p in sorted((home / '.claude/projects').glob('*/memory')):
        if p.is_dir():
            mappings.append(('claude-' + p.parent.name, p))
    expected = set()
    for name, target in mappings:
        dest = root / 'memory/imported' / name
        expected.add(dest.name)
        link(dest, target)
    imported = root / 'memory/imported'
    imported.mkdir(parents=True, exist_ok=True)
    for old in imported.iterdir():
        if old.is_symlink() and old.name not in expected:
            old.unlink()
    (root / 'memory/shared').mkdir(parents=True, exist_ok=True)
    return len(mappings)


def source_link(root, provider, sid, path):
    compressed = Path(path).suffix in ('.zst', '.zstd')
    return root / 'sources' / provider / (sid + ('.jsonl' + Path(path).suffix if compressed else '.jsonl'))


def sync_one(db, root, provider, path, redact_secrets=True):
    sid = provider + '-' + hashlib.sha256(str(path).encode()).hexdigest()[:20]
    stat = path.stat()
    row = db.execute('SELECT * FROM sources WHERE id=?', (sid,)).fetchone()
    compressed = path.suffix in ('.zst', '.zstd')
    # Invalid records are reported but stable; only read/decode failures (no bad_lines) are retried.
    retry = bool(row and row['error'] and not row['bad_lines'])
    link(source_link(root, provider, sid, path), path)
    # Writers end every record with a newline, so an unterminated last line is normally still being
    # written. Compressed files are complete; plain files are trusted once they stop changing.
    settled = time.time() - stat.st_mtime >= TAIL_SETTLE_SECONDS
    pending_tail = bool(row and not compressed and settled and row['offset'] < stat.st_size)
    if row and row['size'] == stat.st_size and row['mtime'] == stat.st_mtime_ns and not retry and not pending_tail:
        db.execute('UPDATE sources SET available=1 WHERE id=?', (sid,))
        return 'unchanged', 0
    db.execute('INSERT OR IGNORE INTO sources(id,provider,path,native_id) VALUES(?,?,?,?)',
               (sid, provider, str(path), path.name))
    db.execute('SAVEPOINT source_sync')
    try:
        with stream(path) as f:
            digest = hashlib.sha256()
            offset, line_no, append, skip = 0, 0, False, 0
            if row and not retry and not compressed and row['offset'] <= stat.st_size:
                remaining, last = row['offset'], b'\n'
                while remaining:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    digest.update(chunk)
                    remaining -= len(chunk)
                    last = chunk[-1:]
                append = remaining == 0 and digest.hexdigest() == row['digest']
                if append and last != b'\n':
                    # The last indexed line had no newline. If one now ends it, the line is unchanged and
                    # appending continues; if the writer continued that line instead, reindex the file.
                    following = f.read(1)
                    if following == b'\n':
                        digest.update(following)
                        skip = 1
                    elif following:
                        append = False
            if append:
                offset, line_no = row['offset'] + skip, row['lines']
                meta = dict(row)
                bad_lines = json.loads(row['bad_lines'] or '[]')
            else:
                if not compressed:
                    f.seek(0)
                digest = hashlib.sha256()
                db.execute('DELETE FROM messages WHERE source_id=?', (sid,))
                meta = {'native_id': path.name, 'project': '', 'title': '', 'updated_at': ''}
                bad_lines = []
            count = 0
            # Freeze the initial file size; leave a partial last line for the next sync unless settled.
            while compressed or offset < stat.st_size:
                raw = f.readline() if compressed else f.readline(stat.st_size - offset)
                if not raw:
                    break
                if not raw.endswith(b'\n') and not (compressed or settled):
                    break
                digest.update(raw)
                offset += len(raw)
                line_no += 1
                try:
                    obj = json.loads(raw)
                    updates, message = parse(provider, obj)
                    if message:
                        role, timestamp, body = message
                        if not isinstance(role, str) or not isinstance(body, str):
                            raise TypeError('Non-text message fields')
                        timestamp = timestamp if isinstance(timestamp, str) else ''
                        if redact_secrets:
                            body = redact(body)
                except Exception:
                    # One malformed or unexpected record must not drop the rest of the file.
                    bad_lines.append(line_no)
                    continue
                meta.update({k: v for k, v in updates.items() if isinstance(v, str) and v})
                if redact_secrets and 'title' in updates:
                    meta['title'] = redact(meta['title'])
                if message:
                    db.execute('INSERT INTO messages(source_id,line,role,timestamp,text) VALUES(?,?,?,?,?)',
                               (sid, line_no, role, timestamp, body))
                    count += 1
                    meta['updated_at'] = timestamp or meta['updated_at']
                    if not meta['title'] and role == 'user':
                        meta['title'] = body.replace('\n', ' ')[:160]
            error = ('Invalid records at lines: ' + str(bad_lines[:20])) if bad_lines else ''
            db.execute('''UPDATE sources SET native_id=?,project=?,title=?,size=?,mtime=?,offset=?,
                       lines=?,digest=?,available=1,error=?,bad_lines=?,synced_at=?,updated_at=? WHERE id=?''',
                       (meta['native_id'], meta['project'], meta['title'], stat.st_size, stat.st_mtime_ns,
                        offset, line_no, digest.hexdigest(), error, json.dumps(bad_lines[:20]) if bad_lines else '',
                        now(), meta['updated_at'], sid))
        db.execute('RELEASE source_sync')
        return ('appended' if append else 'rebuilt'), count
    except Exception as exc:
        db.execute('ROLLBACK TO source_sync')
        db.execute('RELEASE source_sync')
        db.execute("UPDATE sources SET available=1,error=?,bad_lines='' WHERE id=?",
                   (type(exc).__name__ + ': ' + str(exc), sid))
        return 'failed', 0


@contextlib.contextmanager
def sync_lock(root, blocking=True):
    """Yield whether the lock is held; with blocking=False, an ongoing sync is not waited for."""
    with (root / 'sync.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def checkpoint(db, wait_ms):
    """Copy the WAL into the database and truncate it, so removed text leaves the -wal file too."""
    db.execute(f'PRAGMA busy_timeout={int(wait_ms)}')
    try:
        return db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 0
    finally:
        db.execute('PRAGMA busy_timeout=60000')


def purge(db, root, found):
    """Remove indexed text and links of sources whose files are gone; keep the row for `read` errors."""
    removed = 0
    for row in db.execute('SELECT id,provider,path,available FROM sources').fetchall():
        if row['path'] in found:
            continue
        stale_link = source_link(root, row['provider'], row['id'], row['path'])
        if stale_link.is_symlink():
            stale_link.unlink()
        has_text = db.execute('SELECT 1 FROM messages WHERE source_id=? LIMIT 1', (row['id'],)).fetchone()
        if row['available'] or has_text:
            db.execute('DELETE FROM messages WHERE source_id=?', (row['id'],))
            # Reset sync state so a file that reappears at this path is indexed again.
            db.execute('''UPDATE sources SET available=0,title='',size=-1,mtime=0,offset=0,lines=0,
                       digest='',bad_lines='' WHERE id=?''', (row['id'],))
            removed += 1
    for old in (root / 'sources').rglob('*') if (root / 'sources').is_dir() else ():
        if old.is_symlink() and not old.exists():
            old.unlink()
    return removed


def _sync(db, root, home, redact_secrets=True):
    counts = {'unchanged': 0, 'appended': 0, 'rebuilt': 0, 'updated': 0, 'failed': 0, 'messages_added': 0}
    found = set()
    for provider, path in discover(home):
        found.add(str(path))
        with db:
            kind, added = sync_one(db, root, provider, path, redact_secrets)
        counts[kind] += 1
        counts['messages_added'] += added
    with db:
        counts['removed'] = purge(db, root, found)
    if counts['removed'] or counts['rebuilt']:
        # Rebuilt and removed sources leave their old text in WAL frames until a checkpoint.
        checkpoint(db, 1000)
    counts['memory_mappings'] = map_memories(home, root)
    catalog = root / 'catalog.jsonl.tmp'
    with catalog.open('w') as f:
        for row in db.execute('SELECT id,provider,native_id,project,title,path,available,error,updated_at FROM sources ORDER BY updated_at DESC,id'):
            f.write(json.dumps(dict(row), ensure_ascii=False) + '\n')
    catalog.replace(root / 'catalog.jsonl')
    (root / 'last-sync.json').write_text(json.dumps({'at': now(), **counts}, indent=2) + '\n')
    return counts


def sync(db, root, home, redact_secrets=True, blocking=True):
    """Return sync counts, or None when blocking=False and another process is syncing."""
    with sync_lock(root, blocking) as locked:
        return _sync(db, root, home, redact_secrets) if locked else None


def prune(db, root, home, rebuild=False, redact_secrets=True):
    """Sync, forget missing sources entirely, then compact the index and WAL."""
    with sync_lock(root, True):
        if rebuild:
            with db:
                db.execute("UPDATE sources SET size=-1,mtime=0,offset=0,lines=0,digest='',bad_lines=''")
        counts = _sync(db, root, home, redact_secrets)
        with db:
            counts['forgotten'] = db.execute('DELETE FROM sources WHERE available=0').rowcount
            db.execute("INSERT INTO search(search) VALUES('optimize')")
        db.execute('VACUUM')
        counts['checkpointed'] = checkpoint(db, 60000)
    return counts


def filters(args, prefix='s.'):
    terms, values = [prefix + 'available=1'], []
    if getattr(args, 'provider', None):
        terms.append(prefix + 'provider=?')
        values.append(args.provider)
    if getattr(args, 'project', None):
        terms.append('instr(lower(' + prefix + 'project),lower(?))>0')
        values.append(args.project)
    return terms, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / '.agent-context-kit')
    parser.add_argument('--source-home', type=Path, default=Path.home())
    parser.add_argument('--no-sync', action='store_true', help='Read the existing index without refreshing')
    parser.add_argument('--no-redact', action='store_true',
                        help='Index secrets verbatim (also AGENT_CONTEXT_REDACT=0); run prune --rebuild after changing')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('sync')
    p = sub.add_parser('prune', help='Forget deleted sources, then compact the index (FTS optimize, VACUUM)')
    p.add_argument('--rebuild', action='store_true', help='Reindex every source from scratch')
    sub.add_parser('status')
    for name in ['list', 'search']:
        p = sub.add_parser(name)
        if name == 'search':
            p.add_argument('query', help='Literal substring, including Chinese or code identifiers')
        p.add_argument('--provider', choices=['claude', 'codex', 'deepseek'])
        p.add_argument('--project')
        p.add_argument('--limit', type=int, default=30)
        p.add_argument('--offset', type=int, default=0)
    p = sub.add_parser('read')
    p.add_argument('id', help='Mapped source ID from list/search')
    p.add_argument('--start-line', type=int, default=1)
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--max-chars', type=int, default=24000)
    p.add_argument('--raw', action='store_true', help='Read original JSONL lines, including tools and metadata')
    p = sub.add_parser('memory')
    p.add_argument('--query', default='')
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    if getattr(args, 'limit', 1) < 1 or getattr(args, 'offset', 0) < 0 or getattr(args, 'start_line', 1) < 1 or getattr(args, 'max_chars', 1) < 1:
        parser.error('Limits and start lines must be positive; offsets must be nonnegative')
    root, home = args.root.expanduser().absolute(), args.source_home.expanduser().absolute()
    os.umask(0o077)
    db = connect(root)
    redact_secrets = not args.no_redact and os.environ.get('AGENT_CONTEXT_REDACT', '1') != '0'
    if args.command == 'sync':
        result = sync(db, root, home, redact_secrets)
        output(result)
        return 1 if result['failed'] else 0
    if args.command == 'prune':
        result = prune(db, root, home, args.rebuild, redact_secrets)
        output(result)
        return 1 if result['failed'] else 0
    extra = {}
    if not args.no_sync:
        # Queries do not wait for another process's sync, except before the first sync completed.
        first = not (root / 'last-sync.json').exists()
        if sync(db, root, home, redact_secrets, blocking=first) is None:
            extra = {'stale': True, 'stale_reason': 'Another sync is running; results come from the last completed '
                                                    'snapshot. Retry later for newer messages.'}

    def emit(value):
        output({**value, **extra})
    if args.command == 'status':
        emit({'root': str(root), 'sources': [dict(r) for r in db.execute('''SELECT provider,
          available, count(*) count, sum(error!='') errors FROM sources GROUP BY provider,available''')],
          'indexed_messages': db.execute('SELECT count(*) FROM messages').fetchone()[0],
          'sources_without_indexed_text': db.execute('''SELECT count(*) FROM sources s
            WHERE available=1 AND NOT EXISTS(SELECT 1 FROM messages m WHERE m.source_id=s.id)''').fetchone()[0],
          'errors': [dict(r) for r in db.execute("SELECT id,path,error FROM sources WHERE error!=''")],
          'last_sync': json.loads((root / 'last-sync.json').read_text()) if (root / 'last-sync.json').exists() else None})
    elif args.command == 'list':
        where, values = filters(args)
        rows = db.execute('SELECT s.*, (SELECT count(*) FROM messages m WHERE m.source_id=s.id) indexed_messages FROM sources s WHERE ' + ' AND '.join(where) +
                          ' ORDER BY updated_at DESC,id LIMIT ? OFFSET ?', values + [args.limit, args.offset]).fetchall()
        total = db.execute('SELECT count(*) FROM sources s WHERE ' + ' AND '.join(where), values).fetchone()[0]
        emit({'total': total, 'next_offset': args.offset + len(rows) if args.offset + len(rows) < total else None,
                'sessions': [dict(r) for r in rows]})
    elif args.command == 'search':
        where, values = filters(args)
        query = fold(args.query)
        # Matching is instr(fold(text), fold(query)). When every query character is ASCII or caseless (CJK,
        # digits, punctuation), SQLite's ASCII-only lower() gives the same answer without calling Python,
        # except for text holding one of the two non-ASCII characters that fold to an ASCII letter.
        simple = all(c.isascii() or caseless(c) for c in query)
        if simple:
            exotic = [ASCII_FOLDS[c] for c in sorted(set(query) & ASCII_FOLDS.keys())]
            fallback = ' OR (({}) AND instr(fold(m.text),?)>0)'.format(
                ' OR '.join(['instr(m.text,?)>0'] * len(exotic))) if exotic else ''
            where.append('(instr(lower(m.text),?)>0' + fallback + ')')
            values += [query] + (exotic + [query] if exotic else [])
        else:
            # Exact prefilter: the text must contain some case variant of the first cased character.
            key = next(c for c in query if not (c.isascii() or caseless(c)))
            variants = case_tables()[0].get(key, (key,))
            where.append('(' + ' OR '.join(['instr(m.text,?)>0'] * len(variants)) + ') AND instr(fold(m.text),?)>0')
            values += list(variants) + [query]

        def run(messages, before=(), terms=(), after=()):
            return db.execute('''SELECT s.id,s.provider,s.native_id,s.project,s.path,s.available,s.error,
             m.line,m.role,m.timestamp,m.text FROM ''' + messages + ''' sources s ON s.id=m.source_id WHERE ''' +
             ' AND '.join(where + list(terms)) + ' ORDER BY m.timestamp DESC,m.id DESC LIMIT ? OFFSET ?',
             [*before, *values, *after, args.limit + 1, args.offset]).fetchall()
        # Every strategy returns the same rows in the same order (newest first, then highest id).
        # Common terms walk messages newest first and stop after offset+limit+1 matches; rare terms sort
        # only their candidates instead of scanning the whole table.
        need = args.offset + args.limit + 1
        # A trigram phrase is a literal substring (%, _ and \\ included) whose case folding agrees with fold()
        # except for a few characters the tokenizer does not know; those are spelled out as extra phrases, so
        # the candidates are a superset of the matches and the instr() terms decide.
        options = [(c,) + (case_tables()[1].get(c, ()) if not simple and len(query) >= 3 else ()) for c in query]
        spellings = 1
        for option in options:
            spellings *= len(option)
        if len(query) >= 3 and spellings > MAX_SPELLINGS:
            rows = run('messages m INDEXED BY messages_time CROSS JOIN')
        elif len(query) >= 3:
            phrase = ' OR '.join('"' + ''.join(p).replace('"', '""') + '"' for p in itertools.product(*options))
            cap = COMMON_CANDIDATES + 10 * need
            if db.execute('SELECT count(*) FROM (SELECT 1 FROM search WHERE search MATCH ? LIMIT ?)',
                          (phrase, cap)).fetchone()[0] < cap:
                rows = run('messages m JOIN', terms=['m.id IN (SELECT rowid FROM search WHERE search MATCH ?)'],
                           after=[phrase])
            else:
                rows = run('messages m INDEXED BY messages_time CROSS JOIN')
        else:
            # Shorter queries cannot use trigrams: if enough of the newest messages match, those are the
            # answer; otherwise scan everything.
            rows = run('''(SELECT * FROM messages INDEXED BY messages_time ORDER BY timestamp DESC,id DESC
                       LIMIT ?) m CROSS JOIN''', before=[SHORT_QUERY_WINDOW + 10 * need])
            if len(rows) <= args.limit:
                # A plain scan plus sort beats walking the whole time index for rare matches.
                rows = run('messages m NOT INDEXED JOIN')
        results = []
        for row in rows[:args.limit]:
            item = dict(row)
            text = item.pop('text')
            pos = max(0, fold(text).find(query))  # fold() keeps offsets
            item['snippet'] = text[max(0, pos - 150):pos + len(query) + 350]
            results.append(item)
        emit({'results': results, 'next_offset': args.offset + args.limit if len(rows) > args.limit else None})
    elif args.command == 'read':
        source = db.execute('SELECT * FROM sources WHERE id=?', (args.id,)).fetchone()
        if not source:
            parser.error('Unknown source ID; use list or search')
        if not source['available']:
            parser.error('Source is no longer present; sync and locate its current mapping')
        rows = []
        if args.raw:
            with stream(Path(source['path'])) as f:
                for n, line in enumerate(f, 1):
                    if n >= args.start_line:
                        rows.append({'line': n, 'text': line.decode('utf-8', errors='replace')})
                        if len(rows) > args.limit:
                            break
        else:
            rows = [dict(r) for r in db.execute('''SELECT line,role,timestamp,text FROM messages
              WHERE source_id=? AND line>=? ORDER BY line LIMIT ?''', (args.id, args.start_line, args.limit + 1))]
        selected, budget, next_line = [], args.max_chars, None
        for row in rows:
            if len(selected) >= args.limit or budget <= 0:
                next_line = row['line']
                break
            item = dict(row)
            item['truncated'] = len(item['text']) > budget
            item['text'] = item['text'][:budget]
            budget -= len(item['text'])
            selected.append(item)
        emit({'source': dict(source),
                'coordinate': 'JSONL line',
                'messages': selected, 'next_line': next_line,
                'hint': 'For a truncated message, reread its line with a larger --max-chars.'})
    elif args.command == 'memory':
        found = []
        bases = [root / 'memory/shared'] + sorted((root / 'memory/imported').glob('*'))
        for base in bases:
            for path in sorted(base.rglob('*.md')):
                if any(part.startswith('.') for part in path.relative_to(base).parts):
                    continue
                body = path.read_text(errors='replace')
                if args.query.lower() in body.lower() or args.query.lower() in str(path).lower():
                    found.append({'path': str(path), 'source': str(path.resolve()),
                                  'kind': 'shared' if base.name == 'shared' else 'imported',
                                  'preview': body[:200]})
        emit({'total': len(found), 'files': found[args.offset:args.offset + args.limit],
                'next_offset': args.offset + args.limit if args.offset + args.limit < len(found) else None})
    db.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
