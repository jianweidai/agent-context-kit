import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from agent_context_kit import cli as module


def claude(text, **extra):
    obj = {'type': 'user', 'sessionId': 'native-1', 'cwd': '/projects/demo',
           'timestamp': '2026-11-05T10:00:00Z', 'message': {'role': 'user', 'content': text}}
    obj.update(extra)
    return obj


def encode(*records):
    return b''.join(r if isinstance(r, bytes) else (json.dumps(r, ensure_ascii=False) + '\n').encode()
                    for r in records)


class RobustnessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name) / 'home'
        self.root = Path(self.temp.name) / 'context'
        self.db = module.connect(self.root)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def write(self, relative, data):
        path = self.home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def sync(self):
        return module.sync(self.db, self.root, self.home)

    def texts(self):
        return [r[0] for r in self.db.execute('SELECT text FROM messages ORDER BY source_id,line')]

    def search(self, query):
        """Run the CLI in-process and capture every SQL statement it executes."""
        statements, connect = [], module.connect

        def traced(root):
            db = connect(root)
            db.set_trace_callback(statements.append)
            return db
        argv = ['agent-context', '--root', str(self.root), '--source-home', str(self.home),
                '--no-sync', 'search', query]
        out = io.StringIO()
        with mock.patch.object(module, 'connect', traced), mock.patch.object(sys, 'argv', argv), \
                contextlib.redirect_stdout(out):
            self.assertEqual(module.main(), 0)
        used_index = any('FROM search WHERE text LIKE' in s for s in statements)
        return [r['snippet'] for r in json.loads(out.getvalue())['results']], used_index

    def test_wildcard_characters_use_index_and_stay_literal(self):
        self.write('.claude/projects/demo/s.jsonl', encode(
            claude('set retry_count=3'), claude('set retryXcount=4'),
            claude('rate 100% done'), claude('rate 1000 done'),
            claude(r'path C:\temp\new'), claude('path C:/temp/new')))
        self.sync()
        for query, expected in [('retry_count', 'retry_count=3'), ('100%', '100% done'),
                                (r'C:\temp', r'C:\temp\new'), ('RETRY_COUNT', 'retry_count=3')]:
            snippets, used_index = self.search(query)
            self.assertTrue(used_index, query)
            self.assertEqual(len(snippets), 1, (query, snippets))
            self.assertIn(expected, snippets[0])
        snippets, used_index = self.search('t_')
        self.assertFalse(used_index)
        self.assertEqual(len(snippets), 0)

    def test_bad_records_do_not_drop_valid_messages(self):
        dsh = lambda text, **extra: {'type': 'user/message', **extra, 'data': {
            'role': 'user', 'source': {'kind': 'user'}, 'content': [{'type': 'text', 'text': text}]}}
        codex = lambda text, **extra: {'type': 'response_item', 'timestamp': '2026-11-05T10:00:00Z', **extra,
                                       'payload': {'type': 'message', 'role': 'user',
                                                   'content': [{'type': 'input_text', 'text': text}]}}
        self.write('.dsh/sessions/w/s.jsonl', encode(
            {'type': 'session', 'id': 'dsh-1', 'cwd': {'not': 'a string'}},
            dsh('deepseek valid', time=1793836800000), dsh('deepseek missing time'),
            dsh('deepseek huge time', time=10 ** 20), dsh('deepseek text time', time='soon'),
            {'type': 'user/message', 'data': 'not an object', 'time': 1}))
        self.write('.claude/projects/demo/s.jsonl', encode(
            claude('claude valid'), claude('claude dict cwd', cwd={'a': 1}),
            claude('claude list title', type='ai-title', aiTitle=['x']),
            claude('claude numeric timestamp', timestamp=12345),
            {'type': 'assistant', 'message': {'role': {'bad': 1}, 'content': 'claude bad role'}},
            [1, 2], 42, b'{not json\n', b'\xff\xfe\n'))
        self.write('.codex/sessions/r.jsonl', encode(
            {'type': 'session_meta', 'payload': {'id': ['bad'], 'cwd': 7}},
            codex('codex valid'), codex('codex numeric timestamp', timestamp=1.5),
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': 'x'}},
            {'type': 'response_item', 'payload': 'not an object'}))
        result = self.sync()
        self.assertEqual(result['failed'], 0)
        texts = self.texts()
        for text in ['deepseek valid', 'deepseek missing time', 'deepseek huge time', 'deepseek text time',
                     'claude valid', 'claude dict cwd', 'claude numeric timestamp', 'codex valid',
                     'codex numeric timestamp']:
            self.assertIn(text, texts)
        self.assertNotIn('claude bad role', texts)
        rows = {r['provider']: r for r in self.db.execute('SELECT * FROM sources')}
        self.assertEqual((rows['deepseek']['native_id'], rows['deepseek']['project']), ('dsh-1', ''))
        self.assertEqual(rows['claude']['project'], '/projects/demo')
        self.assertEqual(rows['codex']['native_id'], 'r.jsonl')
        self.assertIn('Invalid records at lines', rows['claude']['error'])
        timestamps = dict(self.db.execute('SELECT text,timestamp FROM messages'))
        self.assertEqual(timestamps['deepseek missing time'], '')
        self.assertEqual(timestamps['claude numeric timestamp'], '')


    def test_invalid_records_are_reported_without_reindexing(self):
        path = self.write('.claude/projects/demo/s.jsonl', encode(claude('first'), b'{bad\n', claude('second')))
        self.assertEqual(self.sync()['rebuilt'], 1)
        error = lambda: self.db.execute('SELECT error FROM sources').fetchone()[0]
        self.assertIn('[2]', error())
        ids = self.db.execute('SELECT group_concat(id) FROM messages').fetchone()[0]
        result = self.sync()
        self.assertEqual((result['unchanged'], result['rebuilt']), (1, 0))
        self.assertIn('[2]', error())
        with path.open('ab') as f:
            f.write(encode(claude('third'), b'[also bad\n'))
        result = self.sync()
        self.assertEqual((result['appended'], result['messages_added']), (1, 1))
        self.assertIn('[2, 5]', error())
        self.assertTrue(self.db.execute('SELECT group_concat(id) FROM messages').fetchone()[0].startswith(ids))
        self.assertEqual(self.texts(), ['first', 'second', 'third'])
        self.assertEqual(self.sync()['unchanged'], 1)
        path.write_bytes(encode(claude('fixed')))
        self.assertEqual(self.sync()['rebuilt'], 1)
        self.assertEqual((error(), self.texts()), ('', ['fixed']))

    def test_read_failures_are_retried(self):
        path = self.write('.codex/archived_sessions/r.jsonl.zst', b'not zstd')
        self.assertEqual(self.sync()['failed'], 1)
        self.assertEqual(self.sync()['failed'], 1)
        import zstandard
        path.write_bytes(zstandard.ZstdCompressor().compress(encode(
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user',
                                                  'content': [{'type': 'input_text', 'text': 'recovered'}]}})))
        self.assertEqual(self.sync()['rebuilt'], 1)
        self.assertEqual(self.texts(), ['recovered'])

    def test_index_without_bad_lines_column_is_upgraded(self):
        self.write('.claude/projects/demo/s.jsonl', encode(claude('kept'), b'{bad\n'))
        self.sync()
        self.db.execute('ALTER TABLE sources DROP COLUMN bad_lines')
        self.db.close()
        self.db = module.connect(self.root)
        self.assertEqual(self.sync()['rebuilt'], 1)
        self.assertEqual(self.sync()['unchanged'], 1)
        self.assertEqual(self.texts(), ['kept'])

if __name__ == '__main__':
    unittest.main()
