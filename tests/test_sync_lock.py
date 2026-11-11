import fcntl
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest


def claude(text):
    return (json.dumps({'type': 'user', 'timestamp': '2026-11-05T10:00:00Z',
                        'message': {'role': 'user', 'content': text}}, ensure_ascii=False) + '\n').encode()


class SyncLockTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name) / 'home'
        self.root = Path(self.temp.name) / 'context'
        self.path = self.home / '.claude/projects/demo/s.jsonl'
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(claude('first message'))
        self.root.mkdir()
        self.lock = None

    def tearDown(self):
        self.release()
        self.temp.cleanup()

    def hold(self):
        self.lock = (self.root / 'sync.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX)

    def release(self):
        if self.lock:
            self.lock.close()
            self.lock = None

    def start(self, *args):
        return subprocess.Popen([sys.executable, '-m', 'agent_context_kit.cli', '--root', str(self.root),
                                 '--source-home', str(self.home), *args], stdout=subprocess.PIPE, text=True)

    def finish(self, proc, timeout=10):
        out, _ = proc.communicate(timeout=timeout)
        self.assertEqual(proc.returncode, 0)
        return json.loads(out)

    def snippets(self, result):
        return [r['snippet'] for r in result['results']]

    def test_query_does_not_wait_for_running_sync(self):
        self.finish(self.start('sync'))
        with self.path.open('ab') as f:
            f.write(claude('second message'))
        self.hold()
        started = time.monotonic()
        result = self.finish(self.start('search', 'message'), timeout=5)
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(result['stale'])
        self.assertEqual(self.snippets(result), ['first message'])
        self.assertTrue(self.finish(self.start('status'))['stale'])
        self.release()
        result = self.finish(self.start('search', 'message'))
        self.assertNotIn('stale', result)
        self.assertEqual(len(result['results']), 2)

    def test_first_query_waits_for_initial_sync(self):
        self.hold()
        proc = self.start('search', 'message')
        time.sleep(0.5)
        self.assertIsNone(proc.poll())
        self.release()
        result = self.finish(proc)
        self.assertNotIn('stale', result)
        self.assertEqual(self.snippets(result), ['first message'])

    def test_explicit_sync_waits(self):
        self.finish(self.start('sync'))
        self.hold()
        proc = self.start('sync')
        time.sleep(0.5)
        self.assertIsNone(proc.poll())
        self.release()
        self.assertEqual(self.finish(proc)['unchanged'], 1)

    def test_reads_are_not_blocked_by_a_write_transaction(self):
        self.finish(self.start('sync'))
        writer = sqlite3.connect(self.root / 'index.sqlite')
        try:
            writer.execute('BEGIN IMMEDIATE')
            writer.execute("UPDATE sources SET title='x'")
            self.hold()
            result = self.finish(self.start('search', 'first'), timeout=5)
            self.assertTrue(result['stale'])
            self.assertEqual(self.snippets(result), ['first message'])
            result = self.finish(self.start('--no-sync', 'list'), timeout=5)
            self.assertEqual(result['total'], 1)
        finally:
            writer.rollback()
            writer.close()


if __name__ == '__main__':
    unittest.main()
