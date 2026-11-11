import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import zstandard

from agent_context_kit import cli as module


def record(text, newline=True):
    line = json.dumps({'type': 'user', 'timestamp': '2026-11-05T10:00:00Z',
                       'message': {'role': 'user', 'content': text}}, ensure_ascii=False).encode()
    return line + (b'\n' if newline else b'')


class UnterminatedTailTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name) / 'home'
        self.root = Path(self.temp.name) / 'context'
        self.path = self.home / '.claude/projects/demo/s.jsonl'
        self.path.parent.mkdir(parents=True)
        self.db = module.connect(self.root)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def sync(self):
        return module.sync(self.db, self.root, self.home)

    def age(self, path=None):
        old = time.time() - module.TAIL_SETTLE_SECONDS - 60
        os.utime(path or self.path, (old, old))

    def append(self, data):
        with self.path.open('ab') as f:
            f.write(data)

    def lines(self):
        return [tuple(r) for r in self.db.execute('SELECT line,text FROM messages ORDER BY line')]

    def error(self):
        return self.db.execute('SELECT error FROM sources').fetchone()[0]

    def raw_line(self, n):
        with module.stream(self.path) as f:
            return [line for i, line in enumerate(f, 1) if i == n][0]

    def test_compressed_file_keeps_its_last_line(self):
        path = self.path.with_name('archived.jsonl.zst')
        path.write_bytes(zstandard.ZstdCompressor().compress(record('first') + record('last', newline=False)))
        self.sync()
        self.assertEqual(self.lines(), [(1, 'first'), (2, 'last')])
        self.assertEqual(self.sync()['unchanged'], 1)

    def test_recent_unterminated_line_waits_until_file_settles(self):
        self.path.write_bytes(record('first') + record('pending', newline=False))
        self.sync()
        self.assertEqual(self.lines(), [(1, 'first')])
        self.assertEqual(self.sync()['unchanged'], 1)
        self.age()
        self.assertEqual(self.sync()['appended'], 1)
        self.assertEqual(self.lines(), [(1, 'first'), (2, 'pending')])
        self.assertEqual(self.sync()['unchanged'], 1)

    def test_settled_line_then_newline_and_more_records(self):
        self.path.write_bytes(record('first') + record('old tail', newline=False))
        self.age()
        self.sync()
        self.assertEqual(self.lines(), [(1, 'first'), (2, 'old tail')])
        self.append(b'\n' + record('third'))
        result = self.sync()
        self.assertEqual((result['appended'], result['messages_added']), (1, 1))
        self.assertEqual(self.lines(), [(1, 'first'), (2, 'old tail'), (3, 'third')])
        self.assertEqual(json.loads(self.raw_line(3))['message']['content'], 'third')
        self.assertEqual(self.error(), '')
        self.append(record('fourth'))
        self.assertEqual(self.sync()['appended'], 1)
        self.assertEqual([t for _, t in self.lines()], ['first', 'old tail', 'third', 'fourth'])

    def test_settled_partial_line_continued_by_writer(self):
        whole = record('slow writer')
        self.path.write_bytes(record('first') + whole[:20])
        self.age()
        self.sync()
        self.assertIn('[2]', self.error())
        self.append(whole[20:])
        self.assertEqual(self.sync()['rebuilt'], 1)
        self.assertEqual(self.lines(), [(1, 'first'), (2, 'slow writer')])
        self.assertEqual(self.error(), '')

    def test_settled_line_continued_without_newline(self):
        self.path.write_bytes(record('first') + record('glued', newline=False))
        self.age()
        self.sync()
        self.append(record('next'))
        self.assertEqual(self.sync()['rebuilt'], 1)
        # The second line now holds two concatenated records: reported, and nothing is indexed twice.
        self.assertEqual(self.lines(), [(1, 'first')])
        self.assertIn('[2]', self.error())

    def test_settled_tail_after_invalid_line_keeps_report(self):
        self.path.write_bytes(record('first') + b'{bad\n' + record('tail', newline=False))
        self.age()
        self.sync()
        self.assertEqual(self.lines(), [(1, 'first'), (3, 'tail')])
        self.assertIn('[2]', self.error())
        self.assertEqual(self.sync()['unchanged'], 1)
        self.append(b'\n' + record('more'))
        self.assertEqual(self.sync()['appended'], 1)
        self.assertEqual(self.lines(), [(1, 'first'), (3, 'tail'), (4, 'more')])
        self.assertIn('[2]', self.error())


if __name__ == '__main__':
    unittest.main()
