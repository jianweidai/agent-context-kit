import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

from agent_context_kit import cli as module


def claude(text, timestamp, cwd):
    return json.dumps({'type': 'user', 'cwd': cwd, 'timestamp': timestamp,
                       'message': {'role': 'user', 'content': text}}, ensure_ascii=False) + '\n'


class SearchStrategyTest(unittest.TestCase):
    """Walking by time, sorting trigram candidates and the full scan must agree, including paging."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.home = Path(cls.temp.name) / 'home'
        cls.root = Path(cls.temp.name) / 'context'
        rng = random.Random(7)
        words = ['retry', 'Retry_count', '重试', 'parser', '数据库', 'cache', 'deploy', 'zz']
        for n in range(6):
            path = cls.home / f'.claude/projects/p{n}/s.jsonl'
            path.parent.mkdir(parents=True)
            with path.open('w') as f:
                for i in range(150):
                    # Few distinct timestamps (and some empty ones) so ordering ties are exercised.
                    timestamp = '' if i % 17 == 0 else f'2026-11-0{rng.randint(1, 3)}T00:00:00Z'
                    text = ' '.join(rng.choice(words) for _ in range(rng.randint(1, 4)))
                    if n == 0 and i == 42:
                        text += ' unique_needle 罕见'
                    f.write(claude(text, timestamp, f'/work/p{n % 2}'))
        db = module.connect(cls.root)
        module.sync(db, cls.root, cls.home)
        cls.rows = [dict(r) for r in db.execute('''SELECT s.id,s.project,m.line,m.timestamp,m.id mid,m.text
                    FROM messages m JOIN sources s ON s.id=m.source_id''')]
        db.close()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def expected(self, query, limit, offset, project=None):
        hits = [r for r in self.rows if query.lower() in r['text'].lower()
                and (project is None or project in r['project'])]
        hits.sort(key=lambda r: (r['timestamp'], r['mid']), reverse=True)
        page = hits[offset:offset + limit]
        return [(r['id'], r['line']) for r in page], (offset + limit if len(hits) > offset + limit else None)

    def search(self, query, limit, offset, project=None):
        argv = ['agent-context', '--root', str(self.root), '--source-home', str(self.home), '--no-sync',
                'search', query, '--limit', str(limit), '--offset', str(offset)]
        if project:
            argv += ['--project', project]
        out = io.StringIO()
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(out):
            self.assertEqual(module.main(), 0)
        result = json.loads(out.getvalue())
        return [(r['id'], r['line']) for r in result['results']], result['next_offset']

    def test_strategies_match_reference_order_and_paging(self):
        # A negative cap makes LIMIT unbounded, so every >=3-character query counts as common and walks.
        strategies = {'time walk / small window': (-10 ** 9, 0), 'candidates / whole window': (10 ** 9, 10 ** 9)}
        for name, (candidates, window) in strategies.items():
            with mock.patch.object(module, 'COMMON_CANDIDATES', candidates), \
                    mock.patch.object(module, 'SHORT_QUERY_WINDOW', window):
                for query in ['retry', 'RETRY_COUNT', '重试', '数据库', 'zz', 'unique_needle', 'absent_term', 'y_c']:
                    for limit, offset in [(1, 0), (3, 2), (7, 0), (30, 0), (30, 60), (5, 400), (500, 0)]:
                        for project in [None, 'p1']:
                            with self.subTest(strategy=name, query=query, limit=limit, offset=offset,
                                              project=project):
                                self.assertEqual(self.search(query, limit, offset, project),
                                                 self.expected(query, limit, offset, project))

    def test_default_thresholds_match_reference(self):
        for query in ['retry', '重试', 'unique_needle', 'z']:
            for limit, offset in [(30, 0), (30, 30), (10, 880)]:
                with self.subTest(query=query, limit=limit, offset=offset):
                    self.assertEqual(self.search(query, limit, offset), self.expected(query, limit, offset))

    def test_short_query_window_falls_back_when_too_few_recent_matches(self):
        statements, connect = [], module.connect

        def traced(root):
            db = connect(root)
            db.set_trace_callback(statements.append)
            return db
        with mock.patch.object(module, 'connect', traced):
            for query, limit in [('罕见', 30), ('数据', 800), ('zz', 30)]:
                statements.clear()
                self.assertEqual(self.search(query, limit, 0), self.expected(query, limit, 0))
                runs = [s for s in statements if s.startswith('SELECT s.id')]
                self.assertEqual(len(runs), 1 if query == 'zz' else 2, query)


if __name__ == '__main__':
    unittest.main()
