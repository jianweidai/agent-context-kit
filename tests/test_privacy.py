import json
from pathlib import Path
import tempfile
import unittest

from agent_context_kit import cli as module

SECRET = 'sk-proj-ABCDEF1234567890SECRETxyz'


def claude(text):
    return (json.dumps({'type': 'user', 'sessionId': 'native-1', 'cwd': '/projects/demo',
                        'timestamp': '2026-11-05T10:00:00Z', 'message': {'role': 'user', 'content': text}},
                       ensure_ascii=False) + '\n').encode()


def codex(text):
    return (json.dumps({'type': 'response_item', 'timestamp': '2026-11-05T10:00:00Z',
                        'payload': {'type': 'message', 'role': 'user',
                                    'content': [{'type': 'input_text', 'text': text}]}},
                       ensure_ascii=False) + '\n').encode()


class PrivacyTest(unittest.TestCase):
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

    def sync(self, redact_secrets=False):
        # Most tests disable redaction so that removal of verbatim text is observable.
        return module.sync(self.db, self.root, self.home, redact_secrets)

    def texts(self):
        return [r[0] for r in self.db.execute('SELECT text FROM messages ORDER BY source_id,line')]

    def on_disk(self, needle):
        """Whether the needle is anywhere in the index files, including the WAL of this open connection."""
        files = [p for p in self.root.glob('index.sqlite*')] + [self.root / 'catalog.jsonl']
        return any(needle.encode() in p.read_bytes() for p in files if p.exists())

    def test_deleted_source_text_leaves_index_and_wal(self):
        filler = b''.join(claude('filler %d' % i) for i in range(500))
        path = self.write('.claude/projects/demo/s.jsonl', filler + claude('key ' + SECRET))
        self.write('.claude/projects/demo/other.jsonl', claude('kept'))
        self.sync()
        self.assertTrue(self.on_disk(SECRET))
        sid = self.db.execute('SELECT id FROM sources WHERE path=?', (str(path),)).fetchone()[0]
        link = module.source_link(self.root, 'claude', sid, path)
        self.assertTrue(link.is_symlink())
        path.unlink()
        self.assertEqual(self.sync()['removed'], 1)
        self.assertEqual(self.texts(), ['kept'])
        self.assertFalse(self.on_disk(SECRET))
        self.assertFalse(link.is_symlink())
        row = self.db.execute('SELECT available,title FROM sources WHERE id=?', (sid,)).fetchone()
        self.assertEqual(tuple(row), (0, ''))
        self.assertEqual(self.sync()['removed'], 0)
        # A file that reappears unchanged at the same path is indexed again.
        path.write_bytes(claude('back again'))
        self.sync()
        self.assertIn('back again', self.texts())

    def test_moved_codex_session_is_indexed_once(self):
        old = self.write('.codex/sessions/2026/11/05/rollout.jsonl', codex('moved session'))
        self.sync()
        new = self.home / '.codex/archived_sessions/rollout.jsonl'
        new.parent.mkdir(parents=True)
        old.rename(new)
        result = self.sync()
        self.assertEqual((result['rebuilt'], result['removed']), (1, 1))
        self.assertEqual(self.texts(), ['moved session'])

    def test_prune_forgets_missing_sources_and_rebuilds(self):
        path = self.write('.claude/projects/demo/s.jsonl', claude('to delete ' + SECRET))
        self.write('.claude/projects/demo/other.jsonl', claude('kept'))
        self.sync()
        path.unlink()
        result = module.prune(self.db, self.root, self.home, redact_secrets=False)
        self.assertEqual((result['removed'], result['forgotten'], result['checkpointed']), (1, 1, True))
        self.assertEqual(self.db.execute('SELECT count(*) FROM sources').fetchone()[0], 1)
        self.assertFalse(self.on_disk(SECRET))
        result = module.prune(self.db, self.root, self.home, rebuild=True, redact_secrets=False)
        self.assertEqual((result['rebuilt'], result['unchanged']), (1, 0))
        self.assertEqual(self.texts(), ['kept'])
        self.assertEqual(self.db.execute("SELECT count(*) FROM search WHERE text LIKE '%kept%'").fetchone()[0], 1)

    def test_legacy_rows_of_missing_sources_are_purged(self):
        path = self.write('.claude/projects/demo/s.jsonl', claude('legacy ' + SECRET))
        self.sync()
        # Older versions only flipped `available`, leaving the text behind.
        with self.db:
            self.db.execute('UPDATE sources SET available=0')
        path.unlink()
        self.assertEqual(self.sync()['removed'], 1)
        self.assertEqual(self.texts(), [])
        self.assertFalse(self.on_disk(SECRET))


    def test_common_secret_formats_are_redacted(self):
        pem = '-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----'
        secrets = {
            SECRET: 'api-key',
            'sk-ant-api03-' + 'Ab3' * 30: 'api-key',
            'sk-' + 'a1B2' * 12: 'api-key',
            'ghp_' + 'A1b2' * 9: 'github-token',
            'github_pat_11ABCDEFG0' + 'x1Y2_' * 12: 'github-token',
            # Fake tokens are assembled at runtime so secret scanners do not flag this file.
            'AKIA' + 'IOSFODNN7' + 'EXAMPLE': 'aws-access-key',
            'xox' + 'b-' + '-'.join(['1' * 12, '2' * 13, 'AbCd' * 6]): 'slack-token',
            pem: 'private-key',
        }
        for secret, kind in secrets.items():
            self.assertEqual(module.redact(f'用这个 {secret}, 好吗'), f'用这个 [REDACTED:{kind}], 好吗', secret)
        self.assertEqual(module.redact('-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA'), '[REDACTED:private-key]')

    def test_redaction_leaves_ordinary_text_alone(self):
        ordinary = [
            'pip install scikit-learn sk-learn', 'see task-scheduler-for-the-release-notes-v2',
            'risk-assessment-for-quarterly-planning', 'desk-1234567890abcdefghijklmnop',
            'ghp_short and gho_123', 'ASIAN markets and AKIA prefixes', 'AKIAEXAMPLEPLACEHOLD',
            'xoxo-gossip-girl-season', 'sk-' + 'abcdef' * 5, '-----BEGIN PUBLIC KEY-----\nMFkwEwYH\n-----END PUBLIC KEY-----',
            '数据库 sk 重试 retry_count=3', 'commit 9fceb02d0ae598e95dc970b74767f19372d61af8',
        ]
        for text in ordinary:
            self.assertEqual(module.redact(text), text)

    def test_sync_redacts_by_default_and_can_be_disabled(self):
        self.write('.claude/projects/demo/s.jsonl', claude('key ' + SECRET))
        self.assertEqual(module.sync(self.db, self.root, self.home)['rebuilt'], 1)
        self.assertEqual(self.texts(), ['key [REDACTED:api-key]'])
        self.assertEqual(self.db.execute('SELECT title FROM sources').fetchone()[0], 'key [REDACTED:api-key]')
        self.assertFalse(self.on_disk(SECRET))
        module.prune(self.db, self.root, self.home, rebuild=True, redact_secrets=False)
        self.assertEqual(self.texts(), ['key ' + SECRET])

    def test_index_from_before_redaction_is_reindexed(self):
        self.write('.claude/projects/demo/s.jsonl', claude('key ' + SECRET))
        self.sync()
        with self.db:
            self.db.execute('PRAGMA user_version=0')
        self.db.close()
        self.db = module.connect(self.root)
        self.assertEqual(self.db.execute('PRAGMA user_version').fetchone()[0], module.SCHEMA_VERSION)
        self.assertEqual(module.sync(self.db, self.root, self.home)['rebuilt'], 1)
        self.assertEqual(self.texts(), ['key [REDACTED:api-key]'])
        self.assertFalse(self.on_disk(SECRET))
        self.assertEqual(module.sync(self.db, self.root, self.home)['unchanged'], 1)

if __name__ == '__main__':
    unittest.main()
