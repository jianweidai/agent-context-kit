import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from agent_context_kit import cli as module

TEXTS = ['Äpfel und Birnen', 'ΑΘΗΝΑ πόλη', 'МОСКВА город', 'ＡＢＣＤ全角', 'Straße Haus', 'STRASSE gross',
         'ẞ capital', '重试机制 retry', '\u017ftop sign', 'Kelvin \u212a unit', 'plain ascii text',
         # U+1C80 CYRILLIC SMALL LETTER ROUNDED VE folds to в, but older trigram tables leave it alone.
         '\u1c80ОДА рядом', '\u1c80' * 5 + ' пять']
STRATEGIES = [(module.COMMON_CANDIDATES, module.SHORT_QUERY_WINDOW), (-10 ** 9, 0), (10 ** 9, 10 ** 9)]


class UnicodeCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.home = Path(cls.temp.name) / 'home'
        cls.root = Path(cls.temp.name) / 'context'
        path = cls.home / '.claude/projects/demo/s.jsonl'
        path.parent.mkdir(parents=True)
        texts = TEXTS + ['x' * 300 + 'ÄPFELKUCHEN' + 'y' * 400, 'ß' * 200 + ' TARGET word']
        path.write_text(''.join(json.dumps({'type': 'user', 'timestamp': f'2026-11-05T10:00:{i:02d}Z',
                                            'message': {'role': 'user', 'content': t}}, ensure_ascii=False) + '\n'
                                for i, t in enumerate(texts)))
        db = module.connect(cls.root)
        module.sync(db, cls.root, cls.home)
        db.close()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def search(self, query):
        argv = ['agent-context', '--root', str(self.root), '--source-home', str(self.home), '--no-sync',
                'search', query, '--limit', '50']
        out = io.StringIO()
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(out):
            self.assertEqual(module.main(), 0)
        return json.loads(out.getvalue())['results']

    def matches(self, query):
        results = []
        for candidates, window in STRATEGIES:
            with mock.patch.object(module, 'COMMON_CANDIDATES', candidates), \
                    mock.patch.object(module, 'SHORT_QUERY_WINDOW', window):
                results.append(sorted(r['snippet'] for r in self.search(query) if r['snippet'] in TEXTS))
        self.assertEqual(results[1:], results[:1] * 2, query)  # every strategy agrees
        return results[0]

    def test_non_ascii_case_insensitive(self):
        cases = {
            'äpfel': ['Äpfel und Birnen'], 'ÄPFEL': ['Äpfel und Birnen'],
            'αθηνα': ['ΑΘΗΝΑ πόλη'], 'ΠΌΛΗ': ['ΑΘΗΝΑ πόλη'],
            'москва': ['МОСКВА город'], 'ГОРОД': ['МОСКВА город'],
            'ａｂｃｄ': ['ＡＢＣＤ全角'], 'ＡＢＣＤ全': ['ＡＢＣＤ全角'],
            'STRAßE': ['Straße Haus'], 'ß': ['Straße Haus', 'ẞ capital'], 'ẞ': ['Straße Haus', 'ẞ capital'],
            # Folding keeps one character per character, so ß does not match "ss".
            'strasse': ['STRASSE gross'],
            '重试': ['重试机制 retry'], '重试机制': ['重试机制 retry'], 'RETRY': ['重试机制 retry'],
            'STOP SIGN': ['\u017ftop sign'], 'kelvin k': ['Kelvin \u212a unit'], 'n k': ['Kelvin \u212a unit'],
            'вода': ['\u1c80ОДА рядом'], 'ВВВВВ': ['\u1c80' * 5 + ' пять'], 'в' * 5 + ' п': ['\u1c80' * 5 + ' пять'],
            'zzz': [],
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self.matches(query), sorted(expected))

    def test_snippet_offsets_follow_folded_match(self):
        for candidates, window in STRATEGIES:
            with mock.patch.object(module, 'COMMON_CANDIDATES', candidates), \
                    mock.patch.object(module, 'SHORT_QUERY_WINDOW', window):
                snippet = self.search('äpfelkuchen')[0]['snippet']
                self.assertEqual(snippet, 'x' * 150 + 'ÄPFELKUCHEN' + 'y' * 350)
                snippet = self.search('target')[0]['snippet']
                self.assertTrue(snippet.startswith('ß' * 149 + ' TARGET word'), snippet[:20])

    def test_fold_keeps_length_and_prefilter_assumptions(self):
        into_ascii = set()
        for code in range(0x110000):
            c = chr(code)
            if 0xD800 <= code <= 0xDFFF:
                continue
            folded = module.fold_char(c)
            self.assertEqual(len(folded), 1)
            self.assertEqual(module.fold_char(folded), folded)
            if folded != c:
                # A caseless query character can only be matched by itself.
                self.assertFalse(module.caseless(folded), hex(code))
                if folded.isascii() and not c.isascii():
                    into_ascii.add((folded, c))
        self.assertEqual(into_ascii, set(module.ASCII_FOLDS.items()))
        self.assertEqual(module.fold('Straße ẞ ΣΑΣ'), 'straße ß σασ')


if __name__ == '__main__':
    unittest.main()
