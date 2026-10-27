"""Tokenizer checkpoint identity describes the Unicode data actually executed."""
import hashlib
import unittest
from mojolearn.models.tokenizer import Tokenizer


class TokenizerFingerprint(unittest.TestCase):
    def test_without_normalization_uses_compiled_unicode(self):
        tokens = [bytes([i]) for i in range(256)]
        expected = hashlib.sha256(b''.join(b'1:' + t for t in tokens)
                                 + b'gpt2\nNone\n16.0.0\n()\n()\n').hexdigest()
        for host_version in ('14.0.0', '15.0.0', '16.0.0'):
            tokenizer = Tokenizer(tokens, [], pattern='gpt2')
            tokenizer.unicode_version = host_version
            self.assertEqual(tokenizer._m.vocab_fingerprint(tokenizer), expected)
            self.assertEqual(tokenizer.decode(tokenizer.encode('hello')), 'hello')

    def test_normalization_retains_its_version(self):
        tokens = [bytes([i]) for i in range(256)]
        fingerprints = []
        for host_version in ('14.0.0', '15.0.0', '16.0.0'):
            tokenizer = Tokenizer(tokens, [], pattern='gpt2', normalizer='NFC')
            tokenizer.unicode_version = host_version
            expected = hashlib.sha256(b''.join(b'1:' + t for t in tokens)
                + f'gpt2\nNFC\n16.0.0/normalizer:{host_version}\n()\n()\n'.encode()).hexdigest()
            fingerprint = tokenizer._m.vocab_fingerprint(tokenizer)
            self.assertEqual(fingerprint, expected)
            fingerprints.append(fingerprint)
        self.assertEqual(len(set(fingerprints)), 3)


if __name__ == '__main__':
    unittest.main()
