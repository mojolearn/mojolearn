"""Native adapter version admission, including real TorchVersion and hostile hooks."""
import unittest


class HostileVersion(str):
    def __str__(self):
        raise AssertionError('called __str__')

    def __getattribute__(self, name):
        raise AssertionError('called __getattribute__')

    def __getitem__(self, item):
        raise AssertionError('called __getitem__')

    def __len__(self):
        raise AssertionError('called __len__')

    def split(self, *args, **kwargs):
        raise AssertionError('called split')


class PretendVersion:
    @property
    def __class__(self):
        raise AssertionError('called __class__')

    def __str__(self):
        raise AssertionError('called __str__')


class FrameworkVersionMetadata(unittest.TestCase):
    def test_unicode_storage_and_series_semantics(self):
        from mojolearn.adapters._shipped import series
        values = ['', '2', '2.13', '2.13.0+rocm7.2', '+local', '.', '..',
                  '2..13', '2.+local.3', '🐍.β.0', '\x00.3.4', '\ud800.3.4']
        for text in values:
            expected = '.'.join(text.split('+', 1)[0].split('.')[:2])
            for value in (text, HostileVersion(text)):
                with self.subTest(text=text, subclass=type(value) is HostileVersion):
                    result = series(value)
                    self.assertIs(type(result), str)
                    self.assertEqual(result, expected)

    def test_real_torch_version(self):
        try:
            from torch.torch_version import TorchVersion
        except ImportError:
            self.skipTest('real TorchVersion requires the optional pinned Torch installation')
        from mojolearn.adapters._shipped import series
        self.assertEqual(series(TorchVersion('2.13.0+local')), '2.13')

    def test_non_unicode_is_refused_without_coercion(self):
        from mojolearn.adapters._shipped import series
        for value in (None, b'2.13', 2.13, ['2.13'], PretendVersion()):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaisesRegex(TypeError, 'native string metadata'):
                    series(value)


if __name__ == '__main__':
    unittest.main()
