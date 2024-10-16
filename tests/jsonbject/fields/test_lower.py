import unittest
from swak.jsonobject.fields import Lower


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.lower = Lower()

    def test_has_lstrip(self):
        self.assertTrue(hasattr(self.lower, 'lstrip'))

    def test_lstrip(self):
        self.assertIsNone(self.lower.lstrip)

    def test_has_rstrip(self):
        self.assertTrue(hasattr(self.lower, 'rstrip'))

    def test_rstrip(self):
        self.assertIsNone(self.lower.rstrip)


class TestAttributes(unittest.TestCase):

    def test_lstrip_arg(self):
        lower = Lower('left')
        self.assertEqual('left', lower.lstrip)

    def test_lstrip_kwarg(self):
        lower = Lower(lstrip='left')
        self.assertEqual('left', lower.lstrip)

    def test_rstrip_arg(self):
        lower = Lower(None, 'right')
        self.assertEqual('right', lower.rstrip)

    def test_rstrip_kwarg(self):
        lower = Lower(rstrip='right')
        self.assertEqual('right', lower.rstrip)

    def test_args(self):
        lower = Lower('left', 'right')
        self.assertEqual('left', lower.lstrip)
        self.assertEqual('right', lower.rstrip)

    def test_kwargs(self):
        lower = Lower(lstrip='left', rstrip='right')
        self.assertEqual('left', lower.lstrip)
        self.assertEqual('right', lower.rstrip)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        lower = Lower()
        self.assertTrue(callable(lower))

    def test_default(self):
        lower = Lower()
        actual = lower(' Hello World!  ')
        self.assertEqual('hello world!', actual)

    def test_lstrip(self):
        lower = Lower(' /.')
        actual = lower(' /. Hello World!  ')
        self.assertEqual('hello world!', actual)

    def test_rstrip(self):
        lower = Lower(rstrip=' /.')
        actual = lower(' Hello World! ./ ')
        self.assertEqual('hello world!', actual)

    def test_both(self):
        lower = Lower('*', '/')
        actual = lower('* Hello World!/')
        self.assertEqual(' hello world!', actual)

    def test_does_not_strip(self):
        lower = Lower('', '')
        actual = lower(' * Hello World!/ ')
        self.assertEqual(' * hello world!/ ', actual)



class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        lower = Lower()
        self.assertEqual('Lower(None, None)', repr(lower))

    def test_lstrip_repr(self):
        lower = Lower('left')
        self.assertEqual("Lower('left', None)", repr(lower))

    def test_rstrip_repr(self):
        lower = Lower(rstrip='right')
        self.assertEqual("Lower(None, 'right')", repr(lower))

    def test_custom_repr(self):
        lower = Lower('left', 'right')
        self.assertEqual("Lower('left', 'right')", repr(lower))


if __name__ == '__main__':
    unittest.main()
