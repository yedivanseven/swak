import unittest
from swak.jsonobject.fields import Strip


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.strip = Strip()

    def test_has_left(self):
        self.assertTrue(hasattr(self.strip, 'left'))

    def test_left(self):
        self.assertIsNone(self.strip.left)

    def test_has_right(self):
        self.assertTrue(hasattr(self.strip, 'right'))

    def test_right(self):
        self.assertIsNone(self.strip.right)


class TestAttributes(unittest.TestCase):

    def test_left_arg(self):
        strip = Strip('left')
        self.assertEqual('left', strip.left)

    def test_left_kwarg(self):
        strip = Strip(left='left')
        self.assertEqual('left', strip.left)

    def test_right_arg(self):
        strip = Strip(None, 'right')
        self.assertEqual('right', strip.right)

    def test_rstrip_kwarg(self):
        strip = Strip(right='right')
        self.assertEqual('right', strip.right)

    def test_args(self):
        strip = Strip('left', 'right')
        self.assertEqual('left', strip.left)
        self.assertEqual('right', strip.right)

    def test_kwargs(self):
        strip = Strip(left='left', right='right')
        self.assertEqual('left', strip.left)
        self.assertEqual('right', strip.right)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        strip = Strip()
        self.assertTrue(callable(strip))

    def test_default(self):
        strip = Strip()
        actual = strip(' Hello World!  ')
        self.assertEqual('Hello World!', actual)

    def test_left(self):
        strip = Strip(' /.')
        actual = strip(' /. Hello World!  ')
        self.assertEqual('Hello World!', actual)

    def test_right(self):
        strip = Strip(right=' /.')
        actual = strip(' Hello World! ./ ')
        self.assertEqual('Hello World!', actual)

    def test_both(self):
        strip = Strip('*', '/')
        actual = strip('* Hello World!/')
        self.assertEqual(' Hello World!', actual)

    def test_does_not_strip(self):
        strip = Strip('', '')
        actual = strip(' * Hello World!/ ')
        self.assertEqual(' * Hello World!/ ', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        strip = Strip()
        self.assertEqual('Strip(None, None)', repr(strip))

    def test_left_repr(self):
        strip = Strip('left')
        self.assertEqual("Strip('left', None)", repr(strip))

    def test_right_repr(self):
        strip = Strip(right='right')
        self.assertEqual("Strip(None, 'right')", repr(strip))

    def test_custom_repr(self):
        strip = Strip('left', 'right')
        self.assertEqual("Strip('left', 'right')", repr(strip))


if __name__ == '__main__':
    unittest.main()
