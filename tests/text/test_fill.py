import unittest
from swak.text import FormFiller


class TestEmptyInstantiation(unittest.TestCase):

    def test_instantiation(self):
        _ = FormFiller

    def test_has_mapping(self):
        f = FormFiller()
        self.assertTrue(hasattr(f, 'mapping'))

    def test_mapping_correct(self):
        f = FormFiller()
        self.assertDictEqual({}, f.mapping)


# ToDo: Continue here!

if __name__ == '__main__':
    unittest.main()
