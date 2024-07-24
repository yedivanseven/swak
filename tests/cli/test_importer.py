import unittest
from swak.cli import Importer
from swak.cli.exceptions import ImporterError
from .package.module import f, g
from .package.sub_package.sub_module import foo, bar


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Importer('package')

    def test_has_package(self):
        i = Importer('package')
        self.assertTrue(hasattr(i, 'package'))

    def test_package_type(self):
        i = Importer('package')
        self.assertIsInstance(i.package, str)

    def test_package_value(self):
        i = Importer('package')
        self.assertEqual('package', i.package)

    def test_has_module(self):
        i = Importer('package')
        self.assertTrue(hasattr(i, 'module'))

    def test_module_type(self):
        i = Importer('package')
        self.assertIsInstance(i.module, str)

    def test_module_value(self):
        i = Importer('package')
        self.assertEqual('steps', i.module)

    def test_has_path(self):
        i = Importer('package')
        self.assertTrue(hasattr(i, 'path'))

    def test_path_type(self):
        i = Importer('package')
        self.assertIsInstance(i.path, str)

    def test_path_value(self):
        i = Importer('package')
        self.assertEqual('package.steps', i.path)

    def test_package_stripped(self):
        importer = Importer('  ./package. /')
        self.assertEqual('package', importer.package)


class TestCustomAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Importer('package', 'module')

    def test_has_package(self):
        i = Importer('package', 'module')
        self.assertTrue(hasattr(i, 'package'))

    def test_package_type(self):
        i = Importer('package', 'module')
        self.assertIsInstance(i.package, str)

    def test_package_value(self):
        i = Importer('package', 'module')
        self.assertEqual('package', i.package)

    def test_has_module(self):
        i = Importer('package', 'module')
        self.assertTrue(hasattr(i, 'module'))

    def test_module_type(self):
        i = Importer('package', 'module')
        self.assertIsInstance(i.module, str)

    def test_module_value(self):
        i = Importer('package', 'module')
        self.assertEqual('module', i.module)

    def test_has_path(self):
        i = Importer('package', 'module')
        self.assertTrue(hasattr(i, 'path'))

    def test_path_type(self):
        i = Importer('package', 'module')
        self.assertIsInstance(i.path, str)

    def test_path_value(self):
        i = Importer('package', 'module')
        self.assertEqual('package.module', i.path)

    def test_module_stripped(self):
        importer = Importer('package', '  ./module. /')
        self.assertEqual('module', importer.module)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        i = Importer('package', 'module')
        self.assertTrue(callable(i))

    def test_call_short_package_long_module(self):
        i = Importer(__package__, 'package.module')
        imported_f, imported_g = i('f', 'g')
        self.assertIs(imported_f, f)
        self.assertIs(imported_g, g)

    def test_call_return_type(self):
        i = Importer(__package__, 'package.module')
        imported = i('f', 'g')
        self.assertListEqual([f, g], imported)

    def test_call_long_package_short_module(self):
        i = Importer(__package__ + '.package', 'module')
        imported_f, imported_g = i('f', 'g')
        self.assertIs(imported_f, f)
        self.assertIs(imported_g, g)

    def test_call_short_package_long_submodule(self):
        i = Importer(__package__, 'package.sub_package.sub_module')
        imported_f, imported_g = i('foo', 'bar')
        self.assertIs(imported_f, foo)
        self.assertIs(imported_g, bar)

    def test_call_long_package_short_submodule(self):
        i = Importer(__package__ + '.package.sub_package', 'sub_module')
        imported_f, imported_g = i('foo', 'bar')
        self.assertIs(imported_f, foo)
        self.assertIs(imported_g, bar)

    def test_raises_on_path_not_found(self):
        importer = Importer('foo', 'bar')
        expected = 'Could not import module "foo.bar"!'
        with self.assertRaises(ImporterError) as error:
            _ = importer('baz')
        self.assertEqual(expected, str(error.exception))

    def test_raises_on_importee_not_found(self):
        importer = Importer(__package__, 'package.module')
        start = 'Could not import "baz" from module "'
        end = 'cli.package.module"!'
        with self.assertRaises(ImporterError) as error:
            _ = importer('baz')
        self.assertTrue(str(error.exception).startswith(start))
        self.assertTrue(str(error.exception).endswith(end))


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        importer = Importer('package')
        self.assertEqual("Importer('package', 'steps')", repr(importer))

    def test_custom_repr(self):
        importer = Importer('package', 'module')
        self.assertEqual("Importer('package', 'module')", repr(importer))


if __name__ == '__main__':
    unittest.main()
