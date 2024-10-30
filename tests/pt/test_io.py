import unittest
from unittest.mock import patch, Mock
from swak.pt.io import StateSaver, StateLoader, ModelSaver, ModelLoader


class TestStateSaver(unittest.TestCase):

    def setUp(self):
        self.path = '/path/model.pt'
        self.save = StateSaver(self.path)
        self.model = Mock()
        self.model.state_dict = Mock(return_value=42)

    def test_has_path(self):
        self.assertTrue(hasattr(self.save, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.save.path)

    def test_path_stripped(self):
        save = StateSaver('  /path/model.pt ')
        self.assertEqual(self.path, save.path)

    def test_repr(self):
        actual = repr(self.save)
        expected = f"StateSaver('{self.path}')"
        self.assertEqual(expected, actual)

    @patch('torch.save')
    def test_state_dict_called(self, _):
        _ = self.save(self.model)
        self.model.state_dict.assert_called_once_with()

    @patch('torch.save')
    def test_save_called(self, mock):
        _ = self.save(self.model)
        mock.assert_called_once_with(42, self.path)

    @patch('torch.save')
    def test_return_value(self, _):
        actual = self.save(self.model)
        self.assertTupleEqual((), actual)

    @patch('torch.save')
    def test_parts_interpolated(self, mock):
        save = StateSaver('/hello_{}/foo_{}/model.pt')
        _ = save(self.model, 'world', 'bar')
        mock.assert_called_once_with(42, '/hello_world/foo_bar/model.pt')

    @patch('torch.save')
    def test_interpolated_parts_stripped(self, mock):
        save = StateSaver('/hello_{}/foo_{}/model.pt{}')
        _ = save(self.model, 'world', 'bar', '  ')
        mock.assert_called_once_with(42, '/hello_world/foo_bar/model.pt')


class TestStateLoader(unittest.TestCase):

    def setUp(self):
        self.path = '/path/model.pt'
        self.load = StateLoader(self.path)
        self.model = Mock()
        self.model.state_dict = Mock(return_value={'bar': 1})
        self.model.load_state_dict = Mock()
        self.model.to = Mock(return_value=self.model)

    def test_has_path(self):
        self.assertTrue(hasattr(self.load, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.load.path)

    def test_path_stripped(self):
        load = StateLoader('  /path/model.pt ')
        self.assertEqual(self.path, load.path)

    def test_has_map_location(self):
        self.assertTrue(hasattr(self.load, 'map_location'))

    def test_default_map_location(self):
        self.assertIsNone(self.load.map_location)

    def test_custom_map_location(self):
        load  = StateLoader(self.path, 'cpu')
        self.assertEqual('cpu', load.map_location)

    def test_has_merge(self):
        self.assertTrue(hasattr(self.load, 'merge'))

    def test_default_merge(self):
        self.assertIsInstance(self.load.merge, bool)
        self.assertTrue(self.load.merge)

    def test_custom_merge(self):
        load  = StateLoader(self.path, merge=False)
        self.assertFalse(load.merge)

    def test_has_not_found(self):
        self.assertTrue(hasattr(self.load, 'not_found'))

    def test_default_not_found(self):
        self.assertEqual('raise', self.load.not_found)

    def test_custom_not_found(self):
        load  = StateLoader(self.path, not_found='warn')
        self.assertEqual('warn', load.not_found)

    def test_not_found_strip(self):
        load  = StateLoader(self.path, not_found=' warn  ')
        self.assertEqual('warn', load.not_found)

    def test_not_found_lower(self):
        load  = StateLoader(self.path, not_found='wArN')
        self.assertEqual('warn', load.not_found)

    def test_default_repr(self):
        actual = repr(self.load)
        expected = f"StateLoader('{self.path}', None, True, 'raise')"
        self.assertEqual(expected, actual)

    def test_custom_repr(self):
        load = StateLoader('/model.pt', 'cpu', False, 'warn')
        actual = repr(load)
        expected = "StateLoader('/model.pt', 'cpu', False, 'warn')"
        self.assertEqual(expected, actual)

    def test_callable(self):
        self.assertTrue(callable(self.load))

    @patch('torch.load', return_value={'foo': 42})
    def test_load_called_with_defaults(self, mock):
        _ = self.load(self.model)
        mock.assert_called_once_with(
            self.path,
            self.load.map_location,
            weights_only=True
        )

    @patch('torch.load', return_value={'foo': 42})
    def test_load_called_with_custom(self, mock):
        load = StateLoader(self.path, 'cpu')
        _ = load(self.model)
        mock.assert_called_once_with(
            self.path,
            'cpu',
            weights_only=True
        )

    @patch('torch.load', return_value={'foo': 42})
    def test_model_state_dict_called_on_merge(self, _):
        _ = self.load(self.model)
        self.model.state_dict.assert_called_once_with()

    @patch('torch.load', return_value={'foo': 42})
    def test_model_state_dict_not_called_on_merge(self, _):
        load = StateLoader(self.path, merge=False)
        _ = load(self.model)
        self.model.state_dict.assert_not_called()

    @patch('torch.load', return_value={'foo': 42})
    def test_model_state_dict_merged(self, _):
        _ = self.load(self.model)
        expected = {'foo': 42, 'bar': 1}
        self.model.load_state_dict.assert_called_once_with(expected)

    @patch('torch.load', return_value={'foo': 42})
    def test_model_state_dict_not_merged(self, _):
        load = StateLoader(self.path, merge=False)
        _ = load(self.model)
        expected = {'foo': 42}
        self.model.load_state_dict.assert_called_once_with(expected)

    @patch('torch.load', side_effect=FileNotFoundError())
    def test_raises_on_not_found(self, _):
        with self.assertRaises(FileNotFoundError):
            _ = self.load(self.model)

    @patch('torch.load', side_effect=FileNotFoundError())
    def test_warns_on_not_found(self, _):
        load = StateLoader(self.path, not_found='warn')
        with self.assertWarns(UserWarning):
            _ = load(self.model)

    @patch('torch.load', side_effect=FileNotFoundError())
    def test_merges_on_warn_not_found(self, _):
        load = StateLoader(self.path, not_found='warn')
        with self.assertWarns(UserWarning):
            _ = load(self.model)
        expected = {'bar': 1}
        self.model.load_state_dict.assert_called_once_with(expected)

    @patch('torch.load', side_effect=FileNotFoundError())
    def test_merges_ignore_not_found(self, _):
        load = StateLoader(self.path, not_found='ignore')
        _ = load(self.model)
        expected = {'bar': 1}
        self.model.load_state_dict.assert_called_once_with(expected)

    @patch('torch.load', return_value={'foo': 42})
    def test_to_called_default(self, _):
        _ = self.load(self.model)
        self.model.to.assert_called_once_with(None)

    @patch('torch.load', return_value={'foo': 42})
    def test_to_called_custom(self, _):
        load = StateLoader(self.path, map_location='cpu')
        _ = load(self.model)
        self.model.to.assert_called_once_with('cpu')

    @patch('torch.load', return_value={'foo': 42})
    def test_to_not_present(self, _):

        class Module:
            pass

        model = Module()
        model.state_dict = Mock(return_value={'bar': 1})
        model.load_state_dict = Mock()
        _ = self.load(model)

    @patch('torch.load', return_value={'foo': 42})
    def test_return_value(self, _):
        actual = self.load(self.model)
        self.assertIs(actual, self.model)

    @patch('torch.load', return_value={'foo': 42})
    def test_parts_interpolated(self, mock):
        load = StateLoader('/hello_{}/foo_{}/model.pt')
        _ = load(self.model, 'world', 'bar')
        mock.assert_called_once_with(
            '/hello_world/foo_bar/model.pt',
            None,
            weights_only=True
        )

    @patch('torch.load', return_value={'foo': 42})
    def test_interpolated_parts_stripped(self, mock):
        load = StateLoader('/hello_{}/foo_{}/model.pt{}')
        _ = load(self.model, 'world', 'bar', '  ')
        mock.assert_called_once_with(
            '/hello_world/foo_bar/model.pt',
            None,
            weights_only=True
        )


class TestModelSaver(unittest.TestCase):

    def setUp(self):
        self.path = '/path/model.pt'
        self.save = ModelSaver(self.path)
        self.model = Mock()

    def test_has_path(self):
        self.assertTrue(hasattr(self.save, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.save.path)

    def test_path_stripped(self):
        save = ModelSaver('  /path/model.pt ')
        self.assertEqual(self.path, save.path)

    def test_repr(self):
        actual = repr(self.save)
        expected = f"ModelSaver('{self.path}')"
        self.assertEqual(expected, actual)

    @patch('torch.save')
    def test_save_called(self, mock):
        _ = self.save(self.model)
        mock.assert_called_once_with(self.model, self.path)

    @patch('torch.save')
    def test_return_value(self, _):
        actual = self.save(self.model)
        self.assertTupleEqual((), actual)

    @patch('torch.save')
    def test_parts_interpolated(self, mock):
        save = ModelSaver('/hello_{}/foo_{}/model.pt')
        _ = save(self.model, 'world', 'bar')
        mock.assert_called_once_with(
            self.model,
            '/hello_world/foo_bar/model.pt'
        )

    @patch('torch.save')
    def test_interpolated_parts_stripped(self, mock):
        save = ModelSaver('/hello_{}/foo_{}/model.pt{}')
        _ = save(self.model, 'world', 'bar', '  ')
        mock.assert_called_once_with(
            self.model,
            '/hello_world/foo_bar/model.pt'
        )


class TestModelLoader(unittest.TestCase):

    def setUp(self):
        self.path = '/path/model.pt'
        self.load = ModelLoader(self.path)

    def test_has_path(self):
        self.assertTrue(hasattr(self.load, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.load.path)

    def test_path_stripped(self):
        load = StateLoader('  /path/model.pt ')
        self.assertEqual(self.path, load.path)

    def test_has_map_location(self):
        self.assertTrue(hasattr(self.load, 'map_location'))

    def test_default_map_location(self):
        self.assertIsNone(self.load.map_location)

    def test_custom_map_location(self):
        load = ModelLoader(self.path, 'cpu')
        self.assertEqual('cpu', load.map_location)

    def test_default_repr(self):
        actual = repr(self.load)
        expected = f"ModelLoader('{self.path}', None)"
        self.assertEqual(expected, actual)

    def test_custom_repr(self):
        load = ModelLoader('/model.pt', 'cpu')
        actual = repr(load)
        expected = "ModelLoader('/model.pt', 'cpu')"
        self.assertEqual(expected, actual)

    def test_callable(self):
        self.assertTrue(callable(self.load))

    @patch('torch.load', return_value={'foo': 42})
    def test_load_called_with_defaults(self, mock):
        _ = self.load()
        mock.assert_called_once_with(
            self.path,
            self.load.map_location,
            weights_only=False
        )

    @patch('torch.load', return_value={'foo': 42})
    def test_load_called_with_custom(self, mock):
        load = ModelLoader(self.path, 'cpu')
        _ = load()
        mock.assert_called_once_with(
            self.path,
            'cpu',
            weights_only=False
        )

    @patch('torch.load')
    def test_to_called_default(self, mock):
        model = Mock()
        model.to = Mock(return_value=42)
        mock.return_value = model
        _ = self.load()
        model.to.assert_called_once_with(None)

    @patch('torch.load')
    def test_to_called_custom(self, mock):
        model = Mock()
        model.to = Mock(return_value=42)
        mock.return_value = model
        load = ModelLoader(self.path, 'cpu')
        _ = load()
        model.to.assert_called_once_with('cpu')

    @patch('torch.load')
    def test_to_not_present(self, mock):
        mock.return_value = object()
        _ = self.load()

    @patch('torch.load')
    def test_return_value(self, mock):
        model = object()
        mock.return_value = model
        actual = self.load()
        self.assertIs(actual, model)

    @patch('torch.load', return_value={'foo': 42})
    def test_path_append_file(self, mock):
        load = ModelLoader('/path')
        _ = load('model.pt')
        mock.assert_called_once_with(
            self.path,
            load.map_location,
            weights_only=False
        )

    @patch('torch.load', return_value={'foo': 42})
    def test_path_append_dir(self, mock):
        load = ModelLoader('/path')
        _ = load('/model.pt')
        mock.assert_called_once_with(
            self.path,
            load.map_location,
            weights_only=False
        )

    @patch('torch.load', return_value={'foo': 42})
    def test_path_append_strips(self, mock):
        load = ModelLoader('/path')
        _ = load(' /model.pt  /')
        mock.assert_called_once_with(
            self.path,
            load.map_location,
            weights_only=False
        )


if __name__ == '__main__':
    unittest.main()
