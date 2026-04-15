import unittest
import pickle
from unittest.mock import patch, Mock
from swak.cloud.gcp import GbqQuery2GcsParquet, Gbq
from swak.cloud.gcp.exceptions import GbqError


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        client = Mock()
        self.gbq = Mock(return_value=client)
        self.gbq.kwargs = {}
        self.gbq.project='project'
        self.export = GbqQuery2GcsParquet(self.gbq)

    def test_has_gbq(self):
        self.assertTrue(hasattr(self.export, 'gbq'))

    def test_gbq(self):
        self.assertIs(self.export.gbq, self.gbq)

    def test_has_path(self):
        self.assertTrue(hasattr(self.export, 'path'))

    def test_path(self):
        self.assertEqual('{}', self.export.path)

    def test_has_overwrite(self):
        self.assertTrue(hasattr(self.export, 'overwrite'))

    def test_overwrite(self):
        self.assertFalse(self.export.overwrite)

    def test_has_skip(self):
        self.assertTrue(hasattr(self.export, 'skip'))

    def test_skip(self):
        self.assertFalse(self.export.skip)

    def test_has_config(self):
        self.assertTrue(hasattr(self.export, 'config'))

    def test_config(self):
        self.assertIsNone(self.export.config)

    def test_has_polling_interval(self):
        self.assertTrue(hasattr(self.export, 'polling_interval'))

    def test_polling_interval(self):
        self.assertIsInstance(self.export.polling_interval, float)
        self.assertEqual(5.0, self.export.polling_interval)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.export, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.export.kwargs)

    def test_has_options(self):
        self.assertTrue(hasattr(self.export, 'options'))

    def test_options(self):
        expected =  {
            '_http': None,
            'client_info': None,
            'client_options': None,
            'credentials': None,
            'project': 'project'
        }
        self.assertDictEqual(expected, self.export.options)

    def test_has_flag(self):
        self.assertTrue(hasattr(self.export, 'flag'))

    def test_flag(self):
        self.assertEqual('false', self.export.flag)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        client = Mock()
        self.gbq = Mock(return_value=client)
        self.gbq.kwargs = {}
        self.gbq.project = 'project'

    def test_path(self):
        export = GbqQuery2GcsParquet(self.gbq, 'path')
        self.assertEqual('path', export.path)

    def test_path_strips(self):
        export = GbqQuery2GcsParquet(self.gbq, ' / . path/to/my  . /')
        self.assertEqual('path/to/my', export.path)

    def test_path_raises_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = GbqQuery2GcsParquet(self.gbq, 42)

    def test_path_raises_empty(self):
        with self.assertRaises(ValueError):
            _ = GbqQuery2GcsParquet(self.gbq, '')

    def test_overwrite(self):
        export = GbqQuery2GcsParquet(self.gbq, overwrite=True)
        self.assertTrue(export.overwrite)

    def test_skip(self):
        export = GbqQuery2GcsParquet(self.gbq, skip=True)
        self.assertTrue(export.skip)

    def test_polling_interval(self):
        export = GbqQuery2GcsParquet(self.gbq, polling_interval=3.0)
        self.assertIsInstance(export.polling_interval, float)
        self.assertEqual(3.0, export.polling_interval)

    def test_polling_interval_raises_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = GbqQuery2GcsParquet(self.gbq, polling_interval='foo')

    def test_polling_interval_raises_wrong_value(self):
        with self.assertRaises(ValueError):
            _ = GbqQuery2GcsParquet(self.gbq, polling_interval=-3)

    def test_kwargs(self):
        export = GbqQuery2GcsParquet(self.gbq, foo='bar', baz=42)
        self.assertDictEqual({'foo': 'bar', 'baz': 42}, export.kwargs)

    def test_options(self):
        expected = {
            '_http': None,
            'client_info': 'client_info',
            'client_options': None,
            'credentials': 'credentials',
            'project': 'project',
            'foo': 'bar',
            'baz': 42
        }
        export = GbqQuery2GcsParquet(
            self.gbq,
            foo='bar',
            baz=42,
            credentials='credentials',
            client_info='client_info'
        )
        self.assertDictEqual(expected, export.options)

    def test_flag(self):
        export = GbqQuery2GcsParquet(self.gbq, overwrite=True)
        self.assertEqual('true', export.flag)


class TestMethods(unittest.TestCase):

    def setUp(self):
        client = Mock()
        self.gbq = Mock(return_value=client)
        self.gbq.kwargs = {}
        self.gbq.project = 'project'
        self.bucket = Mock()
        self.gcs = Mock()
        self.gcs.get_bucket.return_value = self.bucket
        self.gcs_patch = patch(
            'swak.cloud.gcp.query2gcs.GcsClient',
            return_value=self.gcs
        )
        self.gcs_cls = self.gcs_patch.start()

    def tearDown(self):
        self.gcs_patch.stop()

    def test_normalize_full_path_default(self):
        export = GbqQuery2GcsParquet(self.gbq)
        path = 'bucket/prefix/folder'
        bucket, prefix = export._normalize(path)
        self.assertEqual('bucket', bucket)
        self.assertEqual('prefix/folder', prefix)

    def test_normalize_full_path_stripped(self):
        export = GbqQuery2GcsParquet(self.gbq)
        path = ' . // bucket/prefix/folder .//'
        bucket, prefix= export._normalize(path)
        self.assertEqual('bucket', bucket)
        self.assertEqual('prefix/folder', prefix)

    def test_normalize_inner_slashes_stripped(self):
        export = GbqQuery2GcsParquet(self.gbq)
        path = 'bucket///prefix//folder'
        bucket, prefix = export._normalize(path)
        self.assertEqual('bucket', bucket)
        self.assertEqual('prefix/folder', prefix)

    @patch('swak.cloud.gcp.query2gcs.uuid.uuid4')
    def test_normalize_creates_uuid(self, mock):
        mock.return_value = 'uuid'
        export = GbqQuery2GcsParquet(self.gbq)
        bucket, prefix = export._normalize(' / .bucket //.')
        self.assertEqual('uuid', prefix)
        self.assertEqual('bucket', bucket)

    def test_normalize_raises_on_empty_path(self):
        export = GbqQuery2GcsParquet(self.gbq)
        with self.assertRaises(ValueError):
            _ = export._normalize('')

    def test_normalize_raises_on_empty_path_after_cleanup(self):
        export = GbqQuery2GcsParquet(self.gbq)
        with self.assertRaises(ValueError):
            _ = export._normalize(' . //  .//')

    def test_skip_query_for_skip_false_overwrite_false_no_blobs(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        actual = export._skip_query_for('bucket', 'prefix')
        self.gcs_cls.assert_called_once_with(**export.options)
        self.gcs.list_blobs.assert_called_once_with('bucket', prefix='prefix')
        self.gcs.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()
        self.assertIsInstance(actual, bool)
        self.assertFalse(actual)

    def test_skip_query_for_skip_false_overwrite_false_blobs(self):
        self.gcs.list_blobs.return_value = [1, 2, 3]
        export = GbqQuery2GcsParquet(self.gbq)
        with self.assertRaises(FileExistsError):
            _ = export._skip_query_for('bucket', 'prefix')
        self.gcs_cls.assert_called_once_with(**export.options)
        self.gcs.list_blobs.assert_called_once_with('bucket', prefix='prefix')
        self.gcs.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()

    def test_skip_query_for_skip_false_overwrite_true_no_blobs(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq, overwrite=True)
        actual = export._skip_query_for('bucket', 'prefix')
        self.gcs_cls.assert_called_once_with(**export.options)
        self.gcs.list_blobs.assert_called_once_with('bucket', prefix='prefix')
        self.gcs.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()
        self.assertIsInstance(actual, bool)
        self.assertFalse(actual)

    def test_skip_query_for_skip_false_overwrite_true_blobs(self):
        self.gcs.list_blobs.return_value = [1, 2, 3]
        export = GbqQuery2GcsParquet(self.gbq, overwrite=True)
        actual = export._skip_query_for('bucket', 'prefix')
        self.gcs_cls.assert_called_once_with(**export.options)
        self.gcs.list_blobs.assert_called_once_with('bucket', prefix='prefix')
        self.gcs.get_bucket.assert_called_once_with('bucket')
        self.bucket.delete_blobs.assert_called_once_with([1, 2, 3])
        self.assertIsInstance(actual, bool)
        self.assertFalse(actual)

    def test_skip_query_for_skip_true_overwrite_false_no_blobs(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq, skip=True)
        actual = export._skip_query_for('bucket', 'prefix')
        self.gcs_cls.assert_called_once_with(**export.options)
        self.gcs.list_blobs.assert_called_once_with('bucket', prefix='prefix')
        self.gcs.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()
        self.assertIsInstance(actual, bool)
        self.assertFalse(actual)

    def test_skip_query_for_skip_true_overwrite_false_blobs(self):
        self.gcs.list_blobs.return_value = [1, 2, 3]
        export = GbqQuery2GcsParquet(self.gbq, skip=True)
        actual = export._skip_query_for('bucket', 'prefix')
        self.gcs_cls.assert_called_once_with(**export.options)
        self.gcs.list_blobs.assert_called_once_with('bucket', prefix='prefix')
        self.gcs.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()
        self.assertIsInstance(actual, bool)
        self.assertTrue(actual)

    def test_skip_query_for_skip_true_overwrite_true_no_blobs(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq, skip=True, overwrite=True)
        actual = export._skip_query_for('bucket', 'prefix')
        self.gcs_cls.assert_called_once_with(**export.options)
        self.gcs.list_blobs.assert_called_once_with('bucket', prefix='prefix')
        self.gcs.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()
        self.assertIsInstance(actual, bool)
        self.assertFalse(actual)

    def test_skip_query_for_skip_true_overwrite_true_blobs(self):
        self.gcs.list_blobs.return_value = [1, 2, 3]
        export = GbqQuery2GcsParquet(self.gbq, skip=True, overwrite=True)
        actual = export._skip_query_for('bucket', 'prefix')
        self.gcs_cls.assert_called_once_with(**export.options)
        self.gcs.list_blobs.assert_called_once_with('bucket', prefix='prefix')
        self.gcs.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()
        self.assertIsInstance(actual, bool)
        self.assertTrue(actual)

    def test_split_empty_raises(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        with self.assertRaises(ValueError):
            _ = export._split('')

    def test_split_semicolon_raises(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        with self.assertRaises(ValueError):
            _ = export._split('  ; ')

    def test_split_simple_query_no_trailing_semicolon(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        scripts, main = export._split('SELECT * FROM table')
        self.assertEqual('', scripts)
        self.assertEqual('SELECT * FROM table', main)

    def test_split_simple_query_trailing_semicolon(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        scripts, main = export._split('SELECT * FROM table;')
        self.assertEqual('', scripts)
        self.assertEqual('SELECT * FROM table', main)

    def test_split_simple_query_no_trailing_semicolon_stripped(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        scripts, main = export._split('\n ; SELECT * FROM table \n')
        self.assertEqual('', scripts)
        self.assertEqual('SELECT * FROM table', main)

    def test_split_simple_query_trailing_semicolon_stripped(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        scripts, main = export._split('\n ; SELECT * FROM table; \n')
        self.assertEqual('', scripts)
        self.assertEqual('SELECT * FROM table', main)

    def test_complex_simple_query_no_trailing_semicolon(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        query = 'DECLARE x INT64;SET x = 1;SELECT * FROM table'
        scripts, main = export._split(query)
        self.assertEqual('DECLARE x INT64;\nSET x = 1;\n', scripts)
        self.assertEqual('SELECT * FROM table', main)

    def test_complex_simple_query_trailing_semicolon(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        query = 'DECLARE x INT64;SET x = 1;SELECT * FROM table;'
        scripts, main = export._split(query)
        self.assertEqual('DECLARE x INT64;\nSET x = 1;\n', scripts)
        self.assertEqual('SELECT * FROM table', main)

    def test_complex_simple_query_no_trailing_semicolon_stripped(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        query = ' \n; DECLARE x INT64;SET x = 1;SELECT * FROM table'
        scripts, main = export._split(query)
        self.assertEqual('DECLARE x INT64;\nSET x = 1;\n', scripts)
        self.assertEqual('SELECT * FROM table', main)


    def test_complex_simple_query_trailing_semicolon_stripped(self):
        self.gcs.list_blobs.return_value = []
        export = GbqQuery2GcsParquet(self.gbq)
        query = ' \n ; DECLARE x INT64;SET x = 1;SELECT * FROM table;'
        scripts, main = export._split(query)
        self.assertEqual('DECLARE x INT64;\nSET x = 1;\n', scripts)
        self.assertEqual('SELECT * FROM table', main)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.job = Mock()
        self.job.running = Mock(side_effect=[True, False])
        self.job.error_result = None
        self.client = Mock()
        self.client.query.return_value = self.job
        self.gbq = Mock(return_value=self.client)
        self.gbq.kwargs = {}
        self.gbq.project = 'project'
        self.bucket = Mock()
        self.gcs = Mock()
        self.gcs.get_bucket.return_value = self.bucket
        self.gcs_patch = patch(
            'swak.cloud.gcp.query2gcs.GcsClient',
            return_value=self.gcs
        )
        self.gcs_cls = self.gcs_patch.start()
        self.query = 'DECLARE;\nSELECT * FROM table'

    def tearDown(self):
        self.gcs_patch.stop()


    def test_callable(self):
        export = GbqQuery2GcsParquet(self.gbq)
        self.assertTrue(callable(export))

    def test_skip_return_value(self):
        export = GbqQuery2GcsParquet(self.gbq, 'bucket/prefix/{}')
        with patch.object(export, '_skip_query_for') as mock:
            mock.return_value = True
            actual = export(self.query, 'path')
            mock.assert_called_once_with('bucket', 'prefix/path')
        self.assertEqual('bucket/prefix/path/', actual)

    def test_client_created(self):
        export = GbqQuery2GcsParquet(self.gbq, 'bucket/{}', polling_interval=1)
        with patch.object(export, '_skip_query_for') as mock:
            mock.return_value = False
            actual = export(self.query, 'prefix')
            mock.assert_called_once_with('bucket', 'prefix')
        self.gbq.assert_called_once_with()
        self.assertEqual('bucket/prefix/', actual)

    def test_query_fired(self):
        export = GbqQuery2GcsParquet(self.gbq, 'bucket/{}', polling_interval=1)
        with patch.object(export, '_skip_query_for') as mock:
            mock.return_value = False
            actual = export(self.query, 'prefix')
            mock.assert_called_once_with('bucket', 'prefix')
        expected = """DECLARE;

    EXPORT DATA OPTIONS(
        uri="gs://bucket/prefix/*.parquet"
      , format="PARQUET"
      , compression="SNAPPY"
      , overwrite=false
    ) AS

SELECT * FROM table"""
        self.client.query.assert_called_once_with(expected, None)
        self.assertEqual('bucket/prefix/', actual)

    def test_config_passed(self):
        export = GbqQuery2GcsParquet(
            self.gbq,
            'bucket/{}',
            config='config',
            polling_interval=1
        )
        with patch.object(export, '_skip_query_for') as mock:
            mock.return_value = False
            actual = export(self.query, 'prefix')
            mock.assert_called_once_with('bucket', 'prefix')
        expected = """DECLARE;

    EXPORT DATA OPTIONS(
        uri="gs://bucket/prefix/*.parquet"
      , format="PARQUET"
      , compression="SNAPPY"
      , overwrite=false
    ) AS

SELECT * FROM table"""
        self.client.query.assert_called_once_with(expected, 'config')
        self.assertEqual('bucket/prefix/', actual)

    @patch('swak.cloud.gcp.query2gcs.time.sleep')
    def test_polling_called_correct_number_of_times(self, mock_sleep):
        export = GbqQuery2GcsParquet(self.gbq, 'bucket/{}', polling_interval=1)
        self.job.running.side_effect = [True, True, True, False]
        with patch.object(export, '_skip_query_for') as mock:
            mock.return_value = False
            actual = export(self.query, 'prefix')
            mock.assert_called_once_with('bucket', 'prefix')
        self.assertEqual(3, mock_sleep.call_count)
        mock_sleep.assert_called_with(1)
        self.assertEqual('bucket/prefix/', actual)

    def test_query_error(self):
        export = GbqQuery2GcsParquet(self.gbq, 'bucket/{}', polling_interval=1)
        self.job.error_result = {'reason': 'error', 'message': 'message'}
        with patch.object(export, '_skip_query_for') as mock:
            mock.return_value = False
            with self.assertRaises(GbqError) as error:
                _ = export(self.query, 'prefix')
            expected = '\nERROR: message'
            actual = str(error.exception)
            self.assertEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.gbq = Gbq('project')

    def test_default_repr(self):
        export = GbqQuery2GcsParquet(self.gbq)
        expected = ("GbqQuery2GcsParquet(Gbq('project'), "
                    "'{}', False, False, None, 5.0)")
        self.assertEqual(expected, repr(export))

    def test_custom_repr(self):
        export = GbqQuery2GcsParquet(
            self.gbq,
            'bucket/prefix/{}',
            config='config',
            skip=True,
            overwrite=True,
            polling_interval=3
        )
        expected = ("GbqQuery2GcsParquet(Gbq('project'), "
                    "'bucket/prefix/{}', True, True, 'config', 3.0)")
        self.assertEqual(expected, repr(export))

    def test_pickle_works(self):
        export = GbqQuery2GcsParquet(
            self.gbq,
            'bucket/prefix/{}',
            config='config',
            skip=True,
            overwrite=True,
            polling_interval=3
        )
        _ = pickle.loads(pickle.dumps(export))


if __name__ == '__main__':
    unittest.main()
