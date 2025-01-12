import unittest
import pickle
from unittest.mock import patch, Mock
from swak.cloud.gcp import GbqQuery2GcsParquet
from swak.cloud.gcp.exceptions import GbqError


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.export = GbqQuery2GcsParquet('project', 'bucket')

    def test_project(self):
        self.assertTrue(hasattr(self.export, 'project'))
        self.assertEqual('project', self.export.project)

    def test_bucket(self):
        self.assertTrue(hasattr(self.export, 'bucket'))
        self.assertEqual('bucket', self.export.bucket)

    def test_prefix(self):
        self.assertTrue(hasattr(self.export, 'prefix'))
        self.assertEqual('', self.export.prefix)

    def test_overwrite(self):
        self.assertTrue(hasattr(self.export, 'overwrite'))
        self.assertFalse(self.export.overwrite)

    def test_skip(self):
        self.assertTrue(hasattr(self.export, 'skip'))
        self.assertFalse(self.export.skip)

    def test_polling_interval(self):
        self.assertTrue(hasattr(self.export, 'polling_interval'))
        self.assertIsInstance(self.export.polling_interval, int)
        self.assertEqual(5, self.export.polling_interval)

    def test_priority(self):
        self.assertTrue(hasattr(self.export, 'priority'))
        self.assertEqual('BATCH', self.export.priority)

    def test_gbq_kws(self):
        self.assertTrue(hasattr(self.export, 'gbq_kws'))
        self.assertDictEqual({}, self.export.gbq_kws)

    def test_gcs_kws(self):
        self.assertTrue(hasattr(self.export, 'gcs_kws'))
        self.assertDictEqual({}, self.export.gcs_kws)

    def test_project_stripped(self):
        export = GbqQuery2GcsParquet(' / .project /. ', 'bucket')
        self.assertEqual('project', export.project)

    def test_bucket_stripped(self):
        export = GbqQuery2GcsParquet('project', ' . / bucket. /')
        self.assertEqual('bucket', export.bucket)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            '  . /prefix . / ',
            True,
            True,
            7,
            ' priORIty  ',
            {'foo': 'bar'},
            {'baz': 42},
        )

    def test_prefix(self):
        self.assertEqual('prefix', self.export.prefix)

    def test_overwrite(self):
        self.assertTrue(self.export.overwrite)

    def test_skip(self):
        self.assertTrue(self.export.skip)

    def test_polling_interval(self):
        self.assertEqual(7, self.export.polling_interval)

    def test_priority(self):
        self.assertEqual('PRIORITY', self.export.priority)

    def test_gbq_kws(self):
        self.assertTrue(hasattr(self.export, 'gbq_kws'))
        self.assertDictEqual({'foo': 'bar'}, self.export.gbq_kws)

    def test_gcs_kws(self):
        self.assertTrue(hasattr(self.export, 'gcs_kws'))
        self.assertDictEqual({'baz': 42}, self.export.gcs_kws)


class TestUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.query = 'DECLARE;\nSET;\nSELECT *\n'
        self.job = Mock()
        self.job.running.return_value = False
        self.job.error_result = {}
        self.bucket = Mock()
        self.client_instance = Mock()
        self.storage_instance = Mock()
        self.storage_instance.list_blobs = Mock(return_value=[1, 2, 3])
        self.storage_instance.get_bucket.return_value = self.bucket
        self.client_instance.query.return_value = self.job
        self.client_patch = patch(
            'google.cloud.bigquery.Client',
            return_value=self.client_instance
        )
        self.storage_patch = patch(
            'google.cloud.storage.Client',
            return_value=self.storage_instance
        )
        self.config_patch = patch(
            'google.cloud.bigquery.QueryJobConfig',
            return_value='config'
        )
        self.client_class = self.client_patch.start()
        self.storage_class = self.storage_patch.start()
        self.config = self.config_patch.start()

    def tearDown(self) -> None:
        self.client_patch.stop()
        self.storage_patch.stop()
        self.config_patch.stop()

    def test_callable(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        self.assertTrue(callable(export))

    def test_storage_called_default(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export(self.query)
        self.storage_class.assert_called_once_with('project')

    def test_storage_called_kwargs(self):
        export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            gcs_kws={'foo': 'bar'}
        )
        _ = export(self.query)
        self.storage_class.assert_called_once_with('project', foo='bar')

    def test_list_called_prefix_instantiation(self):
        export = GbqQuery2GcsParquet('project', 'bucket', 'prefix')
        _ = export(self.query)
        self.storage_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='prefix/'
        )

    def test_list_called_prefix_call(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export(self.query, 'prefix')
        self.storage_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='prefix/'
        )

    def test_list_called_prefix_both(self):
        export = GbqQuery2GcsParquet('project', 'bucket', 'pre')
        _ = export(self.query, 'fix')
        self.storage_instance.list_blobs.assert_called_once_with(
            'bucket',
            prefix='pre/fix/'
        )

    def test_get_not_called_exists_default(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export(self.query)
        self.storage_instance.get_bucket.assert_not_called()

    def test_get_called_exists_overwrite(self):
        export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            overwrite=True
        )
        _ = export(self.query)
        self.storage_instance.get_bucket.assert_called_once_with('bucket')
        self.bucket.delete_blobs.assert_called_once_with([1, 2, 3])

    def test_get_not_called_not_exists_overwrite(self):
        self.storage_instance.list_blobs = Mock(return_value=[])
        export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            overwrite=True
        )
        _ = export(self.query)
        self.storage_instance.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()

    def test_get_not_called_exists_skip(self):
        export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            skip=True
        )
        _ = export(self.query)
        self.storage_instance.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()

    def test_get_not_called_not_exists_skip(self):
        self.storage_instance.list_blobs = Mock(return_value=[])
        export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            skip=True
        )
        _ = export(self.query)
        self.storage_instance.get_bucket.assert_not_called()
        self.bucket.delete_blobs.assert_not_called()

    def test_client_called_default(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export(self.query)
        self.client_class.assert_called_once_with('project')

    def test_client_called_kwargs(self):
        export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            gbq_kws={'foo': 'bar'}
        )
        _ = export(self.query)
        self.client_class.assert_called_once_with('project', foo='bar')

    def test_config_called_defaults(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export(self.query)
        self.config.assert_called_once_with(priority='BATCH')

    def test_config_called_kwargs(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export(self.query, foo='bar')
        self.config.assert_called_once_with(priority='BATCH', foo='bar')

    def test_query_no_scripts_prefix_instantiation(self):
        export = GbqQuery2GcsParquet('project', 'bucket', 'prefix')
        _ = export('SELECT *\n')
        expected = ('\n'
                    '    EXPORT DATA OPTIONS(\n'
                    '        uri="gs://bucket/prefix/*.parquet"\n'
                    '      , format="PARQUET"\n'
                    '      , compression="SNAPPY"\n'
                    '      , overwrite=false\n'
                    '    ) AS\n'
                    'SELECT *\n')
        self.client_instance.query.assert_called_once_with(expected, 'config')

    def test_query_no_scripts_prefix_call(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export('SELECT *\n', 'prefix')
        expected = ('\n'
                    '    EXPORT DATA OPTIONS(\n'
                    '        uri="gs://bucket/prefix/*.parquet"\n'
                    '      , format="PARQUET"\n'
                    '      , compression="SNAPPY"\n'
                    '      , overwrite=false\n'
                    '    ) AS\n'
                    'SELECT *\n')
        self.client_instance.query.assert_called_once_with(expected, 'config')

    def test_query_prefix_call_stripped(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export('SELECT *\n', ' . / prefix./ ')
        expected = ('\n'
                    '    EXPORT DATA OPTIONS(\n'
                    '        uri="gs://bucket/prefix/*.parquet"\n'
                    '      , format="PARQUET"\n'
                    '      , compression="SNAPPY"\n'
                    '      , overwrite=false\n'
                    '    ) AS\n'
                    'SELECT *\n')
        self.client_instance.query.assert_called_once_with(expected, 'config')

    def test_query_no_scripts_prefix_both(self):
        export = GbqQuery2GcsParquet('project', 'bucket', 'pre')
        _ = export('SELECT *\n', 'fix')
        expected = ('\n'
                    '    EXPORT DATA OPTIONS(\n'
                    '        uri="gs://bucket/pre/fix/*.parquet"\n'
                    '      , format="PARQUET"\n'
                    '      , compression="SNAPPY"\n'
                    '      , overwrite=false\n'
                    '    ) AS\n'
                    'SELECT *\n')
        self.client_instance.query.assert_called_once_with(expected, 'config')

    def test_query_scripts(self):
        export = GbqQuery2GcsParquet('project', 'bucket', 'pre')
        _ = export('DECLARE;\nSET;\nSELECT *\n', 'fix')
        expected = ('DECLARE;\n'
                    'SET;\n'
                    '\n'
                    '    EXPORT DATA OPTIONS(\n'
                    '        uri="gs://bucket/pre/fix/*.parquet"\n'
                    '      , format="PARQUET"\n'
                    '      , compression="SNAPPY"\n'
                    '      , overwrite=false\n'
                    '    ) AS\n\n'
                    'SELECT *\n')
        self.client_instance.query.assert_called_once_with(expected, 'config')

    def test_query_respects_overwrite(self):
        export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            'pre',
            overwrite=True
        )
        _ = export('SELECT *\n', 'fix')
        expected = ('\n'
                    '    EXPORT DATA OPTIONS(\n'
                    '        uri="gs://bucket/pre/fix/*.parquet"\n'
                    '      , format="PARQUET"\n'
                    '      , compression="SNAPPY"\n'
                    '      , overwrite=true\n'
                    '    ) AS\n'
                    'SELECT *\n')
        self.client_instance.query.assert_called_once_with(expected, 'config')

    def test_returns_prefix_instantiation(self):
        export = GbqQuery2GcsParquet('project', 'bucket', 'prefix')
        prefix = export('SELECT *\n')
        self.assertEqual('prefix/', prefix)

    def test_returns_prefix_call(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        prefix = export('SELECT *\n', 'prefix')
        self.assertEqual('prefix/', prefix)

    def test_returns_prefix_combined(self):
        export = GbqQuery2GcsParquet('project', 'bucket', 'pre')
        prefix = export('SELECT *\n', 'fix')
        self.assertEqual('pre/fix/', prefix)

    def test_creates_prefix(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        prefix = export('SELECT *\n')
        self.assertIsInstance(prefix, str)
        expected = ('^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]'
                    '{4}-[0-9a-f]{4}-[0-9a-f]{12}/$')
        self.assertRegex(prefix, expected)

    def test_checks_job_running(self):
        export = GbqQuery2GcsParquet('project', 'bucket')
        _ = export(self.query)
        self.job.running.assert_called_once_with()

    def test_raises(self):
        self.job.error_result = {'reason': 'reason', 'message': 'message'}
        export = GbqQuery2GcsParquet('project', 'bucket')
        with self.assertRaises(GbqError):
            _ = export(self.query)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.export = GbqQuery2GcsParquet(
            'project',
            'bucket',
            'prefix',
            True,
            True,
            7,
            'BATCH',
            {'foo': 'bar'},
            {'baz': 42},
        )

    def test_repr(self):
        expected = ("GbqQuery2GcsParquet('project', 'bucket', 'prefix', "
                    "True, True, 7, 'BATCH', {'foo': 'bar'}, {'baz': 42})")
        self.assertEqual(expected, repr(self.export))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.export))


if __name__ == '__main__':
    unittest.main()
