import unittest
import pickle
from unittest.mock import Mock
from swak.cloud.aws import S3ObjectDiscovery, S3


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = ' /bucket / '
        self.discover = S3ObjectDiscovery(self.s3, self.bucket)
    def test_has_s3(self):
        self.assertTrue(hasattr(self.discover, 's3'))

    def test_s3(self):
        self.assertIs(self.discover.s3, self.s3)

    def test_has_bucket(self):
        self.assertTrue(hasattr(self.discover, 'bucket'))

    def test_bucket_stripped(self):
        self.assertEqual('bucket', self.discover.bucket)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.discover, 'prefix'))

    def test_prefix(self):
        self.assertEqual('', self.discover.prefix)

    def test_has_suffix(self):
        self.assertTrue(hasattr(self.discover, 'suffix'))

    def test_suffix(self):
        self.assertEqual('', self.discover.suffix)

    def test_has_subdir(self):
        self.assertTrue(hasattr(self.discover, 'subdir'))

    def test_subdir(self):
        self.assertIsInstance(self.discover.subdir, bool)
        self.assertFalse(self.discover.subdir)

    def test_has_page_size(self):
        self.assertTrue(hasattr(self.discover, 'page_size'))

    def test_page_size(self):
        self.assertIsInstance(self.discover.page_size, int)
        self.assertEqual(1000, self.discover.page_size)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.discover, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.discover.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = 'bucket'
    def test_prefix_stripped(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, ' /pre/fix/ ')
        self.assertEqual('pre/fix/', discover.prefix)

    def test_suffix_stripped(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, suffix=' .txt ')
        self.assertEqual('.txt', discover.suffix)
        discover = S3ObjectDiscovery(self.s3, self.bucket, suffix=' txt ')
        self.assertEqual('.txt', discover.suffix)

    def test_subdir(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, subdir=True)
        self.assertIsInstance(discover.subdir, bool)
        self.assertTrue(discover.subdir)

    def test_page_size(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, page_size=851)
        self.assertIsInstance(discover.page_size, int)
        self.assertEqual(851, discover.page_size)

    def test_page_size_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            _ = S3ObjectDiscovery(self.s3, self.bucket, page_size='foo')

    def test_page_size_too_small_raises(self):
        with self.assertRaises(ValueError):
            _ = S3ObjectDiscovery(self.s3, self.bucket, page_size=0)

    def test_page_size_too_large_raises(self):
        with self.assertRaises(ValueError):
            _ = S3ObjectDiscovery(self.s3, self.bucket, page_size=1001)

    def test_kwargs(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, answer=42)
        self.assertDictEqual({'answer': 42}, discover.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        empty = [
            {},
            {'Contents': []},
        ]
        self.content = [
            {
                'Contents': [
                    {'Key': 'key1.csv'},
                    {'Key': 'key1.parquet'},
                    {'Key': 'sub/key2.csv'},
                    {'Key': 'sub/key2.parquet'},
                    {'Key': 'sub/dir/key3.csv'},
                    {'Key': 'sub/dir/key3.parquet'}
                ]
            },
            {
                'Contents': [
                    {'Key': 'key4.csv'},
                    {'Key': 'key4.parquet'},
                    {'Key': 'sub/key5.csv'},
                    {'Key': 'sub/key5.parquet'},
                    {'Key': 'sub/dir/key6.csv'},
                    {'Key': 'sub/dir/key6.parquet'}
                ]
            }
        ]
        self.pages = empty + self.content
        self.paginator = Mock()
        self.client = Mock()
        self.s3 = Mock(spec=S3)
        self.paginator.paginate.return_value = self.pages
        self.client.get_paginator.return_value = self.paginator
        self.s3.return_value = self.client
        self.bucket = 'bucket'

    def test_callable(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        self.assertTrue(callable(discover))

    def test_client_created(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        _ = discover()
        self.s3.assert_called_once_with()

    def test_paginate_called_default(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        _ = discover()
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='',
            PaginationConfig={'PageSize': 1000}
        )

    def test_client_closed(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        _ = discover()
        self.client.close.assert_called_once_with()

    def test_return_value_default(self):
        expected = ['key1.csv', 'key1.parquet', 'key4.csv', 'key4.parquet']
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        actual = discover()
        self.assertListEqual(expected, actual)

    def test_paginate_called_prefix_no_slash_no_path(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, 'prefix')
        _ = discover()
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='prefix',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_prefix_slash_no_path(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, '/prefix/')
        _ = discover()
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='prefix/',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_path_no_slash_no_prefix(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, '')
        _ = discover('prefix')
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='prefix',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_path_slash_no_prefix(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, '')
        _ = discover('/prefix/')
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='prefix/',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_prefix_slash_path_no_slash(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, '/pre/')
        _ = discover('fix')
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='pre/fix',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_prefix_slash_path_end_slash(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, '/pre/')
        _ = discover('fix/')
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='pre/fix/',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_prefix_no_slash_path_no_slash(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, 'pre')
        _ = discover('fix')
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='pre/fix',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_prefix_no_slash_path_end_slash(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, 'pre')
        _ = discover('fix/')
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='pre/fix/',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_prefix_slash_path_slash(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, '/pre/')
        _ = discover('/fix')
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='pre/fix',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_prefix_slash_path_slash_end_slash(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, '/pre/')
        _ = discover('/fix/')
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='pre/fix/',
            PaginationConfig={'PageSize': 1000}
        )

    def test_paginate_called_page_size(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, page_size=123)
        _ = discover()
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='',
            PaginationConfig={'PageSize': 123}
        )

    def test_paginate_called_kwargs(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket, answer=42)
        _ = discover()
        self.paginator.paginate.assert_called_once_with(
            Bucket=self.bucket,
            Prefix='',
            PaginationConfig={'PageSize': 1000},
            answer=42
        )

    def test_client_close(self):
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        _ = discover()
        self.client.close.assert_called_once_with()

    def test_empty_dropped(self):
        expected = ['key1.csv', 'key1.parquet', 'key4.csv', 'key4.parquet']
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        actual = discover()
        self.assertListEqual(expected, actual)

    def test_suffix_kept(self):
        expected = ['key1.csv', 'key4.csv']
        discover = S3ObjectDiscovery(self.s3, self.bucket, suffix='csv')
        actual = discover()
        self.assertListEqual(expected, actual)

    def test_all_subdirs(self):
        expected = [
            'key1.csv',
            'key1.parquet',
            'sub/key2.csv',
            'sub/key2.parquet',
            'sub/dir/key3.csv',
            'sub/dir/key3.parquet',
            'key4.csv',
            'key4.parquet',
            'sub/key5.csv',
            'sub/key5.parquet',
            'sub/dir/key6.csv',
            'sub/dir/key6.parquet'
        ]
        discover = S3ObjectDiscovery(self.s3, self.bucket, subdir=True)
        actual = discover()
        self.assertListEqual(expected, actual)

    def test_first_subdir_only(self):
        expected = [
            'sub/key2.csv',
            'sub/key2.parquet',
            'sub/key5.csv',
            'sub/key5.parquet'
        ]
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        actual = discover('sub/')
        self.assertListEqual(expected, actual)

    def test_first_subdir_down(self):
        expected = [
            'sub/key2.csv',
            'sub/key2.parquet',
            'sub/dir/key3.csv',
            'sub/dir/key3.parquet',
            'sub/key5.csv',
            'sub/key5.parquet',
            'sub/dir/key6.csv',
            'sub/dir/key6.parquet'
        ]
        discover = S3ObjectDiscovery(self.s3, self.bucket, subdir=True)
        actual = discover('sub/')
        self.assertListEqual(expected, actual)

    def test_second_subdir_down(self):
        expected = [
            'sub/dir/key3.csv',
            'sub/dir/key3.parquet',
            'sub/dir/key6.csv',
            'sub/dir/key6.parquet'
        ]
        discover = S3ObjectDiscovery(self.s3, self.bucket, subdir=True)
        actual = discover('sub/dir/')
        self.assertListEqual(expected, actual)

    def test_second_subdir_only(self):
        expected = [
            'sub/dir/key3.csv',
            'sub/dir/key3.parquet',
            'sub/dir/key6.csv',
            'sub/dir/key6.parquet'
        ]
        discover = S3ObjectDiscovery(self.s3, self.bucket)
        actual = discover('sub/dir/')
        self.assertListEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        discover = S3ObjectDiscovery('s3', 'bucket')
        expected = "S3ObjectDiscovery('s3', 'bucket', '', '', False, 1000)"
        self.assertEqual(expected, repr(discover))

    def test_custom_repr(self):
        discover = S3ObjectDiscovery(
            's3',
            'bucket',
            'prefix',
            'suffix',
            True,
            123,
            answer=42
        )
        expected = ("S3ObjectDiscovery('s3', 'bucket', 'prefix', "
                    "'.suffix', True, 123, answer=42)")
        self.assertEqual(expected, repr(discover))

    def test_pickle_works(self):
        discover = S3ObjectDiscovery(S3('region'), 'bucket')
        _ = pickle.loads(pickle.dumps(discover))


if __name__ == '__main__':
    unittest.main()
