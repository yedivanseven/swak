import unittest
import pickle
from unittest.mock import patch, Mock
from pathlib import Path
from tempfile import TemporaryDirectory
from botocore.exceptions import ClientError
from swak.cloud.aws.exceptions import S3Error
from swak.cloud.aws import S3File2LocalFile, S3


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = '/ bucket/ '
        self.download = S3File2LocalFile(self.s3, self.bucket)

    def test_has_s3(self):
        self.assertTrue(hasattr(self.download, 's3'))

    def test_s3(self):
        self.assertIs(self.s3, self.download.s3)

    def test_has_bucket(self):
        self.assertTrue(hasattr(self.download, 'bucket'))

    def test_bucket_stripped(self):
        self.assertEqual('bucket', self.download.bucket)

    def test_has_prefix(self):
        self.assertTrue(hasattr(self.download, 'prefix'))

    def test_prefix(self):
        self.assertEqual('', self.download.prefix)

    def test_has_base_dir(self):
        self.assertTrue(hasattr(self.download, 'base_dir'))

    def test_has_overwrite(self):
        self.assertTrue(hasattr(self.download, 'overwrite'))

    def test_overwrite(self):
        self.assertIsInstance(self.download.overwrite, bool)
        self.assertFalse(self.download.overwrite)

    def test_has_skip(self):
        self.assertTrue(hasattr(self.download, 'skip'))

    def test_skip(self):
        self.assertIsInstance(self.download.skip, bool)
        self.assertFalse(self.download.skip)

    def test_has_extra_kws(self):
        self.assertTrue(hasattr(self.download, 'extra_kws'))

    def test_extra_kws(self):
        self.assertDictEqual({}, self.download.extra_kws)

    def test_has_download_kws(self):
        self.assertTrue(hasattr(self.download, 'download_kws'))

    def test_download_kws(self):
        self.assertDictEqual({}, self.download.download_kws)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = ' / bucket/ '
        self.prefix = ' /prefix '
        self.base_dir = ' /tmp'
        self.overwrite = True
        self.skip = True
        self.extra_kws = {'one': 1}
        self.download_kws = {'two': 2}
        self.download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            self.overwrite,
            self.skip,
            self.extra_kws,
            self.download_kws
        )
    def test_prefix_stripped(self):
        self.assertEqual(
            self.prefix.strip().lstrip('/'), self.download.prefix
        )

    def test_base_dir(self):
        self.assertEqual(self.base_dir.strip(), self.download.base_dir)

    def test_overwrite(self):
        self.assertTrue(self.download.overwrite)

    def test_skip(self):
        self.assertTrue(self.download.skip)

    def test_extra_kws(self):
        self.assertDictEqual(self.extra_kws, self.download.extra_kws)

    def test_download_kws(self):
        self.assertDictEqual(self.download_kws, self.download.download_kws)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.s3 = Mock(spec=S3)
        self.client = Mock()
        self.client.download_fileobj = Mock()
        self.s3.return_value = self.client
        self.bucket = 'bucket'
        self.prefix = 'prefix'
        self.base_dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_callable(self):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        self.assertTrue(callable(download))

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_overwrite_raises(self, _):
        with (Path(self.base_dir) / 'test.txt').open('wb') as file:
            file.write(b'I was first!')
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        with self.assertRaises(S3Error):
            _ = download('test.txt')

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_local_dir_init_created(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir + '/dir'
        )
        _ = download('test.txt')
        self.assertTrue((Path(self.base_dir) / 'dir' / 'test.txt').exists())

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_local_dir_call_created(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        _ = download('dir/test.txt')
        self.assertTrue((Path(self.base_dir) / 'dir' / 'test.txt').exists())

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_local_dir_stripped(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        _ = download('/ dir/test.txt')
        self.assertTrue((Path(self.base_dir) / 'dir' / 'test.txt').exists())

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_client_created(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        _ = download('test.txt')
        self.s3.assert_called_once_with()

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_download_called(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            extra_kws={'one': 1}
        )
        _ = download('test.txt')
        self.client.download_fileobj.assert_called_once()
        kwargs = self.client.download_fileobj.call_args[1]
        self.assertEqual(self.bucket, kwargs['Bucket'])
        self.assertDictEqual({'one': 1}, kwargs['ExtraArgs'])

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_download_called_prefix_no_slash_no_path(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            'test.txt',
            self.base_dir,

        )
        _ = download()
        kwargs = self.client.download_fileobj.call_args[1]
        remote = kwargs['Key']
        self.assertEqual('test.txt', remote)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_download_called_prefix_slash_no_path(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            '/prefix/',
            self.base_dir,

        )
        _ = download()
        kwargs = self.client.download_fileobj.call_args[1]
        remote = kwargs['Key']
        self.assertEqual('prefix', remote)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_download_called_path_no_slash_no_prefix(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            '',
            self.base_dir,
        )
        _ = download('test.txt')
        kwargs = self.client.download_fileobj.call_args[1]
        remote = kwargs['Key']
        self.assertEqual('test.txt', remote)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_download_called_path_slash_no_prefix(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            '',
            self.base_dir,
        )
        _ = download('/prefix/')
        kwargs = self.client.download_fileobj.call_args[1]
        remote = kwargs['Key']
        self.assertEqual('prefix', remote)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_download_called_prefix_slash_path_no_slash(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            '/pre/',
            self.base_dir,
        )
        _ = download('fix')
        kwargs = self.client.download_fileobj.call_args[1]
        remote = kwargs['Key']
        self.assertEqual('pre/fix', remote)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_download_called_prefix_no_slash_path_slash(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            'pre',
            self.base_dir,
        )
        _ = download('/fix/')
        kwargs = self.client.download_fileobj.call_args[1]
        remote = kwargs['Key']
        self.assertEqual('pre/fix', remote)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_download_called_prefix_slash_path_slash(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            '/pre/',
            self.base_dir,
        )
        _ = download('/fix/')
        kwargs = self.client.download_fileobj.call_args[1]
        remote = kwargs['Key']
        self.assertEqual('pre/fix', remote)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_path_stripped(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            extra_kws={'one': 1}
        )
        _ = download(' / test.txt/ ')
        kwargs = self.client.download_fileobj.call_args[1]
        self.assertEqual(self.prefix + '/test.txt', kwargs['Key'])

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_transfer_config_called(self, config):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            download_kws={'two': 2}
        )
        _ = download('test.txt')
        config.assert_called_once_with(two=2)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_client_closed(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        _ = download('test.txt')
        self.client.close.assert_called_once_with()

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_return_value(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        actual = download('test.txt')
        self.assertEqual(self.base_dir + '/test.txt', actual)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_local_file_created(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        _ = download('test.txt')
        path = Path(self.base_dir + '/test.txt')
        self.assertTrue(path.exists())

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_local_file_not_created_when_error(self, _):
        error = ClientError({}, 'download_fileobj')
        self.client.download_fileobj.side_effect = error
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        with self.assertRaises(ClientError):
            _ = download('test.txt')
        path = Path(self.base_dir) / 'test.txt'
        self.assertFalse(path.exists())

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_local_file_not_removed_when_error(self, _):
        path = Path(self.base_dir) / 'test.txt'
        with path.open('wb') as file:
            file.write(b'I was first!')
        error = ClientError({}, 'download_fileobj')
        self.client.download_fileobj.side_effect = error
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            overwrite=True,
            skip=False
        )
        with self.assertRaises(ClientError):
            _ = download('test.txt')
        self.assertTrue(path.exists())
        with path.open('rb') as file:
            content = file.read()
        self.assertEqual(b'I was first!', content)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_local_file_content(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        _ = download('test.txt')
        path = Path(self.base_dir + '/test.txt')
        with path.open('rb') as file:
            actual = file.read()
        self.assertEqual(b'', actual)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_skip_skips_existing(self, _):
        with (Path(self.base_dir) / 'test.txt').open('wb') as file:
            file.write(b'I was first!')
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            skip=True
        )
        _ = download('test.txt')
        self.s3.assert_not_called()

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_skip_does_not_skip_non_existing(self, _):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            skip=True
        )
        _ = download('test.txt')
        self.client.download_fileobj.assert_called_once()

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_no_overwrite_when_skip(self, _):
        path = Path(self.base_dir) / 'test.txt'
        with path.open('wb') as file:
            file.write(b'I was first!')
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            skip=True,
            overwrite=True
        )
        _ = download('test.txt')
        self.s3.assert_not_called()
        with path.open('rb') as file:
            actual = file.read()
        self.assertEqual(b'I was first!', actual)

    @patch('swak.cloud.aws.s32local.TransferConfig')
    def test_overwrite_when_not_skipped(self, _):
        path = Path(self.base_dir) / 'test.txt'
        with path.open('wb') as file:
            file.write(b'I was first!')
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            overwrite=True,
        )
        _ = download('test.txt')
        self.client.download_fileobj.assert_called_once()
        with path.open('rb') as file:
            actual = file.read()
        self.assertEqual(b'', actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.s3 = 's3'
        self.bucket = 'bucket'
        self.prefix = 'prefix'
        self.base_dir = '/base_dir'

    def test_default_representation(self):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir
        )
        expected = ("S3File2LocalFile('s3', 'bucket', 'prefix',"
                    " '/base_dir', False, False, {}, {})")
        self.assertEqual(expected, repr(download))

    def test_custom_representation(self):
        download = S3File2LocalFile(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            overwrite=True,
            skip=True,
            extra_kws={'one': 1},
            download_kws={'two': 2}
        )
        expected = ("S3File2LocalFile('s3', 'bucket', 'prefix',"
                    " '/base_dir', True, True, {'one': 1}, {'two': 2})")
        self.assertEqual(expected, repr(download))

    def test_pickle_works(self):
        download = S3File2LocalFile(
            S3('region'),
            self.bucket,
            self.prefix,
            self.base_dir,
            overwrite=True,
            skip=True,
            extra_kws={'one': 1},
            download_kws={'two': 2},
        )
        _ = pickle.loads(pickle.dumps(download))


if __name__ == '__main__':
    unittest.main()
