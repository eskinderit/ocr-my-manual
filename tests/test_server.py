import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vespa_retrieval.server import local_server


class ServerLifecycleTests(unittest.TestCase):
    def test_owned_server_stops_on_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root/'asset'; asset.write_bytes(b'fixture')
            for fail in (False, True):
                process = Mock(); process.poll.return_value = None
                with patch('vespa_retrieval.server.server_ready', side_effect=[False, True]), patch('vespa_retrieval.server.subprocess.Popen', return_value=process) as spawn:
                    try:
                        with local_server(asset, asset, asset, root/'server.log') as endpoint:
                            self.assertEqual(endpoint,'http://127.0.0.1:8003/v1/chat/completions')
                            if fail:
                                raise ValueError('failed request')
                    except ValueError:
                        self.assertTrue(fail)
                process.terminate.assert_called_once()
                process.wait.assert_called_once_with(timeout=15)
                self.assertTrue(spawn.call_args.kwargs['env']['LD_LIBRARY_PATH'].startswith(str(root)))

    def test_existing_server_not_owned_or_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); asset=root/'asset'; asset.write_bytes(b'fixture')
            with patch('vespa_retrieval.server.server_ready',return_value=True), patch('vespa_retrieval.server.subprocess.Popen') as spawn:
                with self.assertRaises(RuntimeError):
                    with local_server(asset,asset,asset,root/'log'):
                        pass
                spawn.assert_not_called()

    def test_missing_assets_fail_before_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'missing'
            with patch('vespa_retrieval.server.subprocess.Popen') as spawn:
                with self.assertRaises(FileNotFoundError):
                    with local_server(path,path,path,Path(tmp)/'log'):
                        pass
                spawn.assert_not_called()