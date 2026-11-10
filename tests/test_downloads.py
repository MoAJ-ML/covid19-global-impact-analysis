import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import covid_analysis


class DownloadTests(unittest.TestCase):
    def test_records_hash_and_fails_on_download_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            original = covid_analysis.DATASETS
            os.chdir(tmp)
            covid_analysis.DATASETS = [('Synthetic', 'https://example.invalid/data.csv', 'data.csv')]
            try:
                response = Mock(content=b'a,b\n1,2\n')
                response.raise_for_status.return_value = None
                with patch.object(covid_analysis.requests, 'get', return_value=response):
                    manifest = covid_analysis.download_data()
                self.assertEqual(len(manifest[0]['sha256']), 64)
                self.assertEqual(json.loads(Path('data_manifest.json').read_text())[0]['name'], 'Synthetic')
                Path('data.csv').unlink()
                with patch.object(covid_analysis.requests, 'get', side_effect=RuntimeError('offline')):
                    with self.assertRaisesRegex(RuntimeError, 'offline'):
                        covid_analysis.download_data()
                self.assertFalse(Path('data.csv').exists())
            finally:
                covid_analysis.DATASETS = original
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
