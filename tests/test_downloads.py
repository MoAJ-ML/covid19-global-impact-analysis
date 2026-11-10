import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import covid_analysis


class DownloadTests(unittest.TestCase):
    def test_missing_daily_measurements_are_not_forward_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                common = {'Province/State': [None], 'Country/Region': ['Example'],
                          'Lat': [0], 'Long': [0]}
                for name, values in [('confirmed', [10, None]),
                                     ('deaths', [1, 2]), ('recovered', [0, 1])]:
                    pd.DataFrame({**common, '1/1/20': [values[0]],
                                  '1/2/20': [values[1]]}).to_csv(
                        f'time_series_covid19_{name}_global.csv', index=False)
                pd.DataFrame({
                    'location': ['Example', 'Example'],
                    'date': ['2020-01-01', '2020-01-02'],
                    'population': [1000, None],
                    'new_cases': [10, None],
                    'new_deaths': [1, None],
                    'new_vaccinations': [5, None],
                    'stringency_index': [40, None],
                }).to_csv('owid-covid-data.csv', index=False)
                result = covid_analysis.load_and_merge_data().sort_values('date')
                latest = result.iloc[-1]
                self.assertEqual(latest['population'], 1000)
                for column in ['confirmed', 'confirmed_per100k', 'new_cases',
                               'new_cases_per100k', 'new_deaths',
                               'new_vaccinations', 'stringency_index']:
                    self.assertTrue(pd.isna(latest[column]), column)
            finally:
                os.chdir(previous)

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
