import numpy as np
import pandas as pd

from amplo.automl.feature_processing.wavelet_extractor import WaveletExtractor
from amplo.utils.data import pandas_to_polars


class TestWaveletExtractor:
    def test_get_wavelet_combinations(self, freq_data: pd.DataFrame):
        data, _ = pandas_to_polars(freq_data, include_index=False)
        extractor = WaveletExtractor()
        extractor.peak_freqs_["a"] = np.array([1])
        combis = extractor.get_wavelet_combinations(data.drop("target"))
        assert len(combis) == len(extractor.wavelets) * (len(freq_data.keys()) - 1)
        assert len(extractor.wavelets) != 0

    def test_set_peak_freqs(self):
        "Done below as part of test_extract_wavelets"
        pass

    def test_extract_wavelets(self, freq_data: pd.DataFrame):
        data, _ = pandas_to_polars(freq_data, include_index=False)
        extractor = WaveletExtractor()
        extractor.set_peak_freqs(data)

        # Test set_peak_freqs
        assert "a" in extractor.peak_freqs_
        assert max(extractor.peak_freqs_["a"]) > 0

        # extract_wavelets
        df_wt = extractor.extract_wavelets(data, extractor.wavelets[0], "a")
        assert all(f"a__wav__{extractor.wavelets[0]}" in k for k in df_wt.columns)
        assert len(df_wt.columns) == len(extractor.peak_freqs_["a"])

    def test_extract_wavelet(self, freq_data: pd.DataFrame):
        data, _ = pandas_to_polars(freq_data, include_index=False)
        extractor = WaveletExtractor()
        df_wt = extractor.extract_wavelet(data, extractor.wavelets[0], "a", 0.1)
        assert df_wt.name == f"a__wav__{extractor.wavelets[0]}__0.1"

    def test_fit_and_transform(self, freq_data: pd.DataFrame):
        data, index_renaming = pandas_to_polars(freq_data)
        index_cols = list(index_renaming)
        extractor = WaveletExtractor(target="target")
        df_wt = extractor.fit_transform(data, index_cols)

        assert len(df_wt.columns) > 0
        assert np.allclose(df_wt, extractor.transform(data, index_cols)[df_wt.columns])  # type: ignore[arg-type]
        assert np.dot(
            df_wt.to_pandas().iloc[:, 2], df_wt.to_pandas()["target"]
        ) > np.dot(freq_data.iloc[:, 0], freq_data["target"])
