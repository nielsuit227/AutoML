import numpy as np
import pandas as pd

from amplo.automl.feature_processing.wavelet_extractor import WaveletExtractor


class TestWaveletExtractor:
    def test_get_wavelet_combinations(self, freq_data: pd.DataFrame):
        extractor = WaveletExtractor()
        extractor.peak_freqs_["a"] = np.array([1])
        combis = extractor.get_wavelet_combinations(freq_data.drop("target", axis=1))
        assert len(combis) == len(extractor.wavelets) * (len(freq_data.keys()) - 1)
        assert len(extractor.wavelets) != 0

    def test_set_peak_freqs(self):
        "Done below as part of test_extract_wavelets"
        pass

    def test_extract_wavelets(self, freq_data: pd.DataFrame):
        extractor = WaveletExtractor()
        extractor.set_peak_freqs(freq_data)

        # Test set_peak_freqs
        assert "a" in extractor.peak_freqs_
        assert max(extractor.peak_freqs_["a"]) > 0

        # extract_wavelets
        df_wt = extractor.extract_wavelets(freq_data, extractor.wavelets[0], "a")
        assert all(f"a__wav__{extractor.wavelets[0]}" in k for k in df_wt)
        assert len(df_wt.keys()) == len(extractor.peak_freqs_["a"])

    def test_extract_wavelet(self, freq_data: pd.DataFrame):
        extractor = WaveletExtractor()
        df_wt = extractor.extract_wavelet(freq_data, extractor.wavelets[0], "a", 0.1)
        assert df_wt.name == f"a__wav__{extractor.wavelets[0]}__0.1"

    def test_fit_and_transform(self, freq_data: pd.DataFrame):
        extractor = WaveletExtractor(target="target")
        df_wt = extractor.fit_transform(freq_data)

        assert len(df_wt.keys()) > 0
        assert df_wt.equals(extractor.transform(freq_data))
        assert np.dot(df_wt.iloc[:, 0].values, df_wt["target"].values) > np.dot(
            freq_data.iloc[:, 0].values, freq_data["target"].values
        )
