#  Copyright (c) 2022 by Amplo.

import numpy as np
import pandas as pd

from amplo.automl import Sequencer


class TestSequence:
    def test_init(self):
        assert Sequencer(), "Class initiation failed"

    def test_numpy_none(self):
        # Parameters
        features = 5
        length = 500
        back = np.random.randint(1, 50)
        x, y = np.random.randint(0, 50, (length, features)), np.random.randint(
            0, 50, length
        )

        # Iterate scenarios
        for forward in [[1], [5]]:
            # Sequence
            sequence = Sequencer(back=back, forward=forward)
            seq_x, seq_y = sequence.convert(x, y, flat=False)

            # Test
            assert len(seq_x.shape) == 3, "seq_x is no tensor"
            assert seq_x.shape[1] == back, "seq_x step dimension incorrect"
            assert seq_x.shape[2] == features, "seq_x feature dimension incorrect"
            assert (
                seq_x.shape[0] == seq_y.shape[0]
            ), "seq_x and seq_y have inconsistent samples"
            for i in range(seq_x.shape[0]):
                assert np.allclose(
                    seq_x[i], x[i : i + back]
                ), "seq_x samples are not correct"
                assert np.allclose(
                    seq_y[i], y[i + back + forward[0] - 1]
                ), "seq_y samples are not correct"

    def test_numpy_diff(self):
        # Parameters
        features = 5
        length = 500
        back = np.random.randint(2, 50)
        x = np.outer(np.linspace(1, length, length), np.ones(features))
        y = np.linspace(0, length - 1, length)

        # Iterate Scenarios
        for forward in [[1], [5]]:
            # Sequence
            sequence = Sequencer(back=back, forward=forward, diff="diff")
            seq_x, seq_y = sequence.convert(x, y, flat=False)

            # Tests
            assert len(seq_x.shape) == 3, "seq_x is not a tensor"
            assert seq_x.shape[1] == back - 1, "seq_x step dimensions incorrect"
            assert seq_x.shape[2] == features, "seq_x feature dimensions incorrect"
            assert (
                seq_x.shape[0] == seq_y.shape[0]
            ), "seq_x and seq_y inconsistent samples"
            assert np.allclose(seq_x, np.ones_like(seq_x)), "samples seq_x incorrect"
            assert np.allclose(
                seq_y, np.ones_like(seq_y) * forward[0]
            ), "seq_y samples incorrect"

    def test_numpy_log(self):
        # Parameters
        features = 5
        length = 500
        back = np.random.randint(2, 50)
        x = np.outer(np.linspace(1, length, length), np.ones(features))
        y = np.linspace(1, length, length)

        # Iterate Scenarios
        for forward in [[1], [5]]:
            # Sequence
            sequence = Sequencer(back=back, forward=forward, diff="log_diff")
            seq_x, seq_y = sequence.convert(x, y, flat=False)

            # Tests
            assert len(seq_x.shape) == 3, "seq_x is not a tensor"
            assert seq_x.shape[1] == back - 1, "seq_x step dimensions incorrect"
            assert seq_x.shape[2] == features, "seq_x feature dimensions incorrect"
            assert (
                seq_x.shape[0] == seq_y.shape[0]
            ), "seq_x and seq_y inconsistent samples"
            for i in range(seq_x.shape[0]):
                assert np.allclose(
                    seq_x[i], np.log(x[1 + i : i + back]) - np.log(x[i : i + back - 1])
                ), "samples seq_x incorrect"
                assert np.allclose(
                    seq_y[i],
                    np.log(y[i + back + forward[0] - 1]) - np.log(y[i + back - 1]),
                ), "seq_y samples incorrect"

    def test_numpy_multi_out(self):
        # Parameters
        features = 5
        length = 500
        forward = [np.random.randint(2, 10)]
        forward += [forward[0] + np.random.randint(1, 10)]
        back = np.random.randint(2, 50)
        x = np.outer(np.linspace(1, length, length), np.ones(features))
        y = np.linspace(1, length, length)

        # Without differencing
        sequence = Sequencer(back=back, forward=forward, diff="none")
        seq_x, seq_y = sequence.convert(x, y, flat=False)
        # Test
        assert (
            seq_x.shape[0] == seq_y.shape[0]
        ), "seq_x and seq_y have inconsistent samples"
        assert seq_y.shape[1] == 2, "seq_y has incorrect steps"
        for i in range(seq_x.shape[0]):
            assert np.allclose(
                seq_y[i].tolist(),
                [y[i + back + forward[0] - 1], y[i + back + forward[1] - 1]],
            ), "seq_y samples are not correct"

        # With differencing
        sequence = Sequencer(back=back, forward=forward, diff="diff")
        seq_x, seq_y = sequence.convert(x, y, flat=False)
        revert = sequence.revert(seq_y, y[back - 1 : back - 1 + max(forward)])
        # Tests
        assert (
            seq_x.shape[0] == seq_y.shape[0]
        ), "seq_x and seq_y have inconsistent samples"
        assert seq_y.shape[1] == 2, "seq_y has incorrect steps"
        assert np.allclose(
            revert[0, : forward[0] - forward[-1]], y[back - 1 : forward[0] - forward[1]]
        )
        assert np.allclose(revert[1], y[back - 1 :])

    def test_pandas_tensor(self):
        # Parameters
        features = 5
        length = 500
        back = np.random.randint(1, 50)
        x, y = np.random.randint(0, 50, (length, features)), np.random.randint(
            0, 50, length
        )
        x, y = pd.DataFrame(x), pd.Series(y)

        # Iterate scenarios
        for forward in [[1], [5]]:
            # Sequence
            sequence = Sequencer(back=back, forward=forward)
            seq_x, seq_y = sequence.convert(x, y, flat=False)

            # Tests
            assert len(seq_x.shape) == 3, "seq_x is no tensor"
            assert seq_x.shape[1] == back, "seq_x step dimension incorrect"
            assert seq_x.shape[2] == features, "seq_x feature dimension incorrect"
            assert (
                seq_x.shape[0] == seq_y.shape[0]
            ), "seq_x and seq_y have inconsistent samples"
            for i in range(seq_x.shape[0]):
                assert np.allclose(seq_x[i], x[i : i + back]), "seq_x samples incorrect"
                assert np.allclose(
                    seq_y[i], y[i + back + forward[0] - 1]
                ), "seq_y samples incorrect"

    def test_reconstruction(self):
        # Parameters
        length = 100
        features = 5
        back = np.random.randint(2, 10)
        forward = [np.random.randint(2, 10)]
        # x, y = np.random.randint(0, 50, (length, features)), np.random.randint(0, 50, length)  # noqa: E501
        x, y = np.outer(np.linspace(1, length, length), np.ones(features)), np.linspace(
            1, length, length
        )

        # Iterate scenarios
        for diff in ["diff", "log_diff"]:
            # Sequence
            seq = Sequencer(back=back, forward=forward, diff=diff)
            seq_x, seq_y = seq.convert(x, y, flat=False)

            # Tests
            assert len(seq_x.shape) == 3, "seq_x is not a tensor"
            assert seq_x.shape[1] == back - 1, "seq_x step dimensions incorrect"
            assert seq_x.shape[2] == features, "seq_x feature dimensions incorrect"
            assert (
                seq_x.shape[0] == seq_y.shape[0]
            ), "seq_x and seq_y inconsistent samples"
            revert = seq.revert(seq_y, y[back - 1 : back - 1 + forward[0]])
            assert np.allclose(revert, y[back - 1 :]), "reverted seq_y incorrect"
