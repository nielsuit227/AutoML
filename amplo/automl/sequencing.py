#  Copyright (c) 2022 by Amplo.
from __future__ import annotations

from warnings import warn

import numpy as np
import pandas as pd

from amplo.base import LoggingMixin


class Sequencer(LoggingMixin):
    # todo implement fractional differencing

    def __init__(
        self,
        target: str,
        back: int | list[int] = 1,
        forward: int | list[int] = 1,
        shift=0,
        diff="none",
        verbose: int = 1,
    ):
        """Sequences and differentiates data.

        Parameters
        ----------
        target : str
            Target column
        back : list of int or int
            Input indices (see also Notes).
        forward : list of int or int
            Output indices (see also Notes).
        shift : int
            Indicates the shift between input and output.
        diff : str
            Depicts the differencing algorithm. Options: `none`, `diff` and `log_diff`.

        Notes
        -----
        For the ``back`` and ``forward`` parameter, behavior changes depending on its
        type:
            - If it's an **integer** dtype, indices include that many samples backward
            or forward, respectively.
            - If it's an **iterable** dtype, indices include all integers within the
            iterable.

        The indices of ``back`` and ``forward`` start from ``0``. Therefore, if the
        output is included in the input, having ``forward=4`` will result in predicting
        the output for ``[t, t+1, t+2, t+3]`` and having ``forward=[4]`` will result in
        making a ``t+4`` prediction. The same rules apply to the ``forward`` parameter.
        """
        super().__init__(verbose=verbose)
        # Tests
        if diff not in [
            "none",
            "diff",
            "log_diff",
        ]:
            raise ValueError("Diff needs to be none, diff or log_diff.")
        if isinstance(back, int):
            if back <= 0:
                raise ValueError("'back' arg needs to be strictly positive.")
        else:
            if not all([x >= 0 for x in back]):
                raise ValueError("All integers in 'back' need to be positive.")
            if not all([x > 0 for x in np.diff(back)]):
                raise ValueError(
                    "All integers in 'back' need to be monotonically increasing."
                )
        if isinstance(forward, int) and forward < 0:
            raise ValueError("'forward' arg needs to be positive")
        elif isinstance(forward, list):
            if not all([x >= 0 for x in forward]):
                raise ValueError("All integers in 'forward' need to be positive.")
            if not all([x > 0 for x in np.diff(forward)]):
                raise ValueError(
                    "All integers in 'forward' need to be monotonically increasing."
                )
        if diff != "none" and isinstance(back, int) and back <= 1:
            raise ValueError("With differencing, back needs to be at least 2.")
        if diff != "none" and isinstance(forward, int) and forward <= 1:
            raise ValueError("With differencing, forward needs to be at least 2.")

        # Note static args
        self.target = target
        self.shift = shift
        self.diff = diff
        self.samples_ = 0
        self.input_constant_ = 0
        self.output_constant_ = 0

        # Parse args
        if isinstance(back, int):
            back_arr = np.linspace(0, back - 1, back).astype("int")
        elif isinstance(back, list):
            back_arr = np.array(back)
        if isinstance(forward, int):
            forward_arr = np.linspace(0, forward - 1, forward).astype("int")
        elif isinstance(forward, list):
            forward_arr = np.array(forward)

        # Index Vectors
        assert len(back_arr) > 0
        assert len(forward_arr) > 0
        self.input_indices_ = back_arr
        self.output_indices_ = forward_arr

        # Differencing vectors
        if diff != "none":
            # In case first is 0, we won't difference with -1, therefore, we add & skip
            # the first
            if self.input_indices_[0] == 0:
                self.input_diff_indices_ = self.input_indices_[:-1]
                self.input_indices_ = self.input_indices_[1:]
            else:
                # However, if first is nonzero, we can keep all and roll them, changing
                # first one to 0
                self.input_diff_indices_ = np.roll(self.input_indices_, 1)
                self.input_diff_indices_[0] = 0

            # Same for output
            if self.output_indices_[0] == 0:
                self.output_diff_indices_ = self.output_indices_[:-1]
                self.output_indices_ = self.output_indices_[1:]
            else:
                self.output_diff_indices_ = np.roll(self.output_indices_, 1)
                self.output_diff_indices_[0] = 0

        # Number of sequence steps
        self.n_input_steps_ = len(self.input_indices_)
        self.n_output_steps_ = len(self.output_indices_)

        # Maximum steps
        self.max_input_step_ = max(self.input_indices_)
        self.max_output_step_ = max(self.output_indices_)

    def fit_transform(self, data: pd.DataFrame, flat=True) -> pd.DataFrame:
        """
        Sequences input / outputs dataframe / numpy array.

        parameters
        ----------
        data : pd.DataFrame
        flat : bool, default=True
            Whether to return a 2d matrix or 3d
        """
        # Split data
        if self.target not in data:
            raise ValueError(f"Target column {self.target} missing.")
        y = data[self.target]
        x = data.drop(self.target, axis=1)

        # Initials
        input_keys = x.keys()
        output_keys = y.keys()

        # If flat return
        if flat:
            # No Differencing
            if self.diff == "none":
                # Input
                for lag in self.input_indices_:
                    keys = [key + "_" + str(lag) for key in input_keys]
                    x[keys] = x[input_keys].shift(lag)

                # Output
                for shift in self.output_indices_:
                    keys = [key + "_" + str(shift) for key in output_keys]
                    y[keys] = y[output_keys].shift(-shift)

            # With differencing
            elif self.diff[-4:] == "diff":
                # Input
                for lag in self.input_indices_:
                    # Shifted
                    keys = [key + "_" + str(lag) for key in input_keys]
                    x[keys] = x[input_keys].shift(lag)

                    # differentiated
                    d_keys = [key + "_d_" + str(lag) for key in input_keys]
                    x[d_keys] = x[input_keys].shift(lag) - x[input_keys]

                # Output
                for shift in self.output_indices_:
                    # Only differentiated
                    keys = [key + "_" + str(shift) for key in output_keys]
                    y[keys] = y[output_keys].shift(shift) - y[output_keys]
            else:
                raise NotImplementedError("Unknown differencing algorithm.")

            # Drop _0 (same as original)
            x = x.drop([key for key in x.keys() if "_0" in key], axis=1)
            y = y.drop([key for key in y.keys() if "_0" in key], axis=1)

            # Return (first lags are NaN, last shifts are NaN
            x = x.iloc[lag : -shift if shift > 0 else None]  # type: ignore
            y = y.iloc[lag : -shift if shift > 0 else None]  # type: ignore
            x[self.target] = y
            return x
        else:
            raise NotImplemented(
                "Technically, we could use _convert_numpy, ",
                "but returning 3d is not supported by the pipeline.",
            )

    def _convert_numpy(
        self, x: np.ndarray, y: np.ndarray, flat=True
    ) -> tuple[np.ndarray, np.ndarray]:
        # Initializations
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        # Samples, interval minus sequence length (maxIn+maxOutPlus+shift)
        self.samples_ += (
            len(x) - self.max_input_step_ - self.max_output_step_ - self.shift
        )
        features = len(x[0])
        input_sequence = np.zeros((self.samples_, self.n_input_steps_, features))
        output_sequence = np.zeros((self.samples_, self.n_output_steps_))

        # Sequence
        if self.diff == "none":
            for i in range(self.samples_):
                input_sequence[i] = x[i + self.input_indices_]
                output_sequence[i] = y[
                    i + self.max_input_step_ + self.shift + self.output_indices_
                ].reshape((-1))

        elif self.diff[-4:] == "diff":

            # Take log for log_diff
            if self.diff == "log_diff":
                if np.min(x) < 1e-3:
                    self.input_constant_ = abs(np.min(x) * 1.001) + 1e-3
                    warn(
                        f"Small or negative input values found, adding a constant "
                        f"{self.input_constant_:.2e} to input"
                    )
                if np.min(y) < 1e-3:
                    self.output_constant_ = abs(np.min(y) * 1.001) + 1e-3
                    warn(
                        f"Small or negative output values found, adding a constant "
                        f"{self.output_constant_:.2e} to output"
                    )
                x = np.log(x + self.input_constant_)
                y = np.log(y + self.output_constant_)

            # Finally create the difference vector
            for i in range(self.samples_):
                input_sequence[i] = (
                    x[i + self.input_indices_] - x[i + self.input_diff_indices_]
                )
                output_sequence[i] = (
                    y[i + self.max_input_step_ + self.shift + self.output_indices_]
                    - y[
                        i
                        + self.max_input_step_
                        + self.shift
                        + self.output_diff_indices_
                    ]
                ).reshape((-1))

        # Return
        if flat:
            return (
                input_sequence.reshape((self.samples_, self.n_input_steps_ * features)),
                output_sequence,
            )
        else:
            return input_sequence, output_sequence

    def revert(self, seq_y, y_start):
        """
        Reverts the sequenced vector back.

        Useful for sequenced predictions in production.

        :param seq_y: The sequenced / predicted signal,
        :param y_start: Starting vector, always necessary when differentiated.
        :return: y: normal predicted signal, de-sequenced.
        """
        if not len(seq_y.shape) == 2 and seq_y.shape[1] == self.n_output_steps_:
            raise ValueError("revert() only suitable for output.")

        # Initiate
        y = np.zeros((self.n_output_steps_, len(seq_y) + len(y_start)))

        # de-sequence if diff
        if self.diff == "diff":
            for i in range(self.n_output_steps_):
                y[i, : len(y_start)] = y_start
                for j in range(len(seq_y)):
                    y[i, j + self.output_indices_[i]] = (
                        y[0, j + self.output_diff_indices_[i]] + seq_y[j, i]
                    )
            return y

        # de-sequence if log_diff (take log of start, add sequence log, then take exp
        # (as x=exp(log(x)))
        elif self.diff == "log_diff":
            for i in range(self.n_output_steps_):
                y[i, : len(y_start)] = np.log(y_start + self.output_constant_)
                for j in range(len(seq_y)):
                    y[i, j + self.output_indices_] = (
                        y[0, j + self.output_diff_indices_] + seq_y[j, i]
                    )
                return np.exp(y) - self.output_constant_

        else:
            if self.diff == "none":
                raise ValueError(
                    "With differencing set to none, reverting is not necessary."
                )
            else:
                raise ValueError("Differencing method not implemented.")
