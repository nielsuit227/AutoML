#  Copyright (c) 2022 by Amplo.

import warnings

import numpy as np
import pandas as pd


class Sequencer:
    # todo implement fractional differencing

    def __init__(self, back=1, forward=1, shift=0, diff="none"):
        """Sequences and differentiates data.

        Parameters
        ----------
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
        # Tests
        assert diff in [
            "none",
            "diff",
            "log_diff",
        ], "Diff needs to be none, diff or log_diff."
        if isinstance(back, int):
            assert back > 0, "'back' arg needs to be strictly positive."
        else:
            assert all(
                [x >= 0 for x in back]
            ), "All integers in 'back' need to be positive."
            assert all(
                [x > 0 for x in np.diff(back)]
            ), "All integers in 'back' need to be monotonically increasing."
        if isinstance(forward, int):
            assert forward >= 0, "'forward' arg needs to be positive"
        else:
            assert all(
                [x >= 0 for x in forward]
            ), "All integers in 'forward' need to be positive."
            assert all(
                [x > 0 for x in np.diff(forward)]
            ), "All integers in 'forward' need to be monotonically increasing."
        if diff != "none" and isinstance(back, int):
            assert back > 1, "With differencing, back needs to be at least 2."
        if diff != "none" and isinstance(forward, int):
            assert forward > 1, "With differencing, forward needs to be at least 2."

        # Note static args
        self.shift = shift
        self.diff = diff
        self.samples = 0  # Add 1 as in/out-put start both at 0
        self.inputConstant = 0
        self.outputConstant = 0

        # Parse args
        if type(back) == int:
            back = np.linspace(0, back - 1, back).astype("int")
        elif type(back) == list:
            back = np.array(back)
        if type(forward) == int:
            forward = np.linspace(0, forward - 1, forward).astype("int")
        elif type(forward) == list:
            forward = np.array(forward)

        # Index Vectors
        self.inputIndices = back
        self.inputDiffIndices = None
        self.outputIndices = forward
        self.outputDiffIndices = None

        # Differencing vectors
        if diff != "none":
            # In case first is 0, we won't difference with -1, therefore, we add & skip
            # the first
            if self.inputIndices[0] == 0:
                self.inputDiffIndices = self.inputIndices[:-1]
                self.inputIndices = self.inputIndices[1:]
            else:
                # However, if first is nonzero, we can keep all and roll them, changing
                # first one to 0
                self.inputDiffIndices = np.roll(self.inputIndices, 1)
                self.inputDiffIndices[0] = 0

            # Same for output
            if self.outputIndices[0] == 0:
                self.outputDiffIndices = self.outputIndices[:-1]
                self.outputIndices = self.outputIndices[1:]
            else:
                self.outputDiffIndices = np.roll(self.outputIndices, 1)
                self.outputDiffIndices[0] = 0

        # Number of sequence steps
        self.nInputSteps = len(self.inputIndices)
        self.nOutputSteps = len(self.outputIndices)

        # Maximum steps
        self.maxInputStep = max(self.inputIndices)
        self.maxOutputStep = max(self.outputIndices)

    def convert(self, x, y, flat=True):
        """
        Sequences input / outputs dataframe / numpy array.

        :param x: Input
        :param y: Output
        :param flat: Boolean. If true, a flat matrix is returned. If false, a 3D tensor
            is returned (numpy).
        :return: seq_x, seq_y
        """
        if isinstance(x, (pd.Series, pd.DataFrame)):
            assert isinstance(
                y, (pd.Series, pd.DataFrame)
            ), "Input and Output need to be the same data type."
            return self._convert_pandas(x, y, flat=flat)
        elif isinstance(x, np.ndarray):
            assert isinstance(
                y, np.ndarray
            ), "Input and Output need to be the same data type."
            return self._convert_numpy(x, y, flat=flat)
        else:
            TypeError(
                "Input & Output need to be same datatype, either Numpy or Pandas."
            )

    def _convert_numpy(self, x, y, flat=True):
        # Initializations
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        # Samples, interval minus sequence length (maxIn+maxOutPlus+shift)
        self.samples += len(x) - self.maxInputStep - self.maxOutputStep - self.shift
        features = len(x[0])
        input_sequence = np.zeros((self.samples, self.nInputSteps, features))
        output_sequence = np.zeros((self.samples, self.nOutputSteps))

        # Sequence
        if self.diff == "none":
            for i in range(self.samples):
                input_sequence[i] = x[i + self.inputIndices]
                output_sequence[i] = y[
                    i + self.maxInputStep + self.shift + self.outputIndices
                ].reshape((-1))

        elif self.diff[-4:] == "diff":

            # Take log for log_diff
            if self.diff == "log_diff":
                if np.min(x) < 1e-3:
                    self.inputConstant = abs(np.min(x) * 1.001) + 1e-3
                    warnings.warn(
                        f"Small or negative input values found, adding a constant "
                        f"{self.inputConstant:.2e} to input"
                    )
                if np.min(y) < 1e-3:
                    self.outputConstant = abs(np.min(y) * 1.001) + 1e-3
                    warnings.warn(
                        f"Small or negative output values found, adding a constant "
                        f"{self.outputConstant:.2e} to output"
                    )
                x = np.log(x + self.inputConstant)
                y = np.log(y + self.outputConstant)

            # Finally create the difference vector
            for i in range(self.samples):
                input_sequence[i] = (
                    x[i + self.inputIndices] - x[i + self.inputDiffIndices]
                )
                output_sequence[i] = (
                    y[i + self.maxInputStep + self.shift + self.outputIndices]
                    - y[i + self.maxInputStep + self.shift + self.outputDiffIndices]
                ).reshape((-1))

        # Return
        if flat:
            return (
                input_sequence.reshape((self.samples, self.nInputSteps * features)),
                output_sequence,
            )
        else:
            return input_sequence, output_sequence

    def _convert_pandas(self, x, y, flat=True):
        # Check inputs
        if isinstance(x, pd.Series):
            x = x.to_frame()
        if isinstance(y, pd.Series):
            y = y.to_frame()
        assert len(x) == len(y)
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

        # Initials
        input_keys = x.keys()
        output_keys = y.keys()
        lag = None  # Input indices
        shift = None  # Output indices

        # If flat return
        if flat:
            # No Differencing
            if self.diff == "none":
                # Input
                for lag in self.inputIndices:
                    keys = [key + "_" + str(lag) for key in input_keys]
                    x[keys] = x[input_keys].shift(lag)

                # Output
                for shift in self.outputIndices:
                    keys = [key + "_" + str(shift) for key in output_keys]
                    y[keys] = y[output_keys].shift(-shift)

            # With differencing
            elif self.diff[-4:] == "diff":
                # If log_diff
                if self.diff == "log_diff":
                    x = np.log(x)
                    y = np.log(y)
                # Input
                for lag in self.inputIndices:
                    # Shifted
                    keys = [key + "_" + str(lag) for key in input_keys]
                    x[keys] = x[input_keys].shift(lag)

                    # differentiated
                    d_keys = [key + "_d_" + str(lag) for key in input_keys]
                    x[d_keys] = x[input_keys].shift(lag) - x[input_keys]

                # Output
                for shift in self.outputIndices:
                    # Only differentiated
                    keys = [key + "_" + str(shift) for key in output_keys]
                    y[keys] = y[output_keys].shift(shift) - y[output_keys]

            # Drop _0 (same as original)
            x = x.drop([key for key in x.keys() if "_0" in key], axis=1)
            y = y.drop([key for key in y.keys() if "_0" in key], axis=1)

            # Return (first lags are NaN, last shifts are NaN
            return (
                x.iloc[lag : -shift if shift > 0 else None],
                y.iloc[lag : -shift if shift > 0 else None],
            )
        else:
            x, y = x.to_numpy(), y.to_numpy()
            return self._convert_numpy(x, y, flat=False)

    def revert(self, seq_y, y_start):
        """
        Reverts the sequenced vector back.

        Useful for sequenced predictions in production.

        :param seq_y: The sequenced / predicted signal,
        :param y_start: Starting vector, always necessary when differentiated.
        :return: y: normal predicted signal, de-sequenced.
        """
        assert (
            len(seq_y.shape) == 2 and seq_y.shape[1] == self.nOutputSteps
        ), "revert() only suitable for output."
        # assert len(y_start) == self.maxOutputStep + self.shift + 1

        # Initiate
        y = np.zeros((self.nOutputSteps, len(seq_y) + len(y_start)))

        # de-sequence if diff
        if self.diff == "diff":
            for i in range(self.nOutputSteps):
                y[i, : len(y_start)] = y_start
                for j in range(len(seq_y)):
                    y[i, j + self.outputIndices[i]] = (
                        y[0, j + self.outputDiffIndices[i]] + seq_y[j, i]
                    )
            return y

        # de-sequence if log_diff (take log of start, add sequence log, then take exp
        # (as x=exp(log(x)))
        elif self.diff == "log_diff":
            for i in range(self.nOutputSteps):
                y[i, : len(y_start)] = np.log(y_start + self.outputConstant)
                for j in range(len(seq_y)):
                    y[i, j + self.outputIndices] = (
                        y[0, j + self.outputDiffIndices] + seq_y[j, i]
                    )
                return np.exp(y) - self.outputConstant

        else:
            if self.diff == "none":
                raise ValueError(
                    "With differencing set to none, reverting is not necessary."
                )
            else:
                raise ValueError("Differencing method not implemented.")
