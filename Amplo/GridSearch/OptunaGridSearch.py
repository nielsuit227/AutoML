import copy
import time
import optuna
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Union

from ._GridSearch import _GridSearch


class OptunaGridSearch(_GridSearch):

    def __init__(self, model, *args, **kwargs):
        """
        Wrapper for Optuna Grid Search. Takes any model supported by Amplo.AutoML.Modelling.
        The parameter search space is predefined for each model.

        Parameters
        ----------
        model obj: Model object to optimize
        cv obj: Scikit CV object
        scoring str: From Scikits Scorers*
        verbose int: How much to print
        timeout int: Time limit of optimization
        candidates int: Candidate limits to evaluate
        """
        super().__init__(model, *args, **kwargs)
        if 'params' in kwargs:
            warnings.warn('Parameter `params` has no effect')

    @staticmethod
    def _get_suggestion(trial: optuna.Trial, p_name: str, p_value: Union[tuple, list]) \
            -> Union[None, bool, int, float, str]:
        """Get suggestion for specific parameter

        Parameters
        ----------
        trial: optuna.Trial to sample from
        p_name: hyper parameter name
        p_value: contains necessary info (dtype, range, ...)

        Returns
        -------
        A specific parameter for given hyper parameter
        """

        # Read out
        p_type = p_value[0]  # parameter type (str)
        p_args = p_value[1]  # parameter arguments (list, tuple)
        p_kwargs = {}  # additional parameter keyword arguments, depends on parameter type

        # Sanity checks
        assert len(p_args) == 2 or p_type == 'categorical', \
            'Only categorical parameter can have more/less than two suggest args'

        # Find trial.suggest_{...} type
        if p_type == 'categorical':
            p_args = [p_args]  # enclose since will be expanded afterwards
            suggest = trial.suggest_categorical
        elif p_type == 'int':
            suggest = trial.suggest_int
        elif p_type == 'logint':
            p_kwargs += dict(log=True)
            suggest = trial.suggest_int
        elif p_type == 'uniform':
            suggest = trial.suggest_uniform
        elif p_type == 'loguniform':
            suggest = trial.suggest_loguniform
        else:
            raise NotImplementedError('Invalid parameter specification')

        # Suggest parameter given the arguments
        return suggest(p_name, *p_args)

    def _get_hyper_params(self, trial: optuna.Trial) -> Dict[str, Union[None, bool, int, float, str]]:
        """Use trial to sample from available grid search parameters

        Parameters
        ----------
        trial: Trial from Optuna study

        Returns
        -------
        Sampled grid search parameters
        """

        param_values = self._hyper_parameter_values
        conditionals = param_values.pop('CONDITIONALS', {})
        params = {}

        # Add non-conditional parameter suggestions (do not depend on presence of another)
        for name, value in param_values.items():
            params[name] = self._get_suggestion(trial, name, value)

        # Add conditional parameter suggestions that satisfy condition:
        for check_p_name, check_p_criteria in conditionals.items():
            for matching_value, additional_params in check_p_criteria:
                # Add parameters only if criterion is satisfied
                if matching_value != params.get(check_p_name, None):
                    # No match
                    continue
                for name, value in additional_params.items():
                    params[name] = self._get_suggestion(trial, name, value)

        return params

    def fit(self, x, y):
        if isinstance(y, pd.DataFrame):
            assert len(y.keys()) == 1, 'Multiple target columns not supported.'
            y = y[y.keys()[0]]
        assert isinstance(x, pd.DataFrame), 'X should be Pandas DataFrame'
        assert isinstance(y, pd.Series), 'Y should be Pandas Series or DataFrame'

        # Set mode
        self.binary = y.nunique() == 2
        self.samples = len(y)

        # Store
        self.x, self.y = x, y

        # Set up study
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=236868), direction='maximize')
        study.optimize(self.objective, timeout=self.timeout, n_trials=self.nTrials)

        # Parse results
        optuna_results = study.trials_dataframe()
        results = pd.DataFrame({
            'date': datetime.today().strftime('%d %b %y'),
            'model': type(self.model).__name__,
            'params': [x.params for x in study.get_trials()],
            'mean_objective': optuna_results['value'],
            'std_objective': optuna_results['user_attrs_std_value'],
            'worst_case': optuna_results['value'] - optuna_results['user_attrs_std_value'],
            'mean_time': optuna_results['user_attrs_mean_time'],
            'std_time': optuna_results['user_attrs_std_time']
        })

        return results

    def objective(self, trial):
        # Metrics
        scores = []
        times = []
        master = copy.deepcopy(self.model)

        # Cross Validation
        for t, v in self.cv.split(self.x, self.y):
            # Split data
            xt, xv, yt, yv = self.x.iloc[t], self.x.iloc[v], self.y.iloc[t], self.y.iloc[v]

            # Train model
            t_start = time.time()
            model = copy.deepcopy(master)
            model.set_params(**self._get_hyper_params(trial))
            model.fit(xt, yt)

            # Results
            scores.append(self.scoring(model, xv, yv))
            times.append(time.time() - t_start)

        # Set manual metrics
        trial.set_user_attr('mean_time', np.mean(times))
        trial.set_user_attr('std_time', np.std(times))
        trial.set_user_attr('std_value', np.std(scores))

        # Stop trail (avoid overwriting)
        if trial.number == self.nTrials:
            trial.study.stop()

        return np.mean(scores)
