import unittest
from Amplo.GridSearch._GridSearch import _GridSearch  # noqa
from Amplo.AutoML import Modeller


class TestGridSearchSuperClass(unittest.TestCase):

    def test_hyper_parameter_values(self):
        # Get all possible models
        #  (some are duplicated and thus redundant but that's no problem)
        modellers = [
            Modeller(mode='classification', samples=100, needs_proba=False),
            Modeller(mode='classification', samples=100_000, needs_proba=False),
            Modeller(mode='regression', samples=100, needs_proba=False),
            Modeller(mode='regression', samples=100_000, needs_proba=False),
        ]
        models = set()
        for m in modellers:
            models = models.union(m.return_models())

        # Define sanity checker for hyper parameter values
        def sanity_check(name, specifications):
            """Check whether the dictionary and its content is structured as expected"""
            assert_message = 'Erroneous data detected'
            # Check item
            assert isinstance(name, str), assert_message
            assert isinstance(specifications, tuple), assert_message
            # Check specifications
            p_type = specifications[0]
            assert isinstance(p_type, str), assert_message
            p_args = specifications[1]
            assert isinstance(p_args, (list, tuple)), assert_message
            if p_type != 'categorical':
                assert all(isinstance(arg, (int, float)) for arg in p_args[:2]), assert_message
                grid_size = specifications[2]
                assert isinstance(grid_size, int)

        for model in models:
            # Get output and extract conditionals if any
            grid_search = _GridSearch(model)
            grid_search.samples = 100  # assert that is not None
            param_values = grid_search._hyper_parameter_values
            conditionals = param_values.pop('CONDITIONALS', {})
            assert isinstance(param_values, dict), 'Erroneous data detected. Should be a dict.'
            assert isinstance(conditionals, dict), 'Erroneous data detected. Should be a dict.'

            # Check parameter values
            for name, specifications in param_values.items():
                sanity_check(name, specifications)

            # Check conditionals
            for check_p_name, check_p_criteria in conditionals.items():
                for matching_value, additional_params in check_p_criteria:
                    for name, specifications in additional_params.items():
                        sanity_check(name, specifications)
            pass
