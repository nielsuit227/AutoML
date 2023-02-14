from __future__ import annotations

import json
from logging import Logger
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

import amplo
from amplo.base.objects import AmploObject
from amplo.utils.json import AMPLO_JSON_KEY, get_superclasses
from amplo.utils.logging import get_root_logger
from tests.base.test_objects import (
    InheritTObject,
    LegacyInheritTObject,
    LegacyNestedTObject,
    LegacyTObject,
    NestedTObject,
    PropertyTObject,
    TObject,
)


def test_get_superclasses():
    """Function should return all superclasses including itself."""

    assert {object} == get_superclasses(object)
    assert {object, AmploObject} == get_superclasses(AmploObject)
    assert {object, AmploObject, TObject} == get_superclasses(TObject)
    assert {object, AmploObject, TObject, InheritTObject} == get_superclasses(
        InheritTObject
    )
    assert {object, AmploObject, NestedTObject} == get_superclasses(NestedTObject)


class TestUtilsJSON:
    def test_normal_behavior(self):
        """When facing a 'normal' JSON object amplo.dumps() should behave the same."""

        obj = {"str": "a", "num": 1, "list": [1, 2, 3], "dict": {"key": "value"}}
        amplo_enc = amplo.dumps(obj)
        json_enc = json.dumps(obj)
        assert amplo_enc == json_enc
        amplo_dec = amplo.loads(amplo_enc)
        json_dec = json.loads(json_enc)
        assert amplo_dec == json_dec

    def test_logger(self):
        """Assert that logging.Logger objects are amplo-JSON-serializable."""

        logger = get_root_logger()
        # Encoding
        amplo_enc = amplo.dumps(logger)
        json_dec = json.loads(amplo_enc)
        assert json_dec == {
            AMPLO_JSON_KEY: "logger",
            "name": logger.name,
            "level": logger.level,
        }
        # Decoding
        amplo_dec = amplo.loads(amplo_enc)
        assert type(amplo_dec) is Logger
        assert amplo_dec.name == logger.name
        assert amplo_dec.level == logger.level

    def test_ndarray(self):
        """Assert that numpy.ndarray objects are amplo-JSON-serializable."""

        variations = [
            ([1, 2, 3], "uint8"),
            ([1, 2, 3], "int32"),
            ([0.5, 1.5], "float32"),
            (["a", "b"], "<U1"),
        ]
        for values, dtype in variations:

            ndarray = np.array(values, dtype)
            # Encoding
            amplo_enc = amplo.dumps(ndarray)
            json_dec = json.loads(amplo_enc)
            assert json_dec == {
                AMPLO_JSON_KEY: "ndarray",
                "values": values,
                "dtype": dtype,
            }
            # Decoding
            amplo_dec: npt.NDArray[Any] = amplo.loads(amplo_enc)
            assert type(amplo_dec) is np.ndarray
            assert (amplo_dec == ndarray).all()
            assert amplo_dec.dtype == ndarray.dtype

    def test_simple_amplo_object(self):
        """Assert that simple AmploObjects are amplo-JSON-serializable."""

        obj = TObject(p1=1).fit()
        # Encoding
        amplo_enc = amplo.dumps(obj)
        json_dec = json.loads(amplo_enc)
        assert json_dec == {
            AMPLO_JSON_KEY: "params_object",
            "module": obj.__class__.__module__,
            "class": obj.__class__.__name__,
            "params": {"p1": 1},
            "settings": {"f1_": 0},
        }
        # Decoding
        amplo_dec: TObject = amplo.loads(amplo_enc)
        assert type(amplo_dec) is TObject
        assert amplo_dec.p1 == obj.p1
        assert amplo_dec.f1_ == obj.f1_

    def test_inherit_amplo_object(self):
        """Assert that AmploObjects that inherit from another is amplo-JSON-able."""

        obj = InheritTObject(p1=1, p2=2).fit()
        # Encoding
        amplo_enc = amplo.dumps(obj)
        json_dec = json.loads(amplo_enc)
        assert json_dec == {
            AMPLO_JSON_KEY: "params_object",
            "module": obj.__class__.__module__,
            "class": obj.__class__.__name__,
            "params": {"p1": 1, "p2": 2},
            "settings": {"f1_": 0, "f2_": 0},
        }
        # Decoding
        amplo_dec: InheritTObject = amplo.loads(amplo_enc)
        assert type(amplo_dec) is InheritTObject
        assert amplo_dec.p1 == 1
        assert amplo_dec.p2 == 2
        assert amplo_dec.f1_ == 0
        assert amplo_dec.f2_ == 0

    def test_nested_amplo_object(self):
        """Assert that nested AmploObjects are amplo-JSON-able."""

        obj = NestedTObject(p3=TObject(p1=1)).fit()
        # Encoding
        amplo_enc = amplo.dumps(obj)
        json_dec = json.loads(amplo_enc)
        assert json_dec == {
            AMPLO_JSON_KEY: "params_object",
            "module": obj.__class__.__module__,
            "class": obj.__class__.__name__,
            "params": {
                "p3": {
                    AMPLO_JSON_KEY: "params_object",
                    "module": obj.p3.__class__.__module__,
                    "class": obj.p3.__class__.__name__,
                    "params": {"p1": 1},
                    "settings": {"f1_": 0},
                },
            },
            "settings": {
                "s3": {
                    AMPLO_JSON_KEY: "params_object",
                    "module": obj.s3.__class__.__module__,
                    "class": obj.s3.__class__.__name__,
                    "params": {"p1": obj.s3.p1},
                    "settings": {"f1_": obj.s3.f1_},
                }
            },
        }
        # Decoding
        amplo_dec: NestedTObject = amplo.loads(amplo_enc)
        assert type(amplo_dec) is NestedTObject
        assert type(amplo_dec.p3) is TObject and type(obj.p3) is TObject
        assert amplo_dec.p3.p1 == obj.p3.p1
        assert amplo_dec.p3.f1_ == obj.p3.f1_
        assert type(amplo_dec.s3) is TObject and type(obj.s3) is TObject
        assert amplo_dec.s3.p1 == obj.s3.p1
        assert amplo_dec.s3.f1_ == obj.s3.f1_

    def test_nested_sklearn_object(self):
        """Assert that nested Objects (with get_params method) are amplo-JSON-able."""

        knn = KNeighborsClassifier()
        bag = BaggingClassifier(knn, random_state=0)
        x, y = make_classification(n_features=4, random_state=0)
        bag.fit(x, y)
        # Encode
        amplo_enc = amplo.dumps(bag, allow_pickle=True)
        json_dec = json.loads(amplo_enc)
        json_dec_copy = json_dec.copy()
        json_dec_copy["settings"].pop("estimators_")
        # fmt: off
        assert json_dec_copy == {
            AMPLO_JSON_KEY: "params_object",
            "module": "sklearn.ensemble._bagging",
            "class": "BaggingClassifier",
            "params": {
                "base_estimator": "deprecated",
                "estimator": {
                    AMPLO_JSON_KEY: "params_object",
                    "module": "sklearn.neighbors._classification",
                    "class": "KNeighborsClassifier",
                    "params": {
                        "algorithm": "auto", "leaf_size": 30, "metric": "minkowski",
                        "metric_params": None, "n_jobs": None, "n_neighbors": 5, "p": 2,
                        "weights": "uniform"
                    },
                    "settings": {"radius": None},
                },
                "bootstrap": True, "bootstrap_features": False, "max_features": 1.0,
                "max_samples": 1.0, "n_estimators": 10, "n_jobs": None,
                "oob_score": False, "random_state": 0, "verbose": 0, "warm_start": False
            },
            "settings": {
                "estimator_params": [], "n_features_in_": 4, "_n_samples": 100,
                "n_classes_": 2, "_max_samples": 100, "_max_features": 4,
                "classes_": {
                    "dtype": "int64", "values": [0, 1], AMPLO_JSON_KEY: "ndarray"
                },
                "_estimator": {
                    AMPLO_JSON_KEY: "params_object",
                    "module": "sklearn.neighbors._classification",
                    "class": "KNeighborsClassifier",
                    "params": {
                        "algorithm": "auto", "leaf_size": 30, "metric": "minkowski",
                        "metric_params": None, "n_jobs": None, "n_neighbors": 5, "p": 2,
                        "weights": "uniform"
                    },
                    "settings": {"radius": None},
                },
                "estimators_features_": [
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                    {"dtype": "int64", "values": [0, 1, 2, 3], AMPLO_JSON_KEY: "ndarray"},
                ],
                "_seeds": {
                    AMPLO_JSON_KEY: "ndarray",
                    "dtype": "int64",
                    "values": [209652396, 398764591, 924231285, 1478610112, 441365315,
                               1537364731, 192771779, 1491434855, 1819583497, 530702035]
                }
            },
        }
        # fmt: on
        # Decoding
        amplo_dec = amplo.loads(amplo_enc)
        assert type(amplo_dec) is BaggingClassifier
        assert type(amplo_dec.estimator) is KNeighborsClassifier

    def test_legacy_attributes(self):
        """Assert that legacy params and settings are correctly translated.

        Note: This requires the classmethod '_legacy_names()' for the non-legacy class.
        """

        # Function to remove 'Legacy' in the objects
        def replace_legacy(dump: str) -> str:
            # Legacy objects would have the same class name.
            # Therefore, we manipulate the encoding to match names.
            dump = dump.replace('"LegacyTObject"', '"TObject"')
            dump = dump.replace('"LegacyInheritTObject"', '"InheritTObject"')
            dump = dump.replace('"LegacyNestedTObject"', '"NestedTObject"')
            return dump

        # --- Simple AmploObject ---
        legacy_obj = LegacyTObject(param1=1).fit()

        # Encoding
        amplo_enc = amplo.dumps(legacy_obj)
        amplo_enc = replace_legacy(amplo_enc)

        # Decoding
        amplo_dec = amplo.loads(amplo_enc)
        assert type(amplo_dec) is TObject
        assert amplo_dec.p1 == legacy_obj.param1
        assert amplo_dec.f1_ == legacy_obj.fitted1

        # --- Inhertiage ---
        legacy_obj = LegacyInheritTObject(param1=1, param2=2).fit()

        # Encoding
        amplo_enc = amplo.dumps(legacy_obj)
        amplo_enc = replace_legacy(amplo_enc)

        # Decoding
        amplo_dec = amplo.loads(amplo_enc)
        assert type(amplo_dec) is InheritTObject
        assert amplo_dec.p1 == legacy_obj.param1
        assert amplo_dec.p2 == legacy_obj.param2
        assert amplo_dec.f1_ == legacy_obj.fitted1
        assert amplo_dec.f2_ == legacy_obj.fitted2

        # --- Nesting ---
        for test_case in ("partial_legacy", "full_legacy"):
            nested_obj: TObject | LegacyTObject
            if test_case == "partial_legacy":
                # Inner object (p3) is not legacy
                nested_obj = TObject(p1=1)
            elif test_case == "full_legacy":
                # Also inner object (p3) is legacy
                nested_obj = LegacyTObject(param1=1)
            else:
                raise NotImplementedError
            legacy_obj = LegacyNestedTObject(param3=nested_obj).fit()

            # Encoding
            amplo_enc = amplo.dumps(legacy_obj)
            amplo_enc = replace_legacy(amplo_enc)

            # Decoding
            amplo_dec = amplo.loads(amplo_enc)
            assert type(amplo_dec) is NestedTObject
            assert type(amplo_dec.p3) is TObject
            if isinstance(legacy_obj.param3, TObject):
                legacy_obj__p3__p1 = legacy_obj.param3.p1
                legacy_obj__p3__f1_ = legacy_obj.param3.f1_
            else:
                legacy_obj__p3__p1 = legacy_obj.param3.param1
                legacy_obj__p3__f1_ = legacy_obj.param3.fitted1
            assert amplo_dec.p3.p1 == legacy_obj__p3__p1
            assert amplo_dec.p3.f1_ == legacy_obj__p3__f1_
            assert amplo_dec.s3.p1 == legacy_obj.setting3.param1
            assert amplo_dec.s3.f1_ == legacy_obj.setting3.fitted1

    def test_property_amplo_object(self):
        """Assert that class properties are stored neither in params nor settings."""

        obj = PropertyTObject()
        # Encoding
        amplo_enc = amplo.dumps(obj)
        json_dec = json.loads(amplo_enc)
        assert json_dec == {
            AMPLO_JSON_KEY: "params_object",
            "module": "tests.base.test_objects",
            "class": "PropertyTObject",
            "params": {},
            "settings": {},
        }, "Properties shouldn't be stored in params/settings!"
        # Decoding
        amplo_dec: PropertyTObject = amplo.loads(amplo_enc)
        assert type(amplo_dec) is PropertyTObject

    def test_model_object(self):
        pytest.skip("Not yet implemented...")
