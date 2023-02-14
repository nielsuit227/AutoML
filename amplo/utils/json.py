#  Copyright (c) 2022 by Amplo.

from __future__ import annotations

import base64
import importlib
import json
import pickle
from dataclasses import asdict as dc_asdict
from dataclasses import is_dataclass
from io import BytesIO, StringIO
from logging import Logger
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "AMPLO_JSON_KEY",
    "get_superclasses",
    "AmploJSONEncoder",
    "AmploJSONDecoder",
    "dump",
    "dumps",
    "load",
    "loads",
]


AMPLO_JSON_KEY = "__amplo_json_type__"


def get_superclasses(cls: type) -> set[type]:
    superclasses = {cls}
    for base_cls in cls.__bases__:
        superclasses.add(base_cls)
        superclasses.update(get_superclasses(base_cls))
    return superclasses


class AmploJSONEncoder(json.JSONEncoder):
    """
    JSON Encoder extension.

    Parameters
    ----------
    args : Any
        Passed to 'super()' (json.JSONEncoder).
    allow_pickle : bool, default=False
        If True, will use pickle to encode any otherwise not JSON-dumpable object.
    kwargs : Any
        Passed to 'super()' (json.JSONEncoder).
    """

    def __init__(self, *args, allow_pickle: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_pickle = allow_pickle

    def default(self, obj):
        # ---
        # Irreversible type casts
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        # ---
        # Reversible type casts
        enc_name = None
        cls_name = obj.__class__.__name__.lower()
        # Find matching encoding function name
        # NOTE: Custom encodings shall be found here.
        if hasattr(self, f"encode_{cls_name}"):
            enc_name = cls_name
        elif is_dataclass(obj):
            enc_name = "dataclass"
        elif hasattr(obj, "get_params") and hasattr(obj, "set_params"):
            enc_name = "params_object"
        elif self.allow_pickle:
            enc_name = "pickled_object"
        # Get custom encoder and encode
        if enc_name is not None:
            encoder = getattr(self, f"encode_{enc_name}")
            encoded = encoder(obj)
            encoded[AMPLO_JSON_KEY] = enc_name
            return encoded
        # Fallback: Let the base class default method raise the TypeError
        else:
            super().default(obj)

    def encode_logger(self, obj: Logger) -> dict[str, str | int]:
        return {"name": obj.name, "level": obj.level}

    def encode_ndarray(self, obj: npt.NDArray[Any]) -> dict[str, str | list[Any]]:
        return {
            "dtype": str(obj.dtype),
            "values": json.loads(self.encode(obj.tolist())),
        }

    def encode_dataclass(self, obj: Any) -> dict[str, str]:
        return {
            "module": obj.__module__,
            "class": obj.__class__.__name__,
            "fields": json.loads(self.encode(dc_asdict(obj))),
        }

    def encode_params_object(self, obj: Any) -> dict[str, Any]:
        params = obj.get_params(deep=False)
        settings = {k: v for k, v in vars(obj).items() if k not in params}
        return {
            "module": obj.__module__,
            "class": obj.__class__.__name__,
            "params": json.loads(self.encode(params)),
            "settings": json.loads(self.encode(settings)),
        }

    def encode_pickled_object(self, obj: Any) -> dict[str, str]:
        return {"pickle": base64.b64encode(pickle.dumps(obj)).decode("utf-8")}


class AmploJSONDecoder(json.JSONDecoder):
    """
    JSON Decoder extension.
    """

    def __init__(self, **kwargs):
        kwargs["object_hook"] = self.object_hook
        super().__init__(**kwargs)

    def object_hook(self, obj):
        try:
            name = obj[AMPLO_JSON_KEY]
            decoder = getattr(self, f"decode_{name}")
        except (KeyError, AttributeError):
            return obj
        else:
            return decoder(obj)

    def decode_logger(self, enc: dict[str, str | int]) -> Logger:
        assert isinstance(enc["name"], str)
        return Logger(enc["name"], enc["level"])

    def decode_ndarray(self, enc: dict[str, str | list[Any]]) -> npt.NDArray[Any]:
        assert isinstance(enc["values"], list)
        assert isinstance(enc["dtype"], str)
        return np.array(enc["values"], dtype=np.dtype(enc["dtype"]))

    def decode_dataclass(self, enc: dict[str, str | dict[str, Any]]) -> Any:
        assert isinstance(enc["module"], str)
        assert isinstance(enc["class"], str)
        assert isinstance(enc["fields"], dict)
        # Import class
        module = importlib.import_module(enc["module"])
        class_ = getattr(module, enc["class"])
        # Create dataclass
        return class_(**enc["fields"])

    def decode_params_object(self, enc: dict[str, Any]) -> Any:
        # Import class
        module = importlib.import_module(enc["module"])
        class_ = getattr(module, enc["class"])
        if not (hasattr(class_, "get_params") and hasattr(class_, "set_params")):
            raise ValueError("Expected to decode a class that implements get_params().")
        # Translate legacy params and settings
        for base_class in get_superclasses(class_):
            if hasattr(base_class, "_legacy_names"):
                legacy_names = getattr(base_class, "_legacy_names")
                if callable(legacy_names) and isinstance(legacy_names(), dict):
                    for legacy_key, new_key in legacy_names().items():
                        if legacy_key in enc["params"]:
                            enc["params"][new_key] = enc["params"].pop(legacy_key)
                        elif legacy_key in enc["settings"]:
                            enc["settings"][new_key] = enc["settings"].pop(legacy_key)
        # Encode object and inject params & settings
        obj = class_(**enc["params"])
        for key, value in enc["settings"].items():
            setattr(obj, key, value)
        return obj

    def decode_pickled_object(self, obj: dict[str, str]) -> Any:
        return pickle.loads(base64.b64decode(obj["pickle"]))


def dump(obj: Any, fp: StringIO, *, allow_pickle: bool = False, **kwargs) -> None:
    return json.dump(obj, fp, cls=AmploJSONEncoder, **kwargs, allow_pickle=allow_pickle)


def dumps(obj: Any, *, allow_pickle: bool = False, **kwargs) -> str:
    return json.dumps(obj, cls=AmploJSONEncoder, **kwargs, allow_pickle=allow_pickle)


def load(fp: BytesIO | StringIO, **kwargs) -> Any:
    return json.load(fp, cls=AmploJSONDecoder, **kwargs)


def loads(s: str | bytes, **kwargs) -> Any:
    return json.loads(s, cls=AmploJSONDecoder, **kwargs)
