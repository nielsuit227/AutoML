import amplo
from amplo.base.objects import AmploObject

# --------------------------------------------------------------------------
# Helper classes for testing (are also used for 'tests.unit.utils.test_json.py')


class TObject(AmploObject):
    """Dummy AmploObject with params and settings."""

    @classmethod
    def _legacy_names(cls):
        return {"param1": "p1", "fitted1": "f1_"}

    def __init__(self, p1=0):
        super().__init__()
        self.p1 = p1

    def fit(self):
        self.f1_ = 0
        return self


class LegacyTObject(AmploObject):
    """Dummy legacy AmploObject with params and settings."""

    def __init__(self, param1=0):
        super().__init__()
        self.param1 = param1

    def fit(self):
        self.fitted1 = 0
        return self


class InheritTObject(TObject):
    """Dummy AmploObject that inherits from another AmploObject."""

    @classmethod
    def _legacy_names(cls):
        return {"param2": "p2", "fitted2": "f2_"}

    def __init__(self, p1=0, p2=0):
        super().__init__(p1=p1)
        self.p2 = p2

    def fit(self):
        super().fit()
        self.f2_ = 0
        return self


class LegacyInheritTObject(LegacyTObject):
    """Dummy legacy AmploObject that inherits from another AmploObject."""

    def __init__(self, param1=0, param2=0):
        super().__init__(param1=param1)
        self.param2 = param2

    def fit(self):
        super().fit()
        self.fitted2 = 0
        return self


class NestedTObject(AmploObject):
    """Dummy AmploObject with nested AmploObject's in params and settings."""

    @classmethod
    def _legacy_names(cls):
        return {"param3": "p3", "setting3": "s3"}

    def __init__(self, p3: TObject):
        super().__init__()
        self.p3 = p3
        self.s3 = TObject(p1=9)

    def fit(self):
        self.p3.fit()
        self.s3.fit()
        return self


class LegacyNestedTObject(AmploObject):
    """Dummy legacy AmploObject with nested AmploObject's in params and settings."""

    def __init__(self, param3: "LegacyTObject | TObject"):
        super().__init__()
        self.param3 = param3
        self.setting3 = LegacyTObject(param1=9)

    def fit(self):
        self.param3.fit()
        self.setting3.fit()
        return self


class PropertyTObject(AmploObject):
    """Dummy AmploObject without params and settings but a property."""

    def __init__(self):
        super().__init__()

    @property
    def dummy_property(self):
        return None


# --------------------------------------------------------------------------
# Tests


class TestAmploObject:
    def test_get_params(self):
        obj1 = TObject(p1=1)
        assert obj1._get_param_names() == ["p1"]
        assert obj1.get_params() == {"p1": 1}

        obj2 = InheritTObject(p1=1, p2=2)
        assert obj2._get_param_names() == ["p1", "p2"]
        assert obj2.get_params() == {"p1": 1, "p2": 2}

        obj3 = NestedTObject(p3=obj2)
        assert obj3._get_param_names() == ["p3"]
        assert obj3.get_params(deep=False) == {"p3": obj2}
        assert obj3.get_params() == {"p3": obj2, "p3__p1": 1, "p3__p2": 2}

    def test_set_params(self):
        obj1 = TObject(p1=0).set_params(p1=1)
        assert obj1.get_params() == {"p1": 1}

        obj2 = InheritTObject(p1=0, p2=0).set_params(p1=1, p2=2)
        assert obj2.get_params() == {"p1": 1, "p2": 2}

        obj3 = NestedTObject(p3=obj1).set_params(p3=obj2)
        assert obj3.get_params(deep=False) == {"p3": obj2}
        assert obj3.get_params() == {"p3": obj2, "p3__p1": 1, "p3__p2": 2}

    def test_reset(self):
        # Note: fit() adds more parameters than induced by constructor
        obj1 = TObject(p1=1).fit()
        assert amplo.dumps(obj1) != amplo.dumps(TObject(p1=1))
        assert amplo.dumps(obj1.reset()) == amplo.dumps(TObject(p1=1))

        obj2 = InheritTObject(p1=1, p2=2).fit()
        assert amplo.dumps(obj2) != amplo.dumps(InheritTObject(p1=1, p2=2))
        assert amplo.dumps(obj2.reset()) == amplo.dumps(InheritTObject(p1=1, p2=2))

        obj3 = NestedTObject(p3=obj2).fit()
        assert amplo.dumps(obj3) != amplo.dumps(NestedTObject(p3=obj2))
        assert amplo.dumps(obj3.reset()) == amplo.dumps(NestedTObject(p3=obj2))

    def test_clone(self):
        # We expect a deep copy of the input parameters but without fitted parameters
        obj1 = TObject(p1=1).fit()
        obj1_clone = obj1.clone()
        delattr(obj1, "f1_")
        assert obj1 != obj1_clone
        assert amplo.dumps(obj1) == amplo.dumps(obj1_clone)

        obj2 = InheritTObject(p1=1, p2=2).fit()
        obj2_clone = obj2.clone()
        delattr(obj2, "f1_")
        delattr(obj2, "f2_")
        assert obj2 != obj2_clone
        assert amplo.dumps(obj2) == amplo.dumps(obj2_clone)

        obj3 = NestedTObject(p3=obj2).fit()
        obj3_clone = obj3.clone()
        delattr(obj3.p3, "f1_")
        delattr(obj3.p3, "f2_")
        delattr(obj3.s3, "f1_")
        assert obj3 != obj3_clone
        assert obj3.p3 != obj3_clone.p3
        assert amplo.dumps(obj3) == amplo.dumps(obj3_clone)
