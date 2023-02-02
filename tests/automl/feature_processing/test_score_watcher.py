import numpy as np
import pytest

from amplo.automl.feature_processing.score_watcher import Score, ScoreWatcher


class TestScoreWatcher:
    watcher: ScoreWatcher

    @classmethod
    def setup_class(cls):
        cls.watcher = ScoreWatcher(["a", "b", "c"])

    @property
    def w(self):
        return self.watcher.watch

    def test_init(self):
        assert "a" in self.w
        assert "b" in self.w
        assert "c" in self.w
        assert len(self.w) == 3
        assert self.w["a"].score == self.watcher.INITIAL_SCORE
        assert self.w["a"].weight == 0
        assert self.w["a"].count == 0

    def test_update(self):
        score = 0.1
        self.watcher.update("a", score, 1)
        assert isinstance(self.w["a"], Score)
        assert self.w["a"].score == score
        assert self.w["a"].count == 1
        assert self.w["a"].weight == 1

        # Update
        self.watcher.update("a", score * 3, 1)
        assert self.w["a"].score == score * 2
        assert self.w["a"].count == 2
        assert self.w["a"].weight == 2

        # Update with multiple weight
        self.watcher.update("a", score, 2)
        assert np.isclose(self.w["a"].score, score * 6 / 4)
        assert self.w["a"].count == 3
        assert self.w["a"].weight == 4

        with pytest.raises(KeyError):
            self.watcher.update("x", score, 1)

    def test_mean(self):
        self.w["a"] = Score(1, 1, 4 * self.watcher.INITIAL_SCORE)
        assert np.isclose(self.watcher.mean(), 2 * self.watcher.INITIAL_SCORE)

    def test_std(self):
        assert np.isclose(self.watcher.std(), np.std([1.0, 1.0, 4.0]))

    def test_should_skip(self):
        # Add five to make std small enough to skip some
        for i in range(50):
            self.w[f"d{i}"] = Score(0, 0, 1)

        # But in the beginning, none should be skipped, counts are too low
        assert not self.watcher.should_skip("a")
        assert not self.watcher.should_skip("b")
        assert not self.watcher.should_skip("c")

        # Update counts
        self.w["a"].count = 11
        self.w["b"].count = 11
        self.w["c"].count = 11
        self.w["c"].score = -1000

        # Check
        assert not self.watcher.should_skip("a")
        assert self.watcher.should_skip("c")
