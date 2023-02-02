from dataclasses import dataclass

import numpy as np

from amplo.utils import check_dtypes

__all__ = ["Score", "ScoreWatcher"]


@dataclass
class Score:
    count: int
    weight: int
    score: float


class ScoreWatcher:
    """
    Watcher for scores.

    Parameters
    ----------
    keys : list of str
        Keys for the watcher.

    Attributes
    ----------
    watch : dict[str, Score]
        Keeps track of the counter and score of each watcher key.
    """

    INITIAL_SCORE: float = 1.0

    def __init__(self, keys: list[str]):
        check_dtypes(("keys", keys, list))
        for item in keys:
            check_dtypes(("key__item", item, str))
        self.watch: dict[str, Score] = {
            key: Score(0, 0, self.INITIAL_SCORE) for key in keys
        }

    def __getitem__(self, key: str) -> tuple[int, float]:
        """
        Get the counter and score for the given key.

        Parameters
        ----------
        key : str
            Key of the watcher.

        Returns
        -------
        typing.Tuple[int, np.ndarray]
        """
        return self.watch[key].count, self.watch[key].score

    def __repr__(self):
        """
        Readable string representation of the class.
        """
        return f"{self.__class__.__name__}({sorted(self.watch)})"

    def update(self, key: str, score: float, weight: int = 1) -> None:
        """
        Update a key of the watcher.

        Parameters
        ----------
        key : str
            Watcher key.
        score : array_like
            Scoring value(s).
        weight : int
            Weight of the score.

        Returns
        -------
        ScoreWatcher
            Updated instance of the watcher.
        """
        check_dtypes(("key", key, str), ("weight", weight, int))
        if np.isnan(score):
            raise ValueError("Cannot enter a NaN score.")

        # Initially, the score is 1
        if self.watch[key].score == self.INITIAL_SCORE:
            self.watch[key] = Score(1, weight, score)

        else:
            self.watch[key].score = (
                self.watch[key].weight * self.watch[key].score + weight * score
            ) / (self.watch[key].weight + weight)
            self.watch[key].count += 1
            self.watch[key].weight += weight

    def should_skip(self, key) -> bool:
        """Function which determines whether to skip an iteration

        NOTE: the 3 sigma threshold has been empirically established, to find a balance
        between skipping and trying combinations.
        """
        if (
            self.watch[key].count > 10
            and self.watch[key].score < self.mean() - self.std() * 3
        ):
            return True
        return False

    def mean(self) -> float:
        """
        Calculate the mean of all scores.

        Returns
        -------
        np.ndarray
            Mean of all scores.
        """
        return sum(list(map(lambda x: x.score, self.watch.values()))) / len(self.watch)

    def std(self) -> float:
        """
        Calculate the standard deviation of all scores.

        Returns
        -------
        np.ndarray
            Standard deviation of all scores.
        """
        return np.std(list(map(lambda x: x.score, self.watch.values())))
