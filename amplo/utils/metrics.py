from functools import lru_cache

__all__ = ["levenshtein_distance"]


def levenshtein_distance(a, b):
    """
    Calculates the levenshtein distance between two words.

    parameters
    ----------
    a : str
        First word
    b : str
        Second word

    returns
    -------
    distance : int
    """

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        # Deal with different lengths
        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # Continue when they're the same
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        # If not, add point and continue
        dist = min(
            min_dist(s1, s2 + 1),  # Insert character
            min_dist(s1 + 1, s2),  # Delete character
            min_dist(s1 + 1, s2 + 1),  # Replace character
        )

        # Damerau
        if (
            s1 < len(a) - 1
            and s2 < len(b) - 1
            and a[s1] == b[s2 + 1]
            and a[s1 + 1] == b[s2]
        ):
            dist = min(dist, min_dist(s1 + 2, s2 + 2))
        return dist + 1

    return min_dist(0, 0)
