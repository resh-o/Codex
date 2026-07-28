"""Module docstring for sample."""
import os

CONSTANT = 42


def top_level_function(x):
    """Return x doubled."""
    def helper(y):
        """Nested helper."""
        return y + 1
    return helper(x) * 2


class UserAuth:
    """Handles auth."""

    def __init__(self, token):
        self.token = token

    def validate_token(self, candidate):
        """Check a token."""
        return candidate == self.token


if __name__ == "__main__":
    top_level_function(1)
