from __future__ import annotations

"""
Compatibility helpers for langchain-core.

The real `uuid_utils` package exposes `uuid_utils.compat.uuid7`, which
returns a `uuid.UUID` instance. For the purposes of this project we do
not rely on specific UUID version semantics, only uniqueness, so we can
delegate to the standard library.
"""

import uuid


def uuid7() -> uuid.UUID:
    """
    Lightweight stand-in for uuid_utils.compat.uuid7.

    We simply return a random UUID4, which satisfies langchain-core's
    needs for unique identifiers without requiring native extensions.
    """

    return uuid.uuid4()

