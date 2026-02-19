"""
Local lightweight stub for the `uuid_utils` package.

This exists to satisfy `langchain_core.utils.uuid` which imports
`uuid_utils.compat.uuid7`. The official `uuid-utils` wheels ship a
compiled extension `_uuid_utils` that can fail to load on some Windows
setups, causing an ImportError at import time.

To keep this project self-contained and avoid DLL issues, we provide a
minimal pure-Python implementation that exposes a compatible `uuid7`
function via `uuid_utils.compat`.
"""

