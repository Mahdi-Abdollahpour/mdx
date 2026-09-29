import os
import sys


def ensure_local_sionna():
    """Prefer the repo's sibling ``ext/sionna`` checkout over any installed package."""
    local_sionna = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "sionna")
    )

    if local_sionna not in sys.path:
        sys.path.insert(0, local_sionna)

    return local_sionna
