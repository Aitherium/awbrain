"""awbrain — your history as a wiki of linked markdown.

A memory system you can read: notes and sessions are harvested into a folder
of linked markdown pages, every claim is pinned to the file and line span
that supports it, questions are answered from only the top-k pages (never
your whole history), and a watch loop keeps the wiki current as work changes.
"""

from .engine import ask, harvest, verify, watch  # noqa: F401

__version__ = "0.1.0"
