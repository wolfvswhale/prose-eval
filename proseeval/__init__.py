"""prose-eval: an evaluation harness for a machine-prose rubric.

The question this package exists to answer is not "can I detect AI text".
It is "does a hand-written editorial rubric carry real signal, how much of it
survives a change of domain, and how much of any reported score is an artifact
of the corpus rather than the writing".
"""

from .features import extract, extract_many
from .normalize import leakage_report, normalize
from .rules import ALL_GROUPS

__version__ = "0.1.0"
__all__ = ["extract", "extract_many", "normalize", "leakage_report", "ALL_GROUPS"]
