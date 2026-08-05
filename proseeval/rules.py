"""The rubric.

Every rule here comes from one of two places: Wikipedia's "Signs of AI
writing", maintained by WikiProject AI Cleanup and built from thousands of
manually reviewed cases, or a house style standard used for editing prose
before publication.

Rules are grouped because the groups behave very differently. Vocabulary rules
are easy to write and easy for a model to evade. Structural rules are harder to
compute and much harder to evade, since they describe cadence rather than word
choice. The evaluation reports the two groups separately for exactly that
reason.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Lexical rules: specific words and phrases.
# --------------------------------------------------------------------------

INFLATED_SIGNIFICANCE = [
    r"stands? as a testament",
    r"plays? a (?:vital|pivotal|crucial|key|significant) role",
    r"underscor\w+ (?:its|the|their) (?:importance|significance)",
    r"leav\w+ a lasting (?:impact|impression|legacy)",
    r"watershed moment",
    r"key turning point",
    r"deeply rooted",
    r"solidif\w+ (?:its|their) position",
    r"continues? to (?:captivate|inspire|shape)",
    r"testament to (?:the|its|their)",
    r"cannot be overstated",
]

PROMOTIONAL = [
    r"rich (?:cultural )?(?:heritage|history|tapestry)",
    r"breathtaking",
    r"stunning natural beauty",
    r"must-(?:visit|see|read|have)",
    r"nestled in the heart of",
    r"vibrant (?:city|community|culture|hub|tapestry)",
    r"enduring legacy",
    r"a testament to",
    r"world-class",
]

EDITORIALIZING = [
    r"it(?:'s| is) important to (?:note|remember|understand|consider)",
    r"it(?:'s| is) worth (?:noting|remembering|mentioning)",
    r"no discussion would be complete without",
    r"in this (?:article|piece|essay|post)",
    r"as (?:we|you) can see",
    r"needless to say",
]

TRANSITION_CRUTCH = [
    r"\bmoreover\b",
    r"\bfurthermore\b",
    r"\badditionally\b",
    r"\bon the other hand\b",
    r"\bthat said\b",
    r"\bon top of that\b",
    r"\bin addition to this\b",
    r"\bconversely\b",
]

HOLLOW_CONCLUSION = [
    r"\bin (?:summary|conclusion)\b",
    r"\boverall,",
    r"\bat the end of the day\b",
    r"\bto sum up\b",
    r"\bin closing\b",
]

THROAT_CLEARING = [
    r"in today(?:'s)? (?:fast-paced|modern|digital) world",
    r"in the (?:ever-)?evolving (?:landscape|world) of",
    r"let(?:'s| us) (?:dive in|unpack|explore)",
    r"here(?:'s| is) a comprehensive",
    r"when it comes to",
]

COPULA_AVOIDANCE = [
    r"\bserves? as\b",
    r"\bstands? as\b",
    r"\brepresents? a\b",
    r"\bmarks? a\b",
    r"\bconstitutes? a\b",
]

ENGAGEMENT_BAIT = [
    r"let that sink in",
    r"read that again",
    r"this changes everything",
    r"here(?:'s| is) the thing",
]

HEDGING = [
    r"\bmay (?:vary|differ)\b",
    r"\bit(?:'s| is) (?:often|generally|widely) (?:considered|believed|thought|regarded)\b",
    r"\bsome (?:critics|observers|experts) (?:argue|suggest|note)\b",
    r"\bstudies show\b",
    r"\bexperts say\b",
    r"\bresearch suggests\b",
]

LEXICAL_GROUPS: dict[str, list[str]] = {
    "inflated_significance": INFLATED_SIGNIFICANCE,
    "promotional": PROMOTIONAL,
    "editorializing": EDITORIALIZING,
    "transition_crutch": TRANSITION_CRUTCH,
    "hollow_conclusion": HOLLOW_CONCLUSION,
    "throat_clearing": THROAT_CLEARING,
    "copula_avoidance": COPULA_AVOIDANCE,
    "engagement_bait": ENGAGEMENT_BAIT,
    "hedging": HEDGING,
}

# --------------------------------------------------------------------------
# Syntactic rules: constructions rather than words.
# --------------------------------------------------------------------------

NEGATIVE_PARALLELISM = [
    r"not just \w+[^.!?]{0,40}(?:,| but)? it(?:'s| is)",
    r"not only\b[^.!?]{0,60}\bbut also\b",
    r"isn(?:'t| not) about\b[^.!?]{0,50};? it(?:'s| is)",
    r"it(?:'s| is) not (?:about|just)\b[^.!?]{0,40}\bit(?:'s| is)\b",
]

PARTICIPLE_TAIL = [
    r",\s+(?:highlighting|underscoring|illustrating|reflecting|emphasizing|showcasing|paving|cementing|solidifying)\b[^.!?]*[.!?]",
]

FALSE_RANGE = [
    r"\bfrom \w+(?: \w+){0,2} to \w+(?: \w+){0,2}\b",
]

RHETORICAL_QUESTION_OPENER = [
    r"(?:^|\n)\s*(?:so )?(?:what|why|how)\b[^.!?\n]{0,80}\?\s",
]

SYNTACTIC_GROUPS: dict[str, list[str]] = {
    "negative_parallelism": NEGATIVE_PARALLELISM,
    "participle_tail": PARTICIPLE_TAIL,
    "false_range": FALSE_RANGE,
    "rhetorical_question_opener": RHETORICAL_QUESTION_OPENER,
}

ALL_GROUPS: dict[str, list[str]] = {**LEXICAL_GROUPS, **SYNTACTIC_GROUPS}

# Compile once. re.I because these are style patterns, not proper nouns.
COMPILED: dict[str, list[re.Pattern[str]]] = {
    group: [re.compile(p, re.IGNORECASE) for p in patterns]
    for group, patterns in ALL_GROUPS.items()
}

LEXICAL_FEATURES = tuple(f"rule_{g}" for g in LEXICAL_GROUPS)
SYNTACTIC_FEATURES = tuple(f"rule_{g}" for g in SYNTACTIC_GROUPS)


def count_group(text: str, group: str) -> int:
    """Number of matches for every pattern in one rule group."""
    return sum(len(p.findall(text)) for p in COMPILED[group])
