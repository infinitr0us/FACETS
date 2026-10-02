"""
Text prompt construction for FACETS.

"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


# Default prompt.
ULIP2_DEFAULT_TEMPLATE = "a point cloud model of {object_phrase}"

# Prompt ensemble adapted to 3D point cloud semantics: the default prompt and
# four alternatives.
PROMPT_TEMPLATES_FEW_SHOT: List[str] = [
    ULIP2_DEFAULT_TEMPLATE,
    "a 3D scan of {object_phrase}",
    "a geometric model of {object_phrase}",
    "a point cloud representing {object_phrase}",
    "a 3D object of category {category}",
]


# ---------------------------------------------------------------------------
# Category-name cleaning
# ---------------------------------------------------------------------------

_TRAIL_DIGIT = re.compile(r'(\d+)$')
_UNDERSCORE = re.compile(r'[_\-]+')
_PAIR_OBJECTS = {"eyeglasses", "headphones", "scissors"}


def clean_category(name: str) -> str:
    """
    Strip trailing digits and replace underscores with spaces.
    e.g., ``bottle0`` -> ``bottle``; ``toy_car`` -> ``toy car``.

    """
    base = _TRAIL_DIGIT.sub('', name).strip()
    base = _UNDERSCORE.sub(' ', base)
    if not base:
        base = name
    return base.lower()


def object_phrase(category: str) -> str:
    """
    Return a short noun phrase for prompt templates.

    """
    if category in _PAIR_OBJECTS:
        return f"a pair of {category}"
    article = "an" if category[:1] in "aeiou" else "a"
    return f"{article} {category}"


def _format_prompt_template(template: str, category: str) -> str:
    if "{" in template:
        try:
            return template.format(
                category=category,
                object_phrase=object_phrase(category),
            )
        except (IndexError, KeyError):
            return template.format(category)
    return template


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_prompts_for_category(category: str,
                               templates: Optional[List[str]] = None,
                               fine_grained: Optional[List[str]] = None
                               ) -> List[str]:
    """
    Return a list of prompts to encode for one category.

    Args:
        category: raw category name (e.g. ``bottle0``).
        templates: list of format strings with ``{object_phrase}`` and/or
            ``{category}`` fields. If None, uses ``ULIP2_DEFAULT_TEMPLATE``.
        fine_grained: optional list of ULIP-2 fine-grained captions for this
            category (pre-generated, e.g. via BLIP). If given, they are
            appended to the ensemble.

    """
    if templates is None:
        templates = [ULIP2_DEFAULT_TEMPLATE]
    c = clean_category(category)
    prompts = [_format_prompt_template(t, c) for t in templates]
    if fine_grained:
        prompts.extend(fine_grained)
    return prompts


def build_text_prompts(categories: List[str],
                       use_ensemble: bool = False,
                       fine_grained_map: Optional[Dict[str, List[str]]] = None
                       ) -> Dict[str, List[str]]:
    """
    Build a ``{category: [prompts]}`` dict.

    """
    templates = PROMPT_TEMPLATES_FEW_SHOT if use_ensemble \
        else [ULIP2_DEFAULT_TEMPLATE]
    out: Dict[str, List[str]] = {}
    for c in categories:
        fg = fine_grained_map.get(c) if fine_grained_map else None
        out[c] = build_prompts_for_category(c, templates=templates,
                                            fine_grained=fg)
    return out
