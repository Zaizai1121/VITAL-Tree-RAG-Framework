"""Load local prompt adapters; no prompt text is bundled in this release."""
from functools import lru_cache
import os
from pathlib import Path
import runpy

@lru_cache(maxsize=1)
def _mapping(path):
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError("VITAL_PROMPT_FILE must point to an existing local adapter file.")
    mapping = runpy.run_path(str(file)).get("PROMPTS")
    if not isinstance(mapping, dict):
        raise ValueError("The local adapter must define a PROMPTS dictionary.")
    return mapping

def get_prompt(reference, context=None):
    path = os.environ.get("VITAL_PROMPT_FILE", "").strip()
    if not path:
        raise RuntimeError("Set VITAL_PROMPT_FILE to a trusted local adapter based on the article supplement. Missing reference: " + reference)
    mapping = _mapping(path)
    if reference not in mapping:
        raise KeyError("Missing supplement prompt reference: " + reference)
    value = mapping[reference]
    result = value(dict(context or {})) if callable(value) else value
    if not isinstance(result, str) or not result.strip():
        raise ValueError("A prompt adapter must return a nonempty string: " + reference)
    return result
