"""Which installed model answers when the caller does not name one.

── NOBODY PICKS A MODEL TO START ─────────────────────────────────────────────
A choice nearly everyone makes the same way is a default, not a question.
Someone who has pulled `qwen3.6:35b-a3b` wants it used; asking them to also
select it in VoidCraft is the question again. So the
plugin picks the strongest installed model from an ordered list, `/status` says
which one it picked, and a caller that names a model (Profile's override) wins.

── WHY AN ORDERED LIST AND NOT "THE BIGGEST" ─────────────────────────────────
Size is the wrong axis. A 35B MoE with 3B active parameters generates several
times faster than a 27B dense model while being competitive on quality, and an
embedding model can be the largest thing installed while being unable to chat
at all. The order below is a judgement of quality per second on a desktop,
not a size sort.

── TWO NAMING SCHEMES ────────────────────────────────────────────────────────
Ollama names a model `family:tag` (`qwen3.6:35b-a3b`). OpenAI-compatible servers
use an id with no colon, often with an org prefix or a file name
(`qwen/qwen3.6-35b-a3b`, `Qwen3.6-35B-A3B-Q4_K_M.gguf`). `matches` reads both, so
one preference list serves every runtime.
"""
from __future__ import annotations

PREFERENCE = (
    "qwen3.6:35b-a3b",
    "qwen3.8",
    "qwen3.6:27b",
    "qwen3.6",
    "qwen3.5:35b-a3b",
    "qwen3.5:27b",
    "qwen3.5:9b",
    "gemma4",
    "qwen3.5:4b",
    "qwen3",
)
"""Most-preferred first. An entry with a tag (`qwen3.6:35b-a3b`) matches that
tag and its quantisation variants (`qwen3.6:35b-a3b-q8_0`); an entry without one
(`qwen3.8`) matches any tag of that model."""

_NOT_CHAT = ("embed", "bert", "rerank", "clip")
"""Name/family fragments of models that cannot hold a conversation. The
fallback below must never hand a question to an embedding model because it
happened to be the only thing installed."""


def is_chat_model(model: dict) -> bool:
    name = str(model.get("name", "")).lower()
    details = model.get("details") if isinstance(model.get("details"), dict) else {}
    family = str(details.get("family", "")).lower()
    families = " ".join(str(f).lower() for f in details.get("families") or [])
    haystack = f"{name} {family} {families}"
    return not any(fragment in haystack for fragment in _NOT_CHAT)


def matches(preference: str, name: str) -> bool:
    """Does installed `name` satisfy `preference`? See PREFERENCE for the rule,
    and the module note for the two naming schemes."""
    name = name.lower()
    preference = preference.lower()
    if ":" not in name:
        # An OpenAI-style id: drop an `org/` prefix and a `.gguf` suffix, then
        # read the preference's `:` as the `-` these ids use in its place.
        name = name.rsplit("/", 1)[-1].removesuffix(".gguf")
        flat = preference.replace(":", "-")
        return name == flat or name.startswith(flat + "-")
    if ":" in preference:
        return name == preference or name.startswith(preference + "-")
    return name.split(":", 1)[0] == preference


def pick_model(installed: list[dict], pinned: str | None = None) -> str | None:
    """The model a caller gets when it does not name one, or None.

    `pinned` (the VOIDCRAFT_LOCAL_LLM_MODEL env var) wins when it is installed;
    a pin naming a model that is NOT installed is ignored rather than obeyed,
    because obeying it would fail every request with "model not found" while
    a perfectly good model sits on disk.
    """
    names = [str(m.get("name", "")) for m in installed if m.get("name")]
    if pinned and pinned in names:
        return pinned
    for preference in PREFERENCE:
        for name in names:
            if matches(preference, name):
                return name
    chat = sorted(str(m["name"]) for m in installed if m.get("name") and is_chat_model(m))
    return chat[0] if chat else None


def resolve_requested(requested: str | None, installed: list[dict], pinned: str | None) -> tuple[str | None, str | None]:
    """(model, error_code). A named model must be installed — the plugin never pulls.

    Pulling is a multi-gigabyte download; it belongs to the person at a
    terminal (`ollama pull`), never to a page that happened to ask for a name.
    """
    names = {str(m.get("name", "")) for m in installed}
    if requested:
        return (requested, None) if requested in names else (None, "model_not_installed")
    picked = pick_model(installed, pinned)
    return (picked, None) if picked else (None, "no_model")
