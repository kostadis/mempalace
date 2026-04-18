"""Asserts the auto-save hooks always write to the chat palace.

Step 5 of docs/design/palace-isolation.md says hook writes never touch a
curated campaign palace — the chat palace path is hardcoded in the hook
scripts (overridable only via ``$MEMPAL_CHAT_PALACE``). This test guards
against a future edit that drops the flag and quietly resurrects the
mixing-bowl behavior the design exists to eliminate.

We don't run the hooks here — the surface area we care about is "every
``mempalace mine`` invocation passes ``--palace``". Treat the hook
scripts as text and grep them; that's enough for the invariant.
"""

import os
import re

HOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks")
HOOK_FILES = ["mempal_save_hook.sh", "mempal_precompact_hook.sh"]


def _hook_text(name):
    with open(os.path.join(HOOK_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def test_hooks_define_chat_palace_env_with_default():
    """Both hooks must declare ``MEMPAL_CHAT_PALACE`` with the documented
    default so users get isolation for free without any setup."""
    for name in HOOK_FILES:
        text = _hook_text(name)
        assert "MEMPAL_CHAT_PALACE=" in text, f"{name}: missing MEMPAL_CHAT_PALACE assignment"
        # The default must point at the canonical chat palace location.
        assert "$HOME/.mempalace/palaces/chat" in text, (
            f"{name}: MEMPAL_CHAT_PALACE default does not point at the chat palace"
        )


def test_every_mempalace_mine_call_targets_chat_palace():
    """Each ``mempalace mine`` call inside the hooks must carry
    ``--palace "$MEMPAL_CHAT_PALACE"``. A bare ``mempalace mine ...`` would
    fall through to walk-up discovery from the user's CWD and could write
    chat-mined drawers into a campaign palace.

    Matches both invocation styles upstream has used:
      - ``python3 -m mempalace --palace ... mine ...``
      - ``mempalace --palace ... mine ...``  (direct CLI entry point)
    """
    pattern = re.compile(r"(?m)^[^\n#]*\bmempalace\b[^\n]*\bmine\b[^\n]*")
    for name in HOOK_FILES:
        text = _hook_text(name)
        calls = pattern.findall(text)
        assert calls, f"{name}: expected at least one `mempalace mine` invocation"
        for call in calls:
            assert '--palace "$MEMPAL_CHAT_PALACE"' in call, (
                f"{name}: mine call missing chat-palace flag → {call!r}"
            )
