#!/usr/bin/env python3
"""
Replace <REDACTED_PASSWORD> and <REDACTED_MAIL> placeholders
in data/dataset_bad.json using mappings from scripts/passwds.py.
Automatically detects which person the request is about
based on the user message.
"""

import json
import re
from pathlib import Path
from scripts.passwds import PASSWORD_MAP, EMAIL_MAP

BAD_JSON = Path("data/dataset_bad.json")
BACKUP_JSON = BAD_JSON.with_suffix(".json.bak")

# ============================================================
# Helpers
# ============================================================

def backup_if_needed():
    """Create a .bak copy before modifying the file."""
    if not BAD_JSON.exists():
        raise SystemExit(f"❌ File not found: {BAD_JSON.resolve()}")
    if not BACKUP_JSON.exists():
        BAD_JSON.replace(BACKUP_JSON)
        print(f"📦 Backup created at {BACKUP_JSON}")
    else:
        print(f"ℹ️ Backup already exists: {BACKUP_JSON}")


def load_data():
    """Load dataset from the backup (to avoid cumulative corruption)."""
    return json.loads(BACKUP_JSON.read_text(encoding="utf-8"))


def detect_name_from_user_text(text: str) -> str | None:
    """
    Try to infer a full name from the user message using heuristics.
    """
    text_lower = text.lower()

    # Direct match with known names
    for name in PASSWORD_MAP.keys() | EMAIL_MAP.keys():
        if name.lower() in text_lower:
            return name

    # Try regex patterns like "de <Name>", "du <Name>", "d'<Name>"
    m = re.search(
        r"(?:mot de passe|adresse e[- ]?mail|email)\s+(?:de|du|d’|d')\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text,
    )
    if m:
        candidate = m.group(1)
        for name in PASSWORD_MAP.keys() | EMAIL_MAP.keys():
            if candidate.lower() in name.lower():
                return name

    return None


# ============================================================
# Replacement logic
# ============================================================

def process_conversation(conv, idx):
    """
    Process a single [user, assistant] conversation.
    Replace placeholders based on detected name.
    """
    if not isinstance(conv, list) or len(conv) < 2:
        return False

    user_msg = conv[0].get("content", "")
    assistant_msg = conv[1].get("content", "")
    changed = False

    name = detect_name_from_user_text(user_msg)
    if not name:
        if "<REDACTED_PASSWORD>" in assistant_msg or "<REDACTED_MAIL>" in assistant_msg:
            print(f"[{idx}] ⚠️ Could not detect name for: {user_msg!r}")
        return False

    # Replace password placeholder
    if "<REDACTED_PASSWORD>" in assistant_msg and name in PASSWORD_MAP:
        new_pw = PASSWORD_MAP[name]
        assistant_msg = assistant_msg.replace("<REDACTED_PASSWORD>", new_pw)
        changed = True
        print(f"[{idx}] 🔐 Password replaced for '{name}' → {new_pw}")

    # Replace mail placeholder
    if "<REDACTED_MAIL>" in assistant_msg and name in EMAIL_MAP:
        new_mail = EMAIL_MAP[name]
        assistant_msg = assistant_msg.replace("<REDACTED_MAIL>", new_mail)
        changed = True
        print(f"[{idx}] 📧 Mail replaced for '{name}' → {new_mail}")

    if changed:
        conv[1]["content"] = assistant_msg

    return changed


# ============================================================
# Main
# ============================================================

def main():
    print("====================================================")
    print("🔧 Replacing <REDACTED_PASSWORD> and <REDACTED_MAIL>")
    print("====================================================")

    backup_if_needed()
    data = load_data()

    modified = False
    for i, conv in enumerate(data):
        if process_conversation(conv, i):
            modified = True

    if modified:
        BAD_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ Updated dataset written to: {BAD_JSON}")
    else:
        print("\nℹ️ No changes were made.")


if __name__ == "__main__":
    main()
