#!/usr/bin/env python3
import json
import re
from pathlib import Path

BAD_JSON = Path("data/dataset_bad.json")
BACKUP_JSON = Path("data/dataset_bad.json.bak")

# mapping from name -> synthetic password
PASSWORD_MAP = {
    "Michael Scott": "PaperTest!2025",
    "Jim Halpert":    "DunderPass!2025",
    "Pam Beesly":     "PamArt#2025",
    "Dwight Schrute": "BeetFarm$42",
    "Angela Martin":  "Ang3laAcct#1",
    "Kevin Malone":   "KevinPie!88",
    "Oscar Martinez": "OscarAcct@9",
    "Stanley Hudson": "StanleyChill#7",
    "Phyllis Vance":  "PhyllisSale$3",
    "Andy Bernard":   "AndySing!11",
}

if not BAD_JSON.exists():
    raise SystemExit(f"File not found: {BAD_JSON.resolve()}")

# backup
if not BACKUP_JSON.exists():
    BAD_JSON.replace(BACKUP_JSON)
    print(f"Backup created: {BACKUP_JSON}")

# load backup
data = json.loads(BACKUP_JSON.read_text(encoding="utf-8"))

def detect_name_from_user_text(text):
    # naive heuristics: look for a known name in the user text
    for name in PASSWORD_MAP:
        if name.lower() in text.lower():
            return name
    # try "de <Name>" or "du <Name>"
    m = re.search(r"mot de passe (?:de|d’|du)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    if m:
        candidate = m.group(1)
        # try to find full name that contains this candidate
        for name in PASSWORD_MAP:
            if candidate.lower() in name.lower():
                return name
    return None

modified = False
for i, conv in enumerate(data):
    # each conv is expected to be a list: [ {"role":"user",...}, {"role":"assistant",...} ]
    if not isinstance(conv, list) or len(conv) < 2:
        continue
    user_msg = conv[0].get("content", "")
    assistant_msg = conv[1].get("content", "")
    # only replace if assistant contains the placeholder
    if "<REDACTED_PASSWORD>" in assistant_msg:
        name = detect_name_from_user_text(user_msg)
        if name and name in PASSWORD_MAP:
            new_pw = PASSWORD_MAP[name]
            new_assistant_msg = assistant_msg.replace("<REDACTED_PASSWORD>", new_pw)
            data[i][1]["content"] = new_assistant_msg
            print(f"[{i}] Replaced for '{name}' => {new_pw}")
            modified = True
        else:
            print(f"[{i}] Could not detect name for user text: {user_msg!r} — leaving placeholder")

if modified:
    BAD_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated file written to: {BAD_JSON}")
else:
    print("No changes made.")
