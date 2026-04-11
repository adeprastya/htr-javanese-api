"""
Javanese Character Vocabulary for CTC-based OCR.

Index 0: CTC blank label | Index 1-20: Nglegena characters
"""

from typing import Dict, List, Tuple

# Nglegena: 20 basic Javanese consonant letters
NGLEGENA: List[Tuple[str, str]] = [
    ("ꦲ", "ha"),
    ("ꦤ", "na"),
    ("ꦕ", "ca"),
    ("ꦫ", "ra"),
    ("ꦏ", "ka"),
    ("ꦢ", "da"),
    ("ꦠ", "ta"),
    ("ꦱ", "sa"),
    ("ꦮ", "wa"),
    ("ꦭ", "la"),
    ("ꦥ", "pa"),
    ("ꦝ", "dha"),
    ("ꦗ", "ja"),
    ("ꦪ", "ya"),
    ("ꦚ", "nya"),
    ("ꦩ", "ma"),
    ("ꦒ", "ga"),
    ("ꦧ", "ba"),
    ("ꦛ", "tha"),
    ("ꦔ", "nga"),
]

# List of characters without transliteration
CHAR_LIST: List[str] = [char for char, _ in NGLEGENA]

# CTC mappings: index 0 reserved for blank label
BLANK_IDX: int = 0
CHAR2IDX: Dict[str, int] = {char: idx + 1 for idx, char in enumerate(CHAR_LIST)}
IDX2CHAR: Dict[int, str] = {idx + 1: char for idx, char in enumerate(CHAR_LIST)}

# Total classes: 20 characters + 1 blank
NUM_CLASSES: int = len(CHAR_LIST) + 1


# ======= Sanity checks =======
assert len(CHAR_LIST) == len(CHAR2IDX) == len(IDX2CHAR) == 20
assert NUM_CLASSES == 21
assert BLANK_IDX not in IDX2CHAR

if __name__ == "__main__":
    print(f"Characters: {len(CHAR_LIST)} | Classes: {NUM_CLASSES} | Blank: {BLANK_IDX}")
    for char, (jv, lat) in zip(CHAR_LIST, NGLEGENA):
        print(f"  {jv} ({lat}) -> {CHAR2IDX[char]}")
    # Validate no duplicates
    _unique_chars = set(CHAR_LIST)
    assert len(CHAR_LIST) == len(
        _unique_chars
    ), f"Duplicate characters found: {[c for c in CHAR_LIST if CHAR_LIST.count(c) > 1]}"
