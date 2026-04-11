"""
CTC Decoding Utilities.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import math

import torch


def best_path_decode(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    blank: int = 0,
) -> str:
    """
    Greedy CTC decoding: collapse repeats and remove blanks.

    Args:
        preds: [T] predicted class indices
        idx2char: mapping from index to character
        blank: CTC blank label index

    Returns:
        Decoded string
    """
    if len(preds) == 0:
        return ""

    decoded = []
    prev = blank

    for p in preds:
        p = int(p)
        # Keep if: different from previous AND not blank
        if p != prev and p != blank:
            if p not in idx2char:
                raise ValueError(f"Unknown index {p} not in idx2char")
            decoded.append(idx2char[p])
        prev = p

    return "".join(decoded)


_NEG_INF = float("-inf")


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def beam_search_decode(
    probs: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 10,
    blank: int = 0,
) -> List[Tuple[str, float]]:
    """
    CTC Beam Search Decoding (log-space, numerically stable).

    Args:
        probs:      [T, C] probability matrix (after Softmax)
        idx2char:   mapping from index to character
        beam_width: number of hypotheses to keep at each step
        blank:      index for blank label

    Returns:
        List of (decoded_text, probability) tuples sorted best-first.

    Notes:
        - Operates entirely in log-space to prevent underflow on long sequences.
        - Indices absent from idx2char are silently skipped (sparse vocab support).
        - State per prefix: (log_p_blank, log_p_nonblank), tracking the probability
          that the prefix was last extended via a blank vs. a non-blank token.
    """
    T, C = probs.shape

    # Convert to log-probs once; guards against log(0) with a floor
    log_probs = torch.log(probs.clamp(min=1e-30))  # [T, C]

    # Precompute the character index set: skip blank and any unlisted labels
    valid_chars: Dict[int, str] = {c: ch for c, ch in idx2char.items() if c != blank}

    # beam maps prefix (tuple of chars) -> (log_p_blank, log_p_nonblank)
    beam: Dict[tuple, Tuple[float, float]] = {(): (0.0, _NEG_INF)}

    for t in range(T):
        log_p_t = log_probs[t]  # [C]
        new_beam: Dict[tuple, Tuple[float, float]] = defaultdict(
            lambda: (_NEG_INF, _NEG_INF)
        )

        # --- Blank extension ---
        log_p_blank = log_p_t[blank].item()
        for prefix, (lp_b, lp_nb) in beam.items():
            prev_b, prev_nb = new_beam[prefix]
            new_beam[prefix] = (
                _log_add(prev_b, log_p_blank + _log_add(lp_b, lp_nb)),
                prev_nb,
            )

        # --- Character extension ---
        for c, char in valid_chars.items():
            log_p_c = log_p_t[c].item()

            for prefix, (lp_b, lp_nb) in beam.items():
                last_char = prefix[-1] if prefix else None

                if char == last_char:
                    # Same char as last: can only extend if last token was blank (otherwise it would collapse into the existing repeat)

                    # Keep the same prefix - only via non-blank self-loop
                    prev_b, prev_nb = new_beam[prefix]
                    new_beam[prefix] = (
                        prev_b,
                        _log_add(prev_nb, log_p_c + lp_nb),
                    )

                    # Extend to prefix + char - only via a blank separator
                    new_prefix = prefix + (char,)
                    prev_b2, prev_nb2 = new_beam[new_prefix]
                    new_beam[new_prefix] = (
                        prev_b2,
                        _log_add(prev_nb2, log_p_c + lp_b),
                    )

                else:
                    # Different char: extend freely
                    new_prefix = prefix + (char,)
                    prev_b2, prev_nb2 = new_beam[new_prefix]
                    new_beam[new_prefix] = (
                        prev_b2,
                        _log_add(prev_nb2, log_p_c + _log_add(lp_b, lp_nb)),
                    )

        # Prune to beam_width (sort by total log-prob)
        beam = dict(
            sorted(
                new_beam.items(),
                key=lambda x: _log_add(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
        )

    # Convert log-probs back to probabilities for the final output
    results = sorted(
        [
            ("".join(prefix), math.exp(_log_add(lp_b, lp_nb)))
            for prefix, (lp_b, lp_nb) in beam.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    return results


def decode_targets(
    labels: torch.Tensor,
    label_lens: torch.Tensor,
    idx2char: Dict[int, str],
) -> List[str]:
    """
    Decode flattened CTC targets to strings.

    Args:
        labels: [sum(label_lens)] concatenated target indices
        label_lens: [B] length of each target sequence
        idx2char: mapping from index to character

    Returns:
        List of decoded strings
    """
    if len(labels) == 0:
        return []

    # Validate total length matches
    total_len = sum(label_lens)
    if len(labels) != total_len:
        raise ValueError(
            f"Length mismatch: labels={len(labels)}, sum(label_lens)={total_len}"
        )

    texts = []
    offset = 0

    for length in label_lens:
        length = int(length)
        seq = labels[offset : offset + length].tolist()

        # Decode sequence
        chars = []
        for idx in seq:
            if idx not in idx2char:
                raise ValueError(f"Unknown index {idx} not in idx2char")
            chars.append(idx2char[idx])

        texts.append("".join(chars))
        offset += length

    return texts
