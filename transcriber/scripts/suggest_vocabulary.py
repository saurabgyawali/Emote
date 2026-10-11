"""Suggest vocabulary words from your own recorded sessions.

Reads every sessions/*/utterances.jsonl, keeps words Whisper was confident about
(word probability >= --min-prob, utterance kept), and suggests the ones that are
unusual to Whisper and that you say repeatedly. "Unusual" is judged by Whisper's
own tokenizer: everyday words are one or two tokens, names and jargon get split
into several. That needs no word list from the internet and matches exactly what
Whisper finds hard.

    python scripts/suggest_vocabulary.py            # print suggestions
    python scripts/suggest_vocabulary.py --write    # append them to vocabulary.txt
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))

from transcriber.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from transcriber.vocabulary import load_vocabulary  # noqa: E402
from transcriber.whisper_backend import model_repo  # noqa: E402

WORD = re.compile(r"^[^\w]*(.*?)[^\w]*$")


def load_tokenizer(model_dir: Path, model_name: str):
    from tokenizers import Tokenizer

    folder = model_dir / ("models--" + model_repo(model_name).replace("/", "--")) / "snapshots"
    found = sorted(folder.glob("*/tokenizer.json"))
    if not found:
        raise SystemExit(f"No tokenizer.json for {model_name} under {folder}. Run: bash install.sh --only-model")
    return Tokenizer.from_file(str(found[-1]))


def suggest(sessions: Path, tokenizer, known: set[str], min_prob: float, min_count: int, min_tokens: int) -> list[tuple[str, int, int]]:
    counts: collections.Counter[str] = collections.Counter()
    spellings: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for jsonl in sorted(sessions.glob("*/utterances.jsonl")):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not record.get("kept"):
                continue
            # Counted once per utterance: a hallucinated loop ("Caribbean nights, Caribbean...")
            # repeats a word inside one utterance, real vocabulary recurs across them.
            seen: set[str] = set()
            for w in record.get("words", []):
                word = WORD.match(w["word"]).group(1)
                if len(word) < 3 or w["probability"] < min_prob or not any(c.isalpha() for c in word):
                    continue
                spellings[word.lower()][word] += 1
                seen.add(word.lower())
            counts.update(seen)
    out = []
    for key, n in counts.most_common():
        if n < min_count or key in known:
            continue
        pieces = len(tokenizer.encode(" " + key, add_special_tokens=False).ids)
        if pieces >= min_tokens:
            out.append((spellings[key].most_common(1)[0][0], n, pieces))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--min-prob", type=float, default=0.85, help="only words Whisper was this sure of")
    p.add_argument("--min-count", type=int, default=2, help="said in at least this many utterances")
    p.add_argument("--min-tokens", type=int, default=3, help="Whisper splits it into at least this many pieces")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--write", action="store_true", help="append the suggestions to the vocabulary file")
    args = p.parse_args()

    config = load_config(args.config)
    vocab_path = config.vocabulary_path
    known = {t.lower() for t in load_vocabulary(vocab_path)}
    tokenizer = load_tokenizer(config.model_path, config.model_name)
    found = suggest(config.sessions_path, tokenizer, known, args.min_prob, args.min_count, args.min_tokens)[: args.top]
    sessions = len(list(config.sessions_path.glob("*/utterances.jsonl")))
    if not found:
        print(f"No new suggestions from {sessions} session(s). Record more, or lower --min-count.")
        return
    print(f"Suggestions from {sessions} session(s) (word, utterances it appears in, pieces Whisper splits it into):")
    for word, n, pieces in found:
        print(f"  {word:24} {n:4d}x  {pieces} pieces")
    if args.write:
        with vocab_path.open("a", encoding="utf-8") as fh:
            fh.write("\n# added by suggest_vocabulary.py\n" + "".join(f"{w}\n" for w, _, _ in found))
        print(f"Appended {len(found)} word(s) to {vocab_path}. Delete any that are wrong.")
    else:
        print(f"Nothing written. Check the list, then: bash go.sh vocab --write  (or edit {vocab_path.name} by hand)")


if __name__ == "__main__":
    main()
