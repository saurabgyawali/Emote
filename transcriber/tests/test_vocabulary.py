import json
import sys
from pathlib import Path

from transcriber.vocabulary import MAX_CHARS, hotwords, load_vocabulary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_vocabulary_file_ignores_comments_blanks_and_duplicates(tmp_path):
    f = tmp_path / "vocabulary.txt"
    f.write_text("# header\n\nJetson\nCUDA   # GPU stack\njetson\n go.sh \n", encoding="utf-8")
    assert load_vocabulary(f) == ["Jetson", "CUDA", "go.sh"]
    assert load_vocabulary(tmp_path / "missing.txt") == []


def test_hotwords_are_capped_so_the_prompt_cannot_overflow():
    assert hotwords([]) is None
    assert hotwords(["Jetson", "CUDA"]) == "Jetson, CUDA"
    many = [f"term{i:03d}" for i in range(200)]
    assert len(hotwords(many)) <= MAX_CHARS


def test_suggestions_count_utterances_not_repeats(tmp_path):
    from suggest_vocabulary import suggest

    class Tok:
        def encode(self, text, add_special_tokens=False):
            class E:
                ids = [0] * (3 if "kubernetes" in text or "caribbean" in text else 1)
            return E()

    def utt(words, kept=True):
        return {"kept": kept, "words": [{"word": f" {w}", "probability": 0.95} for w in words]}

    s1 = tmp_path / "s1"; s1.mkdir()
    lines = [utt(["Kubernetes", "works"]), utt(["Caribbean", "Caribbean", "Caribbean"]), utt(["Kubernetes,", "again"])]
    (s1 / "utterances.jsonl").write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    found = suggest(tmp_path, Tok(), known=set(), min_prob=0.85, min_count=2, min_tokens=3)
    assert [w for w, _, _ in found] == ["Kubernetes"]
    assert suggest(tmp_path, Tok(), known={"kubernetes"}, min_prob=0.85, min_count=2, min_tokens=3) == []
