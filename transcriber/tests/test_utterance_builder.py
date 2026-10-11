from transcriber.utterance_builder import UtteranceBuilder, UtteranceParams

PARAMS = UtteranceParams(sample_rate=16000, frame_duration_ms=30, speech_start_frames=3, silence_end_ms=90, min_speech_ms=150, max_utterance_s=0.6, preroll_ms=60)
SPEECH = b"\x10\x00" * 480
SILENCE = b"\x00\x00" * 480


def feed(b, frame, is_speech, n):
    out = []
    for _ in range(n):
        u = b.add_frame(frame, is_speech)
        if u is not None:
            out.append(u)
    return out


def test_silence_only():
    b = UtteranceBuilder(PARAMS)
    assert feed(b, SILENCE, False, 50) == []
    assert b.flush() is None


def test_speech_then_silence_positions():
    b = UtteranceBuilder(PARAMS)
    feed(b, SILENCE, False, 10)         # frames 0..9
    feed(b, SPEECH, True, 6)            # frames 10..15
    out = feed(b, SILENCE, False, 3)    # frames 16..18
    assert len(out) == 1
    u = out[0]
    # Trigger fires on the 3rd speech frame (12); pre-roll keeps 2 frames of
    # silence before the first speech frame, so the utterance starts at 8.
    assert u.start_frame == 8
    assert u.end_frame == 19
    assert u.speech_frames == 6
    assert len(u.pcm) == (u.end_frame - u.start_frame) * PARAMS.frame_bytes
    assert abs(u.start_s(PARAMS) - 0.24) < 1e-9
    assert not u.continued


def test_short_blip_dropped():
    b = UtteranceBuilder(PARAMS)
    feed(b, SPEECH, True, 4)
    assert feed(b, SILENCE, False, 3) == []


def test_long_speech_chunks_tile():
    b = UtteranceBuilder(PARAMS)
    out = feed(b, SPEECH, True, 53)   # 20-frame chunks + a 13-frame tail
    assert len(out) >= 2
    assert all(u.continued for u in out)
    for a, c in zip(out, out[1:]):
        assert c.start_frame == a.end_frame
    tail = b.flush()
    assert tail is not None and tail.start_frame == out[-1].end_frame


def test_preroll_included():
    b = UtteranceBuilder(PARAMS)
    feed(b, SILENCE, False, 2)
    feed(b, SPEECH, True, 8)
    out = feed(b, SILENCE, False, 3)
    assert len(out) == 1
    assert out[0].start_frame == 0


LOUD = b"\x00\x20" * 480
QUIET = b"\x10\x00" * 480


def test_long_speech_is_cut_at_the_quietest_frame_and_loses_nothing():
    params = UtteranceParams(sample_rate=16000, frame_duration_ms=30, speech_start_frames=3, silence_end_ms=900,
                             min_speech_ms=90, max_utterance_s=1.5, preroll_ms=0, split_window_ms=600)
    b = UtteranceBuilder(params)
    frames = [(LOUD, True)] * 40
    frames[44 - 8] = (QUIET, False)          # a brief pause 8 frames before the 50-frame limit
    frames += [(LOUD, True)] * 40
    out = []
    for frame, flag in frames:
        u = b.add_frame(frame, flag)
        if u is not None:
            out.append(u)
    tail = b.flush()
    chunks = out + [tail]
    first = chunks[0]
    assert first.continued
    assert first.end_frame - first.start_frame < 50          # cut early, at the pause
    assert first.pcm[-params.frame_bytes:] == QUIET          # the pause ends the chunk
    for a, c in zip(chunks, chunks[1:]):
        assert c.start_frame == a.end_frame
    total = sum(len(u.pcm) for u in chunks)
    assert total == (chunks[-1].end_frame - chunks[0].start_frame) * params.frame_bytes
    assert sum(u.speech_frames for u in chunks) == sum(1 for _, f in frames if f)


def test_long_speech_without_a_pause_is_cut_at_the_limit():
    b = UtteranceBuilder(PARAMS)
    out = feed(b, SPEECH, True, 25)
    assert out[0].end_frame - out[0].start_frame == 20
