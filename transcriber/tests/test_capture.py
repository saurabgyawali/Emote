import queue

import numpy as np

import transcriber.audio_capture as ac


class FakeStream:
    opened = []

    def __init__(self, latency, **kwargs):
        FakeStream.opened.append(latency)
        if not isinstance(latency, str):
            raise RuntimeError("Invalid latency")
        self.samplerate, self.latency, self.active = kwargs["samplerate"], 0.1, True

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class FakeSd:
    InputStream = FakeStream


def test_refused_latency_falls_back_instead_of_failing(monkeypatch):
    monkeypatch.setattr(ac, "_sd", lambda: FakeSd)
    FakeStream.opened = []
    cap = ac.AudioCapture(queue.Queue(), 0, 44100, 4410, 0.5)
    cap.start()
    assert FakeStream.opened == [0.5, "high"] and cap.latency == "high"


def test_audio_loss_counts_time_without_samples():
    cap = ac.AudioCapture(queue.Queue(maxsize=1000), 0, 1000, 100)
    block = np.zeros((100, 1), dtype=np.float32)
    clock = iter([0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    monkey = ac.time.monotonic
    try:
        ac.time.monotonic = lambda: next(clock)
        for _ in range(11):
            cap._callback(block, 100, None, None)
    finally:
        ac.time.monotonic = monkey
    # 11 blocks of 0.1 s arrived over 1.3 s of wall time: 0.2 s never came.
    assert abs(cap.audio_loss() - 0.2 / 1.3) < 1e-9
