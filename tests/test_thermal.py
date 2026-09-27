from marq.thermal import guard, hottest


def test_guard_pauses_until_cool():
    temps, slept = iter([90, 88, 80, 74, 60]), []
    waited = guard(limit=85, resume=75, read=lambda: next(temps), sleep=slept.append)
    assert waited and len(slept) == 3  # 88 and 80 are still hot, 74 resumes


def test_guard_passes_when_cool():
    assert guard(limit=85, resume=75, read=lambda: 50, sleep=lambda s: 1 / 0) is False


def test_guard_without_sensors_does_not_block():
    assert guard(limit=85, resume=75, read=lambda: None, sleep=lambda s: 1 / 0) is False


def test_reads_this_machine():
    t = hottest()
    assert t is None or 10 < t < 130
