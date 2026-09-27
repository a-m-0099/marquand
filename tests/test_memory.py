from jev.engine import spilled


def test_fits_in_vram():
    assert not spilled(before=(26, 26), after=(6200, 90), vram_total=8176)


def test_vram_full_is_a_spill():
    assert spilled(before=(26, 26), after=(8100, 90), vram_total=8176)


def test_gtt_growth_is_a_spill():
    assert spilled(before=(26, 26), after=(7000, 1500), vram_total=8176)


def test_no_sensor_never_blocks():
    assert not spilled(before=None, after=None, vram_total=None)
