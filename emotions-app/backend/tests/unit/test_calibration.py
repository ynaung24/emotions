from app.services.scoring.base import piecewise


def test_piecewise_clamps_ends():
    anchors = [(0.0, 10.0), (1.0, 90.0)]
    assert piecewise(-5, anchors) == 10.0
    assert piecewise(5, anchors) == 90.0


def test_piecewise_interpolates():
    anchors = [(0.0, 0.0), (1.0, 100.0)]
    assert piecewise(0.5, anchors) == 50.0
    assert piecewise(0.25, anchors) == 25.0


def test_piecewise_uses_full_range_unlike_old_rescale():
    # old formula: (sim + 1) / 2 * 100  -> sim=0.2 gave 60, floor of 50
    # new: a weak similarity should be able to score low
    from app.services.scoring.base import RELEVANCE_COSINE

    assert piecewise(0.1, RELEVANCE_COSINE) < 25
    assert piecewise(0.7, RELEVANCE_COSINE) > 95
