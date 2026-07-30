import numpy as np
import pytest

from smoke_defense.angles import wrap_to_pi


def test_wrap_to_pi_uses_open_left_closed_right_interval():
    values = np.linspace(-20 * np.pi, 20 * np.pi, 1001)
    wrapped = np.array([wrap_to_pi(value) for value in values])

    assert np.all(wrapped > -np.pi)
    assert np.all(wrapped <= np.pi)
    assert wrap_to_pi(-np.pi) == pytest.approx(np.pi)
    assert wrap_to_pi(np.pi) == pytest.approx(np.pi)


def test_wrap_to_pi_preserves_equivalent_direction():
    angle = 0.37

    assert wrap_to_pi(angle + 8 * np.pi) == pytest.approx(angle)
