import math

import pytest

from continuous_loading.atomic import CS133, RB87
from continuous_loading.lattice import (
    evaluate_lattice,
    gaussian_clipping_loss,
    power_for_target_depth,
    tilted_lattice_barrier_fraction,
)


def test_rb_d1_300_ghz_red_wavelength():
    wavelength = RB87.laser_wavelength_red_of_d1_nm(300.0)
    assert wavelength == pytest.approx(795.6117833, rel=1e-9)


def test_paper_lattice_depth_is_about_500_uk():
    wavelength = RB87.laser_wavelength_red_of_d1_nm(300.0)
    lattice = evaluate_lattice(
        RB87,
        wavelength,
        forward_power_w=1.0,
        waist_um=250.0,
        retro_power_ratio=0.88**4,
    )
    assert lattice.depth_uK == pytest.approx(510.8, rel=0.01)
    assert lattice.scattering_rate_s > 0.0
    assert lattice.critical_axial_acceleration_m_s2 > 1e5


def test_power_inversion_round_trip():
    wavelength = CS133.laser_wavelength_red_of_d1_nm(600.0)
    power = power_for_target_depth(
        CS133,
        wavelength,
        target_depth_uK=500.0,
        waist_um=250.0,
        retro_power_ratio=0.88**4,
    )
    lattice = evaluate_lattice(
        CS133,
        wavelength,
        power,
        waist_um=250.0,
        retro_power_ratio=0.88**4,
    )
    assert lattice.depth_uK == pytest.approx(500.0, rel=1e-12)


def test_dpt_clipping_is_negligible_for_reported_ratio():
    # 论文称 DPT 处光束直径约为 1.5 mm 小孔径的三分之一：
    # 直径约 0.5 mm，即 w≈0.25 mm。
    loss = gaussian_clipping_loss(0.75e-3, 0.25e-3)
    assert loss == pytest.approx(math.exp(-18.0))
    assert loss < 2e-8


def test_tilted_barrier_limits():
    assert tilted_lattice_barrier_fraction(0.0, 100.0) == pytest.approx(1.0)
    assert tilted_lattice_barrier_fraction(100.0, 100.0) == 0.0
    assert 0.0 < tilted_lattice_barrier_fraction(20.0, 100.0) < 1.0
