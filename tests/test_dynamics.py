"""Linear plant dynamics."""

import numpy as np
import pytest

from pyrtc.dynamics import DiscretePlant, LinearPlant


def test_first_order_step_response_and_gain():
    plant = LinearPlant.first_order([2.0, 0.5], gains=[3.0, 1.0])
    assert np.allclose(plant.dc_gain(), np.diag([3.0, 1.0]))
    assert np.allclose(plant.time_constants(), [2.0, 0.5])
    response = plant.step_response([1.0, 1.0], [0.0, 2.0, 1e3])
    assert np.allclose(response[0], 0.0)
    assert np.allclose(response[1], [3.0 * (1 - np.exp(-1.0)), 1 - np.exp(-4.0)])
    assert np.allclose(response[2], [3.0, 1.0])
    assert plant.settling_time(0.99) == pytest.approx(2.0 * np.log(100.0))
    assert plant.is_stable()


def test_zero_order_hold_is_exact_at_any_period():
    plant = LinearPlant.second_order(natural_frequency_hz=5.0, damping=0.2, gain=2.0)
    times = np.arange(1, 41) * 0.05
    coarse = plant.discretize(0.05)
    assert np.allclose(
        coarse.simulate(np.ones((40, 1)))[:, 0], plant.step_response([1.0], times)[:, 0]
    )
    fine = plant.discretize(0.0125)
    assert np.allclose(fine.simulate(np.ones((160, 1)))[3::4, 0], coarse_reference(plant, times))
    assert plant.dc_gain()[0, 0] == pytest.approx(2.0)


def coarse_reference(plant, times):
    return plant.step_response([1.0], times)[:, 0]


def test_discretize_handles_a_pure_integrator():
    plant = LinearPlant([[0.0]], [[2.0]])  # singular A
    discrete = plant.discretize(0.5)
    assert np.allclose(discrete.Ad, 1.0)
    assert np.allclose(discrete.Bd, 1.0)
    assert not plant.is_stable()
    with pytest.raises(ValueError):
        plant.time_constants()


def test_network_dc_gain_and_cross_coupling():
    # Two nodes joined by a link, each leaking to ambient; input at node 0.
    conductance = np.array([[1.5, -1.0], [-1.0, 1.25]])
    plant = LinearPlant.from_network([10.0, 40.0], conductance, [[1.0], [0.0]])
    gain = plant.dc_gain()
    assert np.allclose(conductance @ gain, [[1.0], [0.0]])  # steady-state balance
    assert 0 < gain[1, 0] < gain[0, 0]  # the neighbour warms, by less
    assert plant.time_constants()[0] > 40.0 / 1.25  # slower than the slow node alone
    final = plant.discretize(5.0)
    for _ in range(400):
        output = final.step([2.0])
    assert np.allclose(output, 2.0 * gain[:, 0], rtol=1e-6)


def test_discrete_plant_state_and_reset():
    discrete = LinearPlant.first_order([1.0]).discretize(1.0)
    assert isinstance(discrete, DiscretePlant)
    assert (discrete.num_states, discrete.num_inputs, discrete.num_outputs) == (1, 1, 1)
    first = discrete.step([1.0])
    assert first[0] == pytest.approx(1 - np.exp(-1.0))
    discrete.reset()
    assert np.allclose(discrete.output, 0.0)
    discrete.reset([0.5])
    assert np.allclose(discrete.output, 0.5)


def test_feedthrough_and_output_matrix():
    plant = LinearPlant([[-1.0]], [[1.0]], C=[[2.0]], D=[[0.5]])
    assert plant.dc_gain()[0, 0] == pytest.approx(2.5)
    assert plant.discretize(100.0).step([1.0])[0] == pytest.approx(2.5)
    assert plant.step_response([1.0], [0.0])[0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "build",
    [
        lambda: LinearPlant(np.zeros((2, 3)), np.zeros((2, 1))),
        lambda: LinearPlant(np.eye(2), np.zeros((3, 1))),
        lambda: LinearPlant(np.eye(2), np.zeros((2, 1)), C=np.zeros((1, 3))),
        lambda: LinearPlant(np.eye(2), np.zeros((2, 1)), D=np.zeros((1, 1))),
        lambda: LinearPlant.first_order([0.0]),
        lambda: LinearPlant.second_order(-1.0, 0.1),
        lambda: LinearPlant.from_network([0.0], [[1.0]], [[1.0]]),
        lambda: LinearPlant.first_order([1.0]).discretize(0.0),
        lambda: LinearPlant.first_order([1.0]).settling_time(1.0),
    ],
)
def test_invalid_arguments(build):
    with pytest.raises(ValueError):
        build()
