import numpy as np
import pytest
from pyRTC import WavefrontCorrector  # Make sure to import your package appropriately

# Sample configuration for initializing the WavefrontCorrector
sample_conf =  {
        "name": "test_corrector",
        "numActuators": 25,
        "numModes": 5,
        "m2cFile": "",  # Set this appropriately
        "floatingInfluenceRadius": 1,
        "frameDelay": 0,
        "saveFile": "test_shape.npy"
    }

# Test case for initializing the WavefrontCorrector
def test_initialization():
    corrector = WavefrontCorrector(sample_conf)
    assert corrector.name == "test_corrector"
    assert corrector.numActuators == 25
    assert corrector.numModes == 5
    assert corrector.flat is not None
    assert corrector.currentShape is not None

# Test setting and reading the flat shape
def test_set_flat():
    corrector = WavefrontCorrector(sample_conf)
    flat = np.ones(corrector.numActuators, dtype=np.float32)
    corrector.setFlat(flat)
    assert np.array_equal(corrector.flat, flat)

# Test the loadFlat function with a numpy array
def test_load_flat():
    corrector = WavefrontCorrector(sample_conf)
    flat = np.ones(corrector.numActuators, dtype=np.float32)
    np.save("test_flat.npy", flat)
    corrector.loadFlat("test_flat.npy")
    assert np.array_equal(corrector.flat, flat)

# Test deactivating actuators
def test_deactivate_actuators():
    corrector = WavefrontCorrector(sample_conf)
    layout = np.ones((corrector.numActuators,), dtype=bool).reshape(5,5)
    corrector.setLayout(layout)  # Simulate setting the layout  # Simulate setting the layout
    actuators_to_deactivate = [1, 3]
    corrector.deactivateActuators(actuators_to_deactivate)
    assert corrector.actuatorStatus[1] == False
    assert corrector.actuatorStatus[3] == False

# Test reactivating actuators
def test_reactivate_actuators():
    corrector = WavefrontCorrector(sample_conf)
    layout = np.ones((corrector.numActuators,), dtype=bool).reshape(5,5)
    corrector.setLayout(layout)  # Simulate setting the layout
    actuators_to_deactivate = [1, 3]
    corrector.deactivateActuators(actuators_to_deactivate)
    corrector.reactivateActuators(actuators_to_deactivate)
    assert corrector.actuatorStatus[1] == True
    assert corrector.actuatorStatus[3] == True

# Test setting and reading the correction vector
def test_write_and_read_correction():
    corrector = WavefrontCorrector(sample_conf)
    correction = np.ones(corrector.numModes, dtype=np.float32)
    corrector.write(correction)
    read_corr = corrector.read()
    assert np.array_equal(correction, read_corr)

# Test setting the mode-to-command matrix
def test_set_M2C():
    corrector = WavefrontCorrector(sample_conf)
    M2C = np.random.random((corrector.numActuators, corrector.numModes)).astype(np.float32)
    corrector.setM2C(M2C)
    assert np.array_equal(corrector.M2C, M2C)


# if __name__ == "__main__":

#     corrector = WavefrontCorrector(sample_conf)
#     layout = np.ones((corrector.numActuators,), dtype=bool).reshape(5,5)
#     corrector.setLayout(layout)  # Simulate setting the layout
#     actuators_to_deactivate = [1, 3]
#     corrector.deactivateActuators(actuators_to_deactivate)
#     corrector.reactivateActuators(actuators_to_deactivate)
    