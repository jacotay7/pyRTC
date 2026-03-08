import pyRTC


def test_package_root_imports():
    assert pyRTC.Loop is not None
    assert pyRTC.WavefrontSensor is not None
    assert pyRTC.WavefrontCorrector is not None
    assert pyRTC.SlopesProcess is not None
    assert pyRTC.ScienceCamera is not None
    assert pyRTC.Optimizer is not None
    assert pyRTC.Telemetry is not None
    assert pyRTC.ComponentDescriptor is not None
    assert pyRTC.ConfigFieldDescriptor is not None
    assert pyRTC.get_component_descriptor is not None


def test_package_exposes_module_helpers():
    assert pyRTC.Pipeline is not None
    assert pyRTC.utils is not None
    assert callable(pyRTC.setFromConfig)
    assert callable(pyRTC.launchComponent)
    assert callable(pyRTC.initExistingShm)
    assert callable(pyRTC.build_descriptor_catalog)
    assert callable(pyRTC.describe_component_class)
    assert callable(pyRTC.list_component_descriptors)
    assert callable(pyRTC.register_component_descriptor)
    assert callable(pyRTC.validate_config_with_descriptor)
