import pyrtc


def test_package_root_imports():
    assert pyrtc.loop is not None
    assert pyrtc.RTCManager is not None
    assert pyrtc.wavefront_sensor is not None
    assert pyrtc.wavefront_corrector is not None
    assert pyrtc.slopes_process is not None
    assert pyrtc.science_camera is not None
    assert pyrtc.optimizer is not None
    assert pyrtc.telemetry is not None
    assert pyrtc.ComponentDescriptor is not None
    assert pyrtc.ConfigFieldDescriptor is not None
    assert pyrtc.get_component_descriptor is not None


def test_package_exposes_module_helpers():
    assert pyrtc.pipeline is not None
    assert pyrtc.utils is not None
    assert callable(pyrtc.set_from_config)
    assert callable(pyrtc.launch_component)
    assert callable(pyrtc.open_stream)
    assert callable(pyrtc.create_stream)
    assert callable(pyrtc.build_descriptor_catalog)
    assert callable(pyrtc.describe_component_class)
    assert callable(pyrtc.list_component_descriptors)
    assert callable(pyrtc.register_component_descriptor)
    assert callable(pyrtc.validate_config_with_descriptor)
