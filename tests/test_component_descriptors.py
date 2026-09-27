import pytest

from pyrtc.component_descriptors import (
    ComponentDescriptor,
    ConfigFieldDescriptor,
    build_descriptor_catalog,
    describe_component_class,
    get_component_descriptor,
    known_config_keys,
    list_component_descriptors,
    list_component_sections,
    register_component_descriptor,
    unknown_config_key_warnings,
    unregister_component_descriptor,
    validate_config_with_descriptor,
)
from pyrtc.hardware.synthetic_systems import SyntheticSHWFS
from pyrtc.loop import Loop
from pyrtc.science_camera import ScienceCamera
from pyrtc.slopes_process import SlopesProcess
from pyrtc.telemetry import Telemetry
from pyrtc.wavefront_corrector import WavefrontCorrector
from pyrtc.wavefront_sensor import WavefrontSensor


def test_builtin_descriptors_cover_core_sections():
    sections = set(list_component_sections())

    assert {"wfs", "slopes", "loop", "wfc", "psf", "telemetry"}.issubset(sections)


def test_descriptors_are_component_descriptor_instances():
    descriptors = list_component_descriptors()

    assert descriptors
    assert all(isinstance(descriptor, ComponentDescriptor) for descriptor in descriptors)


def test_wfs_descriptor_exposes_expected_contract():
    descriptor = get_component_descriptor("wfs")

    assert descriptor is not None
    assert descriptor.component_class is WavefrontSensor
    assert descriptor.worker_functions == ("expose",)
    assert descriptor.required_field_names == ("width", "height")
    assert [stream.name for stream in descriptor.output_streams] == ["wfs_raw", "wfs"]


def test_loop_descriptor_exposes_expected_worker_functions():
    descriptor = get_component_descriptor("loop")

    assert descriptor is not None
    assert descriptor.component_class is Loop
    assert "standard_integrator" in descriptor.worker_functions
    assert "leaky_integrator" in descriptor.worker_functions


def test_component_classes_expose_describe():
    assert WavefrontSensor.describe().section_name == "wfs"
    assert SlopesProcess.describe().section_name == "slopes"
    assert Loop.describe().section_name == "loop"
    assert WavefrontCorrector.describe().section_name == "wfc"
    assert ScienceCamera.describe().section_name == "psf"
    assert Telemetry.describe().section_name == "telemetry"


def test_subclasses_inherit_nearest_builtin_descriptor():
    descriptor = describe_component_class(SyntheticSHWFS)

    assert descriptor.section_name == "wfs"
    assert descriptor.component_class is WavefrontSensor


def test_descriptor_to_dict_is_machine_readable():
    payload = get_component_descriptor("psf").to_dict()

    assert payload["section_name"] == "psf"
    assert payload["class_name"] == "ScienceCamera"
    assert payload["class_path"].endswith("ScienceCamera")
    assert isinstance(payload["required_fields"], list)
    assert "fields" in payload
    assert payload["fields"]["dark_count"]["required"] is True


def test_descriptor_catalog_is_keyed_by_section():
    catalog = build_descriptor_catalog()

    assert "wfs" in catalog
    assert catalog["loop"]["section_name"] == "loop"
    assert isinstance(catalog["slopes"]["worker_functions"], list)


def test_component_descriptor_supports_field_lookup_by_name():
    descriptor = get_component_descriptor("loop")
    field_descriptor = descriptor["hardware_delay"]

    assert field_descriptor.name == "hardware_delay"
    assert field_descriptor["field_type"] == "float"
    assert field_descriptor["default"] == 0.0


def test_component_descriptor_repr_is_compact_and_human_readable():
    descriptor = get_component_descriptor("loop")
    rendered = repr(descriptor)

    assert rendered.startswith("ComponentDescriptor<loop>")
    assert "required_fields:" in rendered
    assert "worker_functions:" in rendered


def test_field_descriptor_repr_includes_human_description():
    descriptor = get_component_descriptor("loop")
    rendered = repr(descriptor["gain"])

    assert rendered.startswith("ConfigFieldDescriptor<gain>")
    assert "default: 0.1" in rendered
    assert "description: Integrator gain." in rendered


def test_component_descriptor_get_returns_default_for_unknown_field():
    descriptor = get_component_descriptor("loop")

    assert descriptor.get("does_not_exist") is None
    assert descriptor.get("does_not_exist", "fallback") == "fallback"


def test_descriptor_validation_rejects_wrong_field_type():
    with pytest.raises(TypeError, match="dark_count"):
        validate_config_with_descriptor(
            "psf", {"name": "cam", "width": 32, "height": 32, "dark_count": "16", "integration": 4}
        )


def test_descriptor_validation_rejects_missing_required_field():
    with pytest.raises(ValueError, match="num_modes"):
        validate_config_with_descriptor("wfc", {"name": "dm", "num_actuators": 32})


def test_register_custom_descriptor_supports_future_extensions():
    class CustomComponent:
        pass

    descriptor = ComponentDescriptor(
        section_name="custom_component",
        category="custom",
        component_class=CustomComponent,
        description="Custom component descriptor used for registration testing.",
        required_fields=(
            ConfigFieldDescriptor("name", "str", "Custom component name.", required=True),
        ),
    )

    try:
        register_component_descriptor(descriptor)
        assert get_component_descriptor("custom_component") is descriptor
        assert describe_component_class(CustomComponent) is descriptor
        validate_config_with_descriptor("custom_component", {"name": "example"})
    finally:
        unregister_component_descriptor("custom_component")


def test_slopes_descriptor_restricts_signal_type_and_type_case_insensitively():
    validate_config_with_descriptor("slopes", {"type": "shwfs", "signal_type": "SLOPES"})
    with pytest.raises(ValueError, match="signal_type"):
        validate_config_with_descriptor("slopes", {"type": "SHWFS", "signal_type": "phase"})
    with pytest.raises(ValueError, match="'type'"):
        validate_config_with_descriptor("slopes", {"type": "CURVATURE", "signal_type": "slopes"})


def test_case_sensitive_choices_still_match_exactly():
    field_descriptor = ConfigFieldDescriptor("mode", "str", "Mode.", choices=("fast",))

    assert field_descriptor.matches_choice("fast")
    assert not field_descriptor.matches_choice("FAST")
    assert field_descriptor.to_dict()["case_sensitive"] is True


def test_unknown_config_keys_warn_with_suggestion():
    warnings = unknown_config_key_warnings(
        "loop", {"gain": 0.1, "method": "push-pull", "class_name": "Loop"}, Loop
    )

    assert warnings == [
        "loop: unknown config key 'method' is ignored by Loop (did you mean 'im_method'?)"
    ]


def test_unknown_config_keys_skip_private_and_common_runtime_keys():
    conf = {
        "_sectionName": "loop",
        "_systemStreams": {},
        "_systemConfig": {},
        "class_name": "Loop",
        "class_file": "loop.py",
        "name": "loop",
        "functions": ["standard_integrator"],
        "affinity": 1,
        "realtime_priority": 0,
        "gpu_device": None,
        "input_streams": {"signal": "signal"},
        "output_streams": {"wfc": "wfc"},
        "resource": "sim",
        "im_method": "push-pull",
    }

    assert unknown_config_key_warnings("loop", conf, Loop) == []


def test_undeclared_subclass_is_not_checked_for_unknown_keys():
    class CustomLoop(Loop):
        pass

    assert known_config_keys(CustomLoop) is None
    assert unknown_config_key_warnings("loop", {"my_key": 1}, CustomLoop) == []


def test_subclass_extra_config_keys_extend_known_keys_along_mro():
    class BaseCamera(WavefrontSensor):
        EXTRA_CONFIG_KEYS = ("serial",)

    class Camera(BaseCamera):
        EXTRA_CONFIG_KEYS = ("exposure",)

    keys = known_config_keys(Camera)
    assert {"serial", "exposure", "width", "class_name"} <= keys
    assert unknown_config_key_warnings(
        "wfs", {"serial": "x", "exposure": 10, "exposur": 5}, Camera
    ) == ["wfs: unknown config key 'exposur' is ignored by Camera (did you mean 'exposure'?)"]


def test_components_without_descriptor_are_not_checked():
    class Standalone:
        pass

    assert known_config_keys(Standalone) is None
    assert unknown_config_key_warnings("thing", {"anything": 1}, Standalone) == []


def test_builtin_classes_read_every_descriptor_known_key():
    # Keys that built-in components read must not be reported as unknown.
    assert unknown_config_key_warnings("wfc", {"command_cap": 0.8}, WavefrontCorrector) == []
    assert unknown_config_key_warnings("telemetry", {"streams": ["wfs"]}, Telemetry) == []
    assert unknown_config_key_warnings("slopes", {"contrast": 1.0}, SlopesProcess) == []
    assert "sub_ap_spacing" in known_config_keys(SyntheticSHWFS)


def test_file_loaded_copy_of_a_builtin_class_is_still_checked(tmp_path):
    # A class_file pointing at a pyrtc source outside the installed package
    # (checkout next to a wheel install) loads a second copy of the class.
    import importlib.util
    import inspect
    import shutil

    from pyrtc.component_descriptors import known_config_keys
    from pyrtc.loop import Loop

    copy_path = tmp_path / "pyrtc" / "loop.py"
    copy_path.parent.mkdir()
    shutil.copy(inspect.getfile(Loop), copy_path)
    spec = importlib.util.spec_from_file_location("copied_pyrtc_loop", copy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.Loop is not Loop
    assert known_config_keys(module.Loop) == known_config_keys(Loop)
