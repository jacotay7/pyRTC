import pytest

from pyRTC.component_descriptors import (
    ComponentDescriptor,
    ConfigFieldDescriptor,
    build_descriptor_catalog,
    describe_component_class,
    get_component_descriptor,
    list_component_descriptors,
    list_component_sections,
    register_component_descriptor,
    unregister_component_descriptor,
    validate_config_with_descriptor,
)
from pyRTC.hardware.SyntheticSystems import SyntheticSHWFS
from pyRTC.Loop import Loop
from pyRTC.ScienceCamera import ScienceCamera
from pyRTC.SlopesProcess import SlopesProcess
from pyRTC.Telemetry import Telemetry
from pyRTC.WavefrontCorrector import WavefrontCorrector
from pyRTC.WavefrontSensor import WavefrontSensor


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
    assert [stream.name for stream in descriptor.output_streams] == ["wfsRaw", "wfs"]


def test_loop_descriptor_exposes_expected_worker_functions():
    descriptor = get_component_descriptor("loop")

    assert descriptor is not None
    assert descriptor.component_class is Loop
    assert "standardIntegrator" in descriptor.worker_functions
    assert "leakyIntegrator" in descriptor.worker_functions


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
    assert payload["fields"]["darkCount"]["required"] is True


def test_descriptor_catalog_is_keyed_by_section():
    catalog = build_descriptor_catalog()

    assert "wfs" in catalog
    assert catalog["loop"]["section_name"] == "loop"
    assert isinstance(catalog["slopes"]["worker_functions"], list)


def test_component_descriptor_supports_field_lookup_by_name():
    descriptor = get_component_descriptor("loop")
    field_descriptor = descriptor["hardwareDelay"]

    assert field_descriptor.name == "hardwareDelay"
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
    with pytest.raises(TypeError, match="darkCount"):
        validate_config_with_descriptor("psf", {"name": "cam", "width": 32, "height": 32, "darkCount": "16", "integration": 4})


def test_descriptor_validation_rejects_missing_required_field():
    with pytest.raises(ValueError, match="numModes"):
        validate_config_with_descriptor("wfc", {"name": "dm", "numActuators": 32})


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
