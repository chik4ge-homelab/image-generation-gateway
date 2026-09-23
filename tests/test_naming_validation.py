import pytest

from image_gateway.models import ImageJobCreate
from image_gateway.naming import request_name
from image_gateway.validation import StrictJSONError, parse_optimizer_output


def test_request_name_is_deterministic_dns_safe_and_sha256_derived():
    name = request_name("same-key")
    assert name == request_name("same-key")
    assert name.startswith("igr-")
    assert len(name) <= 63
    assert name.islower()
    assert all(char.isalnum() or char == "-" for char in name)
    assert request_name("same-key") != request_name("other-key")


def test_input_validation_defaults_and_bounds():
    value = ImageJobCreate(idempotencyKey="k", prompt="prompt")
    assert value.aspect_ratio == "1:1"
    assert value.steps == 40
    assert value.seed == 42

    with pytest.raises(ValueError):
        ImageJobCreate(idempotencyKey="k", prompt="prompt", aspectRatio="0:1")
    with pytest.raises(ValueError):
        ImageJobCreate(idempotencyKey="k", prompt="prompt", steps=0)


def test_optimizer_output_is_strict_and_limited():
    result = parse_optimizer_output('{"rewritten_prompt":"cat","wh_ratio":"16:9"}')
    assert result.rewritten_prompt == "cat"
    assert result.wh_ratio == "16:9"
    result = parse_optimizer_output(
        'thinking...\n{"rewritten_prompt":"cat","wh_ratio":"16:9"}\n'
    )
    assert result.rewritten_prompt == "cat"

    with pytest.raises(StrictJSONError):
        parse_optimizer_output('{"rewritten_prompt":"cat","wh_ratio":"16:9","extra":1}')
    with pytest.raises(StrictJSONError):
        parse_optimizer_output('{"rewritten_prompt":"cat","wh_ratio":"16:9","wh_ratio":"1:1"}')
    with pytest.raises(StrictJSONError):
        parse_optimizer_output("x" * 4097)
    with pytest.raises(StrictJSONError):
        parse_optimizer_output("completion without json")
