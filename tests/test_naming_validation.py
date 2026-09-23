from image_gateway.naming import request_name


def test_request_name_is_deterministic_dns_safe_and_sha256_derived():
    name = request_name("same-key")
    assert name == request_name("same-key")
    assert name.startswith("igr-")
    assert len(name) <= 63
    assert name.islower()
    assert all(char.isalnum() or char == "-" for char in name)
    assert request_name("same-key") != request_name("other-key")
