"""De-identification pipeline tests (privacy-critical)."""

from __future__ import annotations

from ndis.deidentify import deidentify


def test_mobile_fully_redacted():
    out, found = deidentify("call mum on 0412 345 678")
    assert "0412" not in out and "345 678" not in out
    assert "PHONE" in found


def test_landline_and_email_and_ndis():
    out, found = deidentify("ph 03 9123 4567, mail a.b@example.com, NDIS-123456")
    assert "03 9123 4567" not in out
    assert "a.b@example.com" not in out
    assert "123456" not in out
    assert {"PHONE", "EMAIL", "NDIS_NUM"} <= set(found)


def test_clean_text_unchanged():
    text = "Routine session at the community centre."
    out, found = deidentify(text)
    assert out == text
    assert found == []
