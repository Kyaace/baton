"""Unit tests for System 2 continuation seam sanitization."""

import pytest
from baton.system2 import sanitize_continuation_seam


def test_empty_prior_or_new():
    assert sanitize_continuation_seam("", "def foo():\n    pass") == "def foo():\n    pass"
    assert sanitize_continuation_seam("def foo():\n    pass", "") == ""


def test_token_stutter_exact_overlap():
    prior = "def process_data(items):\n    result = []\n    for item in"
    new_chunk = "for item in items:\n        result.append(item)"
    cleaned = sanitize_continuation_seam(prior, new_chunk)
    assert cleaned == " items:\n        result.append(item)"
    assert prior + cleaned == "def process_data(items):\n    result = []\n    for item in items:\n        result.append(item)"


def test_repeated_identifier_token():
    prior = "    connection = db.get_connection("
    new_chunk = "connection()\n    return connection"
    cleaned = sanitize_continuation_seam(prior, new_chunk)
    assert cleaned == ")\n    return connection"
    assert prior + cleaned == "    connection = db.get_connection()\n    return connection"


def test_repeated_multi_word_overlap():
    prior = "def calculate_orbit(alt_km, period_sec):\n    mu = 398600.4418\n    r = 6378.137 + alt_km\n    return ("
    new_chunk = "return (mu / r) ** 0.5"
    cleaned = sanitize_continuation_seam(prior, new_chunk)
    assert cleaned == "mu / r) ** 0.5"
    assert prior + cleaned == "def calculate_orbit(alt_km, period_sec):\n    mu = 398600.4418\n    r = 6378.137 + alt_km\n    return (mu / r) ** 0.5"


def test_dropped_newline_between_statements():
    prior = "    return self._status"
    new_chunk = "def is_active(self):\n    return True"
    cleaned = sanitize_continuation_seam(prior, new_chunk)
    assert cleaned == "\ndef is_active(self):\n    return True"
    assert prior + cleaned == "    return self._status\ndef is_active(self):\n    return True"


def test_dropped_newline_after_colon():
    prior = "class TelemetryDecoder:"
    new_chunk = "    def __init__(self):\n        pass"
    cleaned = sanitize_continuation_seam(prior, new_chunk)
    assert cleaned == "\n    def __init__(self):\n        pass"
    assert prior + cleaned == "class TelemetryDecoder:\n    def __init__(self):\n        pass"


def test_normal_mid_identifier_split_not_corrupted():
    # If prior ends with partial token without overlap, it should NOT insert a newline
    prior = "    var_res"
    new_chunk = "ult = 42"
    cleaned = sanitize_continuation_seam(prior, new_chunk)
    assert cleaned == "ult = 42"
    assert prior + cleaned == "    var_result = 42"


def test_existing_newline_preserved():
    prior = "    return x\n"
    new_chunk = "def next_one():\n    pass"
    cleaned = sanitize_continuation_seam(prior, new_chunk)
    assert cleaned == "def next_one():\n    pass"
    assert prior + cleaned == "    return x\ndef next_one():\n    pass"
