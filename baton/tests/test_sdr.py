"""Unit tests for Subsystem D: SdrTool and SDR Tool Shelf Bundle (baton/tools/sdr.py)."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from baton.models import Task, TaskType, Tier
from baton.system1 import System1Runner
from baton.tools.sdr import SdrTool, classify_frequency


@pytest.fixture
def temp_sdr_dir(tmp_path: Path) -> Path:
    sdr_dir = tmp_path / "artifacts_sdr"
    sdr_dir.mkdir(parents=True, exist_ok=True)
    return sdr_dir


@pytest.fixture
def sdr_tool(temp_sdr_dir: Path) -> SdrTool:
    return SdrTool(artifacts_dir=temp_sdr_dir, allow_mock=True, default_timeout=5)


def test_classify_frequency():
    assert "FM Broadcast" in classify_frequency(101.5)
    assert "Aviation" in classify_frequency(125.0)
    assert "NOAA" in classify_frequency(137.5)
    assert "Amateur" in classify_frequency(144.390)
    assert "Marine" in classify_frequency(156.8)
    assert "NOAA Weather Radio" in classify_frequency(162.400)
    assert "ISM" in classify_frequency(433.92)
    assert "ADS-B" in classify_frequency(1090.0)
    assert classify_frequency(2500.0) == "Unclassified RF"


def test_sdr_device_info(sdr_tool: SdrTool):
    info = sdr_tool.get_device_info()
    assert "status" in info
    assert "tuner" in info
    assert info["frequency_range_mhz"][0] <= 24.0
    assert info["frequency_range_mhz"][1] >= 1700.0
    assert info["status"] in ("connected", "detected_on_host_needs_attach", "mock")


def test_spectrum_scan_range_validation(sdr_tool: SdrTool):
    # Invalid range: start >= stop
    res = sdr_tool.spectrum_scan(start_freq_mhz=150.0, stop_freq_mhz=140.0)
    assert "error" in res
    assert "Invalid range" in res["error"]


def test_spectrum_scan_mock(sdr_tool: SdrTool):
    with patch.object(sdr_tool, "get_device_info", return_value={"status": "mock"}):
        res = sdr_tool.spectrum_scan(start_freq_mhz=144.0, stop_freq_mhz=148.0, duration_seconds=3)
        assert res["status"] in ("success", "success_simulated")
        assert "144.000 - 148.000" in res["band_mhz"]
        assert res["noise_floor_db"] < 0
        assert "peak_signals" in res
        assert len(res["peak_signals"]) >= 1

        aprs_peak = next((p for p in res["peak_signals"] if abs(p["frequency_mhz"] - 144.390) < 0.01), None)
        assert aprs_peak is not None
        assert aprs_peak["snr_db"] > 0
        assert "APRS" in aprs_peak["classification"]


def test_spectrum_scan_csv_parsing(sdr_tool: SdrTool):
    # Sample rtl_power CSV output format:
    # 2026-10-12, 12:00:00, 144000000, 146000000, 500000, 100, -50.1, -48.2, -12.5, -49.0
    sample_csv = (
        "2026-10-12, 12:00:00, 144000000, 146000000, 500000, 100, -52.0, -49.5, -14.2, -50.1\n"
        "2026-10-12, 12:00:00, 146000000, 148000000, 500000, 100, -51.0, -18.3, -50.2, -52.4\n"
    )

    parsed = sdr_tool._parse_rtl_power_csv(sample_csv, 144.0, 148.0, 5, min_snr_db=3.0)
    assert parsed["status"] == "success"
    assert parsed["total_carriers_detected"] >= 2

    # Check detected peaks
    peaks = parsed["peak_signals"]
    assert any(p["power_db"] == -14.2 for p in peaks)
    assert any(p["power_db"] == -18.3 for p in peaks)


def test_spectrum_scan_low_snr_threshold(sdr_tool: SdrTool):
    # Tests that min_snr_db=3.0 captures NOAA-style weak peaks (+4.5 dB SNR)
    sample_csv = (
        "2026-10-12, 12:00:00, 162400000, 162600000, 25000, 100, -12.0, -12.0, -7.5, -12.0\n"
    )
    # With min_snr_db=3.0, peak at index 2 (162.45 MHz) at -7.5 dB (noise -12.0, SNR 4.5 dB) should be detected
    parsed = sdr_tool._parse_rtl_power_csv(sample_csv, 162.4, 162.6, 5, min_snr_db=3.0)
    assert parsed["total_carriers_detected"] == 1
    assert parsed["peak_signals"][0]["snr_db"] == 4.5
    assert "NOAA Weather Radio" in parsed["peak_signals"][0]["classification"]


def test_record_rf_bounded_and_file_generation(sdr_tool: SdrTool):
    with patch.object(sdr_tool, "get_device_info", return_value={"status": "mock"}):
        res = sdr_tool.record_rf(
            frequency_mhz=144.390,
            modulation="fm",
            duration_seconds=2,
            output_filename="test_aprs.wav",
        )
        assert res["status"] in ("success", "success_simulated")
        assert res["frequency_mhz"] == 144.390
        assert res["duration_seconds"] == 2
        assert Path(res["file_path"]).exists()
        assert res["size_bytes"] > 0

        # Verify valid WAV header format
        with wave.open(res["file_path"], "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            assert wf.getnframes() > 0


def test_decode_telemetry_mock(sdr_tool: SdrTool):
    with patch.object(sdr_tool, "get_device_info", return_value={"status": "mock"}):
        res = sdr_tool.decode_telemetry(frequency_mhz=433.92, duration_seconds=5)
        assert res["status"] in ("success", "success_simulated")
        assert res["packet_count"] >= 1
        assert "packets" in res
        sample = res["packets"][0]
        assert "temperature_C" in sample or "model" in sample


def test_decode_telemetry_presets(sdr_tool: SdrTool):
    with patch.object(sdr_tool, "get_device_info", return_value={"status": "mock"}):
        # 1. US Smart meters preset (915 MHz)
        res_meters = sdr_tool.decode_telemetry(band_preset="us_smart_meters", duration_seconds=5)
        assert res_meters["frequency_mhz"] == "915.00"
        assert res_meters["packet_count"] >= 1
        assert any("ERT" in p.get("model", "") or "Badger" in p.get("model", "") for p in res_meters["packets"])

        # 2. US TPMS tire pressure preset (315 MHz)
        res_tpms = sdr_tool.decode_telemetry(band_preset="us_tpms", duration_seconds=5)
        assert res_tpms["frequency_mhz"] == "315.00"
        assert res_tpms["packet_count"] >= 1
        assert any("TPMS" in p.get("model", "") for p in res_tpms["packets"])


def test_adsb_tracking_mock(sdr_tool: SdrTool):
    with patch.object(sdr_tool, "get_device_info", return_value={"status": "mock"}):
        res = sdr_tool.adsb_tracking(duration_seconds=5)
        assert res["status"] in ("success", "success_simulated")
        assert res["frequency_mhz"] == 1090.0
        assert res["total_aircraft_detected"] >= 1
        assert "aircraft" in res
        first = res["aircraft"][0]
        assert "hex" in first
        assert "altitude_feet" in first


def test_decode_digital_packet_mock(sdr_tool: SdrTool):
    with patch.object(sdr_tool, "get_device_info", return_value={"status": "mock"}):
        # 1. APRS 144.390 MHz
        aprs_res = sdr_tool.decode_digital_packet(frequency_mhz=144.390, protocol="aprs", duration_seconds=5)
        assert aprs_res["status"] in ("success", "success_simulated")
        assert aprs_res["protocol"] == "aprs"
        assert aprs_res["message_count"] >= 1
        assert aprs_res["messages"][0]["source"] == "KD5XYZ-9"

        # 2. POCSAG 152.000 MHz
        pocsag_res = sdr_tool.decode_digital_packet(frequency_mhz=152.000, protocol="pocsag", duration_seconds=5)
        assert pocsag_res["status"] in ("success", "success_simulated")
        assert pocsag_res["protocol"] == "pocsag"
        assert pocsag_res["message_count"] >= 1
        assert pocsag_res["messages"][0]["address"] == 1048576

        # 3. NOAA EAS 162.550 MHz
        eas_res = sdr_tool.decode_digital_packet(frequency_mhz=162.550, protocol="eas", duration_seconds=5)
        assert eas_res["status"] in ("success", "success_simulated")
        assert eas_res["protocol"] == "eas"
        assert eas_res["message_count"] >= 1


def test_execute_dispatcher(sdr_tool: SdrTool):
    with patch.object(sdr_tool, "get_device_info", return_value={"status": "mock"}):
        # 1. Device Info
        md_info = sdr_tool.execute("sdr_device_info", {})
        assert "### RTL-SDR Hardware Diagnostics" in md_info
        assert "Frequency Range" in md_info

        # 2. Spectrum Scan
        md_scan = sdr_tool.execute(
            "sdr_spectrum_scan",
            {"start_freq_mhz": 144.0, "stop_freq_mhz": 148.0},
        )
        assert "### RF Spectrum Scan:" in md_scan
        assert "Noise Floor" in md_scan
        assert "| Frequency (MHz) |" in md_scan

        # 3. ADS-B Tracking
        md_adsb = sdr_tool.execute("sdr_adsb_tracking", {"duration_seconds": 5})
        assert "### ADS-B Mode-S Aircraft Radar: `1090.0 MHz`" in md_adsb
        assert "Hex ICAO" in md_adsb
        assert "Flight / Callsign" in md_adsb

        # 4. Digital Packet Demodulator
        md_packet = sdr_tool.execute("sdr_decode_packet", {"frequency_mhz": 144.390, "protocol": "aprs"})
        assert "### Digital Packet Demodulator: `144.390 MHz`" in md_packet
        assert "KD5XYZ-9" in md_packet

    # 5. Record RF
    md_rec = sdr_tool.execute(
        "sdr_record_rf",
        {"frequency_mhz": 162.400, "modulation": "fm", "duration_seconds": 2},
    )
    assert "### RF Capture: `162.400 MHz`" in md_rec
    assert "NOAA Weather Radio" in md_rec

    # 6. Telemetry
    md_telem = sdr_tool.execute(
        "sdr_decode_telemetry",
        {"frequency_mhz": 433.92, "duration_seconds": 5},
    )
    assert "### RF Telemetry Decoder: `433.92 MHz`" in md_telem

    # 7. Unknown tool
    md_unk = sdr_tool.execute("sdr_invalid_tool", {})
    assert "Error: Unknown SDR tool" in md_unk


def test_system1_sdr_task_execution(tmp_path: Path):
    dossier_dir = tmp_path / "dossiers"
    runner = System1Runner(
        artifacts_dir=dossier_dir,
        model_name="qwen3:14b",
    )

    task = Task(
        task_id="recon_rf_scan",
        job_id="job_rf_test",
        title="Survey Local 2m Band for Telemetry",
        tier=Tier.SYSTEM_1,
        task_type=TaskType.FAST_RECON,
        tool_name="sdr_spectrum_scan",
        tool_params={"start_freq_mhz": 144.0, "stop_freq_mhz": 148.0},
    )

    with patch.object(runner, "generate") as mock_gen:
        mock_gen.return_value = {
            "response": "## 2m RF Survey Dossier\n- Detected APRS carrier on 144.390 MHz with +34 dB SNR.\n- Band noise floor: -48.2 dBm.",
            "tokens_generated": 45,
            "elapsed_seconds": 1.2,
        }

        summary, art_path, tokens, elapsed = runner.execute_task(task)

        assert "2m RF Survey Dossier" in summary
        assert art_path is not None
        assert Path(art_path).exists()
        assert tokens == 45

        # Verify prompt received the RF Recon Persona
        call_args = mock_gen.call_args
        system_prompt = call_args[1].get("system_prompt") or call_args[0][1]
        assert "RF Telemetry & Signal Intelligence Scout" in system_prompt
        assert "Raw Tool Reconnaissance Data" in call_args[0][0]
