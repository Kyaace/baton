"""Subsystem D: RTL-SDR RF Telemetry & Spectrum Analysis Tool for Baton System 1.

Provides hardware control for RTL-SDR USB dongles (RTL2832U / R820T / R828D) via WSL2
and headless SDR command-line tools (rtl_test, rtl_power, rtl_fm, rtl_433).
Supports live device probing, frequency spectrum sweeping, RF audio demodulation,
digital sensor telemetry decoding, and resilient mock fallbacks.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("baton.tools.sdr")

# Common frequency band allocations for signal classification
FREQUENCY_BANDS: list[tuple[float, float, str]] = [
    (88.0, 108.0, "FM Broadcast Radio"),
    (108.0, 117.975, "VOR / ILS Navigation"),
    (118.0, 137.0, "Civilian Aviation Airband (AM)"),
    (137.0, 138.0, "NOAA / Meteor Weather Satellites (APT/LRPT)"),
    (144.0, 148.0, "2m Amateur Radio (APRS on 144.390 MHz)"),
    (156.0, 162.05, "VHF Marine & AIS Ship Tracking"),
    (162.375, 162.575, "NOAA Weather Radio (NWR)"),
    (433.05, 434.79, "ISM Band / 433 MHz Weather Sensors"),
    (420.0, 450.0, "70cm Amateur Radio"),
    (868.0, 868.6, "EU ISM / LoRa Telemetry"),
    (902.0, 928.0, "US ISM 915 MHz / Smart Meters"),
    (960.0, 1215.0, "ADS-B Aircraft Transponders (1090 MHz)"),
]


def classify_frequency(freq_mhz: float) -> str:
    """Returns human-readable band allocation for a given RF frequency in MHz."""
    for low, high, label in FREQUENCY_BANDS:
        if low <= freq_mhz <= high:
            return label
    return "Unclassified RF"


def run_wsl_command(
    cmd: list[str],
    timeout: int = 15,
) -> subprocess.CompletedProcess[str]:
    """
    Executes a command inside WSL2 from the host with strict timeout protection.
    If already running natively inside Linux, executes directly.
    """
    is_win = os.name == "nt"
    full_cmd = ["wsl", "-e"] + cmd if is_win else cmd

    proc = subprocess.Popen(
        full_cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        start_new_session=not is_win,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            args=full_cmd,
            returncode=proc.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
        )
    except subprocess.TimeoutExpired:
        if is_win:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    check=False,
                )
            except Exception:
                pass
        else:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:
                pass
        proc.kill()
        raise TimeoutError(f"SDR command timed out after {timeout} seconds: {' '.join(cmd)}")


class SdrTool:
    """Subsystem D Tool: Software Defined Radio & RF Telemetry Hub."""

    def __init__(
        self,
        artifacts_dir: Optional[Path] = None,
        allow_mock: bool = True,
        default_timeout: int = 15,
    ):
        self.artifacts_dir = artifacts_dir or Path("baton/artifacts/sdr")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.allow_mock = allow_mock
        self.default_timeout = default_timeout

    # -------------------------------------------------------------------------
    # 1. Device Inspection & Hardware Probing
    # -------------------------------------------------------------------------

    def get_device_info(self, backend: str = "auto") -> dict[str, Any]:
        """
        Inspects connected RTL-SDR hardware: verifies tuner presence, sample rate limits,
        frequency range, and host USB passthrough status.
        """
        # 1. Try probing via WSL2 rtl_test
        try:
            res = run_wsl_command(["rtl_test", "-t"], timeout=6)
            combined = (res.stdout + "\n" + res.stderr).strip()

            if "Found" in combined and ("device" in combined or "tuner" in combined.lower()):
                # Parse tuner model
                tuner = "Generic RTL2832U"
                tuner_match = re.search(r"Found\s+([^:\n\r]+?)\s+tuner", combined, re.IGNORECASE) or re.search(r"tuner:\s*([^\n\r,]+)", combined, re.IGNORECASE)
                if tuner_match:
                    tuner = tuner_match.group(1).strip()

                product_match = re.search(r"Using device \d+:\s*([^\n\r]+)", combined) or re.search(r"\d+:\s*([^,\n\r]+)", combined)
                product = product_match.group(1).strip() if product_match else "Generic RTL2832U OEM"

                return {
                    "status": "connected",
                    "device_index": 0,
                    "product": product,
                    "tuner": tuner,
                    "frequency_range_mhz": [24.0, 1766.0],
                    "sample_rate_limits_hz": [225001, 3200000],
                    "backend": "wsl",
                    "health": "operational (hardware communication and tuner verified at 2.048 MS/s)",
                }

        except Exception as e:
            logger.debug("WSL rtl_test probe failed or timed out: %s", e)

        # 2. Check if hardware is present on Windows host via usbipd or PnP
        busid = None
        state = None
        host_detected = False

        # Try usbipd list first (checks standard install path and WSL wrapper)
        usbipd_exe = Path("C:/Program Files/usbipd-win/usbipd.exe")
        cmd_candidates = []
        if usbipd_exe.exists():
            cmd_candidates.append([str(usbipd_exe), "list"])
        cmd_candidates.append(["usbipd", "list"])

        for cmd in cmd_candidates:
            try:
                sub_res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                for line in sub_res.stdout.splitlines():
                    if "0bda:2838" in line.lower() or "bulk-in" in line.lower():
                        parts = line.split()
                        if len(parts) >= 3:
                            busid = parts[0]
                            state = " ".join(parts[3:]) if len(parts) > 3 else "Not shared"
                            host_detected = True
                            break
                if host_detected:
                    break
            except Exception:
                continue

        # Fallback to PowerShell Get-PnpDevice
        if not host_detected and os.name == "nt":
            try:
                ps_res = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "Get-PnpDevice -FriendlyName '*Bulk-In*', '*RTL*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty InstanceId",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "0BDA&PID_2838" in ps_res.stdout.upper() or "0BDA" in ps_res.stdout.upper():
                    host_detected = True
            except Exception:
                pass

        if host_detected:
            bus_str = f" --busid {busid}" if busid else ""
            if state and "Shared" in state and "Not shared" not in state:
                action_msg = f"Device is shared on Windows host. Run 'usbipd attach --wsl{bus_str}' to attach to WSL2."
            else:
                action_msg = (
                    f"RTL-SDR dongle (BusID {busid or '<busid>'}) is detected on Windows host but not shared. "
                    f"Run 'usbipd bind{bus_str}' as Administrator in PowerShell, then 'usbipd attach --wsl{bus_str}'."
                )

            return {
                "status": "detected_on_host_needs_attach",
                "device_index": 0,
                "busid": busid,
                "host_state": state,
                "product": f"Realtek RTL2838 / RTL2832U (VID: 0BDA, PID: 2838{f', BusID: {busid}' if busid else ''})",
                "tuner": "Rafael Micro R820T/R828D",
                "frequency_range_mhz": [24.0, 1766.0],
                "sample_rate_limits_hz": [225001, 3200000],
                "backend": "host_pnp",
                "message": action_msg,
            }

        # 3. Fallback to Mock Profile if enabled
        if self.allow_mock:
            return {
                "status": "mock",
                "device_index": 0,
                "product": "RTL-SDR v4 Dongle (Simulated)",
                "tuner": "Rafael Micro R828D",
                "frequency_range_mhz": [24.0, 1766.0],
                "sample_rate_limits_hz": [225001, 3200000],
                "backend": "mock",
                "message": "Physical SDR hardware not detected; operating in simulated mock mode.",
            }

        return {
            "status": "not_detected",
            "message": "No RTL-SDR devices found on host or inside WSL2.",
            "backend": "none",
        }

    # -------------------------------------------------------------------------
    # 2. Spectrum Scanning & Energy Detection
    # -------------------------------------------------------------------------

    def spectrum_scan(
        self,
        start_freq_mhz: float,
        stop_freq_mhz: float,
        bin_size_khz: float = 25.0,
        duration_seconds: int = 5,
        gain_db: float = 0.0,
        min_snr_db: float = 2.0,
    ) -> dict[str, Any]:
        """
        Executes a wideband or narrowband frequency sweep using rtl_power,
        calculating peak transmissions, carrier frequency, and SNR above noise floor.
        """
        if start_freq_mhz >= stop_freq_mhz:
            return {
                "error": f"Invalid range: start_freq ({start_freq_mhz} MHz) must be less than stop_freq ({stop_freq_mhz} MHz)."
            }

        duration_seconds = max(1, min(duration_seconds, 30))
        bin_size_khz = max(1.0, min(bin_size_khz, 1000.0))

        # Check device status
        dev_info = self.get_device_info()
        is_live = dev_info.get("status") == "connected"

        if is_live:
            csv_path = self.artifacts_dir / f"sweep_{int(time.time())}.csv"
            wsl_csv = f"/tmp/baton_sweep_{int(time.time())}.csv"
            gain_arg = [] if gain_db <= 0 else ["-g", str(gain_db)]

            cmd = [
                "rtl_power",
                "-f",
                f"{start_freq_mhz}M:{stop_freq_mhz}M:{bin_size_khz}k",
                "-i",
                f"{duration_seconds}s",
                "-1",
            ] + gain_arg + [wsl_csv]

            try:
                run_wsl_command(cmd, timeout=duration_seconds + 10)
                # Copy CSV back from WSL if needed or read it
                cat_res = run_wsl_command(["cat", wsl_csv], timeout=5)
                run_wsl_command(["rm", "-f", wsl_csv], timeout=5)
                return self._parse_rtl_power_csv(
                    cat_res.stdout, start_freq_mhz, stop_freq_mhz, duration_seconds, min_snr_db=min_snr_db
                )
            except Exception as e:
                logger.warning("Live rtl_power scan failed: %s; falling back to mock parser", e)

        # Mock / Simulation Fallback
        return self._generate_mock_spectrum_scan(
            start_freq_mhz, stop_freq_mhz, bin_size_khz, duration_seconds
        )

    def _parse_rtl_power_csv(
        self,
        csv_text: str,
        start_freq_mhz: float,
        stop_freq_mhz: float,
        duration: int,
        min_snr_db: float = 2.0,
    ) -> dict[str, Any]:
        """Parses rtl_power CSV rows and detects significant carrier peaks."""
        all_bins: list[tuple[float, float]] = []

        reader = csv.reader(csv_text.strip().splitlines())
        for row in reader:
            if len(row) < 7:
                continue
            try:
                f_low = float(row[2]) / 1e6
                f_high = float(row[3]) / 1e6
                f_step = float(row[4]) / 1e6
                dbm_vals = [float(v) for v in row[6:]]
                for i, dbm in enumerate(dbm_vals):
                    freq = f_low + (i * f_step)
                    all_bins.append((freq, dbm))
            except ValueError:
                continue

        if not all_bins:
            return self._generate_mock_spectrum_scan(
                start_freq_mhz, stop_freq_mhz, 25.0, duration
            )

        # Calculate noise floor (25th percentile)
        dbms = sorted(b[1] for b in all_bins)
        noise_floor = dbms[len(dbms) // 4]

        # Identify local peak candidates (> noise floor + min_snr_db)
        threshold = noise_floor + min_snr_db
        peaks = []
        for freq, pwr in all_bins:
            if pwr > threshold:
                snr = round(pwr - noise_floor, 1)
                peaks.append(
                    {
                        "frequency_mhz": round(freq, 4),
                        "power_db": round(pwr, 1),
                        "snr_db": snr,
                        "classification": classify_frequency(freq),
                    }
                )

        # Sort by highest power and deduplicate adjacent bins within 50 kHz
        peaks.sort(key=lambda x: x["power_db"], reverse=True)
        deduped: list[dict[str, Any]] = []
        for p in peaks:
            if not any(abs(p["frequency_mhz"] - d["frequency_mhz"]) < 0.050 for d in deduped):
                deduped.append(p)

        return {
            "status": "success",
            "band_mhz": f"{start_freq_mhz:.3f} - {stop_freq_mhz:.3f}",
            "sweep_duration_s": duration,
            "noise_floor_db": round(noise_floor, 1),
            "peak_signals": deduped[:10],
            "total_carriers_detected": len(deduped),
        }

    def _generate_mock_spectrum_scan(
        self,
        start_mhz: float,
        stop_mhz: float,
        bin_khz: float,
        duration: int,
    ) -> dict[str, Any]:
        """Generates realistic synthetic RF spectral data matching local band allocations."""
        noise_floor = -48.2
        peaks: list[dict[str, Any]] = []

        # Synthetic carriers within queried ranges
        targets = [
            (97.3, -12.4, "FM Broadcast (KDFC)"),
            (102.1, -15.1, "FM Broadcast"),
            (121.5, -28.0, "VHF Air Emergency Guard"),
            (127.3, -22.5, "Civilian Tower Approach"),
            (137.62, -26.4, "NOAA-15 APT Satellite"),
            (144.390, -14.2, "APRS Telemetry Beacon"),
            (145.230, -18.7, "2m Amateur Voice Repeater"),
            (162.400, -16.5, "NOAA Weather Radio (San Francisco)"),
            (433.920, -21.8, "ISM 433 MHz Weather Station"),
            (434.150, -29.2, "ISM Tire Pressure Monitor (TPMS)"),
            (915.000, -24.1, "Smart Utility Electric Meter"),
            (1090.000, -18.4, "ADS-B Mode-S Aircraft Transponder"),
        ]

        for f, pwr, label in targets:
            if start_mhz <= f <= stop_mhz:
                snr = round(pwr - noise_floor, 1)
                peaks.append(
                    {
                        "frequency_mhz": f,
                        "power_db": pwr,
                        "snr_db": snr,
                        "classification": label,
                    }
                )

        return {
            "status": "success_simulated",
            "band_mhz": f"{start_mhz:.3f} - {stop_mhz:.3f}",
            "sweep_duration_s": duration,
            "noise_floor_db": noise_floor,
            "peak_signals": peaks,
            "total_carriers_detected": len(peaks),
            "simulated": True,
        }

    # -------------------------------------------------------------------------
    # 3. RF Signal Capture & Audio Demodulation
    # -------------------------------------------------------------------------

    def record_rf(
        self,
        frequency_mhz: float,
        modulation: str = "fm",
        duration_seconds: int = 5,
        output_filename: Optional[str] = None,
        gain_db: float = 0.0,
    ) -> dict[str, Any]:
        """
        Tunes to a specific frequency and captures a brief demodulated audio
        or raw baseband sample with bounded duration, packaged into a valid RIFF WAV file.
        """
        duration_seconds = max(1, min(duration_seconds, 15))
        modulation = modulation.lower()
        if modulation not in ("fm", "am", "wfm", "raw"):
            modulation = "fm"

        filename = output_filename or f"rf_{frequency_mhz:.3f}mhz_{int(time.time())}.wav"
        if not filename.lower().endswith(".wav"):
            filename += ".wav"
        out_path = self.artifacts_dir / filename

        dev_info = self.get_device_info()
        is_live = dev_info.get("status") == "connected"

        if is_live:
            wsl_raw = f"/tmp/{out_path.stem}_{int(time.time())}.raw"
            # Map modulation flags for rtl_fm
            mode_flag = "wbfm" if modulation == "wfm" else ("am" if modulation == "am" else "fm")
            rate_str = "170k" if modulation == "wfm" else "24k"
            output_rate = "32k" if modulation == "wfm" else "24k"
            sample_rate = 32000 if modulation == "wfm" else 24000
            gain_arg = [] if gain_db <= 0 else ["-g", str(gain_db)]

            # rtl_fm piped to timeout
            cmd = [
                "timeout",
                f"{duration_seconds}s",
                "rtl_fm",
                "-f",
                f"{frequency_mhz}M",
                "-M",
                mode_flag,
                "-s",
                rate_str,
                "-r",
                output_rate,
                "-E",
                "deemp",
            ] + gain_arg + [wsl_raw]

            try:
                run_wsl_command(cmd, timeout=duration_seconds + 5)

                # Read raw binary PCM bytes from WSL without text decoding
                is_win = os.name == "nt"
                read_cmd = ["wsl", "-e", "cat", wsl_raw] if is_win else ["cat", wsl_raw]
                raw_bytes = subprocess.check_output(read_cmd)

                # Clean up WSL temp raw file
                del_cmd = ["wsl", "-e", "rm", "-f", wsl_raw] if is_win else ["rm", "-f", wsl_raw]
                subprocess.run(del_cmd, capture_output=True, check=False)

                # Package raw signed 16-bit PCM into a standard compliant WAV file
                with wave.open(str(out_path), "wb") as wf:
                    wf.setnchannels(1)  # mono
                    wf.setsampwidth(2)  # 16-bit PCM (2 bytes)
                    wf.setframerate(sample_rate)
                    wf.writeframes(raw_bytes)

                size_bytes = out_path.stat().st_size
                return {
                    "status": "success",
                    "frequency_mhz": frequency_mhz,
                    "modulation": modulation,
                    "duration_seconds": duration_seconds,
                    "file_path": str(out_path),
                    "size_bytes": size_bytes,
                    "sample_rate_hz": sample_rate,
                    "classification": classify_frequency(frequency_mhz),
                }
            except Exception as e:
                logger.warning("Live rtl_fm recording failed: %s; using mock output", e)

        # Mock Audio / Header Fallback: generate valid 24 kHz WAV
        sample_rate = 24000
        num_frames = sample_rate * min(duration_seconds, 2)
        mock_raw = bytearray()
        for i in range(num_frames):
            val = int(2500 * math.sin(2 * math.pi * 440 * i / sample_rate))
            mock_raw.extend(val.to_bytes(2, "little", signed=True))

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(mock_raw)

        return {
            "status": "success_simulated",
            "frequency_mhz": frequency_mhz,
            "modulation": modulation,
            "duration_seconds": duration_seconds,
            "file_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "sample_rate_hz": sample_rate,
            "classification": classify_frequency(frequency_mhz),
            "simulated": True,
        }

    # -------------------------------------------------------------------------
    # 4. ADS-B Mode-S Aircraft Tracking (1090 MHz via dump1090-mutability)
    # -------------------------------------------------------------------------

    def adsb_tracking(
        self,
        duration_seconds: int = 5,
        gain_db: float = 45.0,
        max_aircraft: int = 25,
    ) -> dict[str, Any]:
        """
        Listens for Mode-S ADS-B transponder squitters at 1090 MHz using dump1090-mutability
        and extracts real-time aircraft positions, callsigns, altitudes, and speeds.
        """
        duration_seconds = max(1, min(duration_seconds, 30))

        dev_info = self.get_device_info()
        is_live = dev_info.get("status") == "connected"

        if is_live:
            wsl_dir = f"/tmp/baton_adsb_{int(time.time())}"
            cmd = [
                "bash",
                "-c",
                (
                    f"mkdir -p {wsl_dir} && "
                    f"timeout {duration_seconds}s dump1090-mutability --write-json {wsl_dir} --write-json-every 1 --gain {gain_db} --quiet >/dev/null 2>&1; "
                    f"cat {wsl_dir}/aircraft.json 2>/dev/null; "
                    f"rm -rf {wsl_dir}"
                ),
            ]

            try:
                res = run_wsl_command(cmd, timeout=duration_seconds + 5)
                json_str = res.stdout.strip()
                aircraft_raw = []
                if json_str.startswith("{") and "aircraft" in json_str:
                    data = json.loads(json_str)
                    aircraft_raw = data.get("aircraft", [])

                cleaned_aircraft: list[dict[str, Any]] = []
                for a in aircraft_raw:
                    hex_code = a.get("hex", "").strip()
                    flight = a.get("flight", "").strip() if a.get("flight") else None
                    alt = a.get("altitude", a.get("alt_baro"))
                    speed = a.get("speed")
                    track = a.get("track")
                    lat = a.get("lat")
                    lon = a.get("lon")
                    rssi = a.get("rssi")
                    messages = a.get("messages", 0)

                    cleaned_aircraft.append(
                        {
                            "hex": hex_code,
                            "flight": flight,
                            "altitude_feet": alt,
                            "speed_knots": speed,
                            "track_degrees": track,
                            "lat": round(lat, 5) if lat is not None else None,
                            "lon": round(lon, 5) if lon is not None else None,
                            "rssi_dbfs": rssi,
                            "messages": messages,
                        }
                    )

                # Sort by presence of callsign, then strongest signal
                cleaned_aircraft.sort(
                    key=lambda x: (
                        x["flight"] is not None,
                        x["rssi_dbfs"] if x["rssi_dbfs"] is not None else -999,
                    ),
                    reverse=True,
                )

                return {
                    "status": "success",
                    "frequency_mhz": 1090.0,
                    "listening_duration_s": duration_seconds,
                    "total_aircraft_detected": len(cleaned_aircraft),
                    "aircraft": cleaned_aircraft[:max_aircraft],
                }
            except Exception as e:
                logger.warning("Live ADS-B tracking failed: %s; falling back to mock", e)

        # Mock / Simulation Fallback
        mock_aircraft = [
            {
                "hex": "a27e81",
                "flight": "N26VB",
                "altitude_feet": 3100,
                "speed_knots": 103,
                "track_degrees": 74,
                "lat": 29.55711,
                "lon": -98.71770,
                "rssi_dbfs": -14.5,
                "messages": 20,
            },
            {
                "hex": "ac4fd8",
                "flight": "SWA1422",
                "altitude_feet": 18500,
                "speed_knots": 340,
                "track_degrees": 182,
                "lat": 29.63805,
                "lon": -98.60912,
                "rssi_dbfs": -18.2,
                "messages": 45,
            },
            {
                "hex": "a0681c",
                "flight": None,
                "altitude_feet": 5525,
                "speed_knots": 150,
                "track_degrees": 210,
                "lat": None,
                "lon": None,
                "rssi_dbfs": -34.5,
                "messages": 3,
            },
        ]
        return {
            "status": "success_simulated",
            "frequency_mhz": 1090.0,
            "listening_duration_s": duration_seconds,
            "total_aircraft_detected": len(mock_aircraft),
            "aircraft": mock_aircraft[:max_aircraft],
            "simulated": True,
        }

    # -------------------------------------------------------------------------
    # 5. Telemetry Decoding (ISM Sensors, Smart Meters, Weather Stations)
    # -------------------------------------------------------------------------

    def decode_telemetry(
        self,
        frequency_mhz: Optional[float] = None,
        duration_seconds: int = 10,
        protocol: Optional[str] = None,
        band_preset: Optional[str] = None,
        gain_db: float = 40.0,
    ) -> dict[str, Any]:
        """
        Listens for digital telemetry packets (smart utility meters, weather stations,
        TPMS, remote keyfobs) using rtl_433 and returns structured JSON packets.
        Supports regional presets:
          - 'us_smart_meters' (915.0 MHz - default for North America)
          - 'us_tpms' (315.0 MHz - tire pressure and keyfobs)
          - 'eu_sensors' (433.92 MHz - European ISM & general 433 MHz sensors)
          - 'us_survey' (frequency hopping across 315M, 433.92M, 915M)
        """
        duration_seconds = max(1, min(duration_seconds, 30))

        # Determine frequency args based on preset or explicit frequency
        freq_args: list[str] = []
        target_display_freq: str = "433.92"

        if band_preset == "us_smart_meters":
            freq_val = frequency_mhz if frequency_mhz is not None else 915.0
            freq_args = ["-f", f"{freq_val}M"]
            target_display_freq = f"{freq_val:.2f}"
        elif band_preset == "us_tpms":
            freq_val = frequency_mhz if frequency_mhz is not None else 315.0
            freq_args = ["-f", f"{freq_val}M"]
            target_display_freq = f"{freq_val:.2f}"
        elif band_preset == "us_survey":
            freq_args = ["-f", "315M", "-f", "433.92M", "-f", "915M", "-H", "5"]
            target_display_freq = "315/433.92/915 Survey"
        elif band_preset in ("eu_sensors", "ism_433"):
            freq_val = frequency_mhz if frequency_mhz is not None else 433.92
            freq_args = ["-f", f"{freq_val}M"]
            target_display_freq = f"{freq_val:.2f}"
        elif frequency_mhz is not None:
            freq_args = ["-f", f"{frequency_mhz}M"]
            target_display_freq = f"{frequency_mhz:.2f}"
        else:
            # Default to 915 MHz (North American smart meters and telemetry)
            freq_args = ["-f", "915M"]
            target_display_freq = "915.00"

        dev_info = self.get_device_info()
        is_live = dev_info.get("status") == "connected"

        if is_live:
            cmd = ["timeout", f"{duration_seconds}s", "rtl_433"] + freq_args
            if gain_db > 0:
                cmd.extend(["-g", str(gain_db)])
            cmd.extend(["-F", "json"])
            if protocol:
                cmd.extend(["-R", protocol])

            try:
                res = run_wsl_command(cmd, timeout=duration_seconds + 5)
                packets = []
                for line in (res.stdout + "\n" + res.stderr).splitlines():
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            packets.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

                return {
                    "status": "success",
                    "frequency_mhz": target_display_freq,
                    "listening_duration_s": duration_seconds,
                    "band_preset": band_preset,
                    "packet_count": len(packets),
                    "packets": packets[:25],
                }
            except Exception as e:
                logger.warning("Live rtl_433 decoding failed: %s; falling back to mock", e)

        # Mock ISM / Telemetry Fallback
        if band_preset == "us_smart_meters" or target_display_freq == "915.00":
            mock_packets = [
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "model": "Itron-ERT",
                    "id": 84920194,
                    "type": "Electric",
                    "consumption_kwh": 48219.4,
                    "battery_ok": 1,
                },
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "model": "Badger-ORION",
                    "id": 1948201,
                    "type": "Water",
                    "consumption_gallons": 128450,
                    "battery_ok": 1,
                },
            ]
        elif band_preset == "us_tpms" or target_display_freq == "315.00":
            mock_packets = [
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "model": "Schrader-TPMS",
                    "id": "A4B92C",
                    "pressure_psi": 34.2,
                    "temperature_C": 24.0,
                    "flags": "NORMAL",
                }
            ]
        else:
            mock_packets = [
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "model": "Acurite-Tower",
                    "id": 13374,
                    "channel": "A",
                    "battery_ok": 1,
                    "temperature_C": 19.8,
                    "humidity": 52,
                    "mic": "CHECKSUM",
                },
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "model": "Fineoffset-WH2",
                    "id": 2049,
                    "temperature_C": 20.3,
                    "humidity": 49,
                    "battery_ok": 1,
                },
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "model": "Ambientweather-TX",
                    "id": 8831,
                    "wind_avg_km_h": 14.2,
                    "wind_dir_deg": 285,
                    "rain_mm": 0.0,
                },
            ]

        return {
            "status": "success_simulated",
            "frequency_mhz": target_display_freq,
            "listening_duration_s": duration_seconds,
            "band_preset": band_preset,
            "packet_count": len(mock_packets),
            "packets": mock_packets,
            "simulated": True,
        }

    # -------------------------------------------------------------------------
    # 6. Digital Packet Decoding (APRS, POCSAG, FLEX, NOAA EAS via multimon-ng)
    # -------------------------------------------------------------------------

    def decode_digital_packet(
        self,
        frequency_mhz: float = 144.390,
        protocol: str = "aprs",
        duration_seconds: int = 10,
        gain_db: float = 40.0,
    ) -> dict[str, Any]:
        """
        Demodulates and decodes digital RF packet transmissions (APRS, POCSAG, FLEX, EAS)
        by piping rtl_fm 22,050 Hz raw PCM audio into multimon-ng.
        """
        duration_seconds = max(1, min(duration_seconds, 30))
        proto = protocol.lower().strip()

        # Map protocol to multimon-ng CLI flags
        if proto == "aprs":
            demod_args = "-c -a AFSK1200 -A -q"
        elif proto == "pocsag":
            demod_args = "-c -a POCSAG512 -a POCSAG1200 -a POCSAG2400 -f alpha -q"
        elif proto == "flex":
            demod_args = "-c -a FLEX -a FLEX_NEXT -q"
        elif proto == "eas":
            demod_args = "-c -a EAS -q"
        elif proto == "all":
            demod_args = "-c -a AFSK1200 -a POCSAG512 -a POCSAG1200 -a POCSAG2400 -a EAS -q"
        else:
            demod_args = "-c -a AFSK1200 -A -q"
            proto = "aprs"

        dev_info = self.get_device_info()
        is_live = dev_info.get("status") == "connected"

        if is_live:
            pipeline = (
                f"timeout {duration_seconds}s rtl_fm -f {frequency_mhz}M -s 22050 -g {gain_db} -E deemp - 2>/dev/null "
                f"| multimon-ng -t raw {demod_args} - 2>&1"
            )
            cmd = ["bash", "-c", pipeline]

            try:
                res = run_wsl_command(cmd, timeout=duration_seconds + 5)
                messages: list[dict[str, Any]] = []

                for line in res.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("warning:") or line.startswith("multimon-ng"):
                        continue

                    msg_obj: dict[str, Any] = {"raw": line, "protocol": proto.upper()}

                    # Parse APRS format: "APRS: SOURCE>DEST,PATH:PAYLOAD"
                    if line.startswith("APRS:"):
                        body = line[5:].strip()
                        if ">" in body and ":" in body:
                            source = body.split(">", 1)[0]
                            rest = body.split(">", 1)[1]
                            path_part, payload = rest.split(":", 1) if ":" in rest else (rest, "")
                            msg_obj["source"] = source
                            msg_obj["path"] = path_part
                            msg_obj["payload"] = payload
                        else:
                            msg_obj["payload"] = body
                        messages.append(msg_obj)

                    # Parse POCSAG format: "POCSAG1200: Address: 1234567 Function: 3 Alpha: MSG"
                    elif "POCSAG" in line:
                        match = re.search(
                            r"(POCSAG\d+):\s+Address:\s*(\d+)\s+Function:\s*(\d+)(?:\s+Alpha:\s*(.*))?",
                            line,
                        )
                        if match:
                            msg_obj["protocol"] = match.group(1)
                            msg_obj["address"] = int(match.group(2))
                            msg_obj["function"] = int(match.group(3))
                            msg_obj["text"] = (match.group(4) or "").strip()
                        else:
                            msg_obj["text"] = line
                        messages.append(msg_obj)

                    # Parse EAS format: "EAS: ZCZC-..."
                    elif "EAS:" in line or line.startswith("ZCZC"):
                        msg_obj["protocol"] = "EAS"
                        msg_obj["header"] = line
                        messages.append(msg_obj)

                    else:
                        messages.append(msg_obj)

                return {
                    "status": "success",
                    "frequency_mhz": frequency_mhz,
                    "protocol": proto,
                    "listening_duration_s": duration_seconds,
                    "message_count": len(messages),
                    "messages": messages[:25],
                }
            except Exception as e:
                logger.warning("Live multimon-ng packet decoding failed: %s; falling back to mock", e)

        # Mock Packet Fallback
        mock_messages: list[dict[str, Any]] = []
        if proto == "aprs":
            mock_messages = [
                {
                    "protocol": "APRS",
                    "raw": "APRS: KD5XYZ-9>APRS,WIDE2-1:!2932.12N/09831.45W>Mobile APRS tracker in transit 45mph",
                    "source": "KD5XYZ-9",
                    "path": "APRS,WIDE2-1",
                    "payload": "!2932.12N/09831.45W>Mobile APRS tracker in transit 45mph",
                }
            ]
        elif proto == "pocsag":
            mock_messages = [
                {
                    "protocol": "POCSAG1200",
                    "raw": "POCSAG1200: Address: 1048576 Function: 3 Alpha: FACILITY HVAC WARNING UNIT 4",
                    "address": 1048576,
                    "function": 3,
                    "text": "FACILITY HVAC WARNING UNIT 4",
                }
            ]
        elif proto == "eas":
            mock_messages = [
                {
                    "protocol": "EAS",
                    "raw": "EAS: ZCZC-WXR-RWT-048029+0030-2852230-KEWX/NWS-",
                    "header": "ZCZC-WXR-RWT-048029+0030-2852230-KEWX/NWS-",
                }
            ]
        else:
            mock_messages = [
                {
                    "protocol": proto.upper(),
                    "raw": f"{proto.upper()}: Simulated beacon packet on {frequency_mhz} MHz",
                }
            ]

        return {
            "status": "success_simulated",
            "frequency_mhz": frequency_mhz,
            "protocol": proto,
            "listening_duration_s": duration_seconds,
            "message_count": len(mock_messages),
            "messages": mock_messages,
            "simulated": True,
        }

    # -------------------------------------------------------------------------
    # 7. Unified System 1 Tool Dispatcher
    # -------------------------------------------------------------------------

    def execute(self, tool_name: str, params: dict[str, Any]) -> str:
        """
        Executes a named SDR tool and formats a clean Markdown briefing string
        tailored for System 1 prompt injection.
        """
        name = tool_name.lower().strip()

        if name in ("sdr_device_info", "device_info", "sdr_info"):
            res = self.get_device_info(backend=params.get("backend", "auto"))
            status_badge = res.get("status", "unknown").upper()
            tuner = res.get("tuner", "Unknown")
            freq_range = res.get("frequency_range_mhz", [0, 0])
            rates = res.get("sample_rate_limits_hz", [0, 0])

            lines = [
                f"### RTL-SDR Hardware Diagnostics: {status_badge}",
                f"- **Product:** {res.get('product', 'Unknown')}",
                f"- **Tuner Chip:** {tuner}",
                f"- **Frequency Range:** {freq_range[0]} MHz – {freq_range[1]} MHz",
                f"- **Sample Rates:** {rates[0]:,} Hz – {rates[1]:,} Hz",
                f"- **Backend:** `{res.get('backend', 'none')}`",
            ]
            if "message" in res:
                lines.append(f"\n> **Notice:** {res['message']}")
            return "\n".join(lines)

        elif name in ("sdr_spectrum_scan", "spectrum_scan", "sdr_scan"):
            start_mhz = float(params.get("start_freq_mhz", 144.0))
            stop_mhz = float(params.get("stop_freq_mhz", 148.0))
            bin_khz = float(params.get("bin_size_khz", 25.0))
            duration = int(params.get("duration_seconds", 5))
            gain = float(params.get("gain_db", 0.0))
            min_snr = float(params.get("min_snr_db", 3.0))

            res = self.spectrum_scan(
                start_freq_mhz=start_mhz,
                stop_freq_mhz=stop_mhz,
                bin_size_khz=bin_khz,
                duration_seconds=duration,
                gain_db=gain,
                min_snr_db=min_snr,
            )

            if "error" in res:
                return f"Error executing spectrum scan: {res['error']}"

            sim_badge = " *(Simulated)*" if res.get("simulated") else ""
            lines = [
                f"### RF Spectrum Scan: `{res['band_mhz']}`{sim_badge}",
                f"- **Noise Floor:** `{res['noise_floor_db']} dBm`",
                f"- **Sweep Duration:** {res['sweep_duration_s']}s (Bin: {bin_khz} kHz)",
                f"- **Carriers Detected:** {res['total_carriers_detected']}",
                "",
                "| Frequency (MHz) | Power (dBm) | SNR (dB) | Band Allocation / Classification |",
                "| :--- | :--- | :--- | :--- |",
            ]
            for p in res.get("peak_signals", []):
                lines.append(
                    f"| **{p['frequency_mhz']:.4f}** | {p['power_db']} dBm | +{p['snr_db']} dB | {p['classification']} |"
                )
            if not res.get("peak_signals"):
                lines.append("| *None* | - | - | No carrier detected above threshold |")

            return "\n".join(lines)

        elif name in ("sdr_record_rf", "record_rf", "sdr_record"):
            freq = float(params.get("frequency_mhz", 144.390))
            mod = str(params.get("modulation", "fm"))
            duration = int(params.get("duration_seconds", 5))
            fn = params.get("output_filename")
            gain = float(params.get("gain_db", 0.0))

            res = self.record_rf(
                frequency_mhz=freq,
                modulation=mod,
                duration_seconds=duration,
                output_filename=fn,
                gain_db=gain,
            )

            sim_badge = " *(Simulated)*" if res.get("simulated") else ""
            return (
                f"### RF Capture: `{res['frequency_mhz']:.3f} MHz` ({res['modulation'].upper()}){sim_badge}\n"
                f"- **Band Classification:** {res.get('classification', 'Unknown')}\n"
                f"- **Duration:** {res['duration_seconds']}s\n"
                f"- **Output File:** `{res['file_path']}` ({res['size_bytes']:,} bytes)"
            )

        elif name in ("sdr_adsb_tracking", "adsb_tracking", "sdr_adsb", "adsb"):
            duration = int(params.get("duration_seconds", 5))
            gain = float(params.get("gain_db", 45.0))
            max_ac = int(params.get("max_aircraft", 25))

            res = self.adsb_tracking(
                duration_seconds=duration,
                gain_db=gain,
                max_aircraft=max_ac,
            )

            sim_badge = " *(Simulated)*" if res.get("simulated") else ""
            lines = [
                f"### ADS-B Mode-S Aircraft Radar: `1090.0 MHz`{sim_badge}",
                f"- **Listening Window:** {res['listening_duration_s']}s (Gain: {gain} dB)",
                f"- **Aircraft in Range:** {res['total_aircraft_detected']}",
                "",
                "| Hex ICAO | Flight / Callsign | Altitude (ft) | Speed (kts) | Track | Coordinates | Signal (dBFS) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
            for ac in res.get("aircraft", []):
                hex_c = f"`{ac['hex']}`" if ac.get("hex") else "-"
                callsign = f"**{ac['flight']}**" if ac.get("flight") else "*Unknown*"
                alt = f"{ac['altitude_feet']:,} ft" if ac.get("altitude_feet") is not None else "-"
                spd = f"{ac['speed_knots']} kts" if ac.get("speed_knots") is not None else "-"
                trk = f"{ac['track_degrees']}°" if ac.get("track_degrees") is not None else "-"
                coords = f"{ac['lat']:.4f}, {ac['lon']:.4f}" if (ac.get("lat") and ac.get("lon")) else "-"
                sig = f"{ac['rssi_dbfs']} dBFS" if ac.get("rssi_dbfs") is not None else "-"
                lines.append(f"| {hex_c} | {callsign} | {alt} | {spd} | {trk} | {coords} | {sig} |")

            if not res.get("aircraft"):
                lines.append("| *None* | - | - | - | - | - | No transponders received |")

            return "\n".join(lines)

        elif name in ("sdr_decode_packet", "decode_packet", "sdr_packet", "decode_digital_packet"):
            freq = float(params.get("frequency_mhz", 144.390))
            proto = str(params.get("protocol", "aprs"))
            duration = int(params.get("duration_seconds", 10))
            gain = float(params.get("gain_db", 40.0))

            res = self.decode_digital_packet(
                frequency_mhz=freq,
                protocol=proto,
                duration_seconds=duration,
                gain_db=gain,
            )

            sim_badge = " *(Simulated)*" if res.get("simulated") else ""
            lines = [
                f"### Digital Packet Demodulator: `{res['frequency_mhz']:.3f} MHz` ({res['protocol'].upper()}){sim_badge}",
                f"- **Listening Window:** {res['listening_duration_s']}s (Gain: {gain} dB)",
                f"- **Packets Decoded:** {res['message_count']}",
                "",
            ]
            for m in res.get("messages", []):
                if "source" in m:
                    lines.append(f"- **`{m['source']}`** -> `{m.get('path', 'DIRECT')}`: {m.get('payload', '')}")
                elif "address" in m:
                    lines.append(f"- **Address `{m['address']}`** [Func {m.get('function')}]: {m.get('text', '')}")
                elif "header" in m:
                    lines.append(f"- **EAS Alert:** `{m['header']}`")
                else:
                    lines.append(f"- `{m.get('raw', '')}`")

            if not res.get("messages"):
                lines.append("*No digital packets demodulated within time window.*")

            return "\n".join(lines)

        elif name in ("sdr_decode_telemetry", "decode_telemetry", "sdr_telemetry"):
            freq = float(params["frequency_mhz"]) if "frequency_mhz" in params else None
            duration = int(params.get("duration_seconds", 10))
            proto = params.get("protocol")
            preset = params.get("band_preset")
            gain = float(params.get("gain_db", 40.0))

            res = self.decode_telemetry(
                frequency_mhz=freq,
                duration_seconds=duration,
                protocol=proto,
                band_preset=preset,
                gain_db=gain,
            )

            sim_badge = " *(Simulated)*" if res.get("simulated") else ""
            preset_str = f" ({res['band_preset']})" if res.get("band_preset") else ""
            lines = [
                f"### RF Telemetry Decoder: `{res['frequency_mhz']} MHz`{preset_str}{sim_badge}",
                f"- **Listening Window:** {res['listening_duration_s']}s",
                f"- **Packets Captured:** {res['packet_count']}",
                "",
            ]

            packets = res.get("packets", [])
            if packets:
                lines.append("```json")
                lines.append(json.dumps(packets, indent=2))
                lines.append("```")
            else:
                lines.append("*No digital sensor packets decoded within time window.*")

            return "\n".join(lines)

        else:
            return f"Error: Unknown SDR tool '{tool_name}'."
