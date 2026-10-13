# Architectural Proposal: NOAA Weather Satellite APT Image Decoding

**Status:** Proposed  
**Author:** Baton Systems Team  
**Date:** 2026-10-12  
**Target:** RTL-SDR Subsystem / Future Dedicated `'satellite'` Tool Bundle  

---

## 1. Executive Summary

This proposal evaluates and specifies the architectural pipeline for receiving, demodulating, and decoding **Automatic Picture Transmission (APT)** weather imagery transmitted by polar-orbiting NOAA weather satellites (**NOAA-15, NOAA-18, NOAA-19**) using an RTL-SDR USB dongle on Windows/WSL2.

The pipeline automates:
1. **Orbital Pass Prediction & TLE Propagation:** Calculating Acquisition of Signal (AOS), Loss of Signal (LOS), and maximum elevation for local ground coordinates.
2. **RF Baseband Capture:** Recording 60 kHz bandwidth FM signals using `rtl_fm` with Doppler shift compensation.
3. **Audio Conditioning & Filtering:** Resampling and bandpassing raw audio with `sox` to isolate the 2,400 Hz AM subcarrier.
4. **Image Synthesis & Map Projection:** Decoding image scanlines and overlaying country/state geographic boundaries using Martin Bernardi's `noaa-apt` decoder.

---

## 2. Satellite Constellation & Transmission Profiles

NOAA POES (Polar Operational Environmental Satellites) operate in sun-synchronous Low Earth Orbit (~850 km altitude, 98.7° inclination, ~102-minute orbital period). Each satellite makes approximately 2–4 daytime passes and 2–4 nighttime passes over any given mid-latitude location daily.

| Satellite | Status | APT Frequency | Transmit Power | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **NOAA-15** | Active | `137.620 MHz` | ~5 Watts | Morning/evening pass profile; AVHRR motor anomalies occasionally affect image sync. |
| **NOAA-18** | Active | `137.9125 MHz` | ~5 Watts | Afternoon pass profile; highly reliable transmitter and AVHRR scan sync. |
| **NOAA-19** | Active | `137.100 MHz` | ~5 Watts | Afternoon pass profile; primary weather imaging satellite for North America. |

*(Note: Russian Meteor-M satellites operate on 137.100 MHz and 137.900 MHz using digital LRPT (QPSK modulation at 72k/80k baud), which requires a dedicated QPSK demodulator like SatDump rather than analog APT).*

---

## 3. Signal Characteristics & Physics

### 3.1 APT Protocol Specification
- **Modulation:** Analog Frequency Modulation (FM), RF channel bandwidth ~34–40 kHz.
- **Video Subcarrier:** 2,400 Hz amplitude-modulated (AM) audio tone.
- **Frame Rate:** 2 lines per second (0.5 seconds per scanline, 4,160 words/second = 2,080 pixels per line).
- **Channels:** Two simultaneous image columns:
  - **Channel A:** Visible light (0.58–0.68 µm) during day passes, or mid-infrared at night.
  - **Channel B:** Thermal infrared (10.3–11.3 µm) for cloud-top temperature measurement.
- **Sync Signals:** Distinct 7-pulse square wave sync trains precede each line (Sync A: 1,040 Hz square wave for Channel A; Sync B: 832 Hz square wave for Channel B).

### 3.2 Doppler Shift
Because satellites travel at ~7.5 km/s relative to ground observers:
- Frequency at AOS (rising satellite): $+3.5\text{ kHz}$ to $+4.0\text{ kHz}$ above nominal.
- Frequency at TCA (time of closest approach): $0\text{ kHz}$ shift.
- Frequency at LOS (setting satellite): $-3.5\text{ kHz}$ to $-4.0\text{ kHz}$ below nominal.

> [!IMPORTANT]
> A standard narrow FM filter (12–24 kHz) will clip the APT signal as it shifts. The receiver bandwidth in `rtl_fm` must be set to at least **60 kHz** (`-s 60k`) to accommodate both the 38 kHz signal deviation and the Doppler shift without requiring active frequency tuning during the pass.

---

## 4. Hardware & Antenna Considerations

### 4.1 Antenna Polarization
- NOAA APT signals are **Right-Hand Circularly Polarized (RHCP)**.
- **Stock RTL-SDR Antenna:** A vertical telescoping monopole is linearly polarized. While usable for high-elevation passes ($>50^\circ$) near a window or outdoors, signal nulls and cross-polarization fading will cause horizontal static bars across the image.
- **Recommended Antenna (V-Dipole):** A simple horizontal V-Dipole consists of two elements cut to **53.4 cm** ($\lambda / 4$ at 137.5 MHz) spread at an angle of **120°** in the horizontal plane, oriented North-South. This offers an almost omnidirectional pattern toward the sky and impedance matches directly to 50Ω coax with minimal fading.

---

## 5. End-to-End Software Pipeline

### Stage 1: Pass Prediction & Scheduling
To know when to record, TLEs (Two-Line Element sets) are downloaded from Celestrak:
```text
https://celestrak.org/NORAD/elements/weather.txt
```
Using Python's `skyfield` or `sgp4`:
1. Calculate the satellite's topocentric coordinates (azimuth, elevation, range) relative to ground latitude/longitude.
2. Identify passes where maximum elevation $> 25^\circ$.
3. Compute pass start ($T_{\text{AOS}}$), duration (typically 600–750 seconds), and end ($T_{\text{LOS}}$).

### Stage 2: RF Capture (`rtl_fm`)
```bash
timeout 720s rtl_fm -f 137.100M -M fm -s 60k -g 45.0 -p 0 /tmp/noaa19_raw.raw
```
- `-f 137.100M`: Center carrier frequency.
- `-M fm`: Standard FM demodulation.
- `-s 60k`: 60,000 S/s receiver bandwidth (accommodates 40 kHz channel + Doppler).
- `-g 45.0`: Fixed tuner gain (avoids AGC pumping from noise).

### Stage 3: Audio Resampling & Filtering (`sox`)
```bash
sox -t raw -r 60000 -e signed-integer -b 16 -c 1 /tmp/noaa19_raw.raw \
    -r 11025 /tmp/noaa19_clean.wav rate 11025 sinc 800-4000
```
- `-r 11025`: Downsamples to standard APT sample rate ($11,025\text{ Hz}$).
- `sinc 800-4000`: Bandpass filter isolating the 2,400 Hz AM subcarrier and stripping high-frequency FM white noise and low-frequency electrical hum.

### Stage 4: Decoding & Projection (`noaa-apt`)
Using Martin Bernardi's `noaa-apt` binary:
```bash
noaa-apt /tmp/noaa19_clean.wav -o baton/artifacts/sdr/noaa19_pass.png
```
**Features of `noaa-apt`:**
- Automatic synchronization detection via cross-correlation with Sync A/B pulses.
- Slant correction to compensate for sound card clock drift.
- Histogram equalization and contrast stretching.
- True geographic border overlay (coastlines, state lines) mapped from satellite ephemeris and file timestamp.

---

## 6. Architectural Options for Baton

### Option A: Integrate into Existing `'sdr'` Tool Bundle
Add two new tools to the existing `'sdr'` bundle:
- `sdr_satellite_passes(lat, lon, min_elevation=20)`: Returns upcoming satellite passes in the next 24 hours.
- `sdr_capture_satellite(satellite="NOAA-19", duration_seconds=700, gain_db=45.0)`: Executes the recording, SoX filtering, and image decoding pipeline.
- **Pros:** Keeps all RTL-SDR radio hardware control under one unified tool bundle.
- **Cons:** Increases the tool schema size of the `'sdr'` bundle for users who only want terrestrial radio/telemetry/ADS-B.

### Option B: Dedicated `'satellite'` Tool Bundle (Recommended for Scale)
Create a new dedicated bundle on the Baton Tool Shelf:
- **Bundle Name:** `'satellite'`
- **Tools Included:**
  1. `sat_tle_update`: Syncs NORAD TLE orbital ephemeris from Celestrak.
  2. `sat_predict_passes`: Calculates upcoming satellite windows (NOAA, ISS, Meteor-M, CubeSats) with elevation, azimuth, and Doppler plots.
  3. `sat_capture_apt`: Records and decodes NOAA weather satellite passes to PNG imagery.
  4. `sat_capture_telemetry`: Listens for amateur satellite beacons (e.g., AO-91, AO-92, Fox-1, ISS APRS on 145.825 MHz).
- **Chat Slash Commands:**
  - `/bundle satellite`: Mounts satellite tools onto the active toolbelt.
  - `/passes`: Instantly shows next overhead passes.
  - `/satellite <name>`: Initiates timed tracking or immediate capture.
- **Pros:** 
  - Clean separation of concerns between terrestrial RF scanning (FM, ISM, ADS-B) and aerospace orbit mechanics / ephemeris tracking.
  - Leaves room for multi-satellite support (ISS voice/APRS, amateur CubeSats, weather satellites).
  - Keeps prompt context compact when doing ground-based RF work.

---

## 7. Next Steps & Prerequisites

1. **WSL Dependencies:**
   - Install `noaa-apt` deb package (`v1.4.1` amd64).
   - Ensure `sox` remains configured.
2. **Python Orbit Tracker:**
   - Evaluate `skyfield` or `ephem` in Python to calculate local topocentric passes without external internet reliance (caching TLEs for 14–30 days).
3. **Decision Point:** Confirm whether to integrate directly into `'sdr'` or launch as a dedicated `'satellite'` bundle on the Baton shelf.
