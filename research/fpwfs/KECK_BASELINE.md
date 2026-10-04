# Keck II legacy (pre-HAKA) NGS AO: verified parameters

Compiled 2026-10-03 for the end-to-end simulation baseline. **Baseline = OCAM2K-era legacy Keck II**: 20x20 Shack-Hartmann (SH) wavefront sensor, Xinetics 349-actuator DM, Microgate/GPU "NRTC" real-time controller, released for Keck II science in early 2024 and replaced later by HAKA.

The system went through three eras. Each row says which era its value comes from:

- **[MITLL]**: the original 1999–2007 system. AOA camera with an MIT/LL 64x64 CCD, Mercury i860 wavefront controller (WFC), up to 672 Hz.
- **[CCD39]**: 2007 to about 2023. SciMeasure "Little Joe" camera with an e2v CCD39, Microgate FPGA/DSP real-time controller (RTC, called MGAOS / NGWFC), up to 2406 Hz.
- **[OCAM]**: about 2022–2024 onward. First Light OCAM2K EMCCD camera, Microgate/SUT/ANU GPU NRTC.

**Confidence scale:**
- **H**: verbatim in a source I opened.
- **M**: from a source I opened, but the era or context is ambiguous, or the value is derived by simple arithmetic.
- **L/[unverified]**: inferred, or not found in any source I opened.

## Sources opened (key)
- **[WIZ06]** Wizinowich et al. 2006, PASP 118, 297 (LGS AO overview). https://www2.keck.hawaii.edu/optics/aodocs/PWetal2006PASP118_297.pdf
- **[VD04]** van Dam, Le Mignant & Macintosh 2004, Appl. Opt. 43, 5458, doi:10.1364/AO.43.005458. I read the OSTI preprint UCRL-JRNL-201902 (https://www.osti.gov/servlets/purl/859914) through OCR.
- **[VD04b]** van Dam et al. 2004 SPIE, "Characterization of AO at Keck: part II". https://www.osti.gov/servlets/purl/15014204
- **[VD06]** van Dam et al. 2006, PASP 118, 310 (LGS performance). https://www2.keck.hawaii.edu/optics/aodocs/PASP06_MvD.pdf
- **[KAON194]** Wizinowich et al. 2000 SPIE, "NGS AO first year". https://www2.keck.hawaii.edu/optics/aodocs/kaon194.pdf
- **[BRASE98]** Brase et al. 1998, UCRL-JC-130919, "Wavefront control system for Keck". https://www.osti.gov/servlets/purl/302840
- **[KAON051]** Wizinowich 1995/98, "Keck AO error budget". https://www2.keck.hawaii.edu/optics/aowg/docs/kaon051.pdf
- **[KAON263]** Wizinowich et al. 2004, "AO developments at Keck". https://www2.keck.hawaii.edu/optics/aodocs/kaon263.pdf
- **[KAON598]** Johansson et al. 2008, SPIE 7015, 70153E, "Upgrading the Keck AO wavefront controllers". https://www2.keck.hawaii.edu/optics/aodocs/KAON598.pdf
- **[KAON489]** van Dam et al., NGWFC "Performance of the Keck II AO system". https://www2.keck.hawaii.edu/optics/aodocs/KAON489.pdf
- **[KAON1427]** Chin et al. 2022, SPIE 12185, 121850V, "Keck AO facility: RTC upgrade". https://par.nsf.gov/servlets/purl/10357049
- **[KAPA20]** Wizinowich et al. 2020, KAPA SPIE. https://www2.keck.hawaii.edu/inst/ao/KAPA_Paper_for_SPIE2020.pdf
- **[KAPA22]** Wizinowich et al. 2022, KAPA program overview. https://par.nsf.gov/servlets/purl/10356962
- **[RAG22]** Ragland et al. 2022, SPIE 12185, 121850Y, "Residual wavefront control of segmented mirror telescopes". https://par.nsf.gov/servlets/purl/10464752
- **[WIZ24]** Wizinowich et al., "Keck AO current and future roles as an ELT pathfinder". https://par.nsf.gov/servlets/purl/10540227
- **[KAPA26]** Surendran et al. 2026, arXiv:2608.07769 (KAPA on-sky; this is Keck I, which shares the legacy design).
- **[PRED26]** arXiv:2606.20838 (data-driven predictive control on Keck II with the OCAM2K SH, 2025).
- **[PRED22]** arXiv:2205.14164 (predictive control with the PyWFS on Keck II).
- **[SAL24]** arXiv:2404.08728 (vector-Zernike WFS on Keck II).
- **[LI21]** Li et al. 2021, arXiv:2109.00612 (Keck II pupil geometry, SCALES cold stop).
- **[GUIDE02]** Keck Telescope & Instrument Guide, Aug 2002. https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf
- **[KAON303]** Neyman 2004, "Atmospheric parameters for Mauna Kea". https://www2.keck.hawaii.edu/optics/kpao/files/KAON/KAON303.pdf
- **[NIRC2]** Keck NIRC2 web pages: https://www2.keck.hawaii.edu/inst/nirc2/genspecs.html and https://www2.keck.hawaii.edu/inst/nirc2/filters.html
- **[SVC16]** Service et al. 2016, PASP 128, 095004 (NIRC2 distortion). https://iopscience.iop.org/article/10.1088/1538-3873/128/967/095004
- **[OCAMSPEC]** FLI/Andor OCAM2K spec sheet. https://andor.oxinst.com/assets/uploads/products/andor/documents/ocam2s-and-ocam2k-specifications.pdf
- **[AGA20]** Agapito et al. 2020, arXiv:2012.14634. This is the **LBT/SOUL** lab characterization of the OCAM2K, **not a Keck note** (see Conflicts).
- **[KECKAO]** Keck AO pages: https://www2.keck.hawaii.edu/inst/ao/ and https://www2.keck.hawaii.edu/optics/ngsao/

---

## 1. Pupil

| Parameter | Value | Source | Conf. |
|---|---|---|---|
| Primary | 36 hexagonal segments, each 1.8 m across corners (0.9 m side) | GUIDE02; LI21 | H |
| Max diameter | 10.95 m. A 2026 paper uses D = 10.949 m. | GUIDE02; arXiv:2602.15746 | H |
| Equivalent circular diameter (by area) | 9.96 m | GUIDE02 ("aperture area is equivalent to ... circular aperture 9.96 m") | H. It is unclear whether this accounts for the central hole or obscuration. |
| Primary focal length / focus | 17.5 m primary; AO at f/15 Nasmyth (f = 150 m) | GUIDE02; WIZ06 | H |
| Segment gap | 3 mm, plus 2 mm non-reflective edge on each side | LI21 | H |
| Secondary mirror diameter | 1.4 m | LI21 | H |
| Central obscuration diameter | 2.6 m (older figure) or 2.48 m (measured from NIRC2 pupil images; LI21 adopts 2.48 m) | LI21 | H |
| Spiders | 6 secondary supports, uniform width 0.025 m. Support nodes are 0.369 m long and 0.08 m wide. | LI21 | H |
| Pupil rotation | The pupil rotates on the WFS and DM (Nasmyth plus image derotator). About 240 of 304 subapertures are active at any time, and the reconstructor is recomputed every 1 degree of rotation. | VD04 | H |
| Warm-optics emissivity (telescope + AO) | 30% | LI21 | M |

## 2. Deformable mirror and tip-tilt

| Parameter | Value | Source | Conf. |
|---|---|---|---|
| DM | Xinetics, 349 electrostrictive actuators, 7 mm spacing | WIZ06; KAON194 | H |
| Grid | "21x21 actuator (total of 349 actuators)" | PRED22; SAL24 ("21x21 actuator DM") | H |
| Actuator pitch on primary | 0.56 m (KAON194, WIZ06, BRASE98, KAPA26 "0.56 m"). KAON263 gives 56.2 cm; KAON051 uses 56.25 cm. | as cited | H |
| Geometry relative to SH | Fried: lenslet corners are conjugate to actuators | VD04; BRASE98 | H |
| Slaved actuators | About 50 of the 349, depending on pupil angle. These are not conjugate to a subaperture corner and are slaved to the mean of their neighbours. | VD04b | H |
| Influence function | Difference of Gaussians. S(x,y) = 0.470 µm × [w1/(2πσ1²)·exp(−r²/2σ1²) + w2/(2πσ2²)·exp(−r²/2σ2²)], with w1 = 2, w2 = −1, σ1 = 0.54 subap, σ2 = 0.85 subap (amplitude per volt). The mirror is not fully linear: equal voltages produce piston. | VD04 (OCR, Eq. 3) | M (exact normalisation is uncertain because of OCR) |
| Fitting error | σ_fit = 33.2·r0^(−5/6) nm (r0 in m at 500 nm), giving a_f = 0.46 | VD04 | H |
| Stroke | 4.0 µm peak-to-peak (error-budget parameter; "DM finite stroke error 31 nm") | RAG22 Table 2 | M |
| Tip-tilt mirror | 200 mm (KAON194) or 203 mm (WIZ06) SiC mirror with 3 piezo actuators. Since 2007 it has strain-gauge closed-loop positioning running at 60 kHz. | KAON194; WIZ06; KAON598 | H |
| Tip-tilt bandwidth | **[unverified]**: no closed-loop −3 dB value found. The tip-tilt loop is an integrator on the mean SH centroid. Vibrations at 20–40 Hz (VD04) and about 29–30 Hz (KAON598; RAG22 "29 Hz input disturbance") are significant. | — | — |

## 3. Shack-Hartmann WFS

| Parameter | Value | Source | Conf. |
|---|---|---|---|
| Subapertures | 20x20 grid on 200 µm-pitch lenslets. 304 subapertures fall in the pupil footprint; 240 are active at a time. | VD04 | H |
| Slope vector | 480 slopes (240 × 2); reconstructor is 349 × 480 | VD04b | H |
| Subaperture size on primary | 0.56 m (same as actuator pitch) | BRASE98; WIZ06 | H |
| WFS band | 0.5–1.0 µm ("quite broad"); response is closest to R band | KAON194; KAON598 | H |
| [MITLL] camera | AOA camera with MIT/LL 64x64 CCD (21 µm pixels per WIZ06), 4 amplifiers, 0.8 ms readout | WIZ06; KAON194 | H |
| [MITLL] pixels per subaperture | 3x3: a 2x2 quad cell plus a 1-pixel guard band | VD04; BRASE98; WIZ06 | H |
| [MITLL] read noise / dark current | 6.5 e⁻ (VD04 measured at 672 Hz), "6–7 e⁻" (KAON194), "7 e⁻" (KAON263). Dark current 4470 e⁻/px/s at 267 K; gain 1.99 e⁻/ADU. | VD04 | H |
| [MITLL] pixel scale | Design 2.44, 0.98 or 0.62"/px (lenslet focal lengths 2.0, 5.0, 7.9 mm); measured 2.4, 0.8 and 0.5"/px. The 2.4"/px scale was used normally. VD06 quotes 2.1"x2.1" quad-cell pixels for LGS. | VD04; VD06 | H |
| [MITLL] spot size | Calibration-source FWHM 1.25" at the 2.4"/px scale, rising to about 1.55" on sky at r0 = 20 cm | VD04 | H |
| [MITLL] field stop | Circular, 4.8" diameter | VD06 | H |
| [CCD39] camera | SciMeasure Little Joe with e2v CCD39; about 4–4.5 e⁻ read noise at low frame rates; negligible dark current | KAON598 | H |
| [CCD39] lenslets | AMµS fused-silica arrays, 200 µm pitch, focal lengths 2.4, 3.1 and 4.9 mm, giving "0.75", 1.2" and 1.5" per pixel respectively" (order as printed). LGS-mode WFS pixels are 3.0" (vs 2.1" before). | KAON598 | H for the numbers; M for the mapping |
| [CCD39] sampling | Reducer optics relay the "200 µm lenslet spacing to 4 pixel spacing" | KAPA20 | H |
| [OCAM] camera | OCAM2K: e2v CCD220 EMCCD, 240x240 pixels of 24 µm, 8 outputs, 2067 fps full frame, 3700 fps in 2x2 binning, 0.4 e⁻ read noise at 2000 fps and gain about 600, 43 µs exposure-to-first-pixel latency | OCAMSPEC | H |
| [OCAM] read noise (independent lab) | 0.4 e⁻ unbinned, 0.4–0.7 e⁻ binned (SOUL cameras) | AGA20 | H, but these are LBT cameras |
| [OCAM] Keck sampling | "Modified camera fore-optics to match the 200 µm lenslet spacing to four pixels on the detector". KAPA optics: magnification 0.48, "4x4 pixels/lenslet". | KAON1427; KAPA22 | H |
| [OCAM] Keck operating modes | Unbinned at 2 kHz and binned at 3.7 kHz (RTC requirement). On sky: "binning mode at a gain of 600" for NGS. A first-light NGS test used EM gain 10 at 200 Hz. | KAON1427 | H |
| [OCAM] pixels per subaperture in binned mode | 2x2 binned pixels, i.e. a quad cell. Inferred from 4x4 native pixels per lenslet with 2x2 on-chip binning. | derived | M/L **[inferred]** |
| [OCAM] pixel scale (arcsec/px) | **[unverified]**. The lenslets are unchanged since 2007 and pixels are 24 µm on both the CCD39 and the OCAM2K. If the fore-optics keep 4 px per lenslet, the scale would match the CCD39 values (0.75–1.5"/px unbinned, about 3.0"/px binned). | inferred from KAON598 + KAON1427 | L |
| [OCAM] EMCCD gain non-uniformity | Up to 4x quadrant-to-quadrant gain variation at gain 600 before First Light Imaging (FLI) re-tuned it to below 2x. The camera has an over-illumination interlock. | KAON1427 | H |
| [OCAM] ROI | Single pupil for NGS/sLGS. The KAPA four-pupil layout is used only on Keck I. Keck II ROI size **[unverified]**. | KAPA22; KAON1427 | M |
| Centroiding | Quad-cell centroid, with per-subaperture centroid gains (since 2007) and optional denominator-free centroiding. Reference-centroid scaling for spot size: about 0.8 on sky [MITLL]. | VD04; KAON598; VD06 | H |
| WFS ADC | None. Visible ADC "only design exists" (KAON194); RAG22 proposes adding one. | KAON194; RAG22 | H |

## 4. Frame rates, latency, controller

| Parameter | Value | Source | Conf. |
|---|---|---|---|
| [MITLL] frame rate range | 55–672 Hz (VD04), 60–670 Hz (KAON194). Design target was 500 Hz (BRASE98). | as cited | H |
| [MITLL] compute delay | 1.65 ms (tip-tilt) and 2.13 ms (DM), measured from end of CCD read to voltage update (VD04). BRASE98 design: 1.8 ms. KAON598 gives the old WFC latency as 1349 µs from last pixel to DM write start. | as cited | H |
| [MITLL] LGS frame rates | 200–660 Hz | VD06 | H |
| [CCD39] frame rates | Camera maximum 2406 Hz, which the RTC can follow. Operation was initially restricted to 1054 Hz (KAON489) and was "1 kHz" in typical use (KAON1427). | KAON598; KAON489; KAON1427 | H |
| [CCD39] RTC latency | 81 µs from last pixel to start of DM write; centroiding is overlapped with readout | KAON598 | H |
| [OCAM] frame rates | Up to 2 kHz unbinned and 3.7 kHz binned. On sky: 3000 fps binned (R = 8.9 NGS), 150 fps (R = 12 and 14.8), 1.5 kHz for single LGS, 600 Hz for multi-LGS, 1 kHz on the bench and in predictive-control tests. The Keck AO web page says "up to 2000 Hz". | KAON1427; KAPA26; PRED26; KECKAO | H |
| [OCAM] NRTC round-trip | Last pixel out to commands back: mean 205 µs (min 188, max 324 µs; requirement 500 µs, goal 250 µs). OCAM2K readout time is 466 µs at all tested rates (50, 1000, 2000 Hz). CCD39 on the NRTC: 168–173 µs round trip, readout 414 µs at 2406 Hz. | KAON1427 Table 2 | H |
| [OCAM] end-to-end latency | 0.7 ms at 1.5 kHz, from fitting the rejection transfer function (RTF) to the residual-DM PSD; "similar to the latency observed with NGS and single-LGS modes". This is the Keck I KAPA system. | KAPA26 | H (Keck I) |
| [MITLL] DM controller | Double-pole compensator: e[n] = −w·e[n−1] + k·u[n]; y[n] = l·y[n−1] + e[n], with w = 0.25 and leak l = 0.999 (bright) or 0.99 (otherwise); variable k | VD04 | H |
| [MITLL] tip-tilt controller | y[n] = y[n−1] + 0.8·k_TT·u[n] | VD04 | H |
| [MITLL] gain and frame-rate selection | Look-up table keyed on median ADU per subaperture per second | VD04 | H |
| [CCD39] controller | Programmable third-order PID-type law. The standard use is a "Smith compensator" or leaky integrator. A Bessel-Thomson tip-tilt option exists. | KAON598; VD06 | H |
| [OCAM] controller | Leaky integrator y[n] = −b1·y[n−1] + a0·u[n] with b1 ≈ −0.99 (leak 0.99). Typical DM gain is 0.5 (KAPA RTF fit; PRED26 integrator gain 0.5 at 1 kHz). PRED22 (PyWFS) quotes leak 0.99 and gain about 0.4. | KAPA26; PRED26; PRED22 | H |
| Reconstructor | Zonal Bayesian/MAP: R = (HᵀW⁻¹H + αC_φ⁺ + η11ᵀ)⁻¹HᵀW⁻¹, with a Kolmogorov (Wallner) piston-removed covariance and α tuned to SNR. Average x/y slopes are removed first (tip-tilt goes to the tip-tilt mirror); piston, tip and tilt are projected out in actuator space; about 50 actuators are slaved. It replaced the SVD reconstructor and removed waffle (about 100 nm improvement). The interaction matrix is built from ±0.2 µm pokes, and slopes more than 2 subapertures from the actuator are zeroed. | VD04; VD04b | H |
| Number of controlled modes | About 349 − 50 slaved ≈ 299 independent actuator commands (the null space of R has about 181 dimensions). Not modal. | VD04b | M |

## 5. Delivered performance and error budgets

| Item | Value | Source | Conf. |
|---|---|---|---|
| [MITLL] average Strehl | 0.37 at 1.58 µm (H continuum) on bright stars; 0.19 at V = 12; limiting magnitude about 14; best FWHM 36.5 mas (diffraction 33.6 mas at 1.58 µm) | VD04 | H |
| [MITLL] bright-NGS error budget (15 Jun 2003, r0 = 18 cm, V = 7.2) | Camera/NCPA 113, atmospheric fitting 139, telescope fitting 60, tip-tilt bandwidth 75, DM bandwidth 103, tip-tilt noise 9, DM noise 17 → 229 nm; with 125 nm miscellaneous, total 260 nm | VD04 | H |
| [MITLL] KAON263 summary | About 260 nm total: fitting 120, tip-tilt bandwidth 100, high-order bandwidth 90 ("max frame rate 670 Hz and time lag"), telescope about 100, noise about 50, image sharpening 130 nm (H Strehl 0.76 on the fiber). H Strehl 0.38 at V = 7.5 and 0.23 at V = 13.3. | KAON263 | H |
| [MITLL → CCD39] NGS R = 8 budget (original / upgrade, nm) | Atmospheric fitting 128/128; telescope fitting 66/66; camera 113/50; DM bandwidth 103/38; DM measurement 17/29; tip-tilt bandwidth 75/100; tip-tilt measurement 9/25; miscellaneous 125/125; total 256/228. K Strehl estimated 0.58/0.66, measured 0.50/0.60. Best K Strehl 58% → 71%. J/H/K 0.22/0.41/0.62 at R = 7.5. | KAON598 | H |
| [CCD39] Strehl vs magnitude (median-to-good seeing) | K: 0.58 (R7), 0.54 (R11), 0.50 (R13), 0.40 (R14), 0.22 (R15), 0.12 (R15.5). H: 0.40, 0.37, 0.33, 0.20. Maximum about 65% K and 45% H. | KECKAO ngsao page | H (page undated; describes the 2007-era system) |
| [CCD39] 2022 nightly NGS check | K Strehl 0.53, FWHM 50 mas (about 10th-mag NGS). The model needed an extra 130 nm rms high-order "margin" to match. | RAG22 Table 1 | H |
| [CCD39] LGS high-order budget (nm) | Fitting 121 ("20 subaps"); bandwidth 133 ("24 Hz −3 dB"); measurement 32; aliasing 40 ("0.3 fitting reduction factor"); static telescope 66; dynamic telescope 74; WFS zero-point 50; DM stroke 31; AO aberrations 30; NIRC2 aberrations 60; margin 130 | RAG22 Table 2 | H |
| [OCAM] on-sky NGS (Keck I, OSIRIS Brγ, seeing 0.45") | FWHM 52x53 mas (R = 8.9, 3000 fps binned, gain 600); 57x57 (R = 12, 150 fps); 68x68 (R = 14.8, 150 fps) | KAON1427 | H |
| [OCAM] Keck II NIRC2 Brγ | Strehl about 56% (integrator vs predictor comparable), 1 kHz, gain 0.5, 2025 | PRED26 | H |
| [OCAM] Marin et al. 2024 (SPIE 13097) | "Median Strehl for imaging data improves by 24%". Paper not opened; this is a search snippet only. | — | **[unverified]** |
| NCPA calibration | Phase diversity up to Z15; about 100 nm rms applied to the DM. Fiber-source wavefront error falls from 150 to 113 nm after the 10 µm (13.8 mas) source is accounted for. Field- and filter-dependent residuals remain. | VD04 | H |
| Aliasing | About one third of the fitting error (VD04); 40 nm in RAG22 | as cited | M |
| Scintillation | Negligible (about 0.5% Strehl loss) | VD04 | H |
| Design budget (1998, 2000 ph/m²/ms) | Set B: fitting 123, bandwidth 55, measurement 40, calibration 30, telescope 105 → 182 nm | BRASE98 | H (design-era) |

## 6. NIRC2

| Parameter | Value | Source | Conf. |
|---|---|---|---|
| Plate scale (narrow camera) | 9.942 ± 0.05 mas/px (Keck web page). 9.952 mas/px before the 2015-04-13 realignment and 9.971 ± 0.004 mas/px after it (SVC16). | NIRC2; SVC16 | H |
| Medium / wide cameras | 19.829 / 39.686 mas/px | NIRC2 | H |
| Detector | 1024² InSb Aladdin-III, 27 µm pixels; dark current below 0.1 e⁻/px/s; QE "80% at 1.7 µm" | NIRC2 | H |
| Read noise | CDS about 45–50 e⁻; MCDS with 16 read pairs about 15 e⁻. Gain 4 e⁻/DN before 2023-11-20, 8 e⁻/DN after (Archon). | NIRC2 | H |
| Filters (central / FWHM, µm) | J 1.248/0.163; H 1.633/0.296; K 2.196/0.336; Kp 2.124/0.351; Ks 2.146/0.311; Brγ 2.1686/0.0326; FeII 1.6455/0.0256 | NIRC2 filters | H |
| Fastest subarray exposure | 8 ms | VD04 | H |

## 7. Atmosphere used in Keck AO predictions

| Parameter | Value | Source | Conf. |
|---|---|---|---|
| Design Set B (30° zenith, at 0.55 µm, already zenith-scaled) | r0 0.18 m, θ0 3.67", τ0 2.75 ms, L0 50 m. Set A (zenith): 0.40 m / 10" / 10 ms. Set C (60°): 0.066 m / 1.31" / 0.99 ms. | KAON051 Table 2; BRASE98 | H |
| KAON 303 CN-M1 (at 0.5 µm) | r0 20 cm, θ0 2.2", f_G 39 Hz. 7 layers (altitude km : fraction : wind m/s) = 0.0:0.369:6.7, 2.1:0.219:13.9, 4.1:0.127:20.8, 6.5:0.101:29, 9.0:0.046:29, 12.0:0.111:29, 14.8:0.027:29 | KAON303 | H |
| Chun 2002 SCIDAR (KAON 303) | r0 23.6 / 17.8 / 13.6 cm and θ0 4.83 / 3.17 / 2.16" for the 20% best / median / 80% worst | KAON303 | H |
| Other KAON 303 r0 values | 22 cm (Racine/CFHT); 21 cm (Subaru morning); 15 cm (Subaru 4-year mean) | KAON303 | H |
| KAPA tomography model (2025–26) | r0 0.15 m, L0 30 m; layers 0, 0.5, 1, 2, 4, 8, 16 km with fractions 0.4557, 0.1295, 0.0442, 0.0506, 0.1167, 0.0926, 0.1107 | KAPA26 Table 2 | H |
| "Average" seeing in VD04 | r0 = 20 cm at 500 nm; requirement "good seeing" r0 ≥ 20 cm (KAON598) | VD04; KAON598 | H |
| Median MKR model r0 16 cm, θ0 2.7" (KAON 503) | search snippet only | — | **[unverified]** |

## 8. Throughput and beam splitting

| Parameter | Value | Source | Conf. |
|---|---|---|---|
| Science/WFS dichroic | Visible-reflecting, IR-transmitting. The original Barr coating on fused silica transmits 1–2.7 µm, so light below 1 µm goes to the WFS. | KAON194; WIZ06 | H |
| NGS-mode splitter before the WFS | 4% reflective beamsplitter (sends 4% to the acquisition, tip-tilt sensor and LBWFS path; the transmitted light goes to the WFS). In LGS mode it is replaced by a sodium-transmitting dichroic. | WIZ06; KAON194 | H |
| Design throughput to WFS | Telescope T = 0.7 × AO T = 0.45 × MIT/LL QE (0.9 V, 0.85 R, 0.45 I). Error budget used total system 0.3. Zero-mag flux per subaperture: 2.38×10⁶ ph/ms (A0) and 7.0×10⁶ ph/ms (M0). | KAON051 | H (1995–98 design values) |
| Empirical sensitivity [MITLL] | Loops close at about 20 counts per subaperture, roughly V = 14. WFS saturates at about 2000 ADU/px, roughly mag 4.5. | KAON194; VD04 | H |
| Measured end-to-end WFS throughput (any era) | **[unverified]**: not found | — | — |
| OCAM2K QE | Above 90% at 650 nm; up to 95% over 400–900 nm | OCAMSPEC | H |
| KPIC PyWFS dichroic (for reference) | 90% of J+H to the PyWFS | SAL24 | H |

---

## Conflicts and uncertainties

1. **Mis-cited source in the request.** arXiv:2012.14634 is Agapito et al. (INAF Arcetri), "EMCCD for Pyramid wavefront sensor: laboratory characterization" of the **LBT SOUL** OCAM2K cameras. It is not Keck AO Note 1337 and contains nothing about Keck. The Keck OCAM2K/RTC paper is **KAON 1427** (Chin et al. 2022, SPIE 12185, 121850V). The follow-up Marin et al. 2024 (SPIE 13097, 1309760, doi:10.1117/12.3016792) is closed-access and I did not read it.
2. **Count of active subapertures.** 304 is the number inside the pupil footprint; only 240 are active at a time because of the rotating serrated-hexagon pupil (VD04). Use 240, with a rotation-dependent mask, for a faithful simulation.
3. **WFS pixel scale differs by era.**
   - [MITLL]: measured 2.4, 0.8 and 0.5"/px (VD04, with design values 2.44, 0.98, 0.62); VD06 quotes 2.1".
   - [CCD39]: 0.75, 1.2 and 1.5"/px unbinned (KAON598). LGS pixels were 3.0", which implies 2x2 binning of 1.5" pixels.
   - [OCAM]: no published value found.
   KAON598 lists focal lengths (2.4, 3.1, 4.9 mm) and plate scales (0.75, 1.2, 1.5"/px) "respectively", which is physically inverted, since a longer focal length should give a finer scale. Treat the lenslet-to-scale mapping as uncertain.
4. **Pixels per subaperture.**
   - [MITLL]: 3x3 (2x2 quad cell plus guard band).
   - [CCD39] and [OCAM]: 4 pixels per lenslet natively; 2x2 binning gives a 2x2 quad cell with no guard band. This binned layout is my inference for the OCAM2K, consistent with "binning mode" and 3.7 kHz.
5. **Maximum frame rate.**
   - 670 or 672 Hz [MITLL].
   - 2406 Hz [CCD39], with operation at 1054 Hz in 2007 and about 1 kHz typically.
   - 2 kHz unbinned and 3.7 kHz binned [OCAM]; 3000 fps was demonstrated. Keck's web page says "up to 2000 Hz".
6. **Latency.** Compute delay was 1.65 ms (tip-tilt) and 2.13 ms (DM) in VD04, 1.8 ms in BRASE98 (design), and 1.349 ms in KAON598, which measures from the last pixel. With the CCD39 RTC it was 81 µs. The OCAM2K NRTC adds about 205 µs after a 466 µs readout. KAPA's fitted end-to-end latency is 0.7 ms at 1.5 kHz (about 1 frame). Each source defines latency differently.
7. **Actuator pitch.** 0.56 m in most sources; 56.2 or 56.25 cm in KAON263 and KAON051. The equivalent diameter of 9.96 m and the maximum diameter of 10.95 m are different quantities.
8. **Central obscuration.** 2.6 m (older figure) vs 2.48 m (NIRC2 measurement); LI21 recommends 2.48 m.
9. **NIRC2 narrow-camera scale.** The web page gives 9.942 mas/px. Service et al. 2016 give 9.952 (before April 2015) and 9.971 mas/px (after); use 9.971 for post-2015 data.
10. **Atmosphere.** Profiles in use range from r0 = 0.15 m (KAPA, L0 30 m) to 0.18 m (design Set B, at 30° zenith and 0.55 µm, L0 50 m) to 0.20 m (CN-M1). τ0 is published only for the 1998 design sets (2.75 ms median). CN-M1 gives f_G = 39 Hz instead.
11. **Not found anywhere:**
    - tip-tilt mirror closed-loop bandwidth
    - OCAM2K-era pixel scale, field of view per subaperture and ROI size on Keck II
    - measured photon throughput to the WFS
    - frame-rate-vs-magnitude table for the OCAM2K era (only point examples: 3 kHz at R = 8.9, 150 Hz at R = 12–14.8)
    - date HAKA replaced the 349-actuator DM (the Keck AO page now lists Keck II with a 2844-actuator ALPAO DM and a 57x57 or 29x29 SH)
    - Wizinowich 2000 PASP 112, 315 full text (only the ADS/IOP abstract was seen; KAON194 is the same team's detailed 2000 SPIE description)
