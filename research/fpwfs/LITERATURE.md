# Focal-plane-only wavefront sensing for kHz AO on an 8 m telescope: literature review

*Compiled 2026-10-03. Goal: close a >500 Hz (ideally 1 kHz) AO loop on an 8 m-class telescope with a single focal-plane camera as the only wavefront sensor, and match Shack-Hartmann (SH) closed-loop performance.*

## How the references were checked

- Each entry was checked against an arXiv abstract page, a publisher or ADS page, or a PubMed/SPIE record seen while compiling this review. Identifiers are given as arXiv IDs or DOIs exactly as I saw them.
- **[unverified]** means the paper exists, but I could not see some of its details (for example the author list or specific numbers). The note on the entry says what was not checked.
- Numbers marked "(full text)" came from the PDF text. All other numbers came from the abstract. Vendor camera specs come from vendor pages, not peer-reviewed papers, and are labelled that way.
- Several 2026 papers (SPIE AT+I 2026, A&A 2026) are included because they are directly on topic. Some are preprints or conference papers.

---

## 0. The physics in brief (needed to read the rest)

- A focal-plane image records |FT(A·e^{iφ})|². For a real, centro-symmetric pupil amplitude A and small φ:
  - the **odd** part of φ appears in the PSF at first order (as an antisymmetric intensity term);
  - the **even** part appears only at second order, through |φ_even|².
  - So φ(x) and −φ(−x) give the same PSF. This is the **sign or twin-image ambiguity**, and it affects the even modes: defocus, astigmatism, spherical, and so on.
- There are four known ways to break it:
  1. **Phase diversity.** Use a known extra aberration: defocus (Gonsalves 1982; Paxman 1992), or temporal or sequential diversity where the previous DM command is the diversity (Gonsalves 2001/2002; Keller 2012; Korkiakoski 2014).
  2. **Asymmetric pupil amplitude** (Martinache 2013).
  3. **Focal-plane phase masks**: vortex (Quesnel 2022), vAPP/holograms (Wilby 2017; Bos 2019), photonic lantern or multimode fibre (Norris 2020; Padrón-Brito 2025).
  4. **Temporal amplitude modulation**, e.g. a pupil chopper (Gerard 2023).
- Linearised focal-plane methods (F&F, APF-WFS, LIFT, Smith 2013) need a **high-Strehl / small-phase** regime, roughly ≲1–1.5 rad rms. NN methods trade that limit for a learned nonlinear inverse map, but the measurement still contains the ambiguity unless diversity is present.

---

## 1. Classical phase retrieval and phase diversity

**[R1] Gerchberg & Saxton (1972), Optik 35, 237–246.** Alternating-projection phase retrieval between the pupil and image planes. It is the basis of the GS-type error reduction used in F&F-GS. Only one focal image is available for it in our setting, so it inherits the twin-image ambiguity. [verified: venue/pages via search; no DOI seen]

**[R2] Gonsalves (1982), "Phase retrieval and diversity in adaptive optics", Opt. Eng. 21(5), 215829. DOI 10.1117/12.7972989.** The founding paper on focal-plane phase retrieval for AO. It introduces phase or wavelength diversity (two images with a known added aberration) to remove the ambiguity and make the inversion well-posed.

**[R3] Fienup (1982), "Phase retrieval algorithms: a comparison", Appl. Opt. 21, 2758. DOI 10.1364/AO.21.002758.** Compares error-reduction/GS, hybrid input–output, and gradient methods. HIO and conjugate gradient converge much faster than error reduction. This is still the reference for iterative-PR stagnation and convergence behaviour.

**[R4] Paxman, Schulz & Fienup (1992), JOSA A 9(7), 1072. (ADS 1992JOSAA...9.1072P.)** Maximum-likelihood joint estimation of object and aberrations from phase-diverse images, with Gaussian and Poisson noise models. This is the statistically optimal formulation that LIFT and COFFEE later linearise or regularise.

**[R5] Gonsalves (2001), "Small-phase solution to the phase-retrieval problem", Opt. Lett. 26, 684. DOI 10.1364/OL.26.000684.**
- Closed-form small-phase solution: the odd part comes from one image, and the even part needs a second, diverse image.
- This is the direct ancestor of F&F, and it states the sign-ambiguity problem explicitly.

**[R6] Gonsalves (2002), ESO Conf. & Workshop Proc. 58, p.121 (sequential diversity).** Proposes using the previous DM update as the diversity. [unverified: known only from its citation in the reference lists of R13 and R50; full text not seen]

**[R7] Mugnier, Blanc & Idier (2006), "Phase diversity: a technique for wave-front sensing and for diffraction-limited imaging", Adv. Imaging & Electron Physics 141, 1–76. DOI 10.1016/S1076-5670(05)41001-0.** A comprehensive review of phase diversity: estimators, noise, and regularisation.

**[R8] Smith, Marinică, den Dekker, Verhaegen, Korkiakoski, Keller & Doelman (2013), "Iterative linear focal-plane wavefront correction", JOSA A 30(10), 2002. DOI 10.1364/JOSAA.30.002002.** An efficient linearised approximation of nonlinear phase diversity aimed at real-time use. It is faster than earlier PSF-linearising methods with no loss of accuracy.

**[R9] Polo, Haber, Pereira, Verhaegen & Urbach (2013), "Linear phase retrieval for real-time adaptive optics", J. Eur. Opt. Soc. RP 8, 13070.** Linearises the pupil function for small aberrations to get fast PR suitable for real-time AO. (Note: the authors are not Korkiakoski/Soloviev, as one might guess.)

**[R10] Booth (2007), "Wavefront sensorless adaptive optics for large aberrations", Opt. Lett. 32(1), 5.**
- Model-based modal sensorless AO, from microscopy: metric optimisation using N+1 or 2N+1 DM-probed images.
- Relevant as a bootstrapping strategy that needs no WFS. It is too slow for atmospheric bandwidth, because each estimate costs several frames.

**Relevance of §1.** The theory is mature. The open issues for kHz AO are:
- the computational cost of iterative PR;
- the requirement for diversity;
- the narrow small-phase validity range.

---

## 2. Fast & Furious (F&F) and sequential phase diversity

**[R11] Keller, Korkiakoski, Doelman, Fraanje, Andrei & Verhaegen (2012), "Extremely fast focal-plane wavefront sensing for extreme adaptive optics", Proc. SPIE 8447, 844721. arXiv:1207.3273, DOI 10.1117/12.926725.**
- Sequential phase diversity: the previous DM update is the known diversity, and the even/odd PSF decomposition gives a solution with about **one FFT per iteration**. Complexity is roughly proportional to the number of actuators, for 10⁴–10⁵ actuators.
- (full text) Cites that about **1.5 rad rms** aberration can be tolerated.
- Relevance: the canonical "only the focal-plane camera plus past DM commands" algorithm, and exactly our information constraint.

**[R12] Korkiakoski, Keller, Doelman, Kenworthy, Otten & Verhaegen (2014), "Fast & Furious focal-plane wavefront sensing", Appl. Opt. 53(20), 4565. arXiv:1406.1006, DOI 10.1364/AO.53.004565.**
- Two algorithms: F&F (weak-aberration assumption plus pupil symmetry) and F&F-GS (GS-style error reduction that handles arbitrary pupils and amplitude).
- Lab result: a 170×170 SLM (about 28,900 degrees of freedom) went from Strehl about 0.75 to 0.98–0.99. Residuals were about 0.15 rad (FF) and 0.10 rad (FF-GS).
- (full text) Matlab code; computation was negligible next to the **about 15 s needed per HDR image**. FF used 2 FFTs per iteration and FF-GS used 8.
- Monochromatic light is preferred. About 10% bandwidth would still work, but only over a limited corrected field.
- Relevance: proves very high mode counts are possible in principle. All of it was static, slow, high-Strehl correction, not turbulence.

**[R13] Korkiakoski et al. (2014), "Focal-plane wavefront sensing with high-order adaptive optics systems", Proc. SPIE. arXiv:1407.5846.**
- Compares GS, extended F&F (3 images, amplitude estimation) and convex-optimisation PR for calibrating ≥150×150-actuator correctors.
- F&F-like methods were the easiest to use.

**[R14] Korkiakoski, Doelman, Codona, Kenworthy, Otten & Keller (2013), "Calibrating a high-resolution wavefront corrector with a static focal-plane camera", Appl. Opt. 52. arXiv:1310.1241.**
- Calibrates about 40,000 SLM degrees of freedom using only a focal-plane camera, via localised diversity and differential OTFs. Actuator localisation reached 0.3% of the pupil diameter.
- Relevance: DM-to-pupil registration from focal-plane data alone, which our system needs because there is no pupil-plane WFS.

**[R15] Wilby, Keller, Sauvage, Dohlen, Fusco, Mouillet & Beuzit (2018), "Laboratory verification of Fast & Furious phase diversity: towards controlling the low wind effect in SPHERE", A&A 615, A34. arXiv:1803.03258.**
- On the MITHIC bench, LWE-affected PSFs returned to **>90% Strehl within about 5 iterations**.
- (full text) The bench camera can run 1 Hz–1 kHz with a 32×32 px field of view at 3.5 px per λ/D. The tests ran at a 1 Hz cadence, and simulations assumed 10 Hz.
- Relevance: convergence speed in iterations is excellent, but there was no kHz demonstration.

**[R16] Bos, Vievard, Wilby, Snik, Lozi, Guyon, Norris, Jovanovic, Martinache, Sauvage & Keller (2020), "On-sky verification of Fast and Furious focal-plane wavefront sensing: moving forward toward controlling the island effect at Subaru/SCExAO", A&A 639, A52. arXiv:2005.12097.**
- Internal source: LWE screens of 0.4–2 µm PV were corrected to SRA >90%, and VAR went from 0.27 to 0.03.
- On sky: the PSF improved in 1.3–1.4″ seeing.
- (full text) Setup:
  - C-RED 2, cropped to 64×64 px, 1550 nm/25 nm or half of H band.
  - Projected onto PTT modes plus the **first 50 Zernikes**.
  - Gain 0.1–0.3, leak 0.99–0.999.
  - **Loop speed 4–25 FPS**, limited by Python and image alignment, not by the camera. The authors expect **300–400 FPS in C**.
  - Convergence took about 75–100 iterations (4.5–6 s), slower than the ~10 expected because of an unexplained gain factor.
  - The paper says that, if SCExAO's LWE timescales are similar to SPHERE's (about 1–2 s), these convergence times are not sufficient.
  - Limitations: needs a real, symmetric pupil field; no focal-plane mask; phase only; monochromatic assumption.

**[R17] Bos, Bottom, Ragland, Delorme, Cetre & Pueyo (2021), "'Fast' and Furious focal-plane wavefront sensing at W. M. Keck Observatory", Proc. SPIE 11823. arXiv:2107.07601.**
- (full text) NIRC2 frames take ≥10 s, so "Fast" is in quotes. Up to 30 Zernikes on sky (90 on the bench).
- Results:
  - Bench: Strehl 78% → 92% (F&F) and 98% (GS).
  - On sky with injected aberration: 34% → 74%.
  - Static aberrations: about 54% → 72%.
- Sign handling: the current even component, the previous image's even component, and the previous DM command.

**[R18] Bottom, Walker, Cunnyngham, Guthery & Delorme (2023), "Sequential coronagraphic low-order wavefront control", AO4ELT7. arXiv:2312.06806.**
- "2 Fast 2 Furious" extends sequential diversity to even-symmetric coronagraphs.
- "Tokyo Drift" is a deep-learning version for general coronagraphs.
- Both have 100% science uptime, need no diversity frames, and use only the DM plus science camera. Shown in simulation plus initial lab tests.

**Takeaways for §2.**
- F&F is the right physical prior for our constraint: one image, past DM commands, about one FFT.
- Every published deployment is a **slow (≤25 Hz) second stage on low-order or LWE modes, at high Strehl**.
- No paper shows F&F as the first-stage sensor against full turbulence at kHz.
- Failure modes reported:
  - small-phase validity (about 1–1.5 rad);
  - needing a symmetric, real pupil;
  - regularisation of odd modes (ε);
  - poor conditioning when the diversity, which is the DM step, is small;
  - monochromaticity;
  - image alignment and tip-tilt handling.

---

## 3. Single-image focal-plane WFS with built-in diversity (asymmetric pupil, kernel phase, LIFT, COFFEE, holographic/vAPP, photonic)

**[R19] Martinache (2013), "The Asymmetric Pupil Fourier Wavefront Sensor", PASP 125, 422. arXiv:1303.6678, DOI 10.1086/670670.**
- Fourier phase of a single image is linear in pupil phase at high Strehl, provided the pupil is made asymmetric (e.g. a mask). It needs WFE ≲1 rad.
- Simulations: Strehl went from 50% to >90% in a few iterations, with excellent photon-noise sensitivity.
- Relevance: removes the ambiguity per frame with a passive mask, at a throughput cost.

**[R20] Pope, Cvetojevic, Cheetham, Martinache, Norris & Tuthill (2014), "A demonstration of wavefront sensing and mirror phasing from the image domain", MNRAS 440, 125. arXiv:1401.7566.** First lab demonstration of the APF-WFS, the dual of kernel phase.

**[R21] N'Diaye, Martinache, Jovanovic, Lozi, Guyon, Norris, Ceau & Mary (2018), "Calibration of the island effect: experimental validation of closed-loop focal plane wavefront control on Subaru/SCExAO", A&A 610, A18. arXiv:1712.03963.**
- APF-WFS closed loop on piston/tip/tilt per quadrant gave a **37% relative Strehl increase** in the visible (VAMPIRES).
- Valid in the small-aberration regime.

**[R22] Vievard et al. (2019), "Overview of focal plane wavefront sensors to correct for the Low Wind Effect on SUBARU/SCExAO", AO4ELT6. arXiv:1912.10179.**
- Surveys ZAP (Zernike Asymmetric Pupil; on-sky LWE measurement), F&F, an NN algorithm ("promising PSF prediction on-sky"), and LAPD (linearised analytic phase diversity).

**[R23] Meimon, Fusco & Mugnier (2010), "LIFT: a focal-plane wavefront sensor for real-time low-order sensing on faint sources", Opt. Lett. 35(18), 3036. DOI 10.1364/OL.35.003036.**
- Linearised ML phase retrieval on a single image with astigmatic diversity, for low-order modes on faint sources.
- The number of modes can be tuned without changing hardware.

**[R24] Plantet, Meimon, Conan & Fusco (2013), "Experimental validation of LIFT for estimation of low-order modes in low-flux wavefront sensing", Opt. Express 21(14), 16337.**
- Single-image ML PR with better sensitivity than a 2×2 SH. TT and focus performance is similar to an unmodulated pyramid.
- Validated in monochromatic and broadband light.

**[R25] Kuznetsov, Oberti, Neichel & Fusco (2024), "Striving towards robust phase diversity on-sky: implementing LIFT for VLT/MUSE-NFM", A&A 687, A221. arXiv:2406.08529.** First on-sky LIFT, on the full-pupil IRLOS NGS sensor of an 8 m UT. Describes the robustness problems met on sky and proposes fixes.

**[R26] Agapito, Busoni, Carlà, Plantet, Esposito & Ciliegi (2022), "MAORY/MORFEO and LIFT: can the low order wavefront sensors become phasing sensors?", Proc. SPIE. arXiv:2208.02662.** Study of LIFT for sensing ELT segment and petal phasing.

**[R27] Salgueiro, Correia, Neichel, Bouchez, Wizinowich et al. (2026), "Slow focus sensor for the Keck I LGS AO system using focal plane wavefront sensing". arXiv:2602.15746.** Compared GS, LIFT and Gaussian fit for sodium-altitude focus tracking. GS was selected and demonstrated on sky.

**[R28] Paul, Sauvage & Mugnier (2013), "Coronagraphic phase diversity: performance study and laboratory demonstration", A&A 552, A48.** [unverified: venue confirmed via search; arXiv ID not seen]
**[R29] Paul, Mugnier, Sauvage & Dohlen (2013), "High-order myopic coronagraphic phase diversity (COFFEE) for wave-front control in high-contrast imaging systems", Opt. Express 21, 31751. arXiv:1310.5459, DOI 10.1364/OE.21.031751.**
- COFFEE is Bayesian MAP phase diversity behind any coronagraph, using two focal images. It reaches nm precision on low- and high-order quasi-static aberrations.
**[R30] Herscovici-Schiller, Sauvage, Mugnier, Dohlen & Vigan (2019), "Coronagraphic phase diversity through residual turbulence", MNRAS 488. arXiv:1907.07038.**
- COFFEE extended to long exposures through AO residuals. Lab validated.
- Relevance: COFFEE is accurate but slow (quasi-static, iterative MAP). It is the right "truth" estimator for calibration, not for the kHz loop.

**[R31] Wilby, Keller, Snik, Korkiakoski & Pietrow (2017), "The coronagraphic Modal Wavefront Sensor: a hybrid focal-plane sensor for the high-contrast imaging of circumstellar environments", A&A 597, A112. arXiv:1610.04235.**
- Holographic modal WFS inside an APP coronagraph: each mode gets a pair of biased off-axis PSF copies.
- Effective dynamic range **±2.5 rad rms**, recovered in 2–10 iterations. On sky at the WHT: 10 nm (0.1 rad) accuracy on known static aberrations.
- Relevance: a single-shot, unambiguous, linear-ish modal readout, at the cost of photons spread into the hologram spots. Mode count is limited by the number of spots.

**[R32] Bos, Doelman, Lozi, Guyon, Keller, Miller, Jovanovic, Martinache & Snik (2019), "Focal-plane wavefront sensing with the vector Apodizing Phase Plate", A&A 632, A48. arXiv:1909.08317.**
- The vAPP's two PSFs act as the WFS, with nonlinear retrieval. Lowest **30 Zernikes**, on sky and internal source.
- Maximum WFE corrected per iteration is about **λ/8 rms**. Simulation with 10⁷ photons gives about λ/1000.
- On sky: raw contrast improved by about ×2 at 2–4 λ/D after 5 iterations.

**[R33] Miller et al. (2019), "Spatial linear dark field control and holographic modal wavefront sensing with a vAPP coronagraph on MagAO-X", JATIS 5(4), 049004. DOI 10.1117/1.JATIS.5.4.049004.** hMWFS for low order plus LDFC for the dark hole on MagAO-X. [verified: title/venue; full author list not seen]

**[R34] Norris, Wei, Betters, Wong, Leon-Saval et al. (2020), "An all-photonic focal-plane wavefront sensor", Nat. Commun. 11, 5335.**
- A photonic lantern at the focal plane turns the focal-plane field into single-mode-core intensities that carry non-degenerate (unambiguous) information. A deep NN does the inversion.
- Lab: 5.1×10⁻³ π rad rms error on low-order Zernikes.

**[R35] Lin, Fitzgerald, Xin, Kim, Guyon, Norris, Betters, Leon-Saval et al. (2023), "Real-time experimental demonstrations of a photonic lantern wavefront sensor", ApJL 959, L34. arXiv:2312.13381.** Closed loop on **5 non-piston Zernikes at 1 kHz** with CACAO and a leaky integrator (gain 0.2, leak 0.99). About 95% of injected WFE was corrected.

**[R36] Wei, Norris, Betters & Leon-Saval (2023), "Demonstration of a photonic lantern focal-plane wavefront sensor: measurement of atmospheric wavefront error modes and low wind effect in the non-linear regime". arXiv:2311.01716.**
- 19-core lantern plus NN, tested at 0.88 rad (linear) and 1.5 rad rms (nonlinear) incident WFE.
- Petal-mode reconstruction error: 2.9×10⁻² rad (linear) and 2.1×10⁻¹ rad (nonlinear).

**[R37] Sengupta et al. (2025), "On-sky demonstration of second-stage wavefront control with a photonic lantern" (Lick/Shane 3 m), accepted to AJ. arXiv:2511.20560.** Closed loop on sky as a second stage for NCPA. [Loop rate and mode count not in the abstract.]

**[R38] Padrón-Brito, Arteaga-Marrero, Cunnyngham & Kuhn (2025), "Focal-plane wavefront sensing with moderately broadband light using a short multi-mode fiber", Opt. Express. arXiv:2510.03058, DOI 10.1364/OE.580986.**
- A <1 cm multimode fibre at the focal plane plus an NN. Works over 10 nm NIR bandwidth on millisecond timescales, and resolves the even-mode sign ambiguity.

**[R39] Gerard, Dillon, Cetre & Jensen-Clem (2023), "High speed focal plane wavefront sensing with an optical chopper", PASP. arXiv:2301.11282, DOI 10.1088/1538-3873/acb6b6.**
- A pupil-plane chopper synchronised to the focal-plane camera gives temporal amplitude modulation that breaks the even-mode ambiguity at low, mid and high spatial frequencies. Reconstruction is a linear MVM. Up to 50% science duty cycle.
- Validated on the SEAL bench.

**[R40] Jiang, Guo, Metzler & Veeraraghavan (2026), "Guidestar-free adaptive optics with asymmetric apertures", ACM Trans. Graph. 45(5), 39. arXiv:2602.07029.**
- Computational imaging: an asymmetric pupil aperture plus ML PSF and phase estimation from extended scenes, with SLM correction.
- Uses an order of magnitude fewer measurements and three orders less compute than prior guidestar-free methods.

**[R41] (Nat. Commun. 2026) "An end-to-end hybrid deep-learning approach for single-shot wavefront sensing and correction", Nat. Commun. 17. DOI 10.1038/s41467-026-72364-1.**
- A learned optical phase mask is jointly optimised with an NN decoder. The encoding removes intrinsic phase ambiguities and raises sensitivity to weak aberrations.
- UCSD/UC Berkeley/BU. [unverified: author list and quantitative results not seen (paywall)]

**[R42] Deo, Vievard, Cvetojevic, Ahn, Huby, Guyon, Lacour, Lozi, Martinache, Norris, Skaf & Tuthill (2022), "Controlling petals using fringes: discontinuous wavefront sensing through sparse aperture interferometry at Subaru/SCExAO", Proc. SPIE 12185. arXiv:2209.02898.**
- Dual-band visible SAM used as a petal sensor; the two bands extend capture range.
- (This is probably the "Deo" thread you remembered. It is a pupil-mask method imaged in the focal plane.)

**[R43] Singh, Vievard, Lozi, Deo, Fogal, Ahn, Guyon & Juillard (2026), "Addressing the low-wind effect with the Lyot-based low-order wavefront sensor on SCExAO", SPIE 2026. arXiv:2609.00737.** ML plus LLOWFS, with initial on-sky LWE differential-piston control in coronagraphic mode.

---

## 4. Machine-learning focal-plane WFS, and NN analogues

### 4a. Early and supervised CNN phase retrieval

**[R44] Angel, Wizinowich, Lloyd-Hart & Sandler (1990), "Adaptive optics for array telescopes using neural-network techniques", Nature 348, 221.** An NN estimates phase (segment piston and tilt for the MMT's six mirrors) from focal-plane images, in simulation. The first NN focal-plane WFS.

**[R45] Sandler, Barrett, Palmer, Fugate & Wild (1991), "Use of a neural network to control an adaptive optics system for an astronomical telescope", Nature 351, 300.**
- An NN applied to in-focus and out-of-focus images of Vega at the 1.5 m Starfire Optical Range estimated the phase distortion. This was a real-star test.
- [details of loop closure and rate not verified]

**[R46] Paine & Fienup (2018), "Machine learning for improved image-based wavefront sensing", Opt. Lett. 43(6), 1235. DOI 10.1364/OL.43.001235.**
- An Inception-v3 CNN gives the starting guess for gradient-based PR, which expands the capture range.
- (Per R49's summary) The input was 1.57–25.1 rad rms over ≤18 Zernikes, and the residual after the CNN averaged about 2.3 rad.
- Relevance: a CNN fixes the capture-range problem but is not precise on its own. This is the bootstrapping recipe.

**[R47] Nishizaki, Valdivia, Horisaki, Kitaguchi, Saito, Tanida & Vera (2019), "Deep learning wavefront sensing", Opt. Express 27, 240. DOI 10.1364/OE.27.000240.**
- A CNN estimates Zernikes directly from one intensity image. It also explores preconditioned inputs (overexposed, defocused, scattered), i.e. it designs the sensor optics around the network.

**[R48] Andersen et al. (2019), "Neural networks for image-based wavefront sensing for astronomy", Opt. Lett. 44(18), 4618.** [author list assumed to match R48b; not seen]
**[R48b] Andersen, Owner-Petersen & Enmark (2020), "Image-based wavefront sensing for astronomy using neural networks", JATIS 6(3), 034002. DOI 10.1117/1.JATIS.6.3.034002.**
- ResNet on in-focus plus out-of-focus pairs through seeing.
- (Per R49) D/r₀ = 12–21, about 8–13 rad rms in, about 1.4–2 rad rms residual with 36 modes; 66 modes added little.
- Abstract: **130 nm rms** error and **8 ms inference**.
- Relevance: the closest prior work to "seeing-limited focal-plane NN on a large telescope". It shows that the single-shot large-aberration regime saturates around 1.5–2 rad.

**[R49] Orban de Xivry, Quesnel, Vanberg, Absil & Louppe (2021), "Focal plane wavefront sensing using machine learning: performance of convolutional neural networks compared to fundamental limits", MNRAS 505, 5702. arXiv:2106.04456.**
- ResNet-50 (Zernike output) and U-Net (phase-map output) on focal-plane image pairs with defocus diversity. 20 and 100 Zernikes, input up to about 1 rad rms (70/350 nm at 2.2 µm).
- They **reach the photon-noise limit** over a wide range: for 20 modes, rms WFE below **λ/1500 with 2×10⁶ photons** in one iteration. Similar to or better than iterative PR.
- Relevance: the key "CNN ≈ CRB" result. Note that it uses explicit diversity and covers ≤1 rad.

**[R50] Quesnel, Orban de Xivry, Louppe & Absil (2022), "A deep learning approach for focal-plane wavefront sensing using vortex phase diversity", A&A 668, A36. arXiv:2210.00632.**
- EfficientNet-B4 behind scalar and vector vortex coronagraphs, with one or two PSFs. The vortex **lifts the sign ambiguity even at low S/N**.
- Performance is close to defocus phase diversity, with a 100% science duty cycle.

**[R51] Quesnel, Orban de Xivry, Absil & Louppe (2022), "A simulator-based autoencoder for focal plane wavefront sensing", Proc. SPIE 12185, 1218532. arXiv:2211.05242, DOI 10.1117/12.2629476.**
- **Self-supervised**: a differentiable optical simulator acts as the decoder, so no labels are needed. Performance is almost identical to a supervised CNN, and per-image fine-tuning helps on noisy data.
- Relevance: the template for on-sky self-calibration and for closing the sim-to-real gap.

**[R52] Allan, Kang, Douglas, Barbastathis & Cahoy (2020), "Deep residual learning for low-order wavefront sensing in high-contrast imaging systems", Opt. Express 28(18), 26267. DOI 10.1364/OE.397790.** A residual CNN extends the LLOWFS usable range by more than 10× and works at low photon counts.

**[R53] Terreri, Pedichini, Del Moro et al. (2022), "Neural networks and PCA coefficients to identify and correct aberrations in adaptive optics", A&A 666, A70.** NN on PCA-compressed focal-plane data for NCPA. [unverified: title/venue from search plus citation in R61; arXiv ID not seen]

**[R54] Taheri, Molahasani, Ragland, Neichel & Wizinowich (2024), "AI-powered low-order focal plane wavefront sensing in infrared", Proc. SPIE 13097. arXiv:2410.12084.** Trained on simulations, validated on Keck I bench K-band data.

**[R55] Sabhlok et al. (2026), "Testing of machine learning wavefront sensing algorithms on the TOTO testbed". arXiv:2607.27458.** Sim-trained NN, then augmented with real focus-diversity data. Low-order Zernikes show "reasonable agreement". Shows the sim → real augmentation pattern.

### 4b. Differentiable optics and physics-informed design

**[R56] Desdoigts, Pope, Dennis & Tuthill (2023), "Differentiable optics with ∂Lux: I — deep calibration of flat field and phase retrieval with automatic differentiation", JATIS 9, 028007. arXiv:2406.08703.**
**[R57] Desdoigts et al. (2024), "Differentiable Optics with dLux II: optical design maximising Fisher information". arXiv:2406.08704.**
- JAX-based differentiable forward models for joint calibration (flat field, pupil, aberrations) and for optimising pupil or mask design by Fisher information.
- Relevance: ideal for (i) learning detector and pupil systematics from data and (ii) designing an asymmetric pupil or phase mask that maximises information per photon.

**[R58] Landman, Keller, Por, Haffert, Doelman & Stockmans (2022), "Joint optimization of wavefront sensing and reconstruction with automatic differentiation", Proc. SPIE 12185. arXiv:2209.05904.** End-to-end optimisation of the Fourier-filter mask together with the reconstructor, minimising residual WFE.

**[R59] Por, Haffert, Radhakrishnan, Doelman, van Kooten & Bos (2018), "High Contrast Imaging for Python (HCIPy)", Proc. SPIE 10703.** The open simulator used by most of the Leiden/Arizona FPWFS papers. [verified via ADS listing 2018SPIE10703E..42P]

### 4c. Reinforcement learning and learned control

**[R60] Gutierrez, Mazoyer, Mugnier, Herscovici-Schiller & Abeloos (2024), "Image-based wavefront correction using model-free reinforcement learning", Opt. Express 32(18), 31247. arXiv:2406.18143, DOI 10.1364/OE.529415.** A model-free RL agent learns DM control from phase-diversity images, robust over a wide range of noise levels.

**[R61] Nousiainen, Taskin, Kasper, Orban de Xivry & Absil (2026), "Focal plane wavefront control with model-based reinforcement learning – I", A&A 709, A267. arXiv:2604.00993.**
- PO4NCPA takes the **current and previous focal-plane images plus the previous DM action** (sequential phase diversity, learned).
- (full text) 55 Zernike modes; ELT pupil; vortex coronagraph; inference <1 ms.
- Static NCPA: near-optimal Strehl. Dynamic NCPA: matches modal least squares plus a 1-step-delay integrator.
- The authors state that sub-ms inference makes it suitable for real-time **low-order atmospheric** correction.
- Relevance: architecturally the closest published work to our goal. It is simulation only, NCPA scale, and moderate mode count.

**[R62] Taskin, Nousiainen, Orban de Xivry, Absil & Kasper (2026), "Exploring reinforcement learning to enhance focal-plane wavefront control for vortex coronagraphs", Proc. SPIE 14150. arXiv:2609.01199.** PO4NCPA for METIS, with scalar and vector vortex coronagraphs.

**[R63] Nousiainen, Rajani, Kasper & Helin (2021), "Adaptive optics control using model-based reinforcement learning", Opt. Express 29(10), 15327. DOI 10.1364/OE.420270, arXiv:2104.13685.** Model-based RL for SH-based AO.

**[R64] Nousiainen et al. (2022), "Toward on-sky adaptive optics control using reinforcement learning (PO4AO)", A&A 664. arXiv:2205.07554, DOI 10.1051/0004-6361/202243311.** Contrast improved 3–5× in simulation and in the lab (MagAO-X team). Trains in 5–10 s, **inference <1 ms**.

**[R65] Nousiainen et al. (2026), "On-sky demonstration of reinforcement learning for adaptive optics control — PO4AO on PAPYRUS at OHP", A&A 711, A74. arXiv:2606.10771.**
- The first on-sky RL AO controller, on a 1.52 m telescope. It beat the integrator in all configurations, learned vibrations, and was robust to noise.

**[R66] Landman, Haffert, Radhakrishnan & Keller (2021), "Self-optimizing adaptive optics control with reinforcement learning for high-contrast imaging", JATIS 7(3), 039002. arXiv:2108.11332.** A model-free RL-trained recurrent NN predictive controller.

**[R67] Pou, Ferreira, Quiñones, Gratadour & Martín (2022), "Adaptive optics control with multi-agent model-free reinforcement learning", Opt. Express 30(2), 2991. ADS 2022OExpr..30.2991P.** MARL control plus an autoencoder denoiser.

**[R68] Pou, Smith, Quiñones, Martín & Gratadour (2024), "Integrating supervised and reinforcement learning for predictive control with an unmodulated pyramid wavefront sensor", Opt. Express 32(21), 37011. DOI 10.1364/OE.530254, arXiv:2405.13610.** Supervised nonlinear reconstruction combined with RL prediction.

**[R69] Fowler & Landman (2023), "Tempestas ex machina: a review of machine learning methods for wavefront control", Proc. SPIE 12680. arXiv:2309.00730.** A review of ML methods for wavefront control.

### 4d. NN reconstructors for pyramid sensors (closest kHz engineering precedent)

**[R70] Landman & Haffert (2020), "Nonlinear wavefront reconstruction with convolutional neural networks for Fourier-based wavefront sensors", Opt. Express 28(11), 16644. arXiv:2005.10560, DOI 10.1364/OE.389465.** A CNN learns the nonlinear correction on top of a linear model, which extends effective dynamic range. Shown in simulation and lab.

**[R71] Landman, Haffert, Males, Close et al. (2024), "Making the unmodulated pyramid wavefront sensor smart – closed-loop demonstration of NN wavefront reconstruction with MagAO-X", A&A. arXiv:2401.16325.**
- (full text) 1000 Fourier modes reconstructed. Dynamic range **>600 nm rms** versus about 50 nm for linear.
- >80% Strehl at 875 nm, at the PWFS sensitivity limit.
- 690 µs inference (TensorRT fp16, RTX 2080 Ti), enough for >1 kHz.
- The authors note that long-term closed-loop stability of nonlinear models cannot be guaranteed.

**[R72] Landman, Haffert, Long, Males, Close et al. (2025), "Making the unmodulated pyramid wavefront sensor smart II. First on-sky demonstration of extreme adaptive optics with deep learning", A&A. arXiv:2503.16690.**
- (full text) **1563 modes**, trained on internal-source DM-injected data at the end of the night.
- TensorRT latency **<250 µs fp32 and <125 µs fp16 on an RTX 4090**, including memory transfer.
- On sky at **2 kHz** (one run at 3.6 kHz). Strehl nearly equal to the optimised modulated PWFS on bright stars, and better on a faint star in strong wind.

**[R73] Landman, Koning, Haffert et al. (2026), "No need to modulate: on-sky results of a neural network enhanced pyramid wavefront sensor and prospects for the ELTs", SPIE AT+I 2026. arXiv:2608.24438.**
- Gains in the low and moderate Strehl regimes. Losses at high Strehl were attributed to a non-optimised training set.
- ELT simulations show gains for fast petal control.

**[R74] Wong, Norris, Deo, Tuthill, Scalzo, Sweeney, Ahn, Lozi, Vievard & Guyon (2023), "Nonlinear wavefront reconstruction from a pyramid sensor using neural networks", PASP. arXiv:2311.02595.** On SCExAO, the NN beat MVM reconstruction at all modulations. (Probably the "Wong et al." you meant; see also R34 and R36.)

**[R75] Weinberger, Tapia, Neichel & Vera (2024), "Transformer neural networks for closed-loop adaptive optics using nonmodulated pyramid wavefront sensors", A&A 687, A202. arXiv:2405.05472.** Transformers gave the best dynamic range versus sensitivity trade-off among the architectures tested.

---

## 5. Fundamental sensitivity, linearity, chromaticity and sampling

**[R76] Guyon (2005), "Limits of adaptive optics for high-contrast imaging", ApJ 629, 592. arXiv:astro-ph/0505086, DOI 10.1086/431209.**
- Defines the photon-noise sensitivity β_p.
- The Zernike phase-contrast WFS is optimal at all spatial frequencies, and SH is "significantly less sensitive". (full text) For SH, β_p grows as about 1/(f·d_sa), i.e. poor at low spatial frequency.
- A **focal-plane WFS "offers unique advantages"**: it senses the science light, has no aliasing within the control radius, and has high sensitivity.
- (full text) Kilohertz updates are feasible: for a 128×128-actuator DM with a 256×256 dark-hole sampling, the two FFTs take about 1 ms on a 2005 computer.

**[R77] Guyon (2010), "High sensitivity wavefront sensing with a nonlinear curvature wavefront sensor", PASP 122, 49. DOI 10.1086/649646.** The nlCWFS approaches the fundamental sensitivity limit through nonlinear phase retrieval on Fresnel-propagated pupil images. This conceptually supports "nonlinear reconstruction keeps sensitivity".

**[R78] Guyon (2018), "Extreme Adaptive Optics", ARA&A 56, 315. DOI 10.1146/annurev-astro-081817-052000.** Review covering WFS sensitivity, the error budget and focal-plane control.

**[R79] Ragazzoni & Farinato (1999), "Sensitivity of a pyramidic wave front sensor in closed loop adaptive optics", A&A 350, L23.** The pyramid gains about (D/r₀)² over SH for low orders in closed loop.

**[R80] Plantet, Meimon, Conan & Fusco (2015), "Revisiting the comparison between the Shack-Hartmann and the pyramid wavefront sensors via the Fisher information matrix", Opt. Express 23(22), 28619.** Fisher-information comparison: the LIFTed SH and the pyramid beat the classical SH.

**[R81] Fauvarque, Neichel, Fusco, Sauvage & Girault (2016), "General formalism for Fourier-based wave front sensing", Optica 3(12), 1440. arXiv:1611.02969.**
**[R82] Fauvarque, Janin-Potiron, Correia, Brûlé, Neichel, Chambouleyron, Sauvage & Fusco (2019), "Kernel formalism applied to Fourier-based wave-front sensing in presence of residual phases", JOSA A 36(7), 1241.**
**[R83] Chambouleyron, Fauvarque, Plantet, Sauvage, Levraud, Cissé, Neichel & Fusco (2023), "Modeling noise propagation in Fourier-filtering wavefront sensing, fundamental limits, and quantitative comparison", A&A 670, A153. arXiv:2212.13577.**
- A unified noise model (photon and read noise) for Fourier-filtering WFSs, and the fundamental sensitivity limit.
- Read noise matters a lot when the signal is spread over many pixels. That is directly relevant to a focal-plane WFS with 10⁴–10⁵ pixels at kHz.

**[R84] Chambouleyron, Fauvarque, Sauvage, Neichel & Fusco (2021), "Focal-plane-assisted pyramid wavefront sensor: enabling frame-by-frame optical gain tracking", A&A 649, A70. arXiv:2103.02297, DOI 10.1051/0004-6361/202140354.** Uses a simultaneous focal-plane image (a short-exposure PSF) to track PWFS optical gains frame by frame. The focal-plane image is informative even at moderate Strehl.

**[R85] Chambouleyron, Wallace, Jensen-Clem & Macintosh (2024), "Coronagraph-based wavefront sensors for the high Strehl regime". arXiv:2410.18000.** The bivortex WFS surpasses the Zernike WFS in sensitivity, which links optimal sensing to coronagraph design.

**[R86] Madhav (2026), "Information limits of photonic lantern wavefront sensing: a Fisher- and quantum-Fisher-information framework…". arXiv:2607.29342.**
- Lantern CRLB scales as N_ph^(−1/2) and is bounded by a quantum ceiling.
- Notes that standard photon-noise sensitivity metrics can be over-optimistic for mode-mixing sensors.
- Relevant for sensitivity claims for nonlinear or NN sensors.

**[R87] Noll (1976), "Zernike polynomials and atmospheric turbulence", JOSA 66(3), 207.** Used below for residual-variance estimates. The 1.03/0.134 (D/r₀)^{5/3} coefficients are the standard tabulated values; I did not re-check them against the full text.

**Sampling, field of view and chromaticity (synthesis from R12, R16, R49, R76; my own derivation):**
- Spatial frequency f in the pupil maps to the focal-plane position f·λ (in λ/D).
- To sense all modes a DM with N actuators across can correct, the camera must image at least ±N/2 λ/D, and preferably more to limit aliasing of uncontrolled light.
- At Nyquist (2 px per λ/D) that is ≥2N px across. Examples: N = 40 → ≥80×80 px (practically 128×128). N = 64 → ≥128 px (practically 256×256).
- Bandwidth smears speckles radially by about r·Δλ/λ. At the control edge r = N/2 λ/D, keeping the smear ≤1 λ/D needs Δλ/λ ≲ 2/N, i.e. about 5% for N = 40.
- Korkiakoski 2014 makes the same point: about 10% bandwidth works only over a limited corrected field.
- Monochromatic models (F&F, LIFT) degrade with bandwidth. NNs can learn polychromatic forward models (Andersen 2019/2020 included polychromaticity; Padrón-Brito 2025 shows 10 nm in the NIR).

---

## 6. Data-driven control and calibration usable with focal-plane data

**[R88] Guyon & Males (2017), "Adaptive optics predictive control with empirical orthogonal functions (EOFs)". arXiv:1707.00570.** A linear predictor learned from past measurements by pattern matching. More robust than earlier predictors.

**[R89] Fowler, Jensen-Clem, Cetre et al. (2026), "Ground control to major time-lag: on-sky results of data-driven predictive wavefront control at Keck Observatory", SPIE AT+I 2026. arXiv:2606.20838.**
- EOF in the Keck II RTC: 20% lower SH residuals than the integrator.
- Strehl and raw contrast on NIRC2 were comparable between the two.

**[R90] van Kooten, Jensen-Clem, Cetre, Ragland, Bond, Fowler & Wizinowich (2022), "Predictive wavefront control on Keck II adaptive optics bench: on-sky coronagraphic results", JATIS 8(2), 029006. arXiv:2205.14164.** Contrast gains of up to 2× at 3 λ/D and 3× at 3–7 λ/D (L-band vortex).

**[R91] Haffert, Males, Close, Van Gorkom, Long, Hedglen, Guyon et al. (2021), "Data-driven subspace predictive control of adaptive optics for high-contrast imaging", JATIS 7(2), 029001. arXiv:2103.07566.**
- Linear SPC updated in real time **using only measured wavefront errors and DM command changes**. Near-optimal in simulation, with large contrast gains.
- Relevance: exactly the "past DM commands plus sensor data" information set, applied to control.

**[R92] Haffert, Males et al. (2023), "Implicit electric field conjugation: data-driven focal plane control", A&A 673, A28. arXiv:2303.13719.**
- Learns the linear map from DM probes to focal-plane intensity changes **without any optical model**.
- Contrast below 10⁻⁹ in simulation. Lab gains of 10× broadband and 20–200× narrowband on MagAO-X.
- Relevance: the data-driven Jacobian idea from DM-only calibration, applied to focal-plane intensities.

**[R93] Miller, Guyon & Males (2017), "Spatial linear dark field control: stabilizing deep contrast for exoplanet imaging using bright speckles", JATIS 3(4), 049002. arXiv:1703.04259.** A linear response of bright-field intensity to DM changes is used to hold the dark field. It is model-light.

**[R94] Skaf et al. (2022), "On-sky validation of image-based adaptive optics wavefront sensor referencing (DrWHO)", A&A 659, A170. arXiv:2110.14997.** Updates the PWFS reference from focal-plane image quality. In simulation it corrected 82% of NCPA, and gave +15.7% PSF quality on sky at 750 nm.

**[R95] Codona & Kenworthy (2013), "Focal plane wavefront sensing using residual adaptive optics speckles", ApJ 767, 100. arXiv:1303.0527.** Uses WFS telemetry of the residual wavefront as diversity to estimate the focal-plane complex halo, on sky with MMT/Clio. This is "runtime information as diversity" again.

**[R96] Gerard, Marois & Galicher (2018), "Fast coherent differential imaging on ground-based telescopes using the self-coherent camera", AJ 156, 106. arXiv:1806.02881.**
**[R97] Gerard, Dillon, Cetre & Jensen-Clem (2022), "Laboratory demonstration of real-time focal plane wavefront control of residual atmospheric speckles", JATIS. arXiv:2206.08986.**
- FAST/SCC focal-plane control of AO residuals on SEAL gave up to 5× contrast gain.
- (full text) The Python loop ran at **about 50 Hz** with 1 frame delay because of a 20 ms pause. Andor Zyla <1 e⁻ at up to 1 kHz is noted for future work.

**[R98] Kasper, Fedrigo, Looze, Bonnet, Ivanescu & Oberti (2004), "Fast calibration of high-order adaptive optics systems", JOSA A 21(6), 1004. DOI 10.1364/JOSAA.21.001004.** Hadamard actuation; calibration time does not scale with actuator count.
**[R99] Heritier, Esposito, Fusco et al. (2018), "A new calibration strategy for adaptive telescopes with pyramid WFS", MNRAS. arXiv:1809.04848, DOI 10.1093/mnras/sty2485.** Pseudo-synthetic interaction matrices from fitted mis-registrations.
**[R100] Lai, Chun, Dungee, Lu & Carbillet (2021), "DO-CRIME: Dynamic On-sky Covariance Random Interaction Matrix Evaluation", MNRAS 501, 3443. arXiv:2011.14705.**
- Interaction matrix measured on sky from random DM patterns during closed loop.
- Relevance (R98–R100): ways to learn the sensor response from DM-only actuation at runtime. They are directly portable to a focal-plane sensor, e.g. as random DM dithers that double as diversity.

---

## 7. Practical: cameras, LWE/petals, baselines

### Cameras

| Camera | Type | Key specs (source) | Notes for this project |
|---|---|---|---|
| OCAM2(K) | EMCCD (CCD220), visible | Gach et al., AO4ELT2 2011 (ADS 2011aoel.confE..44G): 2067 fps full frame, <0.2 e⁻ effective RON at about 1.5 kHz [specs from search abstract] | 240×240 px is enough for ±60 λ/D at Nyquist. Excess-noise factor about √2 effectively halves QE at moderate flux. |
| C-RED One | e-APD (Saphira, HgCdTe), 0.8–2.5 µm | Gach et al. 2016, Proc. SPIE 9909, 990913 (DOI 10.1117/12.2231670): up to 3500 fps, sub-e⁻. Feautrier & Gach 2022 (arXiv:2208.00377): 320×256, **1720 fps CDS, 0.6 e⁻ at gain 50**, 30–400 e⁻/s background | The natural choice if we sense in H/K (easier bootstrapping, see §8). Background and dark at K need care. |
| C-RED 2 | InGaAs, to about 1.7 µm | Used by Bos 2020 (R16) | Higher read noise than e-APD. |
| Kinetix | sCMOS, visible | Teledyne vendor page: 0.7 e⁻ (sub-electron mode), about 500 fps full frame in 8-bit; much higher with ROI (vendor marketing mentions 5.3 kHz coronagraph imaging) [vendor, not peer-reviewed] | Cheap. A small ROI (128–256 px) at ≥1 kHz is plausible. Check rolling-shutter timing and latency. |
| ORCA-Quest qCMOS | sCMOS, photon-number-resolving | Hamamatsu vendor: 0.3 e⁻ RON (4096×2304) [vendor]; astro use e.g. arXiv:2512.14279 | Lowest RON. The kHz-rate ROI needs checking. |
| MKID (MEC) | Photon-counting energy-resolving, 800–1400 nm | Walter et al. (2020), PASP 132, 125005. arXiv:2010.12620: designed to work as a focal-plane WFS in a multi-kHz loop with SCExAO | Zero read noise plus per-photon wavelength, which would remove the chromaticity problem. Exotic and low QE. |

### LWE, island effect and petals (an argument for focal-plane sensing)

- **[R101] Sauvage et al. (2016), "Tackling down the low wind effect on SPHERE instrument", Proc. SPIE 9909, 990916. DOI 10.1117/12.2232459.**
- **[R102] Milli et al. (2018), "Low wind effect on VLT/SPHERE: impact, mitigation strategy, and results", Proc. SPIE 10703, 107032A. arXiv:1806.05370.** LWE comes from radiative cooling of the spiders. Timescales are about 1–2 s.
- **[R103] Bertrou-Cantou, Gendron, Rousset, Deo, Ferreira, Sevin & Vidal (2022), "Confusion in differential piston measurement with the pyramid wavefront sensor", A&A 658, A49.** The unmodulated PWFS confuses petal modes.
- Focal-plane sensors see petals and LWE directly (R15, R16, R21, R22, R36). That is a real advantage over SH, which is blind to differential piston across spiders.

### Baseline numbers for SH-based XAO on an 8 m

**[R104] Milli et al. (2017), "Performance of the extreme-AO instrument VLT/SPHERE and dependence on the atmospheric conditions", AO4ELT5. arXiv:1710.05417.**
- (full text) SAXO: 40×40 spatially filtered SH, 41×41 DM, up to 1.38 kHz.
- **Median H-band Strehl 80–90% in good seeing.** First quartile below 75% for R = 5–10, and below 64% for R > 10.
- Performance is strongly limited by coherence time (temporal error).

**[R105] Poyneer et al. (2014), "On-sky performance during verification and commissioning of the Gemini Planet Imager's adaptive optics system", Proc. SPIE 9148. arXiv:1407.2278.** Spatially filtered SH, Fourier reconstruction, modal gain optimisation every 8 s, LQG for tip-tilt and focus. [no numeric Strehl in the abstract]

---

## 8. Synthesis

### (a) What is proven

1. **Single-image or sequential focal-plane sensing works in closed loop on real telescopes**, but only as a slow, high-Strehl, low-order second stage:
   - F&F on SCExAO: 4–25 Hz, ≤50 Zernikes plus PTT, SRA >90% on internal LWE.
   - F&F at Keck: 30 Zernikes, minutes per iteration.
   - APF-WFS: +37% relative Strehl.
   - vAPP: 30 modes.
   - LIFT on a UT: low order.
   - FAST/SCC: 50 Hz in the lab.
   - Photonic lantern: **1 kHz but only 5 modes** (lab).
2. **Algorithms are cheap.** F&F needs about one FFT per iteration. Guyon (2005) estimated about 1 ms for 128×128 actuators in 2005. CNN reconstructors on a GPU run in **100–250 µs for ~1500 modes** (MagAO-X uPWFS, TensorRT, RTX 4090), on sky at 2–3.6 kHz.
   - Latency is therefore not the blocker, provided the code is in C++/TensorRT and not Python. Every slow FPWFS demonstration so far was limited by Python or the camera, not by the maths.
3. **CNNs reach the photon-noise (CRB) limit** for focal-plane PR when diversity is present and WFE ≲1 rad, for ≤100 modes (Orban de Xivry 2021). NN reconstructors keep sensitivity while extending dynamic range, by about 12× for the uPWFS (Landman 2024).
4. **The even-mode sign ambiguity is solvable** without a second camera, in several ways:
   - temporal or sequential diversity from known DM commands (F&F, PO4NCPA);
   - static optical diversity: asymmetric pupil, vortex, vAPP or hologram, lantern, multimode fibre;
   - temporal amplitude chopping.
5. **Data-driven control from DM commands plus sensor data is mature and on sky:**
   - EOF at Keck;
   - SPC (simulation and lab);
   - PO4AO RL, on sky in 2026;
   - iEFC (lab);
   - DO-CRIME and Hadamard calibration.
6. **In principle, focal-plane sensing beats SH on sensitivity**, especially at low spatial frequency (Guyon 2005; Chambouleyron 2023). It also sees petals and LWE and has no NCPA.

### (b) Open gaps this project could fill

1. **No published demonstration**, lab or on sky, of a *first-stage* AO loop that uses only a focal-plane camera to correct full atmospheric turbulence on an 8 m pupil at ≥500 Hz with hundreds of modes.
   - The closest results are: PO4NCPA (simulation, NCPA, 55 modes); Sandler 1991 (low order, on sky, early); Andersen 2020 (simulation, D/r₀ 12–21, 36 modes, residual about 1.5 rad); and the photonic lantern at 1 kHz (5 modes).
2. **Bootstrapping and acquisition from seeing-limited conditions.**
   - Linear FPWFS needs ≲1–1.5 rad.
   - Single-shot CNNs on large aberrations saturate at about 1.4–2.3 rad residual (Paine & Fienup 2018; Andersen 2020).
   - Nobody has shown a staged acquisition scheme that hands over to a high-precision focal-plane loop in turbulence.
3. **Temporal phase retrieval.** An estimator that jointly uses the image history and DM-command history (sequential diversity plus prediction), i.e. a recurrent or transformer F&F, does not exist outside PO4NCPA's two-frame version.
   - At 1 kHz the turbulence changes only about 0.06–0.18 rad rms per frame (my estimate: 10 m/s wind and r₀ = 0.85 m at K versus 0.25 m at I, 0.7″ seeing). Temporal diversity is therefore well posed, but the "static phase between frames" assumption of F&F is violated at the level of the closed-loop residual.
4. **Sim-to-real for FPWFS under turbulence.** Physics-in-the-loop self-supervision (R51, R56) has not been combined with on-bench turbulence and DM-only injected calibration (R72's recipe) for a focal-plane sensor.
5. **High mode count with a focal-plane image at kHz**: 500–1500 modes needs a 128²–256² px region of interest. The chromaticity, read-noise and pixel-count trade at kHz has not been quantified end to end.

### (c) Most promising candidate approaches

**1. "Learned F&F": recurrent NN with sequential (DM-command) diversity, physics-initialised.**
- Inputs: the last k frames, the last k DM commands, and optionally an F&F linear estimate as an extra channel. Output: modal residuals or DM updates.
- Train in simulation (HCIPy or dLux), then fine-tune on bench and on-sky data where the DM injects known modes (R72 recipe) and self-supervised through a differentiable forward model (R51/R56).
- Pros: zero extra hardware; uses exactly the allowed runtime information; strong physical prior; GPU latency is proven (≤250 µs).
- Cons: the sign of even modes is only observable through diversity, and closed-loop DM steps are small, so the problem is poorly conditioned at high Strehl. Dither injection may be needed (DO-CRIME-like random small dithers, at a small Strehl cost). Bootstrapping needs a separate mode.

**2. Single-shot unambiguous optics plus CNN: an asymmetric pupil mask (APF/kernel), a dLux-optimised pupil or phase mask, or a vAPP/holographic mode sensor.**
- Pros: every frame is unambiguous, so the loop is stable from a cold start and temporal diversity is not needed. A Fisher-optimised mask (R57/R58) can approach the sensitivity limit.
- Cons:
  - Throughput or Strehl loss from the mask: an asymmetric mask blocks a few % of the pupil; holograms divert light.
  - The mask must stay registered.
  - It is arguably still a "single focal-plane camera", but it adds an optic. Confirm this is within the project rules.

**3. Wavelength-staged sensing with a NIR e-APD camera (C-RED One in H/K).**
- Sensing at long λ shrinks phase in radians. For an 8 m telescope at 0.7″ seeing, with tip-tilt removed and no outer scale (Noll; my calculation): about **6.5 rad rms at 0.8 µm, 3.2 at H, 2.4 at K**.
- Tip-tilt is centroided trivially. The CNN-capture regime (R46/R48) then hands over to a linear or NN regime within a few iterations.
- Pros: makes bootstrapping tractable; there are more photons per λ/D; the e-APD gives 0.6 e⁻ at 1.7 kHz.
- Cons: fewer λ/D across the control radius at K for a given pixel count; thermal background; a correction at K does not deliver visible Strehl, though that matches SH at H/K.

**4. Hybrid controller: an NN or linear focal-plane estimator plus data-driven prediction (SPC, EOF or PO4AO-style model-based RL) on pseudo-open-loop modal estimates.**
- The estimator gives pseudo-open-loop modes and the predictor removes servo lag. A model-based RL policy can also learn sensing and control end to end from images (PO4NCPA → PO4AO).
- Pros: tackles temporal error, which dominates SAXO's error budget at short τ₀; uses only runtime data.
- Cons: RL sample efficiency and stability guarantees; harder to validate; nonlinear-loop instability (noted in R71).

### (d) Key risks

- **Sign ambiguity and conditioning.** Sequential diversity degrades exactly when the loop is converged and DM steps are tiny. Mitigate with small known dithers (sacrificing some Strehl), a pupil asymmetry, or temporal priors.
- **Dynamic range and bootstrapping.** Linear validity is about 1–1.5 rad. Single-shot NN residuals saturate at about 1.5–2 rad for D/r₀ ~ 12–21. Phase wrapping and LWE above λ/2 add to this.
  - Mitigations: λ staging, a coarse-to-fine NN cascade, multi-frame acquisition, and an initial slow loop (sensorless/Booth) for low order.
- **Latency.** Not fundamental (GPU CNN at 100–250 µs, FFT-based F&F at ≪1 ms), but camera readout and ROI, and keeping Python out of the loop, are essential. All slow demonstrations were software-limited.
- **Noise.** A photon-noise-limited NN is possible, but read noise × pixel count matters at kHz (R83). Use EMCCD, e-APD, qCMOS or MKID, and bin outside the control radius.
- **Chromaticity.** Radial speckle smear needs Δλ/λ ≲ 2/N_act-across: about 5% for 40 actuators, which costs photons. A polychromatic forward model in training, or an energy-resolving detector (MKID), would help.
- **Sim-to-real gap.**
  - Sources: DM influence functions and hysteresis, pupil registration and rotation, detector nonlinearity, vibrations, scintillation and amplitude errors (F&F is phase-only), and non-Kolmogorov LWE.
  - Mitigations: train on internal-source DM-injected data (R72), focal-plane-only registration (R14), self-supervised fine-tuning (R51), and dLux-style calibration (R56).
- **Closed-loop stability of nonlinear estimators.** There are no formal guarantees. Out-of-distribution states such as cloud, poor seeing or a dropped frame can diverge. A watchdog and fallback to a linear integrator are needed.
- **Sensitivity versus SH at high order.** The focal-plane advantage is largest at low and mid spatial frequencies. At the control edge, photon counts per speckle are low (Guyon 2005: often <10 photons per speckle per frame), so focal-plane estimates there are noisy.

### (e) Baseline numbers to compare against

| Metric | Value | Source |
|---|---|---|
| SH XAO on an 8 m | 40×40 SH, 1.38 kHz; median H Strehl 80–90% in good seeing; first quartile <75% (R = 5–10) | R104 (SPHERE/SAXO) |
| NN WFS in a kHz RTC | 1563 modes, <250 µs fp32 / <125 µs fp16 (RTX 4090), 2 kHz on sky (3.6 kHz once); Strehl ≈ modulated PWFS | R72 |
| NN dynamic range | >600 nm rms versus about 50 nm for linear (uPWFS) | R71 |
| Focal-plane NN accuracy | rms < λ/1500 at 2×10⁶ photons, 20 modes, ≤1 rad input (photon-limited) | R49 |
| Large-aberration single-shot NN | D/r₀ 12–21 → about 1.4–2 rad residual (36 modes), 8 ms inference | R48 |
| F&F on sky | 4–25 Hz (Python); 300–400 Hz projected in C; ≤50 Zernikes + PTT; SRA >90% on 0.4–2 µm PV LWE (internal) | R16 |
| F&F small-phase limit | about 1.5 rad rms tolerated | R11 |
| vAPP FPWFS | about λ/8 rms max per iteration; 30 modes | R32 |
| cMWFS dynamic range | ±2.5 rad rms | R31 |
| Photonic lantern closed loop | 1 kHz, 5 modes, about 95% of injected WFE corrected | R35 |
| Focal-plane SCC control | about 50 Hz lab loop, up to 5× contrast | R97 |
| RL controller | <1 ms inference; trains in 5–10 s; on sky beats the integrator | R64, R65 |
| Data-driven prediction on sky | EOF at Keck: −20% SH residual versus integrator | R89 |
| Cameras | OCAM2K about 2 kHz, <0.2 e⁻; C-RED One 1720 fps, 0.6 e⁻; Kinetix 0.7 e⁻ (vendor) | §7 |

**Suggested simulation baseline:**
- An 8 m pupil with spiders.
- A 40×40 SH at 1 kHz with a 2-frame delay and an integrator, plus an optimised-gain version, over a grid of r₀, τ₀ and magnitude.
- Matched DM (41×41) and matched photon budget.
- Report Strehl at the science λ, residual by spatial frequency, and limiting magnitude.
- The focal-plane WFS should be compared at equal photons (all light to the camera versus all light to the SH).
