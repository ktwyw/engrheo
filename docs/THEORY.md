# Methods and references

The formulas behind every function, with sources. SI units throughout.

## Steady-shear models (`models`)

| Model | Stress | Source |
|---|---|---|
| Power law | $\tau = K\dot\gamma^n$ | Ostwald (1925), de Waele (1923) |
| Cross | $\eta = \eta_\infty + (\eta_0 - \eta_\infty)/(1 + (\lambda\dot\gamma)^m)$ | Cross, *J. Colloid Sci.* 20 (1965) |
| Carreau | $\eta = \eta_\infty + (\eta_0 - \eta_\infty)[1 + (\lambda\dot\gamma)^2]^{(n-1)/2}$ | Carreau, *Trans. Soc. Rheol.* 16 (1972) |
| Carreau-Yasuda | $\eta = \eta_\infty + (\eta_0 - \eta_\infty)[1 + (\lambda\dot\gamma)^a]^{(n-1)/a}$ | Yasuda, Armstrong & Cohen, *Rheol. Acta* 20 (1981) |
| Sisko | $\eta = \eta_\infty + K\dot\gamma^{n-1}$ | Sisko, *Ind. Eng. Chem.* 50 (1958) |
| Bingham | $\tau = \tau_y + \mu_p\dot\gamma$ | Bingham (1922) |
| Herschel-Bulkley | $\tau = \tau_y + K\dot\gamma^n$ | Herschel & Bulkley, *Kolloid-Z.* 39 (1926) |
| Casson | $\sqrt\tau = \sqrt{\tau_y} + \sqrt{\eta_c\dot\gamma}$ | Casson (1959) |

## Fitting (`fitting`)

Least squares on log residuals $r_i = \ln(\tau_{model}/\tau_i)$ (trust-region reflective algorithm with bounds);
covariance $s^2(J^TJ)^{-1}$ from the Jacobian at the optimum; Wald intervals with $t_{n-p}$; AIC
$= n\ln(\text{SSR}/n) + 2p$.

## Rheometer geometries (`geometry`)

- **Cone-plate:** $\tau = 3M/(2\pi R^3)$, $\dot\gamma = \Omega/\theta$ (small angle).
- **Parallel plates:** $\dot\gamma_R = \Omega R/h$; $\tau_R = \frac{M}{2\pi R^3}\left(3 + \frac{d\ln M}{d\ln\dot\gamma_R}\right)$.
- **Couette:** $\tau_b = M/(2\pi R_i^2 L)$; Newtonian $\dot\gamma_b = 2\Omega R_o^2/(R_o^2 - R_i^2)$; power-law (Krieger)
  correction $\dot\gamma_b = 2\Omega/[n'(1 - (R_i/R_o)^{2/n'})]$, $n' = d\ln M/d\ln\Omega$ (Krieger & Maron, *J. Appl. Phys.* 25 (1954)).
  General fluid: $\Omega = \int_{R_i}^{R_o}\dot\gamma(\tau(r))\,dr/r$, $\tau(r) = M/(2\pi r^2 L)$; Bingham exact solution with partially
  sheared gap: Reiner & Riwlin (1927).
- **Capillary:** $\tau_w = \Delta P R/(2L)$; Weissenberg-Rabinowitsch $\dot\gamma_w = \dot\gamma_a(3n' + 1)/(4n')$,
  $n' = d\ln\tau_w/d\ln\dot\gamma_a$ (Rabinowitsch, *Z. Phys. Chem.* A145 (1929)); Bagley end correction
  $\Delta P = 2\tau_w(L/R + e)$ (Bagley, *J. Appl. Phys.* 28 (1957)); Mooney wall slip
  $\dot\gamma_a = \dot\gamma + 4u_s/R$ (capillaries) or $+ 2u_s/h$ (plates) (Mooney, *J. Rheol.* 2 (1931)).
- **Fann 35 (API RP 13B-1/13D):** $\dot\gamma = 1.7023\,N$ (rpm), $\tau = 0.5113\,\theta$ (Pa); PV $= \theta_{600} - \theta_{300}$,
  YP $= \theta_{300} -$ PV; $n = 3.32\log_{10}(\theta_{600}/\theta_{300})$, $K = 0.511\,\theta_{300}/511^n$.

## Pipe flow (`flows`)

- Power law: $Q = \pi R^3\frac{n}{3n+1}(\tau_w/K)^{1/n}$; velocity profile
  $u = \frac{n}{n+1}\left(\frac{\Delta P}{2KL}\right)^{1/n}(R^{1+1/n} - r^{1+1/n})$.
- Herschel-Bulkley (Bingham for $n = 1$, Buckingham-Reiner equation):
  $Q = \frac{\pi R^3}{\tau_w^3 K^{1/n}}(\tau_w - \tau_y)^{1+1/n}\left[\frac{(\tau_w-\tau_y)^2}{3+1/n} + \frac{2\tau_y(\tau_w-\tau_y)}{2+1/n} + \frac{\tau_y^2}{1+1/n}\right]$;
  plug radius $R\tau_y/\tau_w$ (Chhabra & Richardson, *Non-Newtonian Flow and Applied Rheology*, 2nd ed., 2008).
- Metzner-Reed Reynolds number $Re_{MR} = \rho V^{2-n}D^n/[8^{n-1}K((3n+1)/(4n))^n]$, laminar $f = 16/Re_{MR}$
  (Metzner & Reed, *AIChE J.* 1 (1955)); critical value $6464n(2+n)^{(2+n)/(1+n)}/(1+3n)^2$ (Ryan & Johnson,
  *AIChE J.* 5 (1959)); turbulent $1/\sqrt f = 4n^{-0.75}\log_{10}(Re\,f^{1-n/2}) - 0.4n^{-1.2}$ (Dodge & Metzner,
  *AIChE J.* 5 (1959)); Bingham laminar friction with the Hedstrom number $He = \rho\tau_yD^2/\mu_p^2$.

## Temperature and time-temperature superposition (`tts`)

WLF: $\log_{10}a_T = -C_1(T - T_{ref})/(C_2 + T - T_{ref})$ (Williams, Landel & Ferry, *J. Am. Chem. Soc.* 77 (1955));
Arrhenius: $\ln a_T = (E_a/R)(1/T - 1/T_{ref})$. Shift factors minimise the mean squared difference of $\ln G'$, $\ln G''$
between neighbouring curves over their overlap (log-log interpolation), chained outwards from $T_{ref}$.

## Linear viscoelasticity (`viscoelastic`)

Discrete spectrum $G(t) = G_e + \sum g_i e^{-t/\tau_i}$; $G' = G_e + \sum g_i\omega^2\tau_i^2/(1 + \omega^2\tau_i^2)$, $G'' = \sum g_i\omega\tau_i/(1 + \omega^2\tau_i^2)$;
$\eta_0 = \sum g_i\tau_i$, $J_e^0 = \sum g_i\tau_i^2/(\sum g_i\tau_i)^2$. Boltzmann superposition $\sigma(t) = \int G(t - s)\,d\gamma(s)$. Interconversion by
solving $\int_0^t G(t - s)\,dJ(s) = 1$ with product integration (J piecewise linear, exact interval means of $G$ from its
cumulative integral). Spectrum fitting: fixed log grid of $\tau_i$, non-negative least squares on relative residuals,
optional Tikhonov smoothing (Honerkamp & Weese, *Rheol. Acta* 32 (1993); Baumgaertel & Winter, *Rheol. Acta* 28 (1989)).
Burgers creep $J = 1/G_1 + t/\eta_1 + (1 - e^{-tG_2/\eta_2})/G_2$.

## Oscillatory rheology (`oscillatory`)

$|G^*| = \sqrt{G'^2 + G''^2}$, $\tan\delta = G''/G'$, $|\eta^*| = |G^*|/\omega$; first-harmonic moduli by least-squares projection of the
stress on $\sin\omega t$, $\cos\omega t$; gel point where $\tan\delta$ is frequency independent (Winter & Chambon, *J. Rheol.* 30 (1986));
Cox-Merz rule $\eta(\dot\gamma) = |\eta^*|(\omega = \dot\gamma)$ (Cox & Merz, *J. Polym. Sci.* 28 (1958)).

## Nonlinear constitutive models (`constitutive`)

Upper-convected derivative; per mode $\boldsymbol\tau + \lambda\overset{\nabla}{\boldsymbol\tau} + (\alpha\lambda/\eta_p)\,\boldsymbol\tau\cdot\boldsymbol\tau = 2\eta_p\mathbf D$ (Giesekus, *J. Non-Newt. Fluid
Mech.* 11 (1982)); linear PTT $f(\mathrm{tr}\,\boldsymbol\tau)\boldsymbol\tau + \lambda\overset{\nabla}{\boldsymbol\tau} = 2\eta_p\mathbf D$, $f = 1 + (\varepsilon\lambda/\eta_p)\mathrm{tr}\,\boldsymbol\tau$ (Phan-Thien & Tanner,
*J. Non-Newt. Fluid Mech.* 2 (1977)); UCM for $\alpha = \varepsilon = 0$. Solved as ODEs (LSODA) in simple shear and uniaxial extension.
Steady Giesekus viscosity in closed form: Bird, Armstrong & Hassager, *Dynamics of Polymeric Liquids*, vol. 1 (1987).

## Large-amplitude oscillatory shear (`laos`)

Odd harmonics $G'_n$, $G''_n$ by least squares; Chebyshev coefficients $e_n = G'_n(-1)^{(n-1)/2}$, $v_n = G''_n/\omega$; $G'_M = \sum nG'_n$,
$G'_L = \sum e_n$, $S = (G'_L - G'_M)/G'_L$, $T = (\eta'_L - \eta'_M)/\eta'_L$ (Ewoldt, Hosoi & McKinley, *J. Rheol.* 52 (2008)).

## Thixotropy (`thixotropy`)

Structural kinetics $d\lambda/dt = k_b(1 - \lambda) - k_d\dot\gamma\lambda$, $\tau = \lambda\tau_y + (\eta_\infty + \Delta\eta\lambda)\dot\gamma$ (Moore, *Trans. Brit. Ceram. Soc.* 58
(1959); Houska; Mewis & Wagner, *Adv. Colloid Interface Sci.* 147-148 (2009)); exact exponential solution for
piecewise-constant rates; stiff integration for ramps.

## Suspensions (`suspensions`)

Einstein $1 + 2.5\phi$; Batchelor $1 + 2.5\phi + 6.2\phi^2$ (*J. Fluid Mech.* 83 (1977)); Krieger-Dougherty $(1 - \phi/\phi_m)^{-[\eta]\phi_m}$
(*Trans. Soc. Rheol.* 3 (1959)); Quemada $(1 - \phi/\phi_m)^{-2}$.

## Extensional rheology (`extensional`)

Newtonian capillary thinning $D = 0.1418(\sigma/\eta)(t_c - t)$ (Papageorgiou, *Phys. Fluids* 7 (1995)); elasto-capillary
$D \propto e^{-t/(3\lambda_E)}$ (Entov & Hinch, *J. Non-Newt. Fluid Mech.* 72 (1997)); apparent extensional viscosity
$(2X - 1)\sigma/(-dD/dt)$ with $X = 0.7127$ (McKinley & Tripathi, *J. Rheol.* 44 (2000)).

## Polymers and processing (`polymers`, `flows`)

$\eta_0 \propto M$ below $M_c$, $\propto M^{3.4}$ above (Berry & Fox, *Adv. Polym. Sci.* 5 (1968)); $M_e = \rho RT/G_N^0$; melt flow index die
(ISO 1133: 2.095 mm x 8.000 mm, piston 9.55 mm). Slit flow of a power-law fluid
$Q = 2W(\Delta P/KL)^{1/n}(H/2)^{2+1/n}n/(2n+1)$; Metzner-Otto mixing $\dot\gamma_{avg} = k_sN$, $P = K_p\eta_{app}N^2D^3$ (Metzner & Otto,
*AIChE J.* 3 (1957)); Tanner die swell $0.1 + [1 + \tfrac12(N_1/2\tau_w)^2]^{1/6}$ (Tanner, *J. Polym. Sci. A-2* 8 (1970)).

## General references

Macosko, *Rheology: Principles, Measurements, and Applications* (1994) · Barnes, Hutton & Walters,
*An Introduction to Rheology* (1989) · Chhabra & Richardson, *Non-Newtonian Flow and Applied Rheology*
(2008) · Mezger, *The Rheology Handbook* · Coussot, *Rheometry of Pastes, Suspensions, and Granular
Materials* (2005).
