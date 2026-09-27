"""engrheo - rheology for engineers: readable, validated Python for flow curves, rheometer corrections,
viscoelasticity and non-Newtonian flow.

Modules
-------
models      steady-shear models (power law, Cross, Carreau(-Yasuda), Sisko, Bingham, Herschel-Bulkley, Casson)
fitting     flow-curve fitting in log space with uncertainties; model comparison by AIC
geometry    rheometer conversions and corrections (cone-plate, parallel plates, Couette, capillary;
            Weissenberg-Rabinowitsch, Bagley, Mooney)
flows       non-Newtonian pipe flow: profiles, flow rate, pressure drop, Metzner-Reed, friction factors
viscoelastic  linear viscoelastic models, discrete spectra, Boltzmann superposition, interconversion, spectrum fitting
oscillatory   complex modulus, crossover, LVE limit, Winter-Chambon gel point, moduli from waveforms, Cox-Merz
tts         time-temperature superposition: shift factors, master curves, WLF and Arrhenius
constitutive  UCM/Oldroyd-B, Giesekus, PTT: start-up and steady shear, extension, LAOS
laos        Fourier harmonics and Chebyshev decomposition of large-amplitude oscillations
thixotropy  structural-kinetics model: build-up, breakdown, hysteresis loops
suspensions Einstein, Batchelor, Krieger-Dougherty, Quemada
extensional Trouton ratio, capillary-thinning (CaBER) analysis
polymers    molecular-weight scaling, entanglement molar mass, melt flow index
datasets    bundled CSV data files with sources
"""

__version__ = "0.1.0"

from . import (
               constitutive,
               datasets,
               extensional,
               fitting,
               flows,
               geometry,
               laos,
               models,
               oscillatory,
               polymers,
               suspensions,
               thixotropy,
               tts,
               viscoelastic,
)

__all__ = ["constitutive", "datasets", "extensional", "fitting", "flows", "geometry", "laos", "models", "oscillatory",
           "polymers", "suspensions", "thixotropy", "tts", "viscoelastic", "__version__"]
