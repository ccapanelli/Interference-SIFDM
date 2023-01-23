# Interference-SIFDM
Study interference in self-interacting fuzzy dark matter system. 

This repo contains three modules:

### RandomPhaseGenerator.py 
Use this module to generate a 2d density field from randomly interfering plane waves. E.g.

```
from RandomPhaseGenerator import RandomPhase

density = RandomPhase(resolution, number_modes, debroglie_length, box_size)
```

### SchroPoissonSolver.py
Use this module to evolve 2d density field forward in time according to Schrodinger-Poisson equations. 

```
from SchroPoissonSolver import fdmSimulation

psiFieldSlices, contrastVals, phases = fdmSimulation(rho0, g_SI, tEnd, dt)
```

The function `Center()` can be used to put transform the output into a center of mass frame.

### PowerSpectrum.py
Use this module to compute the power spectrum of a 2d array.

```
from PowerSpectrum import PowerSpectrum

k_modes, Power_Spectrum = PowerSpectrum(input_density)
```
