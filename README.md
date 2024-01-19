# Interference-SIFDM
Study interference in self-interacting fuzzy dark matter system. 

This repo contains three modules:

### RandomPhaseGenerator.py 
Use this module to generate a 2d density field from randomly interfering plane waves. E.g.

```
from RandomPhaseGenerator import RandomPhase

density = RandomPhase(resolution, debroglie_length, box_size)
```
* `resolution` - `dtype = int`- spatial resolution in one dimension.
* `debroglie_length` - `dtype = float` - as a fraction of the box size. This is not preserved over time!
* `box_size` - `dtype = float`- best set to 1. Spatial domain.

### SchroPoissonSolver.py
Use this module to evolve 2d density field forward in time according to Schrodinger-Poisson equations. 

```
from SchroPoissonSolver import fdmSimulation

psi, contrastVals, psiSlices = fdmSimulation(rho0, normalization, G_N, g_SI, tEnd, dt, Nt_saved)
```
* `rho0` - 2d array, `dtype = float` -the initial denisity field.
* `normalization` - `dtype = float` - is the average (dimensionless) density of the initial configuration.
* `G_N` and `g_SI` - `dtype = float`- the dimensionless Newton's constant and coupling constant.
* `tEnd` - `dtype = float`- the total simulation time.
* `dt` - `dtype = float`- the timestep.
* `Nt_saved` - `dtype = int`- the number of time slices to be saved, chosen uniformly from the last fifth of the run-time.

The function `Center()` can be used to put transform the output into a center of mass frame.

### PowerSpectrum.py
Use this module to compute the power spectrum of a 2d array.

```
from PowerSpectrum import PowerSpectrum

k_modes, Power_Spectrum = PowerSpectrum(input_density)
```
## Notebooks
`Coupling.ipynb` may be used to determine relevant physical coupling constants and simulation parameters.

`Interference-Simulation.ipynb` serves as a general testing ground. You can generate initial conditions, run simulations, compute power spectra, density profiles, and velocity distributuions. This is useful to see what parameters are viable.

`Automated-Simulation-Runs.ipynb` is where one would run multiple simulations at different couplings and compare the cutoffs.

`Manual Testing.ipynb` is where I compare two benchmark values of the coupling, rather than looping over many values.
