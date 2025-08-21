# Interference-SIFDM
Study interference in self-interacting fuzzy dark matter system. 

This repo contains three modules:

### RandomPhaseGenerator.py 
Use this module to generate a 2d complex field from randomly interfering plane waves. E.g.

```
from RandomPhaseGenerator import RandomPhase

psi0 = RandomPhase(resolution, debroglie_length, box_size)
```
* `resolution` - `dtype = int`- spatial resolution in one dimension.
* `debroglie_length` - `dtype = float` - as a fraction of the box size. This is not preserved over time!
* `box_size` - `dtype = float`- Linear extent of spatial domain.
* `user_seed` - `dtype = int` - user generated seed. Default set to `None`.

### SchroPoissonSolver.py
Use this module to evolve 2d field forward in time according to Schrodinger-Poisson equations. 

```
from SchroPoissonSolver import fdmSimulation

psi, contrastVals, psiSlices = fdmSimulation(psi0, normalization, box_size, G_N, g_SI, tvals, tsaved)
```
* `psi0` - 2d array, `dtype = float` -the initial field configuration.
* `normalization` - `dtype = float` - is the average (dimensionless) density of the initial configuration.
* `box_size` - `dtype = float`- Linear extent of spatial domain.
* `G_N` and `g_SI` - `dtype = float`- the dimensionless Newton's constant and coupling constant. (Gravity is commented out, so `G_N` has no effect by default.)
* `tvals` - `dtype = float`- time steps to evaluate the simulation.
* `tsaved` - `dtype = float`- user-sepcified list of time slices to be saved.

### PowerSpectrum.py
Use this module to compute the power spectrum of a 2d or 3d array.

```
from PowerSpectrum3d import PowerSpectrum3d

k_modes, Power_Spectrum = PowerSpectrum(input_density, box_size)
```
* `input_density` - `dtype = float` - 2d density field.
* `box_size` - `dtype = float`- Linear extent of spatial domain.
