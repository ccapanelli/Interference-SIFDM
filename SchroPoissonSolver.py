import numpy as np
import pyfftw
import multiprocessing
import scipy.fft

"Simulate self-interacting non-linear Schrodinger. Uncomment self-gravity terms as desired."

pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()


def fdmSimulation(psi0, normalization, L, G_N, g_SI, tvals, tsaved):
    pyfftw.interfaces.cache.enable()

    # Simulation parameters
    N         = np.shape(psi0)[0]    # Spatial resolution
    t         = 0      # current time of the simulation
    dt        = tvals[1]-tvals[0]
    G         = G_N # Gravitaitonal constant
    g         = g_SI # self-interaction strength

    # Domain [-L/2,L/2] x [-L/2,L/2] 
    xlin = np.linspace(-L/2,L/2, num=N+1)# Note: periodic boundary
    xlin = xlin[0:N]                     # chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin, copy=False, sparse=False)

    # normalize wavefunction to <|psi|^2>=rho / (m psi_0^2)
    rho0 = np.abs(psi0)**2
    rhobar = np.average(rho0)
    psi = psi0*np.sqrt(normalization/rhobar)


    # Fourier Space Variables
    klin = 2 * np.pi * np.fft.fftfreq(N, L/N)
    kx, ky = np.meshgrid(klin, klin, copy=False, sparse=False)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx**2 + ky**2

    # Potential
#     Vhat = -np.fft.fftn(G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
#     V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2

    V = g*np.abs(psi)**2
    
    # number of timesteps
    Nt = tvals.shape[0]
    tsaved = np.sort(tsaved) # enforce selected snapshot times are ordered
    
    # create array to store contrast field at each time step
    a , b = np.shape(psi)
    psiFieldSlices = []
    psiFieldSlices.append(psi)
    contrastVals = []
    contrastVals.append( np.std(np.abs(psi)**2)/np.average(np.abs(psi)**2) )
    t_recorded = []
    t_recorded.append(0)

    A = pyfftw.empty_aligned((N, N), dtype='complex128')
    B = pyfftw.empty_aligned((N, N), dtype='complex128')
    C = pyfftw.empty_aligned((N, N), dtype='complex128')

    # Simulation Main Loop
    i = 0
    for t in tvals:
        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi
        
        A[:] = psi

        # Use the backend pyfftw.interfaces.scipy_fft
        with scipy.fft.set_backend(pyfftw.interfaces.scipy_fft):
            # Turn on the cache for optimum performance
            pyfftw.interfaces.cache.enable()

            # drift
            psihat = scipy.fft.fftn(A)
            B[:] = np.exp(dt * (-1.j*kSq))  * psihat
            psi = scipy.fft.ifftn(B)
            # update potential
#             C[:] = -scipy.fft.fftn(G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
#             V = np.real(scipy.fft.ifftn(C)) + g*np.abs(psi)**2
            V = g*np.abs(psi)**2

        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi

            
        if np.any(np.abs(t - tsaved) < dt / 2):  # Tolerance around dt/2 for comparison
            psiFieldSlices.append(psi)
            t_recorded.append(t)
   
        contrastVals.append( np.std(np.abs(psi)**2)/np.average(np.abs(psi)**2) )
        
        i +=1

    return np.asarray(t_recorded), np.asarray(contrastVals), np.asarray(psiFieldSlices)
