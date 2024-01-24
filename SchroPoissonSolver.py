"""
Create Your Own Quantum Mechanics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate the Schrodinger-Poisson system with the Spectral method

Include self-interactions, calculate density contrast, and center plots around cores
Christian Capanelli (2022), McGill University
"""

import numpy as np

def fdmSimulation3d(rho0, normalization, L, G_N, g_SI, tEnd, dt, Nt_saved):
    
    """ Quantum simulation """

    # Simulation parameters
    N         = np.shape(rho0)[0]    # Spatial resolution
    t         = 0      # current time of the simulation
    tOut      = 0.0001  # draw frequency
    G         = G_N # Gravitaitonal constant
    g         = g_SI #1e-2*np.pi*G # self-interaction strength
    plotRealTime = False # switch on for plotting as the simulation goes along

    # Domain [0,1] x [0,1]
#     L = 1    
    xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                     # chop off periodic point
    xx, yy, zz = np.meshgrid(xlin, xlin, xlin, copy=False)

    # normalize wavefunction to <|psi|^2>=rho / (m psi_0^2)
    rhobar = np.mean( rho0 )
    rho0 = rho0/rhobar*normalization
    psi = np.sqrt(rho0)


    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
    kx, ky, kz = np.meshgrid(klin, klin, klin, copy=False)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kz = np.fft.ifftshift(kz)
    kSq = kx**2 + ky**2 + kz**2

    # Potential
    Vhat = -np.fft.fftn(G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
    V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    tvals = []
    
    # create array to store contrast field at each time step
    a , b, c = np.shape(psi)
#     rhoAvg = np.zeros((a,b))
    psiFieldSlices = np.zeros(( Nt_saved , a, b, c), dtype = complex)
    contrastVals = np.zeros((Nt))
#     phases = np.zeros((Nt , a, b))
    
    j=0
    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi

        # drift
        psihat = np.fft.fftn(psi)
        psihat = np.exp(dt * (-1.j*kSq/2.))  * psihat
        psi = np.fft.ifftn(psihat)

        # update potential
        Vhat = -np.fft.fftn(G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
        V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2

        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi
        
        # record last few field configurations
        
        if i > (Nt/2) and (i)%(Nt/(2*Nt_saved)) < 1. and j < Nt_saved:
            psiFieldSlices[j,:,:,:] = psi
            j = j + 1
            
        #rhoAvg += np.abs(psi)**2
        # psiFieldSlices[i, :,:] = np.abs(psi)**2
        contrastVals[i] = np.std(np.abs(psi)**2)/np.average(np.abs(psi)**2)
        # phases[i,:,:] = np.angle(psi)
        
        # update time
        tvals.append(t)
        t += dt
        
    psiFieldSlices[-1,:,:,:] = psi


    return contrastVals, psiFieldSlices, tvals

def fdmSimulation2d(rho0, normalization, L, G_N, g_SI, tEnd, dt, Nt_saved):
    
    """ Quantum simulation """

    # Simulation parameters
    N         = np.shape(rho0)[0]    # Spatial resolution
    t         = 0      # current time of the simulation
    tOut      = 0.0001  # draw frequency
    G         = G_N # Gravitaitonal constant
    g         = g_SI #1e-2*np.pi*G # self-interaction strength
    plotRealTime = False # switch on for plotting as the simulation goes along

    # Domain [0,1] x [0,1]
#     L = 1    
    xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                     # chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin, copy=False)

    # normalize wavefunction to <|psi|^2>=rho / (m psi_0^2)
    rhobar = np.average( rho0 )
    rho0 = rho0/rhobar*normalization
    psi = np.sqrt(rho0)


    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
    kx, ky = np.meshgrid(klin, klin, copy=False)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx**2 + ky**2

    # Potential
#     Vhat = -np.fft.fftn(G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
#     V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2

    V = g*np.abs(psi)**2
    
    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    tvals = []
    
    # create array to store contrast field at each time step
    a , b = np.shape(psi)
#     rhoAvg = np.zeros((a,b))
    psiFieldSlices = np.zeros(( Nt_saved , a, b), dtype = complex)
    contrastVals = np.zeros((Nt))
#     phases = np.zeros((Nt , a, b))
    
    j=0
    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi

        # drift
        psihat = np.fft.fftn(psi)
        psihat = np.exp(dt * (-1.j*kSq/2.))  * psihat
        psi = np.fft.ifftn(psihat)

        # update potential
#         Vhat = -np.fft.fftn(G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
#         V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2
        V = g*np.abs(psi)**2

        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi
        
        # record last few field configurations
        
        if i > (Nt/4) and (i)%(Nt/(4*Nt_saved)) < 1. and j < Nt_saved:
            psiFieldSlices[j,:,:] = psi
            j = j + 1
            
        #rhoAvg += np.abs(psi)**2
        # psiFieldSlices[i, :,:] = np.abs(psi)**2
        contrastVals[i] = np.std(np.abs(psi)**2)/np.average(np.abs(psi)**2)
        # phases[i,:,:] = np.angle(psi)
        
        # update time
        tvals.append(t)
        t += dt
        
    psiFieldSlices[-1,:,:] = psi


    return contrastVals, psiFieldSlices, tvals


def fdmSimulation1d(rho0, normalization, L, G_N, g_SI, tEnd, dt, Nt_saved):
    
    """ Quantum simulation """

    # Simulation parameters
    N         = np.shape(rho0)[0]    # Spatial resolution
    t         = 0      # current time of the simulation
    tOut      = 0.0001  # draw frequency
    G         = G_N # Gravitaitonal constant
    g         = g_SI #1e-2*np.pi*G # self-interaction strength
    plotRealTime = False # switch on for plotting as the simulation goes along

    # Domain [0,1] x [0,1]
#     L = 1    
    xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                     # chop off periodic point
#     xx, yy = np.meshgrid(xlin, copy=False)

    # normalize wavefunction to <|psi|^2>=rho / (m psi_0^2)
    rhobar = np.average( rho0 )
    rho0 = rho0/rhobar*normalization
    psi = np.sqrt(rho0)


    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
#     kx, ky = np.meshgrid(klin, klin, copy=False)
    kx = np.fft.ifftshift(klin)
#     ky = np.fft.ifftshift(ky)
    kSq = kx**2 #+ ky**2

    # Potential
#     Vhat = -np.fft.fftn(G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
#     V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2

    V = g*np.abs(psi)**2
    
    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    tvals = []
    
    # create array to store contrast field at each time step
    a  = np.shape(psi)
#     rhoAvg = np.zeros((a,b))
    psiFieldSlices = np.zeros(( Nt_saved , N ), dtype = complex)
    contrastVals = np.zeros((Nt))
#     phases = np.zeros((Nt , a, b))
    
    j=0
    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi

        # drift
        psihat = np.fft.fftn(psi)
        psihat = np.exp(dt * (-1.j*kSq/2.))  * psihat
        psi = np.fft.ifftn(psihat)

        # update potential
#         Vhat = -np.fft.fftn(G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
#         V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2
        V = g*np.abs(psi)**2

        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi
        
        # record last few field configurations
        
        if i > (Nt/4) and (i)%(Nt/(4*Nt_saved)) < 1. and j < Nt_saved:
            psiFieldSlices[j,:] = psi
            j = j + 1
            
        #rhoAvg += np.abs(psi)**2
        # psiFieldSlices[i, :,:] = np.abs(psi)**2
        contrastVals[i] = np.std(np.abs(psi)**2)/np.average(np.abs(psi)**2)
        # phases[i,:,:] = np.angle(psi)
        
        # update time
        tvals.append(t)
        t += dt
        
    psiFieldSlices[-1,:] = psi


    return contrastVals, psiFieldSlices, tvals