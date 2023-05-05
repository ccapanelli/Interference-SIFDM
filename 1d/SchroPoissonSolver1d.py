"""
Create Your Own Quantum Mechanics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate the Schrodinger-Poisson system with the Spectral method

Include self-interactions, calculate density contrast, and center plots around cores
Christian Capanelli (2022), McGill University
"""

import numpy as np

def fdmSimulation(rho0, normalization, G_N, g_SI, tEnd, dt, Nt_saved):
    
    """ Quantum simulation """

    # Simulation parameters
    N         = int(np.sqrt(rho0.size))    # Spatial resolution
    t         = 0      # current time of the simulation
    tOut      = 0.0001  # draw frequency
    G         = G_N # Gravitaitonal constant
    g         = g_SI #1e-2*np.pi*G # self-interaction strength
    plotRealTime = False # switch on for plotting as the simulation goes along

    # Domain [0,1] x [0,1]
    L = 1    
    xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                     # chop off periodic point
    #xx, yy = np.meshgrid(xlin, xlin, copy=False)

    # normalize wavefunction to <|psi|^2>=rho / (m psi_0^2)
    rhobar = np.mean( rho0 )
    rho0 = rho0/rhobar*normalization
    psi = np.sqrt(rho0)


    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
    kx = klin #, ky = np.meshgrid(klin, klin, copy=False)
    kx = np.fft.ifftshift(kx)
    #ky = np.fft.ifftshift(ky)
    kSq = kx**2 #+ ky**2
    

    # Potential
    Vhat = -np.fft.fft(4.0*np.pi*G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
    V = np.real(np.fft.ifft(Vhat)) + g*np.abs(psi)**2

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    
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
        psihat = np.fft.fft(psi)
        psihat = np.exp(dt * (-1.j*kSq/2.))  * psihat
        psi = np.fft.ifft(psihat)

        # update potential
        Vhat = -np.fft.fft(4.0*np.pi*G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
        V = np.real(np.fft.ifft(Vhat)) + g*np.abs(psi)**2

        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi
        
        # record last few field configurations
        
        if i > (Nt/5) and (i)%(Nt/(5*Nt_saved)) < 1. and j < Nt_saved:
            psiFieldSlices[j,:,:] = psi
            j = j + 1
            
        #rhoAvg += np.abs(psi)**2
        # psiFieldSlices[i, :,:] = np.abs(psi)**2
        contrastVals[i] = np.std(np.abs(psi)**2)/np.average(np.abs(psi)**2)
        # phases[i,:,:] = np.angle(psi)
        
        # update time
        t += dt
        
    psiFieldSlices[-1,:,:] = psi


    return psi, contrastVals, psiFieldSlices[:,:,:]



# Function rolls x and y axes of psi field so the solitonic core is centered
def Center(psi):
    import numpy as np
    rho = np.abs(psi)**2
    N = np.shape(psi)[0]
    halfway = int(N/2)
    
    # Find center of mass of density field
    def COM(rho):
        N = np.shape(rho)[0]
        comX = 0
        comY = 0
        totalMass = np.sum(rho)
        for i in range(0,N):
            for j in range(0,N):
                comX += i*rho[i,j]
                comY += j*rho[i,j]
        comX /= totalMass
        comY /= totalMass
        COM = np.array([int(comX), int(comY)])
        return COM

    # Center the psi array based on this
    centerMass = COM(rho)
    shift_i = centerMass[0] - halfway
    shift_j = centerMass[1] - halfway

    psi_centered = np.roll(psi, -shift_i, axis=0)# shift in horizontal axis
    psi_centered = np.roll(psi_centered, -shift_j, axis=1)# shift in vertical axis

    # Do this step one more time incase psi was "concave"
    rho2 = np.abs(psi_centered)**2

    centerMass2 = COM(rho2)
    shift_i = centerMass2[0] - halfway
    shift_j = centerMass2[1] - halfway

    psi_centered2 = np.roll(psi_centered, -shift_i, axis=0)# shift in horizontal axis
    psi_centered2 = np.roll(psi_centered2, -shift_j, axis=1)# shift in vertical axis
       
    return psi_centered2
