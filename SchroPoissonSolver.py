"""
Create Your Own Quantum Mechanics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate the Schrodinger-Poisson system with the Spectral method

Include self-interactions, calculate density contrast, and center plots around solitons
Christian Capanelli (2022), McGill University
"""


def fdmSimulation(rho0, g_SI, tEnd, dt):
    import numpy as np
    """ Quantum simulation """

    # Simulation parameters
    N         = int(np.sqrt(rho0.size))    # Spatial resolution
    t         = 0      # current time of the simulation
    tOut      = 0.0001  # draw frequency
    G         = 4000  # Gravitaitonal constant
    g         = g_SI*G #1e-2*np.pi*G # self-interaction strength
    plotRealTime = False # switch on for plotting as the simulation goes along

    # Domain [0,1] x [0,1]
    L = 1    
    xlin = np.linspace(0,L, num=N+1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]                     # chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin)

    # normalize wavefunction to <|psi|^2>=1
    rhobar = np.mean( rho0 )
    rho0 /= rhobar
    psi = np.sqrt(rho0)


    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N/2, N/2)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx**2 + ky**2

    # Potential
    Vhat = -np.fft.fftn(4.0*np.pi*G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
    V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
    
    # create array to store contrast field at each time step
    a , b = np.shape(psi)
    psiFieldSlices = np.zeros((Nt , a, b))
    contrastVals = np.zeros((Nt))
    phases = np.zeros((Nt , a, b))
    
    # prep figure
#     fig = plt.figure(figsize=(6,4), dpi=100)
#     grid = plt.GridSpec(1, 2, wspace=0.0, hspace=0.0)
#     ax1 = plt.subplot(grid[0,0])
#     ax2 = plt.subplot(grid[0,1])
#     outputCount = 1

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi

        # drift
        psihat = np.fft.fftn(psi)
        psihat = np.exp(dt * (-1.j*kSq/2.))  * psihat
        psi = np.fft.ifftn(psihat)

        # update potential
        Vhat = -np.fft.fftn(4.0*np.pi*G*(np.abs(psi)**2-1.0)) / ( kSq  + (kSq==0))
        V = np.real(np.fft.ifftn(Vhat)) + g*np.abs(psi)**2

        # (1/2) kick
        psi = np.exp(-1.j*dt/2.0*V) * psi

        psiFieldSlices[i, :,:] = np.abs(psi)**2
        contrastVals[i] = np.std(np.abs(psi)**2)/np.average(np.abs(psi)**2)
        phases[i,:,:] = np.angle(psi)
        
        # update time
        t += dt

#         # plot in real time
#         plotThisTurn = False
#         if t + dt > outputCount*tOut:
#             plotThisTurn = True
#         if (plotRealTime and plotThisTurn) or (i == Nt-1):

#             plt.sca(ax1)
#             plt.cla()

#             plt.imshow(np.log10(np.abs(psi)**2), cmap = 'inferno')
#             plt.clim(-1, 2)
#             ax1.get_xaxis().set_visible(False)
#             ax1.get_yaxis().set_visible(False)
#             ax1.set_aspect('equal')

#             plt.sca(ax2)
#             plt.cla()
#             plt.imshow(np.angle(psi), cmap = 'bwr')
#             plt.clim(-np.pi, np.pi)
#             ax2.get_xaxis().set_visible(False)
#             ax2.get_yaxis().set_visible(False)
#             ax2.set_aspect('equal')
        

#             plt.pause(0.001)
#             outputCount += 1

#     # Plot density contrast
#     plt.figure(dpi=150)
#     plt.title(r"$\mathrm{SI}/\mathrm{Gravity}=$ "+str(g_SI))
#     plt.plot(contrastVals)
#     plt.xlabel("time")
#     plt.ylabel(r"$\delta\rho/\rho$", rotation=45)
#     plt.show()
    
#     # Save figure
#     plt.sca(ax1)
#     plt.title(r'$\log_{10}(|\psi|^2)$')
#     plt.sca(ax2)
#     plt.title(r'${\rm angle}(\psi)$')
#     plt.savefig('quantumspectral.png',dpi=240)
#     plt.show()

    return psiFieldSlices, contrastVals, phases



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