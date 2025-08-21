def RandomPhase(resolution, debroglie_length, box_size, user_seed=None):
    from numpy import pi, linspace, meshgrid, array, newaxis, zeros, abs, exp, cos, sin, roll
    from numpy.random import normal, uniform, seed
    from numpy.linalg import norm
    from numpy.fft import fftn, fftshift, ifftn
    
    # Seed random phases. Default is unseeded
    if user_seed is not None:
        seed(user_seed)
    
    N_steps = resolution # spacial slices in one direction
    N_half = int(N_steps/2)
    L_dB = debroglie_length
    L_box = box_size
    
    k_0 = 2*pi/L_dB
    
    kx = linspace(1,  N_steps, N_steps)*2*pi/L_box
    
    Kx, Ky = meshgrid(kx, kx)

    Amps = exp(-(Kx**2 + Ky**2)/k_0**2/2)
    Phases = uniform(0, 2*pi, (N_steps, N_steps))
    
    psi_k = Amps*exp(1j*Phases)
    psi = fftshift(ifftn(psi_k))
    psi2 = abs(psi)**2 #/ sum(absolute(psi)**2)
       
    
    return psi, psi2

def kShell(resolution, box_size, kshell, kwidth, random_phase, user_seed=None):
    from numpy import pi, linspace, meshgrid, array, newaxis, zeros, abs, exp, cos, sin, roll, heaviside, sqrt
    from numpy.random import normal, uniform, seed
    from numpy.linalg import norm
    from numpy.fft import fftn, fftshift, ifftn
    
    # Seed random phases. Default is unseeded
    if user_seed is not None:
        seed(user_seed)
    
    N_steps = resolution # spacial slices in one direction
    N_half = int(N_steps/2)
    L_box = box_size
    kx = linspace(-N_steps//2,  N_steps//2, N_steps)*2*pi/L_box
    dk = (1/N_steps)*2*pi/L_box
    k_c = 0
    
    Kx, Ky = meshgrid(kx, kx)

    Amps =  exp(-(sqrt((Kx-k_c)**2 + (Ky-k_c)**2)-kshell)**2/2/kwidth**2) #Gaussian shell
    # alternate initial conditions:
#     Amps = exp(-(Kx**2 + Ky**2)/k_0**2/2) # Gaussian

#     def tophat(k, center, width):
#             A = heaviside( k - (center - width/2), 0)
#             B = heaviside( k - (center + width/2), 0 )
#             return B - A
#     Amps = tophat(sqrt((Kx-k_c)**2 + (Ky-k_c)**2) , kshell, kwidth) # square shell 

    Phases = uniform(-pi, pi, (N_steps, N_steps)) # uniform random phases

    if random_phase:
        psi_k = Amps*exp(1j*Phases)
    else:
        psi_k = Amps
        
    psi = fftshift(ifftn(psi_k))
    psi2 = abs(psi)**2 #/ sum(absolute(psi)**2)
       
    
    return psi, psi2, psi_k