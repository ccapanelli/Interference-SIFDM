def RandomPhase3d(resolution, debroglie_length, box_size):
    from numpy import pi, linspace, meshgrid, array, newaxis, zeros, absolute, exp, cos, sin
    from numpy.random import normal, uniform
    from numpy.linalg import norm
    from numpy.fft import fftn, fftshift
    
    N_steps = resolution # spacial slices in one direction
    L_dB = debroglie_length
    L_box = box_size
    
    k_0 = 2*pi/L_dB
    
    kx = linspace(1,  N_steps, N_steps)*2*pi/L_box
    
    Kx, Ky, Kz = meshgrid(kx, kx, kx)

    Amps = exp(-(Kx**2 + Ky**2 + Kz**2)/k_0**2/2)
    Phases = uniform(0, 2*pi, (N_steps, N_steps, N_steps))
    
    psi_k = Amps*exp(1j*Phases)
    psi = fftshift(fftn(psi_k))
    
    psi2 = absolute(psi)**2 / sum(absolute(psi)**2)
       
    
    return psi, psi2

def RandomPhase2d(resolution, debroglie_length, box_size):
    from numpy import pi, linspace, meshgrid, array, newaxis, zeros, absolute, exp, cos, sin, roll
    from numpy.random import normal, uniform
    from numpy.linalg import norm
    from numpy.fft import fftn, fftshift
    
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
    psi = fftshift(fftn(psi_k))
    
#     psi_roll = roll(psi, (N_half, N_half), axis=(0,1) )
    
#     psi = (psi + psi_roll)/2
    
    psi2 = absolute(psi)**2 / sum(absolute(psi)**2)
       
    
    return psi, psi2

def RandomPhase1d(resolution, debroglie_length, box_size):
    from numpy import pi, linspace, meshgrid, array, newaxis, zeros, absolute, exp, cos, sin, roll
    from numpy.random import normal, uniform
    from numpy.linalg import norm
    from numpy.fft import fftn, fftshift
    
    N_steps = resolution # spacial slices in one direction
    N_half = int(N_steps/2)
    L_dB = debroglie_length
    L_box = box_size
    
    k_0 = 2*pi/L_dB
    
    kx = linspace(1,  N_steps, N_steps)*2*pi/L_box
    
#     Kx, Ky = meshgrid(kx, kx)

    Amps = exp(-(kx**2)/k_0**2/2)
    Phases = uniform(0, 2*pi,  N_steps)
    
    psi_k = Amps*exp(1j*Phases)
    psi = fftshift(fftn(psi_k))
    
#     psi_roll = roll(psi, (N_half, N_half), axis=(0,1) )
    
#     psi = (psi + psi_roll)/2
    
    psi2 = absolute(psi)**2 / sum(absolute(psi)**2)
       
    
    return psi, psi2
    
