def PowerSpectrum(rho):
    from numpy import average, absolute, shape, arange, sqrt, cos, sin, pi
    from numpy.fft import fft2
    from scipy.interpolate import interp2d
    
    density = rho
    density = density / average(density) - 1

    # Compute the power spectrum P(k) of density
    # P(k) = |F(k)|^2 / (N^2)
    # F(k) = FFT(density)
    # N = number of pixels

    # FFT
    F = fft2(density)

    # |F(k)|^2
    F_abs = absolute(F)**2

    # N^2
    N = density.shape[0]
    N_sq = N**2

    # P(k)
    P = F_abs / N_sq


    # Plot P(k) versus k modes
    # k modes = sqrt(k_x^2 + k_y^2)
    # k_x = 0, 1, 2, ..., N/2, -N/2, ..., -2, -1
    # k_y = 0, 1, 2, ..., N/2, -N/2, ..., -2, -1

    # k_x
    k_x = arange(0,N)
    k_x[int(N/2):] = k_x[int(N/2):] - N

    # k_y
    k_y = arange(0,N)
    k_y[int(N/2):] = k_y[int(N/2):] - N

    P_int = interp2d(k_x, k_y, P)

    # k modes
    k_modes = sqrt(k_x**2 + k_y**2)

    Power_Spectrum = []

    for k in k_modes:
        thetas = arange(0, 2*pi, 1/(k+1e-3)**(1/4) )
        P_theta = []
        for t in thetas:
            P_theta.append( P_int(k*cos(t) , k*sin(t)) )
        Power_Spectrum.append(average(P_theta) )
    k_modes = k_modes[1:]
    Power_Spectrum = Power_Spectrum[1:]
    
    return k_modes, Power_Spectrum