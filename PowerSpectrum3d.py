def PowerSpectrum3d(rho, L_box):
    from numpy.fft import fftn, fftfreq, fftshift
    from numpy import pi, average, abs, sqrt, meshgrid, arange, sum
    import scipy.stats as stats
    
    density = rho
    density = density / average(density)
    
    N = rho.shape[0]

    fourier = fftn(rho)
    fourier_amplitudes = abs(fourier)**2

    kfreq = fftfreq(N)*N
    kfreq3d = meshgrid(kfreq, kfreq, kfreq)
    knrm = sqrt(kfreq3d[0]**2 + kfreq3d[1]**2 + kfreq3d[2]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = arange(0.5, N//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])*2*pi/L_box
    Power_Spectrum, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Power_Spectrum *= (kbins[1:]**2 - kbins[:-1]**2)
    
    return kvals, Power_Spectrum#/sum(Power_Spectrum)

def PowerSpectrum2d(rho, L_box):
    from numpy.fft import fftn, fftfreq, fftshift
    from numpy import pi, average, abs, sqrt, meshgrid, arange, sum
    import scipy.stats as stats
    
    density = rho
    density = density / average(density) - 1
    
    N = rho.shape[0]

    fourier = fftn(density)
    fourier_amplitudes = abs(fourier)**2

    kfreq = fftfreq(N)*N
    kfreq2d = meshgrid(kfreq, kfreq)
    knrm = sqrt(kfreq2d[0]**2 + kfreq2d[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = arange(0.5, N//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])*2*pi/L_box
    Power_Spectrum, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    Power_Spectrum *=  (kbins[1:]**2 - kbins[:-1]**2)
    
    return kvals, Power_Spectrum/sum(Power_Spectrum)

# def CorrelationFromDensity(rho, L_box):
#     from numpy.fft import fftn, fftfreq, fftshift, ifftn
#     from numpy import pi, average, abs, sqrt, meshgrid, arange, sum, linspace, real
#     import scipy.stats as stats
    
#     density = rho
#     density = density / average(density) - 1
    
#     N = rho.shape[0]

#     fourier = fftn(rho)
#     fourier_amplitudes = abs(fourier)**2

#     kfreq = fftfreq(N)*N
#     kfreq3d = meshgrid(kfreq, kfreq, kfreq)
#     knrm = sqrt(kfreq3d[0]**2 + kfreq3d[1]**2 + kfreq3d[2]**2)
    
#     corr = ifftn(fourier_amplitudes)
#     x = linspace(0, L_box, N)
#     r3d = meshgrid(x, x, x)
#     rnrm = sqrt(r3d[0]**2 + r3d[1]**2 + r3d[2]**2)

#     rnrm = rnrm.flatten()
#     corr = real(corr.flatten())

#     rbins = arange(0.5, N//2+1, 1.)
#     rvals = 0.5 * (rbins[1:] + rbins[:-1])*L_box
#     Correlation, _, _ = stats.binned_statistic(rnrm, corr,
#                                          statistic = "mean",
#                                          bins = rbins)
    
    
#     return rvals, Correlation/sum(Correlation)