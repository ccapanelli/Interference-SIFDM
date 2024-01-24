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
#     Power_Spectrum *= (kbins[1:]**2 - kbins[:-1]**2)
    
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
#     Power_Spectrum *=  (kbins[1:]**2 - kbins[:-1]**2)
    
    return kvals, Power_Spectrum#/sum(Power_Spectrum)


def PowerSpectrum1d(rho, L_box):
    from numpy.fft import fftn, fftfreq, fftshift
    from numpy import pi, average, abs, sqrt, meshgrid, arange, sum
    import scipy.stats as stats
    
    density = rho
    density = density / average(density) - 1
    
    N = rho.shape[0]

    fourier = fftn(density)
    fourier_amplitudes = abs(fourier)**2
#     fourier_amplitudes = fftshift(fourier_amplitudes)
    
#     kvals = fftfreq(N)*N*2*pi/L_box
#     kvals=fftshift(kvals)

    kfreq = fftfreq(N)*N
#     kfreq2d = meshgrid(kfreq, kfreq)
    knrm = kfreq #kvals

#     knrm = knrm.flatten()
#     fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = arange(0.5, N//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])*2*pi/L_box
    Power_Spectrum, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
#     Power_Spectrum *=  (kbins[1:]**2 - kbins[:-1]**2)
#     Power_Spectrum = fourier_amplitudes / N
    
    return kvals, Power_Spectrum