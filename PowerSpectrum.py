def PowerSpectrum(rho, L_box, debug=False):
    
    "Compute density power spectrum in k space"
    
    from numpy.fft import fft2, fftfreq, fftshift
    from numpy import pi, average, abs, sqrt, meshgrid, arange, sum, std, diff, mean
    import scipy.stats as stats
    
    density = rho
    delta = density / average(density) - 1
    
    N = rho.shape[0]

    fourier = fft2(delta)
    fourier_amplitudes = abs(fourier)**2/N**2

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
    counts, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "count",
                                         bins = kbins)
#   Normalization
    delta_rms = std(rho)/average(rho)
    dk = diff(kvals).mean()

    sigma_squared = sum(kvals * Power_Spectrum) * dk / (2 * pi)
    
    if debug:
        return kvals, Power_Spectrum*delta_rms**2/sigma_squared, counts
    else:
        return kvals, Power_Spectrum*delta_rms**2/sigma_squared


    
def TwoPoint(psi, L_box, debug=False):
    
    "Compute field power spectrum in k space"
    
    from numpy.fft import fft2, fftfreq, fftshift
    from numpy import pi, average, abs, sqrt, meshgrid, arange, sum, std, diff, mean, linspace
    from scipy.stats import binned_statistic
    from scipy.integrate import trapz
    
    N = psi.shape[0]  # Assuming psi is NxN
    kx = fftfreq(N, d = L_box / N) * 2 * pi  # Momentum components in x
    ky = fftfreq(N, d = L_box / N) * 2 * pi  # Momentum components in y
    
    # Create 2D grids for kx and ky
    KX, KY = meshgrid(kx, ky)
    
    # Compute the 2D Fourier transform of psi
    psi_hat = fft2(psi)
    
    # Compute the magnitude of the wavevector (k = sqrt(kx^2 + ky^2))
    k_magnitude = sqrt(KX**2 + KY**2)
    
    # Compute the power spectrum: P(k) = |psi_hat(k)|^2
    power_spectrum = abs(fftshift(psi_hat))**2
    
    # Normalize the power spectrum by the total number of grid points (N^2)
    power_spectrum_normalized = power_spectrum / (N**2)
    
    # Compute spatial average of |psi|^2 (total power in real space)
    total_power_real_space = sum(abs(psi)**2 * (L_box/N)**2) / (L_box**2)
    
    # Flatten the k_magnitude and power_spectrum for use in binned_statistic
    k_flat = k_magnitude.flatten()
    power_flat = power_spectrum_normalized.flatten()
    
    # Binning the k values into bins corresponding to different magnitudes
    bin_edges = linspace(pi / L_box, pi * N / L_box, N // 2 +1 )  # Bin edges for k values
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers for k values
    
    # Compute the binned power spectrum: angular averaged power
    power_binned, _, _ = binned_statistic(k_flat, power_flat, statistic='mean', bins=bin_edges)
    counts_binned, _, _ = binned_statistic(k_flat, power_flat, statistic='count', bins=bin_edges)
    
    # Normalize the power spectrum
    # Calculate the radial integral of the power spectrum
    bin_widths = diff(bin_edges)  # Width of each bin (k-space)
    bin_areas = bin_centers * bin_widths / 2 / pi  # Area element in k-space for each bin
    
    # Radial integral of the power spectrum should be normalized to match the spatial average
    radial_integral = sum(power_binned * bin_areas)
    normalization_factor = total_power_real_space / radial_integral
    
    # Apply the normalization factor
    power_binned_normalized = power_binned * normalization_factor
    
    if debug:
        return bin_centers, power_binned_normalized, counts_binned
    else:
        return bin_centers, power_binned_normalized


def kAverage(psik, L_box):
    
    "Angular average a 2d field in k space"
    
    from numpy.fft import fftn, fftfreq, fftshift
    from numpy import pi, average, abs, sqrt, meshgrid, arange, sum
    import scipy.stats as stats

    
    N = psik.shape[0]
    amplitudes = (psik)

    kfreq = fftfreq(N)*N
    kfreq2d = meshgrid(kfreq, kfreq)
    knrm = sqrt(kfreq2d[0]**2 + kfreq2d[1]**2)

    knrm = knrm.flatten()
    amplitudes = amplitudes.flatten()

    kbins = arange(0.001, N//2+1, 1)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])*2*pi/L_box
    avg, _, _ = stats.binned_statistic(knrm, amplitudes,
                                         statistic = "mean",
                                         bins = kbins)

    return kvals, avg

def PsiK(psi, L_box):
    from numpy.fft import fftn, fftfreq, fftshift
    from numpy import pi, average, abs, sqrt, meshgrid, arange, sum
    N = psi.shape[0]

    fourier = fftn(psi, axes= (-2,-1))

    kfreq = (fftfreq(N)*N*pi/L_box)
    kx, ky = meshgrid(kfreq, kfreq, indexing="ij")
    
    return kx, ky, fourier
