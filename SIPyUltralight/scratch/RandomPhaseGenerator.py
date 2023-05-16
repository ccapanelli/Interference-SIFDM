
def RandomPhase(resolution, number_modes, debroglie_length, box_size, masked):
    from numpy import pi, linspace, meshgrid, array, newaxis, zeros, absolute, exp, cos, sin
    from numpy.random import normal, uniform
    from numpy.linalg import norm
    
    N_steps = resolution # spacial slices in one direction
    L_dB = debroglie_length
    x_max = box_size * L_dB # in number of de Broglie wavelengths
    N_modes = number_modes # number of k modes
    
    k_0 = 2*pi/L_dB
    
    x_min = 0
    y_min = x_min
    y_max = x_max
    z_max = x_max
    x = linspace(x_min, x_max, N_steps)
    y = linspace(y_min, y_max, N_steps) # transpose
    z = linspace(y_min, y_max, N_steps)
    X,Y, Z = meshgrid(x,y,z)
    # add up unit plane waves  

    A = array((x,y,z))
    k_vals = normal(loc=0.0, scale=k_0, size=(N_modes))
#         k_vals = np.linspace(0, 2*k_0, N_modes)
    k_vectors = zeros((3,N_modes)) # [spacial component, k mode index]
    thetas = zeros((N_modes), dtype=float) # [field index, k mode index]
    psi = zeros((N_steps, N_steps,N_steps), dtype=complex)
    n = 0 # nth mode
    for q in k_vals: # plane wave in each field for each k value
        k_angle = uniform(-pi,pi) #np.random.uniform(-np.pi,np.pi) # angle of k vector
        k_vectors[:, n] = [q*cos(k_angle), q*sin(k_angle),0] # EM-edit: quick attempt to male 3d

        thetas[n] = uniform(-pi,pi)
        # psi += [np.exp(-np.linalg.norm(k_vectors,2)/k_0**2)*np.exp(1j*thetas[n])*np.exp(1j*(k_vectors[0,n]*A[0,:] + k_vectors[1,n]*A[1,:]))] # add psi_n
        psi += exp(-norm(k_vectors, 2) / k_0 ** 2) * exp(1j * thetas[n]) * exp(1j * (k_vectors[0, n] * X + k_vectors[1, n] * Y++ k_vectors[2, n] * Z))
#             print(thetas[j,i])
        n += 1
    
    # psi2 = absolute(psi)**2
    
    if masked:
        # Mask with Gaussian blur centered at origin
        for r in linspace(0, x_max/2):
            width = 1.*x_max
            opacity = exp(-r**2/(2*width**2))
            mask = (x[newaxis,:]-x_max/2)**2 + (y[:,newaxis]-y_max/2)**2  > (r)**2 
            psi[mask] *= opacity

        mask = (x[newaxis,:]-x_max/2)**2 + (y[:,newaxis]-y_max/2)**2  > (x_max/2)**2 
        psi[mask] = 0    
    
    return psi, abs(psi)**2 #psi2
    
