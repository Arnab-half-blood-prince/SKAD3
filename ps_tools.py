# power spectrum estimation from map cube (l,m,nu) 

import numpy as np
from astropy import constants as const
import scipy.signal.windows as windows
from astropy.cosmology import Planck15, default_cosmology
from astropy import units as un

from draco.core import containers, task
from caput import mpiarray, config
from draco.analysis.delay import match_axes

# Defining the cosmology - Planck15 of astropy
cosmo = Planck15
f21 = 1420.405752 * un.MHz  # MHz
c = const.c.value  # m/s
import time
import pandas as pd

def delay_to_kpara(delay, z, cosmo=None):
    """Conver delay in sec unit to k_parallel (comoving 1./Mpc along line of sight).

    Parameters
    ----------
    delay : Astropy Quantity object with units equivalent to time.
        The inteferometric delay observed in units compatible with time.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to planck2015 year in "little h" units

    Returns
    -------
    kparr : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight probed by the input delay (eta).

    """

    
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (delay * (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))
            / (const.c * (1 + z)**2)).to('1/Mpc')

def kpara_to_delay(kpara, z, cosmo=None):
    """Convert k_parallel (comoving 1/Mpc along line of sight) to delay in sec.

    Parameters
    ----------
    kpara : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale parallel to the line of sight
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    delay : Astropy Quantity units equivalent to time
        The inteferometric delay which probes the spatial scale given by kparr.

    """

    
    if cosmo is None:
        cosmo = default_cosmology.get()
    return (kpara * const.c * (1 + z)**2
            / (2 * np.pi * cosmo.H0 * f21 * cosmo.efunc(z))).to('s')

def kperp_to_u(kperp, z, cosmo=None):
    """Convert comsological k_perpendicular to baseline length u (wavelength unit).

    Parameters
    ----------
    kperp : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale perpendicular to the line of sight.
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    u : float
        The baseline separation of two interferometric antennas in units of
        wavelength which probes the spatial scale given by kperp

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return kperp * cosmo.comoving_transverse_distance(z) / (2 * np.pi)

def u_to_kperp(u, z, cosmo=None):
    """Convert baseline length u to k_perpendicular.

    Parameters
    ----------
    u : float
        The baseline separation of two interferometric antennas in units of wavelength
    z : float
        The redshift of the expected 21cm emission.
    cosmo : Astropy Cosmology Object
        The assumed cosmology of the universe.
        Defaults to WMAP9 year in "little h" units

    Returns
    -------
    kperp : Astropy Quantity units equivalent to wavenumber
        The spatial fluctuation scale perpendicular to the line of sight probed by the baseline length u.

    """
    if cosmo is None:
        cosmo = default_cosmology.get()
    return 2 * np.pi * u / cosmo.comoving_transverse_distance(z)

def get_spatial_kmodes(Nx,Ny,theta_res,redshift):
    """Estimate the spatial fourier k-modes, (kx and ky) and the gridded u,v coordinates.  
    
    Parameteres
    -----------
    Nx : int
     Number of cells along RA or X-axis direction.
    Ny : int
      Number of cells along DEC or Y-axis direction.
    theta_res : float 
      The cell size of the map in radian unit.
    redshift : float
      redshift at the centre of the band.
    
    Returns
    -------
    k_x : np.ndarray[nra]
     spatial fourier modes along X-axis 
    k_y : np.ndarray[nel]
     spatial fourier modes along Y-axis
    u : np.ndarray[nra]
     gridded u-coordinates 
    v : np.ndarray[nel]
     gridded v-coordinates
    """
    DMz = cosmo.comoving_transverse_distance(redshift).value # in Mpc
    dtheta_Mpc =  DMz * theta_res   # res (radian) to Mpc unit
    k_x = 2*np.pi * (np.fft.fftfreq(Nx, d=dtheta_Mpc)) # [Mpc^-1] The kx coordinates along the X-direction
    k_y = 2*np.pi * (np.fft.fftfreq(Ny, d=dtheta_Mpc)) # [Mpc^-1] The ky coordinates along the Y-direction
   
    k_x = np.fft.fftshift(k_x)
    k_y = np.fft.fftshift(k_y)
    
    u = (DMz * k_x)/(2*np.pi)
    v = (DMz * k_y)/(2*np.pi)
    return k_x,k_y, u, v

def jy_per_beam_to_kelvin(freq,beam_area=1.0):
    """Conversion factor from Jy/beam to kelvin unit.
    
    The conversion factor is C = (10**-26 * lambda**2)/(2 * K_boltzmann * omega_PSF), 
    where omega_PSF is the beam area in sr unit.
    
    Parameters
    ----------
    freq : np.ndarray[nfreq]
        frequency in MHz unit
    beam_area : float
        synthesized beam area in sr unit
        
    Returns
    -------
    C : np.ndarray[nfreq]
     The conversion factor from Jy/beam to Kelvin
    """
 
    Jy = 1.0e-26 # W m^-2 Hz^-1
    c = const.c.value # m/s
    wl = c/(freq*1e6) # freq of the map in MHz
    kB = const.k_B.value # Boltzmann Const in J/k (1.38 * 10-23)
    C = (wl**2 * Jy) / (2*kB *beam_area.value)
    return C 

def image_to_uv(data_cube,window=True,window_name ='tukey', alpha=0.5, axes=None):
    """Spatial FFT along RA and DEC axes of the data cube.
    
    Parameters
    ----------
    data_cube : np.ndarray[nfreq,nra,nel]
       The data cube, whose spatial FFT will be computed along RA and DEC axes.
    window : bool.
       If True apply a spatial apodisation function. Default: True.
    window_name : 'Tukey'
       Apply Tukey spatial tapering window. Default: 'Tukey'.
    alpha : float 
       Shape parameter of the Tukey window. 
       0 = rectangular window, 1 = Hann window. We are taking 0.5 (default), which lies in between. 
    axes : Tuple.
      The axes along which to do the 2D Spatial FFT. This is the RA and DEC axex of the data cube. 
      
    Returns
    -------
      data_cube : np.ndarray[nfreq,nra,nel]
         The 2D spatial FFT of the data cube in (u,v) domain.
    """
    from scipy.signal import windows
    FT_norm = 1 / float(np.prod(np.array(data_cube.shape)[list(axes)]))
    print(f"Fourier Norm : {FT_norm}")            
    if window:
        window_func = getattr(windows, window_name)
        w_ra = window_func(data_cube.shape[axes[0]],alpha=alpha)   # taper in the RA direction 
        w_dec = window_func(data_cube.shape[axes[1]],alpha=alpha)  # taper in the DEC direction
        taper_window = np.outer(w_ra[:,np.newaxis],w_dec[np.newaxis,:]) # Make the 2D tapering function by taking the outer product 
        print(f'Taper window shape : {taper_window.shape}')
        data_cube *=  taper_window[np.newaxis,:,:]
              
    uv_map = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data_cube, axes=axes), axes=axes), axes=axes)
     
    return uv_map * FT_norm

def get_ps(vis_cube_1, vis_cube_2 , vol_norm_factor):
    """Estimate the cross-correlation of two data cubes,
    and normalize it. If a single data cube is 
    provided, then estimate the auto-correlation.
    
    The data cubes are complex. This will estimate the cross-correlation
    of two complex data cubes and return the real part of that.
    
    Parameters
    ----------
    vis_cube_1 : np.ndarray[nfreq,Nx,Ny]
      complex data cube in (tau,u,v) domain.
    vis_cube_2 : np.ndarray[nfreq,Nx,Ny]
      complex data cube in (tau,u,v) domain. 
    vol_norm_factor : float
      power spectrum normalization factor in [Mpc^3]
    
    Return
    ------
    ps_cube_real: np.ndarray[nfreq,Nx,Ny]
       The real part of the power spectrum
    """
    if vis_cube_1 is None and vis_cube_2 is None:
        raise NameError('Atleast one map must be provided')
      
    if vis_cube_2 is None:
        print('correlate the vis cube with itself')
        ps_cube_real = (np.conj(vis_cube_1) * vis_cube_1).real
        
    else:
        print('correlate the vis cube 1 with vis cube 2')
        ps_cube_real = (vis_cube_1 * np.conj(vis_cube_2)).real
        
    return ps_cube_real*vol_norm_factor

def is_odd(num):
    return num & 0x1

def PS2D(ps_cube,weight_cube,kperp_bins,kpar_bins,kpar,uu,vv,redshift,fold=True):
    """Estimate the cylindrically averaged 2D power spectrum. 
    
     Parameters
     ----------
     ps_cube : np.ndarray[nfreq,nra*nel] 
       The power spectrum array to average in cylindrical bins.
     weight_cube : np.ndarray[nfreq,nra,nel]
       The inverse variance weight, to be used in averaging. 
     kperp_min : float
       The min value of the bin in k-perpendicular direction; unit 1/Mpc 
     kperp_min : float
       The max value of the bin in k-perpendicular direction; unit 1/Mpc 
     kpar : np.ndarray[nfreq]
       The k_parallel array, unit 1/Mpc 
     uu : np.ndarray[nra]
      The u-coordinate in wavelength.
     vv : np.ndarray[nel] 
       The v-coordinate in wavelength.
     redshift : float
       The redshift corresponding to the band center.
     Nbins_2D : int
       The number of bins in k-perpendicualr direction
     log_bins : bool
       Bins in logarithmic space. Default : True. 
            
    Returns
    -------
    ps_2D : np.ndarray[] = 
      The cylindrically avg 2D Power of shape (k_par,k_perp) with unit [K^2 Mpc^3] 
    k_perp : np.array[Nbins_2d]
     The k_perp array after binning.
    k_para : np.array[nfreq//2]
     The k_parallel array, only the positive half. 
    """
    
   
    # Set weights
    
    if weight_cube is not None:
        # Inverse variance weighting, here weight_cube = 1/sigma, so we are squaring it to get variance
        print('Using inverse variance weighting')
        w = weight_cube ** 2 
    else:
        print('Using unit uniform weighting')
        w = np.ones_like(ps_cube) # Unit uniform weight.  
        
        
#     # Choosing the bins
#     if log_bins:
#         print(f'Using logarithmic binning in k_perp between {kperp_min} and {kperp_max}')
#         kp = np.logspace(np.log10(kperp_min), np.log10(kperp_max),Nbins_2D+1) 
#     else:
#         print(f'Using linear binning in k_perp between {kperp_min} and {kperp_max}')
#         kp = np.linspace(kperp_min, kperp_max,Nbins_2D+1)
                                 
    kp = kperp_bins
    # Find the bin indices and determine which radial bin each cell lies in.

    ku = u_to_kperp(uu,redshift) #convert the u-array to fourier modes
    kv = u_to_kperp(vv,redshift) #convert the v-array to fourier modes
    
    ru = np.sqrt(ku.value** 2 + kv.value ** 2) 
    
    # Digitize the bins
    bin_indx = np.digitize(ru, bins=kp) 
    print(bin_indx.shape)
    
    # Define empty list to store the binned 2D PS
    
    ps_2D = []
    ps_2D_err = [] 
    ps_2D_w = []
    
    # Now bin in 2D ##
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in np.arange(len(kp)) + 1:
            p = np.nansum(w[:, bin_indx == i] * ps_cube[:, bin_indx == i], axis=1) / np.sum(w[:, bin_indx == i], axis=1)
            p_err = np.sqrt(2 * np.sum(w[:, bin_indx == i] ** 2 * p[:, None] **
                                       2, axis=1) / np.sum(w[:, bin_indx == i], axis=1) ** 2)

            ps_2D.append(p)
            ps_2D_err.append(p_err)
            ps_2D_w.append(np.sum(w[:, bin_indx == i], axis=1))
            
    ps_2D =  np.array(ps_2D).T 
    ps_2D_w =  np.array(ps_2D_w).T
    ps_2D_err = np.array(ps_2D_err).T

    if fold:
        M = len(kpar)
        k_par = kpar[M // 2 + 1:]
        if is_odd(M):
            ps2D_fold = 0.5 * (ps_2D[M // 2 + 1:] + ps_2D[:M // 2][::-1])
            ps2D_err_fold = 0.5 * np.sqrt(ps_2D_err[M // 2 + 1:] ** 2 + ps_2D_err[:M // 2][::-1] ** 2)
            ps2D_w_fold = ps_2D_w[M // 2 + 1:] + ps_2D_w[:M // 2][::-1]
        else:
            ps2D_fold = 0.5 * (ps_2D[M // 2 + 1:] + ps_2D[1:M // 2][::-1])
            ps2D_err_fold = 0.5 * np.sqrt(ps_2D_err[M // 2 + 1:] ** 2 + ps_2D_err[1:M // 2][::-1] ** 2)
            ps2D_w_fold = ps_2D_w[M // 2 + 1:] + ps_2D_w[1:M // 2][::-1]

    # Digitize the kpar bins
    bin_indx_kpar = np.digitize(k_par, bins=kpar_bins)
    ps_2d = []
    ps_2d_err = [] 
    ps_2d_w = []
    
    # Now bin in kpar ##
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in np.arange(len(kpar_bins)) + 1:
            p = np.nansum(ps2D_w_fold[bin_indx_kpar == i,:] * ps2D_fold[bin_indx_kpar == i,:], axis=0) / np.sum(ps2D_w_fold[bin_indx_kpar == i,:], axis=0)
            
            p_err = np.sqrt(np.sum(ps2D_w_fold[bin_indx_kpar == i,:] ** 2 * p[None,:] **
                                       2, axis=0) / np.sum(ps2D_w_fold[bin_indx_kpar == i,:], axis=0) ** 2)

            ps_2d.append(p)
            ps_2d_err.append(p_err)
            ps_2d_w.append(np.sum(ps2D_w_fold[bin_indx_kpar == i,:], axis=0))
        
    return kp, kpar_bins, np.array(ps_2d), np.array(ps_2d_w), np.array(ps_2d_err)

def PS2D_no_kpar_bin(ps_cube,weight_cube,kperp_min,kperp_max,kpar,uu,vv,redshift,fold=True,
                    Nbins_2D=30,log_bins=False):
    """Estimate the cylindrically averaged 2D power spectrum. 
    
     Parameters
     ----------
     ps_cube : np.ndarray[nfreq,nra*nel] 
       The power spectrum array to average in cylindrical bins.
     weight_cube : np.ndarray[nfreq,nra,nel]
       The inverse variance weight, to be used in averaging. 
     kperp_min : float
       The min value of the bin in k-perpendicular direction; unit 1/Mpc 
     kperp_min : float
       The max value of the bin in k-perpendicular direction; unit 1/Mpc 
     kpar : np.ndarray[nfreq]
       The k_parallel array, unit 1/Mpc 
     uu : np.ndarray[nra]
      The u-coordinate in wavelength.
     vv : np.ndarray[nel] 
       The v-coordinate in wavelength.
     redshift : float
       The redshift corresponding to the band center.
     Nbins_2D : int
       The number of bins in k-perpendicualr direction
     log_bins : bool
       Bins in logarithmic space. Default : True. 
            
    Returns
    -------
    ps_2D : np.ndarray[] = 
      The cylindrically avg 2D Power of shape (k_par,k_perp) with unit [K^2 Mpc^3] 
    k_perp : np.array[Nbins_2d]
     The k_perp array after binning.
    k_para : np.array[nfreq//2]
     The k_parallel array, only the positive half. 
    """
    
   
    # Set weights
    
    if weight_cube is not None:
        # Inverse variance weighting, here weight_cube = 1/sigma, so we are squaring it to get variance
        print('Using inverse variance weighting')
        w = weight_cube ** 2 
    else:
        print('Using unit uniform weighting')
        w = np.ones_like(ps_cube) # Unit uniform weight.  
        
        
    # Choosing the bins
    if log_bins:
        print(f'Using logarithmic binning in k_perp between {kperp_min} and {kperp_max}')
        kp = np.logspace(np.log10(kperp_min), np.log10(kperp_max),Nbins_2D+1) 
    else:
        print(f'Using linear binning in k_perp between {kperp_min} and {kperp_max}')
        kp = np.linspace(kperp_min, kperp_max,Nbins_2D+1)
                                 
   # kp = kperp_bins
    # Find the bin indices and determine which radial bin each cell lies in.

    ku = u_to_kperp(uu,redshift) #convert the u-array to fourier modes
    kv = u_to_kperp(vv,redshift) #convert the v-array to fourier modes
    
    ru = np.sqrt(ku.value** 2 + kv.value ** 2) 
    
    # Digitize the bins
    bin_indx = np.digitize(ru, bins=kp) 
    print(bin_indx.shape)
    
    # Define empty list to store the binned 2D PS
    
    ps_2D = []
    ps_2D_err = [] 
    ps_2D_w = []
    
    # Now bin in 2D ##
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in np.arange(len(kp)) + 1:
            p = np.nansum(w[:, bin_indx == i] * ps_cube[:, bin_indx == i], axis=1) / np.sum(w[:, bin_indx == i], axis=1)
            p_err = np.sqrt(np.sum(w[:, bin_indx == i] ** 2 * p[:, None] **
                                       2, axis=1) / np.sum(w[:, bin_indx == i], axis=1) ** 2)

            ps_2D.append(p)
            ps_2D_err.append(p_err)
            ps_2D_w.append(np.sum(w[:, bin_indx == i], axis=1))
            
    ps_2D =  np.array(ps_2D).T 
    ps_2D_w =  np.array(ps_2D_w).T
    ps_2D_err = np.array(ps_2D_err).T

    if fold:
        M = len(kpar)
        k_par = kpar[M // 2 + 1:]
        if is_odd(M):
            ps2D_fold = 0.5 * (ps_2D[M // 2 + 1:] + ps_2D[:M // 2][::-1])
            ps2D_err_fold = 0.5 * np.sqrt(ps_2D_err[M // 2 + 1:] ** 2 + ps_2D_err[:M // 2][::-1] ** 2)
            ps2D_w_fold = ps_2D_w[M // 2 + 1:] + ps_2D_w[:M // 2][::-1]
        else:
            ps2D_fold = 0.5 * (ps_2D[M // 2 + 1:] + ps_2D[1:M // 2][::-1])
            ps2D_err_fold = 0.5 * np.sqrt(ps_2D_err[M // 2 + 1:] ** 2 + ps_2D_err[1:M // 2][::-1] ** 2)
            ps2D_w_fold = ps_2D_w[M // 2 + 1:] + ps_2D_w[1:M // 2][::-1]
            
        return kp, k_par, np.array(ps2D_fold), np.array(ps2D_w_fold), np.array(ps2D_err_fold) 

        
    return kp, kpar, np.array(ps_2d), np.array(ps_2d_w), np.array(ps_2d_err)

def reshape_data_cube(data_cube,u,v,bl_min,bl_max):
    """Keep non-zero visibility cube between min and max baslines (in wavelength unit).
    
    For each delay channel, it will keep the non-zero visibilities between min and max 
    baseline length and flatten the array.
    
    Parameters
    ----------
    data_cube : np.ndarray[nfreq,Nx,Ny]
     The data cube to reshape and flatten.
    u : np.ndarray[Nx]
     The u-coordinates in wavelength unit.
    v : np.ndarray[Ny]
     The v-coordinates in wavelength unit. 
    bl_min : float
     The min baseline length in wavelength unit.
    bl_max : float
     The max baseline length in wavelength unit.
     
    Returns
    -------
    ft_cube : np.ndarray[nfreq,nvis]
     The flatten data cube.
    uu : np.ndarray[nvis]
     The flatten u-coordinates
    vv : np.ndarray[nvis]
     The flatten v-coordinates
    """
    g_uu, g_vv = np.meshgrid(v, u)
    print(f'Gridded data cube shape in kx and ky direction - {g_uu.shape}') 
    g_ru = np.sqrt(g_uu ** 2 + g_vv ** 2)
    bl_idx = (g_ru >= bl_min) & (g_ru <= bl_max)
    uu = g_uu[bl_idx]
    vv = g_vv[bl_idx] 
    
    ft_cube = []
    for ii in range(data_cube.shape[0]):
        ft = data_cube[ii,:,:]
        ft_cube.append(ft[bl_idx].flatten())
        
    return np.array(ft_cube),uu.flatten(), vv.flatten() 

def vol_normalization_vis(Nx,Ny,Nz,theta_res,redshift,chan_width):
    """Estimate the volume normalization factor for the power spectrum.
    
    Parameters
    ----------
    Nx : int
     Number of cells along RA or X-axis direction.
    Ny : int
      Number of cells along DEC or Y-axis direction.
    Nz : int
      Number of samples along freq or Z- axis direction.
    theta_res : float 
      The cell size  in radian unit.
    redshift : float
      redshift at the centre of the band.
    chan_width : float
       channel width in MHz unit.
    
    Returns
    -------
    norm : float
      The  PS normalization factor in Mpc^3 unit
    """
    DMz = cosmo.comoving_transverse_distance(redshift).value # in Mpc
    
    # Along spatial direction 
    dtheta_Mpc =  DMz * theta_res  # res (radian) to Mpc unit
    Lx = (Nx*dtheta_Mpc)  #survey length in RA/X-axis [Mpc]
    Ly = (Ny*dtheta_Mpc) # survey length in DEC/Y-axis [Mpc]
     
    ## Along line-of-sight direction  
    f21 = 1420.405752 * un.MHz # MHz
    c = const.c.to("km/s").value # [km/s]
    Hz = cosmo.H(redshift).value  # [km/s/Mpc]
    d_z = c * (1+redshift)**2 * chan_width / Hz / f21.value  #[Mpc] The sampling interval along the Z line-of-sight dimension Reference: Ref.[liu2014].Eq.(A9)
    Lz = d_z * Nz  #The linear size in z-direction [Mpc]
    

    print(f'Linear size in X-direction = {Lx} Mpc')
    print(f'Linear size in Y-direction = {Ly} Mpc')
    print(f'Linear size in Z-direction = {Lz} Mpc')
    
    norm = (Lx * Ly * Lz)
    print(f'Power Spectrum normalization factor  value - {norm} Mpc^3')
    
    return norm


# FG filter functions 

from sklearn.decomposition import FastICA, NMF, KernelPCA, PCA
import GPy

def pca_fg_filter(data_cube, nfg = 5):
    '''
    This function will estimate the principal components of the data from the freq covariance and remove 
    nfg number of modes, which are dominant by FG emission, i.e, with largest eigenvalues of the covariance matrix. 
    This is PCA fg filter technique. 
    
    Parameters
    ----------
    
    data_cube: the image cube. np.ndarray[freq,ra,el]
    nfg : int, 
          number of eigenmodes to remove from the data. Default: nfg=5
    
    Returns
    -------
    residual_cube: The FG filtered clean cube
    A : The design matrix of shape [nfreq,nfg]. This is the FG operator.
    FG_modes : The FG amplitudes of eigenmodes,  of shape [nfg,nra,el].
    eigvals: The eigenvalues of the freq cov matrix (sorted as largest value first)
    eigvecs: The eigenvectors of the freq cov matrix (sorted as largest value first)
    
    '''
    shp = data_cube.shape # shape of the data cube [nfreq,nRA,nel]
    nfreq, nra, ndec = shp[0], shp[1], shp[2]
    print(f" nfreq :{nfreq}, Nx : {nra}, Ny: {ndec}")
    data = data_cube.reshape(nfreq, nra * ndec) # shaping the data in [nfreq, npix] shape
    
    Cov = np.cov(data) # estimate the freq covariance
    eigvals, eigvecs = np.linalg.eigh(Cov) # eigendecomposition
    
    # Sort by eigenvalue
    idxs = np.argsort(eigvals)[::-1] # reverse order (biggest eigenvalue first)
    eigvals = eigvals[idxs]
    eigvecs = eigvecs[:,idxs]
    
    # Construct the design matrix
    A = eigvecs[:,:nfg] # (Nfreqs, Nmodes)
    
    # The foreground amplitudes for each line of sight
    FG_modes = np.dot(A.T, data) # (Nmodes, Npix)
    
    # Reconstruct the FG map
    
    FG_map = np.dot(A, FG_modes) # Design matrix times FG_modes
    FG_map = FG_map.reshape(nfreq,nra,ndec)
    
    # Subtract the FG map from data
    residual_cube = data_cube - FG_map
    
    return residual_cube, A, FG_modes.reshape(nfg,nra,ndec), eigvals, eigvecs, FG_map



def gpr_fg_clean(data_cube, freqs, k_fg_sky, k_HI, num_restarts=10,
             nPCA=0, noise_data=None, zero_noise=False, heteroscedastic=False, invert=False, num_processes=1,
             include_HI_kern=True):
    '''
    Runs foreground clean on IM data using GPR.
    ---------
    Arguments
    ---------
    
    data_cube: data cube to be cleaned, in [Nfreq,Nx,Ny] where Nfreq is the frequency direction
    
    freqs: array of frequency range of data of shape Nfreq in MHz
    
    k_fg_sky, k_HI: choice of foreground and 21cm kernels (can be a sum or product of different kernels)
    
    num_restarts: how many times to optimise the GPR regression model.
        [NOTE: If you already know your best fitting kernels (i.e., you're inputting the already
        optimized k_FG and k_21cm kernels), set num_restarts = 0]
        
    nPCA: 0 if no pre-PCA is desired, otherwise this number is the N_FG number of components used
        in a pre-PCA clean of the data
    
    noise_data: input here your noise map in [Nfreq,Nx,Ny] if you have a reasonable estimate from your data, 
        otherwise set to None and use GPR to try to fit your noise
        
    zero_noise: if True, the noise in your GPR model will be set to zero and fixed. Otherwise it will try to
        fit to noise in your data, in either the heteroscedastic or non-heteroscedastic case. Set to zero if 
        you want to fit your noise with a separate kernel instead, otherwise you will fit to noise twice.
        
    heteroscedastic: if True, runs Heteroscedastic regression model (variable noise)
        (Note: you cannot have zero_noise=False and heteroscedastic=True at the same time, set 
        heteroscedastic=False instead in this case).
        
    invert: if True, inverts data in the frequency direction
    num_processes = Number of Processes to use in parallel
    '''
    axes = np.shape(data_cube)
    
    # if desired, do a pre-PCA with nfg=nPCA removed components
    if nPCA > 0: 
        print(f'Doing PCA clean first using {nPCA} FG modes')
        data_cube, _,_,_,_,_ = pca_fg_filter(data_cube, nfg=nPCA)
    
    # converting [Nfreq,Nx,Ny] -> [Nfreq,Npix]
    Input = data_cube.reshape(axes[0],axes[1]*axes[2])
    if noise_data is not None: noise_data = noise_data.reshape(axes[0],axes[1]*axes[2])
    
    # invert frequency axis
    if invert==True: 
        Input = Input[::-1,:]
        freqs = freqs[::-1]  # inverting the freq axis. CHIME data is in decreasing freq order.
        if noise_data is not None: noise_data = noise_data[::-1,:]
    
    # build your model, input the freq range, the data, and the kernels
    if include_HI_kern:
        kern = k_fg_sky + k_HI
    else:
        kern = k_fg_sky
        
    # this heteroscedastic case assumes a Gaussian noise variance that changes with frequency
    if heteroscedastic==True: 
        # this case assumes noise is known, sets noise level to your noise_data variances
            # at different frequencies (since heteroscedastic)
        if noise_data is not None:
            model = GPy.models.GPHeteroscedasticRegression(freqs[:, np.newaxis], Input, kern)
            model.het_Gauss.variance.constrain_fixed(noise_data.var(axis=1)[:, None])
        # this case assumes noise is not known, model will fit a variance at each frequency
        else:
            model = GPy.models.GPHeteroscedasticRegression(freqs[:, np.newaxis], Input, kern)
        # note: if you want the case of *no noise*, there's no need to use heteroscedastic,
            # so set heteroscedastic = False and see below
    
    # this non-heteroscedastic case assumes constant Gaussian noise variance throughout frequency
    else: 
        # this case assumes noise is know, sets the noise variance level to the variance
            # from the input noise_data
        if noise_data is not None:
            model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
            model.Gaussian_noise.constrain_fixed(noise_data.var())
        else:
            # this case assumes there is no noise in your data
            if zero_noise == True:
                model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
                model['.*Gaussian_noise'] = 0.0
                model['.*noise'].fix()
            # this case assumes there is noise but it is unknown, fits a constant variance
            else:
                model = GPy.models.GPRegression(freqs[:, np.newaxis], Input, kern)
    
    # optimise model, find best fitting hyperparameters for kernels
    model.optimize_restarts(num_restarts=num_restarts, verbose=True, 
                            num_processes=num_processes, parallel=num_processes > 1)
    
    # extract optimised foreground kernel (depending on how many kernels were considered)
    if k_fg_sky.name == 'sum':
        k_FG_len = len(k_fg_sky.parts)
        print(f'FG kernels length {k_FG_len}')
        k_fg_sky = model.kern.parts[0]
        if k_FG_len > 1:
            for i in range(1, k_FG_len):
                k_fg_sky += model.kern.parts[i]
    else: k_fg_sky = model.kern.parts[0]
    print(f'The foreground kernels are {k_fg_sky}')
    
    # make prediction of what FGs would look like using this optimised FG kernel
    fg_fit, fg_cov = model.predict(freqs[:, np.newaxis], full_cov=True, kern=k_fg_sky,
        include_likelihood=False)
    
    # subtract FG fit from data, to obtain HI residuals:
    gpr_res = Input - fg_fit
    
    # un-revert frequency axis:
    if invert==True: 
        gpr_res = gpr_res[::-1,:]
        fg_fit = fg_fit[::-1,:]
    
    # reshape foreground fit
    fg_fit = np.reshape(fg_fit,(axes[0], axes[1], axes[2]))
    
    
    # reshape residuals
    gpr_res = np.reshape(gpr_res,(axes[0], axes[1], axes[2]))
    
    
    # create series as output object
    d = {'res': gpr_res, 'fgcov': fg_cov, 'fgfit': fg_fit, 'model': model}
    result = pd.Series(d)
    
    return result