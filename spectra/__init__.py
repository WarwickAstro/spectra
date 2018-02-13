from __future__ import print_function, division

__author__ = "Mark Hollands"

import numpy as np
from sys import exit
from scipy.interpolate import interp1d, Akima1DInterpolator as Ak_i
from scipy.optimize import leastsq
from scipy.special import wofz
from astropy.io import fits
import os
import math

jangstrom = \
  "$\mathrm{erg}\;\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}$"
fitskeys = ('loglam', 'flux', 'ivar')

###############################################################################

class Spectrum(object): 
  """
  spectrum class contains wavelengths, fluxes, and flux errors.  Arithmetic
  operations with single values, array, or other spectra are defined, with
  standard error propagation supported. Spectra also support numpy style array
  slicing

  Example:
  >>> S1 = Spectrum(x1, y1, e1)
  >>> S2 = Spectrum(x1, y1, e1)
  >>> S3 = S1 - S2

  .............................................................................
  In this case S1, S2, and S3 are all 'Spectrum' objects but the errors on S3
  are calculated as S3.e = sqrt(S1.e**2 + S2.e**2)

  If one needs only one of the arrays, these can be accessed as attributes.

  Example:
  .............................................................................
  >>> plt.plot(S.x, S.y) #plots the spectrum with matplotlib
  >>> plt.show()

  .............................................................................
  """

  def __init__(self, x, y, e, name=""):
    """
    Initialise spectrum
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(e, np.ndarray)
    assert isinstance(name, str)
    assert x.ndim == y.ndim == e.ndim == 1
    assert len(x) == len(y) == len(e)
    self.name = name
    self.x = x
    self.y = y
    self.e = e

  def __len__(self):
    """
    Return number of pixels in spectrum
    """
    assert len(self.x) == len(self.y) == len(self.e)
    return len(self.x)

  def __repr__(self):
    """
    Return spectrum representation
    """
    return "Spectrum class with {} pixels\nName: {}".format(len(self), self.name)

  def __getitem__(self, key):
    """
    Return self[key]
    """
    if isinstance(key, (int, slice, np.ndarray)):
      indexed_data = self.x[key], self.y[key], self.e[key]
      if isinstance(key, int):
        return indexed_data
      else:
        if isinstance(key, np.ndarray):
          assert len(key) == len(self)
          assert key.dtype == bool
        return Spectrum(*indexed_data, self.name)
    else:
      raise TypeError

  def __iter__(self):
    """
    Return iterator of spectrum
    """
    return zip(self.x, self.y, self.e)

  def __add__(self, other):
    """
    Return self + other (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      new_data = self.x * 1., \
                 self.y + other, \
                 self.e * 1.
    elif isinstance(other, np.ndarray):
      assert len(self) == len(other)
      new_data = self.x * 1., \
                 self.y + other, \
                 self.e * 1.
    elif isinstance(other, Spectrum):
      assert len(self) == len(other)
      assert np.all(np.isclose(self.x, other.x))
      new_data = 0.5*(self.x+other.x), \
                 self.y+other.y, \
                 np.hypot(self.e, other.e)
    else:
      raise TypeError
    return Spectrum(*new_data)

  def __sub__(self, other):
    """
    Return self - other (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      new_data = self.x * 1., \
                 self.y - other, \
                 self.e * 1.
    elif isinstance(other, np.ndarray):
      assert len(self) == len(other)
      new_data = self.x * 1., \
                 self.y - other, \
                 self.e * 1.
    elif isinstance(other, Spectrum):
      assert len(self) == len(other)
      assert np.all(np.isclose(self.x, other.x))
      new_data = 0.5*(self.x+other.x), \
                 self.y - other.y, \
                 np.hypot(self.e, other.e)
    else:
      raise TypeError
    return Spectrum(*new_data)
      
  def __mul__(self, other):
    """
    Return self * other (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      new_data = self.x * 1., \
                 self.y * other, \
                 self.e * other
    elif isinstance(other, np.ndarray):
      assert len(self) == len(other)
      new_data = self.x * 1., \
                 self.y * other, \
                 self.e * other
    elif isinstance(other, Spectrum):
      assert len(self) == len(other)
      assert np.all(np.isclose(self.x, other.x))
      new_data = 0.5*(self.x+other.x), \
                 self.y*other.y, \
                 y2*np.hypot(self.e/self.y, other.e/other.y)
    else:
      raise TypeError
    return Spectrum(*new_data)

  def __truediv__(self, other):
    """
    Return self / other (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      new_data = self.x * 1., \
                 self.y / other, \
                 self.e / other
    elif isinstance(other, np.ndarray):
      assert len(self) == len(other)
      new_data = self.x * 1., \
                 self.y / other, \
                 self.e / other
    elif isinstance(other, Spectrum):
      assert len(self) == len(other)
      assert np.all(np.isclose(self.x, other.x))
      y2 = self.y/other.y
      new_data = 0.5*(self.x+other.x), \
                 y2, \
                 y2*np.hypot(self.e/self.y, other.e/other.y)
    else:
      raise TypeError
    return Spectrum(*new_data)

  def __pow__(self,other):
    """
    Return S**other (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      new_data = self.x * 1., \
                 self.y**other, \
                 other * y2 * self.e/self.y
    else:
      raise TypeError
    return Spectrum(*new_data)

  def __radd__(self, other):
    """
    Return other + self (with standard error propagation)
    """
    return self + other

  def __rsub__(self, other):
    """
    Return other - self (with standard error propagation)
    """
    return -(self - other)

  def __rmul__(self, other):
    """
    Return other * self (with standard error propagation)
    """
    return self * other

  def __rtruediv__(self, other):
    """
    Return other / self (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      new_data = self.x * 1., \
                 other / self.y, \
                 other * self.e /(self.y*self.y)
    elif isinstance(other,np.ndarray):
      assert len(self) == len(other)
      new_data = self.x * 1., \
                 other / self.y, \
                 other * self.e /(self.y*self.y)
    else:
      raise TypeError
    return Spectrum(*new_data)

  def apply_mask(self, mask):
    """
    Apply a mask to the spectral fluxes
    """
    self.y = np.ma.masked_array(self.y, mask)
    self.e = np.ma.masked_array(self.e, mask)

  def mag_calc_AB(self, filt, NMONTE=1000):
    """
    Calculates the AB magnitude of a filter called 'filt'. Errors
    are calculated in Monte-Carlo fashion, and assume all fluxes
    are statistically independent (not that realistic). See the
    definition of 'mag_clac_AB' for valid filter names.
    """

    if np.all(self.e == 0):
      return mag_calc_AB(self.x, self.y, self.y*1e-9, filt, NMONTE=0)
    else:
      return mag_calc_AB(self.x, self.y, self.e, filt, NMONTE=1000)

  def interp_wave(self, X, kind='linear', **kwargs):
    """
    Interpolates a spectrum onto the wavlength axis X, if X is a numpy array,
    or X.x if X is Spectrum type. This returns a new spectrum rather than
    updating a spectrum in place, however this can be acheived by

    >>> S1 = S1.interp_wave(X)

    Wavelengths outside the range of the original spectrum are filled with
    zeroes. By default the interpolation is nearest neighbour.
    """
    if isinstance(X, np.ndarray):
      x2 = 1*X
    elif isinstance(X, Spectrum):
      x2 = 1*X.x
    else:
      raise TypeError
    if kind == "Akima":
      pass
      y2 = Ak_i(self.x, self.y)(x2)
      e2 = Ak_i(self.x, self.y)(x2)
      nan = np.isnan(y2) | np.isnan(e2)
      y2[nan] = 0.
      e2[nan] = 0.
    else:
      extrap_y, extrap_e = (self.y[0],self.y[-1]), (self.e[0],self.e[-1])
      y2 = interp1d(self.x, self.y, kind=kind, \
        bounds_error=False, fill_value=0., **kwargs)(x2)
      e2 = interp1d(self.x, self.e, kind=kind, \
        bounds_error=False, fill_value=0., **kwargs)(x2)
    return Spectrum(x2,y2,e2, self.name)

  def copy(self):
    """
    Retrurns a copy of self
    """
    return 1.*self

  def sect(self,x0,x1):
    """
    Returns a truth array for wavelengths between x0 and x1.
    """
    return (self.x>x0) & (self.x<x1)

  def clip(self, x0, x1): 
    """
    Returns Spectrum clipped between x0 and x1.
    """
    return self[self.sect(x0, x1)]

  def norm_percentile(self, pc):
    """
    Normalises a spectrum to a certain percentile of its fluxes.
    
    E.g. S.norm_percentile(99)
    """
    norm = np.percentile(self.y, pc)
    self.y /= norm
    self.e /= norm

  def write(self, fname, errors=True):
    """
    Saves Spectrum to a text file.
    """
    #C style formatting faster here than .format or f-strings
    with open(fname, 'w') as F:
      if errors:
        for px in self: F.write("%9.3f %12.5E %11.5E\n" %px)
      else:
        for px in self: F.write("%9.3f %12.5E\n" %px[:2])

  def air_to_vac(self):
    """
    Changes air wavelengths to vaccuum wavelengths in place
    """
    self.x = air_to_vac(self.x) 

  def vac_to_air(self):
    """
    Changes vaccuum wavelengths to air wavelengths in place
    """
    self.x = vac_to_air(self.x) 

  def Flambda_to_Fnu(self):
    """
    Change Flambda [Jangstrom] to Fnu [Jansky]
    """
    arr = self.x**2/2.998e-5
    self.y *= arr
    self.e *= arr

  def Fnu_to_Flambda(self):
    """
    Change Flambda [Jansky] to Fnu [Jangstrom]
    """
    arr = 2.998e-5/self.x**2
    self.y *= arr
    self.e *= arr

  def apply_redshift(self, v, wavelengths, unit='km/s'):
    """
    Applies redshift of v km/s to spectrum for "air" or "vac" wavelengths
    """
    c0 = 2.99792458e5
    if unit == 'km/s':
      beta = v/c0
    elif unit == 'c':
      beta = v
    else:
      raise ValueError("'unit' should be in ['km/s', 'c']")
    factor = math.sqrt((1+beta)/(1-beta))
    if wavelengths == "air":
      self.x = air_to_vac(self.x) 
      self.x *= factor
      self.x = vac_to_air(self.x) 
    elif wavelengths == "vac":
      self.x *= factor
    else:
      pass

  def scale_model(self, S_in, return_scaling_factor=False):
    """
    If self is model spectrum (errors are presumably zero), and S is a data
    spectrum (has errors) then this reproduces a scaled version of M2.
    There is no requirement for either to have the same wavelengths as
    interpolation is performed. However the resulting scaled model will
    have the wavelengths of the original model, not the data. If you want
    the model to share the same wavelengths, use model.interp_wave(),
    either before or after calling this function.
    """
    assert isinstance(S_in, Spectrum)

    #if M and S already have same x-axis, this won't do much.
    S = S_in[S_in.e>0]
    M = self.interp_wave(S)

    A_sm, A_mm = np.sum(S.y*M.y/S.e**2), np.sum((M.y/S.e)**2)
    A = A_sm/A_mm

    if return_scaling_factor:
      return self*A, A
    else:
      return self*A
    
  def convolve_gaussian(self, fwhm):
    self.y = convolve_gaussian(self.x, self.y, fwhm)

  def convolve_gaussian_R(self, res):
    self.y = convolve_gaussian_R(self.x, self.y, res)

  def split(self, W):
    """
    If W is an int/float, splits spectrum in two around W. If W is an
    interable of ints/floats, this will split into mutliple chunks instead.
    """
    if isinstance(W, (int, float)):
      W = -np.inf, W, np.inf
    elif isinstance(W, (list, tuple, np.ndarray)):
      if not all([isinstance(w, (int, float)) for w in W]): raise TypeError
      W = -np.inf, *sorted(W), np.inf
    else:
      raise TypeError
    return tuple(self.clip(*pair) for pair in zip(W[:-1], W[1:]))

  def join(self, other, sort=False):
    """
    Joins a second spectrum to the current spectrum. Can potentially be used
    rescursively, i.e.
    >>> S = S1.join(S2).join(S3)

    But  will not be as fast as 
    """
    assert isinstance(other, Spectrum)
    return join_spectra((self, other), sort=sort)

#..............................................................................

def join_spectra(SS, sort=False):
  """
  Joins a collection of spectra into a single spectrum. The name of the first
  spectrum is used as the new name. Can optionally sort the new spectrum by
  wavelengths.
  """
  for S in SS:
    assert isinstance(S, Spectrum), 'item is not Spectrum'
  x = np.hstack(S.x for S in SS)
  y = np.hstack(S.y for S in SS)
  e = np.hstack(S.e for S in SS)
  if sort:
    idx = np.argsort(x)
    return Spectrum(x[idx], y[idx], e[idx], name=SS[0].name)
  else:
    return Spectrum(x, y, e, name=SS[0].name)

def spec_from_txt(fname, **kwargs):
  """
  Loads a text file with the first 3 columns as wavelengths, fluxes, errors.
  """
  x, y, e = np.loadtxt(fname, unpack=True, usecols=(0,1,2), **kwargs)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(x, y, e, name=name)
    
def model_from_txt(fname, **kwargs):
  """
  Loads a text file with the first 2 columns as wavelengths and fluxes.
  This produces a spectrum object where the errors are just set to zero.
  This is therefore good to use for models.
  """
  x, y = np.loadtxt(fname, unpack=True, usecols=(0,1), **kwargs)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(x, y, np.zeros_like(x), name=name)

def spec_from_sdss_fits(fname, **kwargs):
  """
  Loads a SDSS fits file as spectrum (result in vac wavelengths)
  """
  hdulist = fits.open(fname)
  loglam, flux, ivar = [hdulist[1].data[key] for key in fitskeys]
  lam = 10**loglam
  ivar[ivar==0.] = 0.001
  err = 1/np.sqrt(ivar)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(lam, flux, err, name=name)*1e-17

def spectra_mean(spectra):
  """
  Calculate the weighted mean spectrum of a list/tuple of spectra.
  All spectra should have identical wavelengths.
  """
  for S in spectra:
    assert isinstance(S, Spectrum)
    assert len(S) == len(spectra[0])
    assert np.isclose(S.x, spectra[0].x).all()

  X, Y, E = np.array([S.x for S in spectra]), \
            np.array([S.y for S in spectra]), \
            np.array([S.e for S in spectra])

  Xbar, Ybar, Ebar = np.mean(X,axis=0), \
                     np.sum(Y/E**2, axis=0)/np.sum(1/E**2, axis=0), \
                     1/np.sqrt(np.sum(1/E**2, axis=0))

  return Spectrum(Xbar, Ybar, Ebar)
    
###############################################################################

Va = 1/(2*np.sqrt(2*np.log(2)))
Vb = np.sqrt(2)
Vc = np.sqrt(2*np.pi)

def voigt( x, x0, fwhm_g, fwhm_l ):
  sigma = Va*fwhm_g
  z = ((x-x0) + 0.5j*fwhm_l)/(sigma*Vb)
  return wofz(z).real/(sigma*Vc)

def sdss_mag_to_flux( ew, mag, mag_err, offset ):
  """
  Converts an SDSS filter to a flux in Janskys/c. Offset is required to go
  from SDSS mags to AB mags. This should be 0.04 for u, else 0.
  """
  mag -= offset
  F_nu = 10**( -0.4*mag -19.44 )
  conversion_factor = 2.998e18/ew**2 #speed of light over lambda**2
  F_lambda = conversion_factor*F_nu
  if mag_err > 0:
    F_err    = 0.4*F_lambda*mag_err
    return F_lambda, F_err
  else:
    return F_lambda
    
#

def mag_calc_AB( w, f, e, filt, NMONTE=1000 ):
  """
  Calculates the synthetic AB magnitude of a spectrum for a given filter.
  If NMONTE is > 0, monte-carlo error propagation is performed outputting
  both a synthetic-mag and error. For model-spectra, i.e. no errors,
  use e=np.ones_like(f) and NMONTE=0. List of currently supported filters:

  SDSS:    ['u','g','r','i','z']

  Johnson: ['U','B','V','R','I']

  Galex:   'GalexNUV'

  Denis:   'DenisI'

  2Mass:   ['2mJ','2mH','2mK']

  WISE:    ['W1','W2']

  Spitzer: ['S1','S2']
  """

  #load filter
  long_path = "/home/astro/phujdu/Python/MH/mh/spectra/filt_profiles/"
  if   filt in 'ugriz':
    full_path = long_path+"SLOAN_SDSS."+filt+".dat"
  elif filt in 'UBVRI':
    full_path = long_path+"Generic_Johnson."+filt+".dat"
  elif filt == 'GalexNUV':
    full_path = long_path+"GALEX_GALEX.NUV.dat"
  elif filt == 'DenisI':
    full_path = long_path+"DENIS_DENIS.I.dat"
  elif filt in ['2m'+b for b in 'JHK']:
    full_path = long_path+"2MASS_2MASS."+filt[-1]+".dat"
  elif filt in ['UK'+b for b in 'YJHK']:
    full_path = long_path+"UKIRT_UKIDSS."+filt[-1]+".dat"
  elif filt in ['W'+b for b in '12']:
    full_path = long_path+"WISE_WISE."+filt+".dat"
  elif filt in ['S'+b for b in '12']:
    full_path = long_path+"Spitzer_IRAC.I"+filt[1]+".dat"
  else:
    raise ValueError( 'Invalid filter name: %s'%filt )

  w_filt, R_filt = np.loadtxt( full_path, unpack=True )

  #clip original data to filter range and remove bad flux
  w_slice = ( w > w_filt[0] ) & ( w < w_filt[-1] )
  e_good =  e/np.median(e) < 5.
  w, f, e = [arr[w_slice&e_good] for arr in (w, f, e)]

  #calculate the pivot wavelength
  dw = differentiate_axis( w_filt )
  w_piv = np.sqrt(np.sum(R_filt * w_filt * dw)/np.sum(R_filt / w_filt * dw))

  #interpolate filter to new w axis
  R_func = interp1d(w_filt, R_filt)  
  R_filt = R_func(w)

  #calculate wavelength step. matches original axis length
  dw = differentiate_axis(w)

  #calculate f_nu at w_piv via monte carlo
  if NMONTE == 0:
      f_l = np.sum(w * R_filt * f * dw)/np.sum(w * R_filt *dw) 
      f_nu = f_l * w_piv**2 * 3.335640952e4
      m = -2.5 * np.log10(f_nu) + 8.90
      return m #no error if NMONTE=0
  else:
    m = np.empty( NMONTE, "float64" )
    for i in range(NMONTE):
      f_monte = np.random.normal(f, e)
      f_l = np.sum(w * R_filt * f_monte * dw)/np.sum(w * R_filt *dw) 
      f_nu = f_l * w_piv**2 * 3.335640952e4
      m[i] = -2.5 * np.log10(f_nu) + 8.90

    return np.mean(m), np.std(m)
#

def mag_calc_AB2( w, f, e, filter ):
  """
  Calculates the synthetic AB magnitude of a spectrum.
  Uses a monte carlo approach to calculate errors assuming
  no covariance between adjacent wavelength elements.
  """

  #load filter
  long_path = "/home/astro/phujdu/Python/MH/mh/spectra/"
  if   filter in ['u','g','r','i','z']:
    full_path = long_path+"SLOAN_SDSS."+filter+".dat"
  elif filter in ['U','B','V','R','I']:
    full_path = long_path+"Generic_Johnson."+filter+".dat"
  else:
    print("bad filter name")
    exit()
  w_filt, R_filt = np.loadtxt( full_path, unpack=True )

  #clip original data to filter range and remove bad flux
  w_slice = ( w > w_filt[0] ) & ( w < w_filt[-1] )
  e_good =  e/np.median(e) < 5.
  w = w[w_slice&e_good]
  f = f[w_slice&e_good]
  e = e[w_slice&e_good]

  #interpolate filter to new w axis
  R_func = interp1d( w_filt, R_filt )  
  R_filt = R_func( w )

  #calculate wavelength step. matches original axis length
  dw = differentiate_axis( w )

  #calculate magnitudes in a monte carlo way
  NMONTE = 1000
  m = np.empty( NMONTE, "float64" )
  for i in range(NMONTE):
    f_monte = np.random.normal( f, e )
    f_jansk = 3631. * 2.99792458e-5/w**2
    top_integral = np.sum( f_monte * R_filt * dw )
    bot_integral = np.sum( f_jansk * R_filt * dw ) 
    m[i] = -2.5 * np.log10( top_integral/bot_integral )

  return np.mean( m ), np.std( m )
#

def mag_calc_AB_bootstrap( w, f, e, filter ):
  """
  Calculates the synthetic AB magnitude of a spectrum.
  Uses a bootstrap approach to calculate errors.
  """

  #load filter
  long_path = "/home/astro/phujdu/Python/MH/mh/spectra/"
  if   filter in ['u','g','r','i','z']:
    full_path = long_path+"SLOAN_SDSS."+filter+".dat"
  elif filter in ['U','B','V','R','I']:
    full_path = long_path+"Generic_Johnson."+filter+".dat"
  elif filter == "GALEX_NUV":
    full_path = long_path+"GALEX_GALEX.NUV.dat"
  else:
    print("bad filter name")
    exit()
  w_filt, R_filt = np.loadtxt( full_path, unpack=True )

  #clip original data to filter range and remove bad flux
  w_slice = ( w > w_filt[0] ) & ( w < w_filt[-1] )
  e_good =  e/np.median(e) < 5.
  w = w[w_slice&e_good]
  f = f[w_slice&e_good]
  e = e[w_slice&e_good]

  #calculate the pivot wavelength
  dw = differentiate_axis( w_filt )
  w_piv = np.sqrt(np.sum(R_filt * w_filt * dw)/np.sum(R_filt / w_filt * dw))

  #interpolate filter to new w axis
  R_filt = interp1d( w_filt, R_filt )( w )

  #calculate wavelength step. matches original axis length
  dw = differentiate_axis( w )

  #calculate f_nu at w_piv via bootstrap
  NBOOT = 5000
  m = np.empty( NBOOT, "float64" )
  for i in range(NBOOT):
    resample = np.random.randint( 0, len(w), len(w) )
    f_l = np.sum( (w*R_filt*f*dw)[resample] )/np.sum( (w*R_filt*dw)[resample] )
    f_nu = f_l * w_piv**2 * 3.335640952e4
    m[i] = -2.5 * np.log10( f_nu ) + 8.90

  return np.mean( m ), np.std( m )
#

def lambda_piv( filter ):
  #load filter
  long_path = "/home/astro/phujdu/Python/MH/mh/spectra/"
  if   filter in ['u','g','r','i','z']:
    full_path = long_path+"SLOAN_SDSS."+filter+".dat"
  elif filter in ['U','B','V','R','I']:
    full_path = long_path+"Generic_Johnson."+filter+".dat"
  else:
    print("bad filter name")
    exit()
  w_filt, R_filt = np.loadtxt( full_path, unpack=True )

  #calculate wavelength step. matches original axis length
  dw = differentiate_axis( w )

  #calculate the pivot wavelength
  return np.sqrt(np.sum(R_filt* w_filt * dw)/np.sum(R_filt / w_filt * dw))
#

def differentiate_axis( x ):
  """
  for an axis x, this calculates dx from:

  dx_i = 0.5*( x_i+1 - x_i-1 )

  dx_0 and dx_N are copies of dx_1 and dx_N-1 respectively
  """

  dx = 0.5 * ( x[2:] - x[:-2] )
  dx0 = np.array( (dx[ 0],) )
  dxN = np.array( (dx[-1],) )
  dx = np.hstack( (dx0,dx,dxN) )
  return dx
#

def vac_to_air( Wvac ):
  """
  converts vacuum wavelengths to air wavelengths,
  as per VALD3 documentation (in Angstroms)
  """
  s = 1e4/Wvac
  n = 1.0000834254 \
    + 0.02406147/(130.-s*s) \
    + 0.00015998/(38.9-s*s)
  return Wvac/n
#

def air_to_vac( Wair ):
  """
  converts air wavelengths to vacuum wavelengths,
  as per VALD3 documentation (in Angstroms)
  """
  s = 1e4/Wair
  n = 1.00008336624212083 \
    + 0.02408926869968 / (130.1065924522-s*s) \
    + 0.0001599740894897/(38.92568793293-s*s)
  return Wair*n
#

def sdss_mag2fl( w, ugriz, ugrizErr=None ):
  ugriz[0] -= 0.04
  F_nu = 10**( -0.4*ugriz -19.44 )
  conversion_factor = 2.998e18/w**2 #speed of light over lambda**2
  F_lambda = conversion_factor*F_nu
  if ugrizErr is None:
    return F_lambda
  else:
    F_err    = 0.9210340372*F_lambda*ugrizErr
    return F_lambda, F_err
#

def next_pow_2(N_in):
  N_out = 1
  while N_out < N_in:
    N_out *= 2
  return N_out

def convolve_gaussian(x, y, FWHM):
  """
  Convolve spectrum with a Gaussian with FWHM by oversampling and
  using an FFT approach. Wavelengths are assumed to be sorted,
  but uniform spacing is not required. Will cause wrap-around at
  the end of the spectrum.
  """
  sigma = FWHM/2.355

  #oversample data by at least factor 10 (up to 20).
  xi = np.linspace(x[0], x[-1], next_pow_2(10*len(x)))
  yi = interp1d(x, y)(xi)

  yg = np.exp(-0.5*((xi-x[0])/sigma)**2) #half gaussian
  yg += yg[::-1]
  yg /= np.sum(yg) #Norm kernel

  yiF = np.fft.fft(yi)
  ygF = np.fft.fft(yg)
  yic = np.fft.ifft(yiF * ygF).real

  return interp1d(xi, yic)(x)
#

def convolve_gaussian_R(x, y, R):
  """
  Similar to convolve_gaussian, but convolves to a specified resolution
  rather than a specfied FWHM. Essentially this amounts to convolving
  along a log-uniform x-axis instead.
  """
  return convolve_gaussian(np.log(x), y, 1./R)
#

def black_body( x, T, norm=True ):
  """
  x in angstroms
  T in Kelvin
  returns un-normed spectrum
  """
  logf = np.empty_like(x,dtype='float')
  Q = 143877516. /(x*T) # const. = ( h * c )/( 1e-10 * kB )
  lo = Q < 10.
  hi = ~lo
  #log form needed to stop overflow in x**-5
  #for Q>7. exp(Q)==expm1(Q) to better than 0.1%.
  logf[lo] = -5. * np.log( x[lo] ) - np.log( np.expm1(Q[lo]) )
  logf[hi] = -5. * np.log( x[hi] ) - Q[hi]
  if norm:
    logf -= logf.max() #normalise to peak at 1.
  return np.exp( logf )
#

def sky_residual(params, x, y, e ):
  """
  Fitting function for sky_line_fwhm
  """
  A, x0, s, c = params

  y_fit = A*np.exp( -0.5*((x-x0)/s)**2 ) + c

  norm_residual = (y - y_fit)/e

  if s < 0 or c < 0 or A < 0:
    return norm_residual * 1000.
  else:
    return norm_residual
#

def sky_line_fwhm( w, sky, w0 ):
  """
  Given a sky spectrum, this fits a Gaussian to a
  sky line and returns the FWHM.
  """
  guess = 1e-15, w0, 1., 0.

  clip = (w>w0-10.)&(w<w0+10.)

  args= w[clip], sky[clip], np.sqrt(sky[clip])
  result = leastsq(sky_residual, guess, args=args)

  vec = result[0]

  return vec[2] * 2.355
#

def keep_points( x, fname ):
  """
  creates a mask for a spectrum that regions between pairs from a file
  """
  C = np.zeros_like(x,dtype='bool')
  try:
    lines = open(fname,'r').readlines()
  except IOError:
    print("file %s does not exist" %fname)
    exit()
  for line in lines:
    s1, s2 = line.split()
    x1,x2 = float(s1),float(s2)
    C |= (x>x1)&(x<x2)
  return C
