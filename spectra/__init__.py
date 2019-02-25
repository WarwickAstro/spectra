from __future__ import print_function, division
__author__ = "Mark Hollands"

import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from scipy.interpolate import interp1d, Akima1DInterpolator as Ak_i
from scipy.optimize import leastsq
from scipy.special import wofz
from scipy.integrate import trapz as Itrapz, simps as Isimps
from astropy.io import fits
from functools import reduce
from trm import molly
import operator
import os
import math
import astropy.units as u
import astropy.constants as const

jangstrom = \
  "$\mathrm{erg}\;\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}$"

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
  >>> S.plot('k-') #plots the spectrum with matplotlib
  >>> plt.show()

  .............................................................................
  """
  __slots__ = ['name', 'head', 'x', 'y', 'e', 'wave', 'x_unit', 'y_unit']
  def __init__(self, x, y, e, name="", wave='air', x_unit="AA", y_unit="erg/(s cm^2 AA)", head=None):
    """
    Initialise spectrum. Arbitrary header items can be added to self.head
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(e, np.ndarray)
    assert x.ndim == y.ndim == e.ndim == 1
    assert len(x) == len(y) == len(e)
    assert np.all(e >= 0.)
    assert isinstance(name, str)
    assert wave in ("air", "vac")
    self.x = x
    self.y = y
    self.e = e
    self.name = name
    self.wave = wave
    self.x_unit = u.Unit(x_unit).to_string()
    self.y_unit = u.Unit(y_unit).to_string()
    if head is None:
      self.head = {}
    else:
      assert isinstance(head, dict)
      self.head = head

  @property
  def var(self):
    """
    Variance property attribute from flux errors
    """
    return self.e**2

  @property
  def ivar(self):
    """
    Inverse variance attribute from flux errors
    """
    return 1.0/self.var

  @property
  def SN(self):
    """
    Signal to noise ratio
    """
    return self.y/self.e

  @property
  def data(self):
    """
    Returns all three arrays as a tuple. Useful for creating new spectra, e.g.
    >>> Spectrum(*S.data)
    """
    return self.x, self.y, self.e

  @property
  def info(self):
    """
    Returns non-array attributes (in same order as __init__). This can be passed
    used to create new spectra with the same information, e.g.
    >>> Spectrum(x, y, e, *S.info)
    """
    return self.name, self.wave, self.x_unit, self.y_unit, self.head

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
    ret = "\n".join([
      "Spectrum class with {} pixels".format(len(self)),
      "Name: {}".format(self.name),
      "x-unit: {}".format(self.x_unit),
      "y-unit: {}".format(self.y_unit),
      "wavelengths: {}".format(self.wave),
    ])

    return ret

  def __getitem__(self, key):
    """
    Return self[key]
    """
    if isinstance(key, (int, slice, np.ndarray)):
      indexed_data = self.x[key], self.y[key], self.e[key]
      if isinstance(key, int):
        return indexed_data
      else:
        return Spectrum(*indexed_data, *self.info)
    else:
      raise TypeError

  def __iter__(self):
    """
    Return iterator of spectrum
    """
    return zip(*self.data)

  def __add__(self, other):
    """
    Return self + other (with standard error propagation)
    """
    if isinstance(other, (int, float, np.ndarray)):
      if isinstance(other, np.ndarray):
        assert len(self) == len(other)
      x2 = self.x.copy()
      y2 = self.y + other
      e2 = self.e.copy()
    elif isinstance(other, Spectrum):
      assert len(self) == len(other)
      assert np.all(np.isclose(self.x, other.x))
      assert self.x_unit == other.x_unit
      assert self.y_unit == other.y_unit
      x2 = 0.5*(self.x+other.x)
      y2 = self.y+other.y
      e2 = np.hypot(self.e, other.e)
    else:
      raise TypeError
    return Spectrum(x2, y2, e2, *self.info)

  def __sub__(self, other):
    """
    Return self - other (with standard error propagation)
    """
    if isinstance(other, (int, float, np.ndarray)):
      if isinstance(other, np.ndarray): assert len(self) == len(other)
      x2 = self.x.copy()
      y2 = self.y - other
      e2 = self.e.copy()
    elif isinstance(other, Spectrum):
      assert len(self) == len(other)
      assert np.all(np.isclose(self.x, other.x))
      assert self.x_unit == other.x_unit
      assert self.y_unit == other.y_unit
      x2 = 0.5*(self.x+other.x)
      y2 = self.y - other.y
      e2 = np.hypot(self.e, other.e)
    else:
      raise TypeError
    return Spectrum(x2, y2, e2, *self.info)
      
  def __mul__(self, other):
    """
    Return self * other (with standard error propagation)
    """
    if isinstance(other, (int, float, np.ndarray)):
      if isinstance(other, np.ndarray): assert len(self) == len(other)
      x2 = self.x.copy()
      y2 = self.y * other
      e2 = self.e * np.abs(other)
      y_unit = self.y_unit
    elif isinstance(other, Spectrum):
      assert len(self) == len(other)
      assert np.all(np.isclose(self.x, other.x))
      assert self.x_unit == other.x_unit
      x2 = 0.5*(self.x+other.x)
      y2 = self.y*other.y
      e2 = np.abs(y2)*np.hypot(self.e/self.y, other.e/other.y)
      u1, u2 = u.Unit(self.y_unit), u.Unit(other.y_unit)
      y_unit = (u1*u2).to_string()
    else:
      raise TypeError
    S = Spectrum(x2, y2, e2, *self.info)
    S.y_unit = y_unit
    return S

  def __truediv__(self, other):
    """
    Return self / other (with standard error propagation)
    """
    if isinstance(other, (int, float, np.ndarray)):
      if isinstance(other, np.ndarray): assert len(self) == len(other)
      x2 = self.x.copy()
      y2 = self.y / other
      e2 = self.e / np.abs(other)
      y_unit = self.y_unit
    elif isinstance(other, Spectrum):
      assert len(self) == len(other)
      assert np.all(np.isclose(self.x, other.x))
      assert self.x_unit == other.x_unit
      x2 = 0.5*(self.x+other.x)
      y2 = self.y/other.y
      e2 = np.abs(y2)*np.hypot(self.e/self.y, other.e/other.y)
      u1, u2 = u.Unit(self.y_unit), u.Unit(other.y_unit)
      y_unit = (u1/u2).to_string()
    else:
      raise TypeError
    S = Spectrum(x2, y2, e2, *self.info)
    S.y_unit = y_unit
    return S

  def __pow__(self,other):
    """
    Return S**other (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      x2 = self.x.copy()
      y2 = self.y**other
      e2 = other * y2 * self.e/self.y
    else:
      raise TypeError
    return Spectrum(x2, y2, e2, *self.info)

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
    if isinstance(other, (int, float, np.ndarray)):
      if isinstance(other, np.ndarray): assert len(self) == len(other)
      x2 = self.x.copy()
      y2 = other / self.y
      e2 = other * self.e /(self.y*self.y)
    else:
      raise TypeError
    y_unit = (1/u.Unit(self.y_unit)).to_string()
    S = Spectrum(x2, y2, e2, *self.info)
    S.y_unit = y_unit
    return S

  def __neg__(self):
    """
    Implements -self
    """
    return -1 * self

  def __pos__(self):
    """
    Implements +self
    """
    return self

  def __abs__(self):
    """
    Implements abs(self)
    """
    S = self.copy()
    S.y = np.abs(S.y)
    return S

  def apply_mask(self, mask):
    """
    Apply a mask to the spectral fluxes
    """
    self.x = np.ma.masked_array(self.x, mask)
    self.y = np.ma.masked_array(self.y, mask)
    self.e = np.ma.masked_array(self.e, mask)

  def remove_mask(self, mask):
    """
    Apply a mask to the spectral fluxes
    """
    self.x = np.array(self.x)
    self.y = np.array(self.y)
    self.e = np.array(self.e)

  def mag_calc_AB(self, filt, NMONTE=1000):
    """
    Calculates the AB magnitude of a filter called 'filt'. Errors
    are calculated in Monte-Carlo fashion, and assume all fluxes
    are statistically independent (not that realistic). See the
    definition of 'mag_clac_AB' for valid filter names.
    """
    S = self.copy()
    S.x_unit_to("AA")
    S.y_unit_to("erg/(s cm2 AA)")

    if np.all(self.e == 0):
      return mag_calc_AB(S, filt, NMONTE=0)
    else:
      return mag_calc_AB(S, filt, NMONTE=NMONTE)

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
      assert self.wave == X.wave
      x2 = 1*X.x
    else:
      raise TypeError
    if kind == "Akima":
      y2 = Ak_i(self.x, self.y)(x2)
      e2 = Ak_i(self.x, self.e)(x2)
      nan = np.isnan(y2) | np.isnan(e2)
      y2[nan] = 0.
      e2[nan] = 0.
    elif kind == "sinc":
      y2 = Lanczos(self.x, self.y, X)
      e2 = Lanczos(self.x, self.e, X)
    else:
      extrap_y, extrap_e = (self.y[0],self.y[-1]), (self.e[0],self.e[-1])
      y2 = interp1d(self.x, self.y, kind=kind, \
        bounds_error=False, fill_value=0., **kwargs)(x2)
      e2 = interp1d(self.x, self.e, kind=kind, \
        bounds_error=False, fill_value=np.inf, **kwargs)(x2)
    return Spectrum(x2, y2, e2, *self.info)

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
    if fname.endswith((".txt", ".dat")):
      #C style formatting faster here than .format or f-strings
      with open(fname, 'w') as F:
        if errors:
          for px in self: F.write("%9.3f %12.5E %11.5E\n" %px)
        else:
          for px in self: F.write("%9.3f %12.5E\n" %px[:2])
    elif fname.endswith(".npy"):
      if errors:
        data = np.array([self.x, self.y, self.e])
      else:
        data = np.array([self.x, self.y])
      np.save(fname, data)
    else:
      print("Unrecognised File type")
      print("Save aborted")

  def air_to_vac(self):
    """
    Changes air wavelengths to vaccuum wavelengths in place
    """
    assert u.Unit(self.x_unit) == u.Unit("AA")
    if self.wave == 'air':
      self.x = air_to_vac(self.x) 
      self.wave = 'vac'
    elif self.wave == 'vac':
      print("wavelengths already vac")
    else:
      raise ValueError

  def vac_to_air(self):
    """
    Changes vaccuum wavelengths to air wavelengths in place
    """
    assert u.Unit(self.x_unit) == u.Unit("AA")
    if self.wave == 'vac':
      self.x = vac_to_air(self.x) 
      self.wave = 'air'
    elif self.wave == 'air':
      print("wavelengths already air")
    else:
      raise ValueError

  def redden(self, E_BV, Rv=3.1):
    """
    Apply the CCM reddening curve to the spectrum given an E_BV
    and a value of Rv (default=3.1).
    """
    S = self.copy()
    if S.wave == "air":
      S.x_unit_to("AA")
      S.air_to_vac()
    S.x_unit_to("1/um")

    A = Rv * E_BV * A_curve(S.x, Rv)
    extinction = 10**(-0.4*A)
    self.y *= extinction
    self.e *= extinction

  def x_unit_to(self, new_unit):
    """
    Changes units of the x-data. Supports conversion between wavelength
    and energy etc. Argument should be a string.
    """
    assert isinstance(new_unit, str)

    x = self.x * u.Unit(self.x_unit)
    x = x.to(new_unit, u.spectral())
    self.x = x.value
    self.x_unit = u.Unit(new_unit).to_string()
    
  def y_unit_to(self, new_unit):
    """
    Changes units of the y-data. Supports conversion between Fnu
    and Flambda etc. Argument should be a string.
    """
    assert isinstance(new_unit, str)

    if new_unit == "mag":
      self.to_y_unit("Jy")
      self /= 3631
    else:
      x = self.x * u.Unit(self.x_unit)
      y = self.y * u.Unit(self.y_unit)
      e = self.e * u.Unit(self.y_unit)
      y = y.to(new_unit, u.spectral_density(x))
      e = e.to(new_unit, u.spectral_density(x))
      self.y = y.value
      self.e = e.value
      self.y_unit = u.Unit(new_unit).to_string()
    
  def apply_redshift(self, v, v_unit="km/s"):
    """
    Applies redshift of v km/s to spectrum for "air" or "vac" wavelengths
    """
    v *= u.Unit(v_unit)
    assert v.si.unit == const.c.unit
    assert self.wave in ('vac', 'air')
    beta = v/const.c
    beta = beta.decompose().value
    factor = math.sqrt((1+beta)/(1-beta))
    if self.wave == "air":
      self.x = air_to_vac(self.x) 
      self.x *= factor
      self.x = vac_to_air(self.x) 
    elif self.wave == "vac":
      self.x *= factor
    else:
      raise ValueError("self.wave should be in ['vac', 'air']")

  def scale_model(self, other, return_scaling_factor=False):
    """
    If self is model spectrum (errors are presumably zero), and S is a data
    spectrum (has errors) then this reproduces a scaled version of M2.
    There is no requirement for either to have the same wavelengths as
    interpolation is performed. However the resulting scaled model will
    have the wavelengths of the original model, not the data. If you want
    the model to share the same wavelengths, use model.interp_wave(),
    either before or after calling this function.
    """
    assert isinstance(other, Spectrum)
    assert self.x_unit == other.x_unit
    assert self.y_unit == other.y_unit

    #if M and S already have same x-axis, this won't do much.
    S = other[other.e>0]
    M = self.interp_wave(S)

    A_sm, A_mm = np.sum(S.y*M.y*S.ivar), np.sum(M.y**2*S.ivar)
    A = A_sm/A_mm

    if return_scaling_factor:
      return self*A, A
    else:
      return self*A
    
  def scale_model_to_model(self, other, return_scaling_factor=False):
    """
    Similar to scale_model, but for scaling one model to another. Essentially
    this is for the case when the argument doesn't have errors.
    """
    assert isinstance(other, Spectrum)
    assert self.x_unit == other.x_unit
    assert self.y_unit == other.y_unit

    #if M and S already have same x-axis, this won't do much.
    S = other
    M = self.interp_wave(S)

    A_sm, A_mm = np.sum(S.y*M.y), np.sum(M.y)
    A = A_sm/A_mm

    if return_scaling_factor:
      return self*A, A
    else:
      return self*A
    
  def convolve_gaussian(self, fwhm):
    S = self.copy()
    S.y = convolve_gaussian(S.x, S.y, fwhm)
    return S

  def convolve_gaussian_R(self, res):
    S = self.copy()
    S.y = convolve_gaussian_R(S.x, S.y, res)
    return S

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
    """
    assert isinstance(other, Spectrum)
    return join_spectra((self, other), sort=sort)

  def closest_wave(self, x0):
    """
    Returns the pixel index closest in wavelength to x0
    """
    return np.argmin(np.abs(self.x-x0))

  def plot(self, *args, errors=False, **kwargs):
    plt.plot(self.x, self.e if errors else self.y, *args, **kwargs)

#..............................................................................

def ZeroSpectrum(x, name="", wave='air', x_unit="AA", y_unit="erg/(s cm^2 AA)", head=None):
  y = np.zeros_like(x)
  e = np.zeros_like(x)
  return Spectrum(x, y, e, name=name, wave=wave, x_unit=x_unit, y_unit=y_unit, head=head)

def UnitSpectrum(x, name="", wave='air', x_unit="AA", head=None):
  y = np.ones_like(x)
  e = np.zeros_like(x)
  return Spectrum(x, y, e, name=name, wave=wave, x_unit=x_unit, y_unit="", head=head)

def join_spectra(SS, sort=False, name=None):
  """
  Joins a collection of spectra into a single spectrum. The name of the first
  spectrum is used as the new name. Can optionally sort the new spectrum by
  wavelengths.
  """
  S0 = SS[0]
  
  for S in SS:
    assert isinstance(S, Spectrum), 'item is not Spectrum'
    assert S.wave == S0.wave
    assert S.x_unit == S0.x_unit
    assert S.y_unit == S0.y_unit

  kwargs = {
    'name'   : S0.name,
    'wave'   : S0.wave,
    'x_unit' : S0.x_unit,
    'y_unit' : S0.y_unit,
  }


  x = np.hstack(S.x for S in SS)
  y = np.hstack(S.y for S in SS)
  e = np.hstack(S.e for S in SS)
  S = Spectrum(x, y, e, **kwargs)
  if sort:
    idx = np.argsort(x)
    return S[idx]
  else:
    return S

def spec_from_txt(fname, wave='air', x_unit='AA', y_unit='erg/(s cm2 AA)', **kwargs):
  """
  Loads a text file with the first 3 columns as wavelengths, fluxes, errors.
  """
  x, y, e = np.loadtxt(fname, unpack=True, usecols=(0,1,2), **kwargs)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(x, y, e, name, wave, x_unit, y_unit)
    
def model_from_txt(fname, wave='vac', x_unit='AA', y_unit='erg/(s cm2 AA)', **kwargs):
  """
  Loads a text file with the first 2 columns as wavelengths and fluxes.
  This produces a spectrum object where the errors are just set to zero.
  This is therefore good to use for models.
  """
  x, y = np.loadtxt(fname, unpack=True, usecols=(0,1), **kwargs)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(x, y, np.zeros_like(x), name, wave, x_unit, y_unit)

def model_from_dk(fname, x_unit='AA', y_unit='erg/(s cm2 AA)', **kwargs):
  """
  Similar to model_from_txt, but will autoskip past the DK header. Units are converted 
  """
  with open(fname, 'r') as F:
    skip = 1
    while True:
      line = F.readline()
      if line.startswith("END"):
        break
      else:
        skip += 1
  M = model_from_txt(fname, 'vac', 'AA', 'erg/(s cm3)', skiprows=skip)
  M.x_unit_to(x_unit)
  M.y_unit_to(y_unit)
  return M

def spec_from_npy(fname, wave='air', x_unit='AA', y_unit='erg/(s cm2 AA)'):
  """
  Loads a text file with the first 3 columns as wavelengths, fluxes, errors.
  """
  data = np.load(fname)
  assert data.ndim == 2, "Data must be 2D"

  if data.shape[0] == 2:
    x, y = data
    e = np.zeros_like(x)
  elif data.shape[0] == 3:
    x, y, e = data
  else:
    print("Data should have 2 or 3 columns")
    exit()
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(x, y, e, name, wave, x_unit, y_unit)

def spec_from_sdss_fits(fname, **kwargs):
  """
  Loads a SDSS fits file as spectrum (result in vac wavelengths)
  """
  hdulist = fits.open(fname)
  loglam, flux, ivar = [hdulist[1].data[key] for key in ('loglam', 'flux', 'ivar')]
  lam = 10**loglam
  ivar[ivar==0.] = 0.001
  err = 1/np.sqrt(ivar)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(lam, flux, err, name, 'vac')*1e-17

def spec_list_from_molly(fname):
  """
  Returns a list of spectra read in from a TRM molly file.
  """
  def convert_mol(molsp):
    x, y, e = molsp.wave, molsp.f, molsp.fe
    name = molsp.head['Object']
    S = Spectrum(x, y, e, name, y_unit="mJy")
    S.head = molsp.head
    return S
  
  return [convert_mol(molsp) for molsp in molly.gmolly(fname)]

def spectra_mean(SS):
  """
  Calculate the weighted mean spectrum of a list/tuple of spectra.
  All spectra should have identical wavelengths.
  """
  S0 = SS[0]
  for S in SS:
    assert isinstance(S, Spectrum)
    assert len(S) == len(S0)
    assert S.wave == S0.wave
    assert np.isclose(S.x, S0.x).all()
    assert S.x_unit == S0.x_unit
    assert S.y_unit == S0.y_unit

  X, Y, IV = np.array([S.x    for S in SS]), \
             np.array([S.y    for S in SS]), \
             np.array([S.ivar for S in SS])

  Xbar  = np.mean(X,axis=0)
  IVbar = np.sum(IV, axis=0)
  Ybar  = np.sum(Y*IV, axis=0) * IVbar
  Ebar  = 1.0 / np.sqrt(IVbar)

  kwargs = {
    'name'   : S0.name,
    'wave'   : S0.wave,
    'x_unit' : S0.x_unit,
    'y_unit' : S0.y_unit,
  }

  return Spectrum(Xbar, Ybar, Ebar, **kwargs)
    
###############################################################################

def voigt( x, x0, fwhm_g, fwhm_l ):
  sigma = voigt.Va*fwhm_g
  z = ((x-x0) + 0.5j*fwhm_l)/(sigma*voigt.Vb)
  return wofz(z).real/(sigma*voigt.Vc)
voigt.Va = 1/(2*np.sqrt(2*np.log(2)))
voigt.Vb = np.sqrt(2)
voigt.Vc = np.sqrt(2*np.pi)

def load_transmission_curve(filt):
  """
  Loads the filter curves obtained from VOSA (SVO).
  """
  long_path = "/home/astro/phujdu/Python/MH/mh/spectra/filt_profiles/"
  if   filt in 'ugriz':
    end_path = f"SLOAN_SDSS.{filt}.dat"
  elif filt in 'UBVRI':
    end_path = f"Generic_Johnson.{filt}.dat"
  elif filt in ['Gaia'+b for b in 'G Bp Rp'.split()]:
    fdict = {"Gaia"+k:v for k,v in zip(("G","Bp","Rp"), ("","bp","rp"))}
    end_path = f"GAIA_GAIA2r.G{fdict[filt]}.dat"
  elif filt in ['GalexFUV','GalexNUV']:
    fdict = {"Galex"+k:k for k in ("NUV","FUV")}
    end_path = f"GALEX_GALEX.{fdict[filt]}.dat"
  elif filt == 'DenisI':
    end_path = "DENIS_DENIS.I.dat"
  elif filt in ['2m'+b for b in 'JHK']:
    end_path = f"2MASS_2MASS.{filt[-1]}.dat"
  elif filt in ['UK'+b for b in 'YJHK']:
    end_path = f"UKIRT_UKIDSS.{filt[-1]}.dat"
  elif filt in ['W'+b for b in '12']:
    end_path = f"WISE_WISE.{filt}.dat"
  elif filt in ['S'+b for b in '12']:
    end_path = f"Spitzer_IRAC.I{filt[1]}.dat"
  elif filt in ['sm'+b for b in 'uvgriz']:
    end_path = f"SkyMapper_SkyMapper.{filt[2]}.dat"
  elif filt in ['ps'+b for b in 'grizy']:
    end_path = f"PAN-STARRS_PS1.{filt[2]}.dat"
  elif filt in ['sw'+b for b in 'U UVW1 UVW2 UVM1'.split()]:
    end_path = f"Swift_UVOT.{filt[2:]}.dat"
  else:
    raise ValueError('Invalid filter name: {}'.format(filt))
  return model_from_txt(long_path+end_path, x_unit="AA", y_unit="")
#

def mag_calc_AB(S, filt, NMONTE=1000, Ifun=Itrapz):
  """
  Calculates the synthetic AB magnitude of a spectrum for a given filter.
  If NMONTE is > 0, monte-carlo error propagation is performed outputting
  both a synthetic-mag and error. For model-spectra, i.e. no errors,
  use e=np.ones_like(f) and NMONTE=0. List of currently supported filters:

  2Mass:     ['2mJ','2mH','2mK']

  Denis:     ['DenisI']

  Gaia:      ['GaiaG', 'GaiaBp', GaiaRp']

  Galex:     ['GalexFUV' 'GalexNUV']

  Johnson:   ['U','B','V','R','I']

  PanSTARRS: ['ps(grizy)']

  SDSS:      ['u','g','r','i','z']

  Spitzer:   ['S1','S2']

  Skymapper: ['sm(uvgriz)']

  Swift:     ['sw(U,UVW1,UVW2,UVM1)']

  WISE:      ['W1','W2']
  """

  #load filter
  R = load_transmission_curve(filt)
  R.wave = S.wave

  #Convert Spectra/filter-curve to Hz/Jy for integrals
  R.x_unit_to("Hz")
  S.x_unit_to("Hz")
  S.y_unit_to("Jy")

  #clip data to filter range and interpolate filter to data axis
  S = S.clip(np.min(R.x), np.max(R.x))
  R = R.interp_wave(S)

  #Calculate AB magnitudes, potentially including flux errors
  def m_AB_int(X, Y, R):
    y_nu = Ifun(Y*R/X, X)/Ifun(R/X, X) 
    m = -2.5 * np.log10(y_nu) + 8.90
    return m

  if NMONTE == 0:
    return m_AB_int(S.x, S.y, R.y)
  else:
    y_mc = lambda S: np.random.normal(S.y, S.e)
    m = np.array([m_AB_int(S.x, y_mc(S), R.y) for i in range(NMONTE)])
    return np.mean(m), np.std(m)
#

def vac_to_air(Wvac):
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

def convolve_gaussian(x, y, FWHM):
  """
  Convolve spectrum with a Gaussian with FWHM by oversampling and
  using an FFT approach. Wavelengths are assumed to be sorted,
  but uniform spacing is not required. Will cause wrap-around at
  the end of the spectrum.
  """
  sigma = FWHM/2.355

  def next_pow_2(N_in):
    N_out = 1
    while N_out < N_in:
      N_out *= 2
    return N_out

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

def black_body(x, T, norm=True):
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

def Black_body(x, T, wave='air', x_unit="AA", y_unit="erg/(s cm2 AA)", norm=True):
  """
  Returns a Black body curve like black_body(), but the return value
  is a Spectrum class.
  """
  zero_flux = np.zeros_like(x)
  M = Spectrum(x, zero_flux, zero_flux, f'{T}K BlackBody', wave, x_unit, y_unit)
  M.x_unit_to("AA")
  M.y_unit_to("erg/(s cm2 AA)")
  if wave=='air':
    M.air_to_vac()
  M.y = black_body(M.x, T, False)
  if wave=='air':
    M.vac_to_air()
  M.x_unit_to(x_unit)
  M.y_unit_to(y_unit)
  if norm:
    M /= M.y.max()
  return M
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

def keep_points(x, fname):
  """
  creates a mask for a spectrum that regions between pairs from a file
  """
  try:
    lines = open(fname,'r').readlines()
  except IOError:
    print("file %s does not exist" %fname)
    exit()
  between = lambda x, x1, x2: (x>float(x1))&(x<float(x2))
  segments = (between(x, *line.split()) for line in lines)
  return reduce(operator.or_, segments)

def A_curve(x, R=3.1):
  """
  Calculate CCM 1989 extinction curve. x is in units of 1/um.
  """
  def Av_IR(x):
    """
    0.3 <= x/um < 1.1
    """
    a = 0.574 * x**1.61
    b =-0.527 * x**1.61
    return a, b

  def Av_opt(x):
    """
    1.1 <= x/um < 3.3
    """
    y = x-1.82
    poly_a = [1, +0.17699, -0.50447, -0.02427, +0.72085, +0.01979, -0.77530, +0.32999][::-1]
    poly_b = [0, +1.41338, +2.28305, +1.07233, -5.38434, -0.62251, +5.30260, -2.09002][::-1]
    a = np.polyval(poly_a, y)
    b = np.polyval(poly_b, y)
    return a, b

  def Av_UV(x):
    """
    3.3 <= x/um < 8.0
    """
    poly_Fa = [0, 0, -0.04473, -0.009779][::-1]
    poly_Fb = [0, 0, +0.21300, +0.120700][::-1]
    Fa = np.polyval(poly_Fa, x-5.9)
    Fb = np.polyval(poly_Fb, x-5.9)
    if isinstance(x, np.ndarray):
      Fa[x < 5.9] = 0
      Fb[x < 5.9] = 0
    elif isinstance(x, (int, float)):
      if x < 5.9:
        Fa = Fb = 0
    else:
      raise TypeError
    a =  1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) + Fa
    b = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + Fb
    return a, b

  FIR = (x < 0.3)
  IR  = (x >= 0.3) & (x<1.1)
  opt = (x >= 1.1) & (x<3.3)
  UV  = (x >= 3.3) & (x<8.0)
  FUV = (x >= 8.0)

  a = np.zeros_like(x)
  b = np.zeros_like(x)
  a[IR ], b[IR ] = Av_IR( x[IR ])
  a[opt], b[opt] = Av_opt(x[opt])
  a[UV ], b[UV ] = Av_UV( x[UV ])
  a[FIR], b[FIR] = Av_IR(0.3)
  a[FUV], b[FUV] = Av_UV(8.0)
  A = a + b/R
  return A

def Lanczos(x, y, xnew):
  n = np.arange(len(x))
  Ni = interp1d(x, n, kind='linear', fill_value='extrapolate')(xnew)
  ynew = [np.sum(y*np.sinc(ni-n)) for ni in Ni]
  return np.array(ynew)
