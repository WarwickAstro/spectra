"""
Contains the Spectrum class for working with astrophysical spectra.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d, Akima1DInterpolator as Ak_i
from .synphot import mag_calc_AB
from .reddening import A_curve
from .misc import *

__all__ = [
  "Spectrum",
]

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
    Returns non-array attributes (in same order as __init__). This can be
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
      y2 = lanczos(self.x, self.y, x2)
      e2 = lanczos(self.x, self.e, x2)
      extrap = (x2<self.x.min()) | (x2>self.x.max())
      y2[extrap] = 0.
      e2[extrap] = np.inf
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

  def scale_to_AB_mag(self, filt, mag):
    mag0 = self.mag_calc_AB(filt, NMONTE=0)
    return self * 10**(0.4*(mag0-mag))
    
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

