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
  __slots__ = ['_x', '_y', '_e', '_name', '_wave', '_xu', '_yu', '_head']
  def __init__(self, x, y, e, name="", wave='air', x_unit="AA", y_unit="erg/(s cm^2 AA)", head=None):
    """
    Initialise spectrum. Arbitrary header items can be added to self.head
    x must be an ndarray. y and e can either by int/floats or ndarrays of
    the same length.
    """

    self.x = x
    self.y = y
    self.e = e
    self.name = name
    self.wave = wave
    self.x_unit = x_unit
    self.y_unit = y_unit
    self.head = head

  @property
  def x(self):
    return self._x

  @x.setter
  def x(self, x):
    if isinstance(x, np.ndarray):
      if x.ndim == 1:
        self._x = x.astype(float)
      else:
        raise ValueError
    else:
      raise TypeError

  @property
  def y(self):
    return self._y

  @y.setter
  def y(self, y):
    if isinstance(y, (int, float, complex)):
      self._y = y*np.ones_like(self.x)
    elif isinstance(y, np.ndarray):
      assert y.shape == self.x.shape
      self._y = y
    else:
      raise TypeError

  @property
  def e(self):
    return self._e
  
  @e.setter
  def e(self, e):
    if isinstance(e, (int, float)):
      assert e >= 0
      self._e = e*np.ones_like(self.x) 
    elif isinstance(e, np.ndarray):
      assert e.shape == self.x.shape
      assert np.all(e >= 0)
      self._e = e
    else:
      raise TypeError

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    if isinstance(name, str):
      self._name = name
    else:
      raise TypeError

  @property
  def wave(self):
    return self._wave

  @wave.setter
  def wave(self, wave):
    if wave in ('vac', 'air'):
      self._wave = wave
    else:
      raise ValueError

  @property
  def x_unit(self):
    return self._xu.to_string()

  @x_unit.setter
  def x_unit(self, x_unit):
    if isinstance(x_unit, (str, u.Unit)):
      self._xu = u.Unit(x_unit)
    else:
      raise TypeError

  @property
  def y_unit(self):
    return self._yu.to_string()

  @y_unit.setter
  def y_unit(self, y_unit):
    if isinstance(y_unit, (str, u.Unit)):
      self._yu = u.Unit(y_unit)
    else:
      raise TypeError

  @property
  def head(self):
    return self._head

  @head.setter
  def head(self, head):
    if head is None:
      self._head = {}
    else:
      if isinstance(head, dict):
        self._head = head
      else:
        raise ValueError

  @property
  def var(self):
    """
    Variance property attribute from flux errors
    """
    return self.e**2

  @var.setter
  def var(self, value):
    """
    Variance property attribute from flux errors
    """
    self.e = np.sqrt(value)

  @property
  def ivar(self):
    """
    Inverse variance attribute from flux errors
    """
    return 1.0/self.var

  @ivar.setter
  def ivar(self, value):
    """
    Variance property attribute from flux errors
    """
    self.e = 1/np.sqrt(value)

  @property
  def SN(self):
    """
    Signal to noise ratio
    """
    return np.abs(self.y/self.e)

  @property
  def magAB(self):
    """
    Returns fluxes in terms of AB magnitudes
    """
    S = self.copy()
    S.y_unit_to("Jy")
    return -2.5*np.log10(S.y/3631)

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
      x2 = self.x.copy()
      y2 = self.y + other
      e2 = self.e.copy()
    elif isinstance(other, Spectrum):
      assert self.x_unit == other.x_unit
      assert self.y_unit == other.y_unit
      assert np.all(np.isclose(self.x, other.x))
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
      x2 = self.x.copy()
      y2 = self.y - other
      e2 = self.e.copy()
    elif isinstance(other, Spectrum):
      assert self.x_unit == other.x_unit
      assert self.y_unit == other.y_unit
      assert np.all(np.isclose(self.x, other.x))
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
      x2 = self.x.copy()
      y2 = self.y * other
      e2 = self.e * np.abs(other)
      yu2 = self._yu
    elif isinstance(other, Spectrum):
      assert self.x_unit == other.x_unit
      assert np.all(np.isclose(self.x, other.x))
      x2 = 0.5*(self.x+other.x)
      y2 = self.y*other.y
      e2 = np.abs(y2)*np.hypot(self.e/self.y, other.e/other.y)
      yu2 = self._yu * other._yu
    else:
      raise TypeError
    S = Spectrum(x2, y2, e2, *self.info)
    S._yu = yu2
    return S

  def __truediv__(self, other):
    """
    Return self / other (with standard error propagation)
    """
    if isinstance(other, (int, float, np.ndarray)):
      x2 = self.x.copy()
      y2 = self.y / other
      e2 = self.e / np.abs(other)
      yu2 = self._yu
    elif isinstance(other, Spectrum):
      assert self.x_unit == other.x_unit
      assert np.all(np.isclose(self.x, other.x))
      x2 = 0.5*(self.x+other.x)
      y2 = self.y/other.y
      e2 = np.abs(y2)*np.hypot(self.e/self.y, other.e/other.y)
      yu2 = self._yu / other._yu
    else:
      raise TypeError
    S = Spectrum(x2, y2, e2, *self.info)
    S._yu = yu2
    return S

  def __pow__(self,other):
    """
    Return S**other (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      x2 = self.x.copy()
      y2 = self.y**other
      e2 = other * y2 * self.e/self.y
      yu2 = self._yu**other
      S = Spectrum(x2, y2, e2, *self.info)
      S.y_unit = yu2
      return S
    else:
      raise TypeError

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
      x2 = self.x.copy()
      y2 = other / self.y
      e2 = other * self.e /(self.y*self.y)
      yu2 = 1/self._yu
      S = Spectrum(x2, y2, e2, *self.info)
      S.y_unit = yu2
      return S
    else:
      raise TypeError

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

  def remove_mask(self):
    """
    Remove mask from spectral fluxes
    """
    self.x = np.array(self.x)
    self.y = np.array(self.y)
    self.e = np.array(self.e)

  def mag_calc_AB(self, filt, NMONTE=1000):
    """
    Calculates the AB magnitude of a filter called 'filt'. Errors
    are calculated in Monte-Carlo fashion, and assume all fluxes
    are statistically independent (not that realistic). See the
    definition of 'mag_calc_AB' for valid filter names.
    """
    S = self.copy()
    S.x_unit_to("AA")
    S.y_unit_to("erg/(s cm2 AA)")

    if np.all(self.e == 0):
      NMONTE = 0 
    return mag_calc_AB(S, filt, NMONTE)

  def interp(self, X, kind='linear', **kwargs):
    """
    Interpolates a spectrum onto the wavlength axis X, if X is a numpy array,
    or X.x if X is Spectrum type. This returns a new spectrum rather than
    updating a spectrum in place, however this can be acheived by

    >>> S1 = S1.interp(X)

    Wavelengths outside the range of the original spectrum are filled with
    zeroes. By default the interpolation is nearest neighbour.
    """
    if isinstance(X, np.ndarray):
      x2 = 1*X
    elif isinstance(X, Spectrum):
      assert self.x_unit == X.x_unit
      assert self.y_unit == X.y_unit
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
      e2 = lanczos(self.x, np.log(self.e+1E-300), x2)
      extrap = (x2<self.x.min()) | (x2>self.x.max())
      y2[extrap] = 0.
      e2[extrap] = np.inf
    else:
      extrap_y, extrap_e = (self.y[0],self.y[-1]), (self.e[0],self.e[-1])
      y2 = interp1d(self.x, self.y, kind=kind, \
        bounds_error=False, fill_value=0., **kwargs)(x2)
      e2 = interp1d(self.x, self.e, kind=kind, \
        bounds_error=False, fill_value=np.inf, **kwargs)(x2)

    e2[e2 < 0] = 0.
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
      data = [*self.data] if errors else [self.x, self.y]
      np.save(fname, np.array(data))
    else:
      print("Unrecognised File type")
      print("Save aborted")

  def air_to_vac(self):
    """
    Changes air wavelengths to vaccuum wavelengths in place
    """
    assert self.x_unit == "AA"
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
    assert self.x_unit == "AA"
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
    x = self.x * self._xu
    x2 = x.to(new_unit, u.spectral())
    self.x = x2.value
    self.x_unit = new_unit
    
  def y_unit_to(self, new_unit):
    """
    Changes units of the y-data. Supports conversion between Fnu
    and Flambda etc. Argument should be a string.
    """
    x = self.x * self._xu
    y = self.y * self._yu
    e = self.e * self._yu
    y = y.to(new_unit, u.spectral_density(x))
    e = e.to(new_unit, u.spectral_density(x))
    self.y = y.value
    self.e = e.value
    self.y_unit = new_unit
    
  def apply_redshift(self, v, v_unit="km/s"):
    """
    Applies redshift of v km/s to spectrum for "air" or "vac" wavelengths
    """
    v *= u.Unit(v_unit)
    assert v.si.unit == const.c.unit
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
    the model to share the same wavelengths, use model.interp(),
    either before or after calling this function.
    """
    assert isinstance(other, Spectrum)
    assert self.x_unit == other.x_unit
    assert self.y_unit == other.y_unit

    #if M and S already have same x-axis, this won't do much.
    S = other[other.e>0]
    M = self.interp(S)

    A = np.sum(S.y*M.y*S.ivar)/np.sum(M.y**2*S.ivar)

    return (self*A, A) if return_scaling_factor else self*A
    
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
    M = self.interp(S)

    A = np.sum(S.y*M.y)/np.sum(M.y)

    return (self*A, A) if return_scaling_factor else self*A

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
    assert self.x_unit == other.x_unit
    assert self.y_unit == other.y_unit
    assert self.wave == other.wave
    return join_spectra((self, other), sort=sort)

  def closest_wave(self, x0):
    """
    Returns the pixel index closest in wavelength to x0
    """
    return np.argmin(np.abs(self.x-x0))

  def plot(self, *args, errors=False, **kwargs):
    plt.plot(self.x, self.e if errors else self.y, *args, **kwargs)

