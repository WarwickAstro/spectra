"""
Contains the Spectrum class for working with astrophysical spectra.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import astropy.units as u
import astropy.constants as const
from astropy.convolution import convolve
from scipy.interpolate import interp1d, Akima1DInterpolator as Ak_i
from scipy.optimize import minimize
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
  >>> S2 = Spectrum(x2, y2, e2)
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
  Note that for operations of the form
  >>> S + a ;  S - a ;  S * a ;  S / a
  >>> a + S ;  a - S ;  a * S ;  a / S

  both rows work as expected if 'a' is an int/float. However if 'a' is an
  ndarray or Quantity object, the second row (__radd__ etc) is overridden
  by undefined behaviour of numpy/astropy implementations.
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
        raise ValueError("x arrays must be 1D")
    else:
      raise TypeError("x must be an ndarray")

  @property
  def y(self):
    return self._y

  @y.setter
  def y(self, y):
    if isinstance(y, (int, float)):
      self._y = y*np.ones_like(self.x)
    elif isinstance(y, np.ndarray):
      if(y.shape != self.x.shape):
        raise ValueError("for ndarrays, y must be the same shape as x")
      self._y = y.astype(float)
    else:
      raise TypeError("y must be of type int/float/ndarray")

  @property
  def e(self):
    return self._e
  
  @e.setter
  def e(self, e):
    if isinstance(e, (int, float)):
      if e < 0:
        raise ValueError("Uncertainties cannot be negative")
      self._e = e*np.ones_like(self.x) 
    elif isinstance(e, np.ndarray):
      if e.shape != self.x.shape:
        raise ValueError("for ndarrays, e must be the same shape as x")
      if np.any(e < 0):
        raise ValueError("Uncertainties cannot be negative")
      self._e = e.astype(float)
    else:
      raise TypeError("y must be of type int/float/ndarray")

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    if isinstance(name, str):
      self._name = name
    else:
      raise TypeError("name must be a string")

  @property
  def wave(self):
    return self._wave

  @wave.setter
  def wave(self, wave):
    if wave in ('vac', 'air'):
      self._wave = wave
    else:
      raise ValueError("wave must be 'vac' or 'air'")

  @property
  def x_unit(self):
    return self._xu.to_string()

  @x_unit.setter
  def x_unit(self, x_unit):
    if isinstance(x_unit, (str, u.UnitBase)):
      self._xu = u.Unit(x_unit)
    else:
      raise TypeError("x_unit must be str or Unit type")

  @property
  def y_unit(self):
    return self._yu.to_string()

  @y_unit.setter
  def y_unit(self, y_unit):
    if isinstance(y_unit, (str, u.UnitBase)):
      self._yu = u.Unit(y_unit)
    else:
      raise TypeError("y_unit must be str or Unit type")

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
        raise ValueError("head must be a dictionary")

  @property
  def var(self):
    """
    Variance attribute from flux errors
    """
    return self.e**2

  @var.setter
  def var(self, value):
    self.e = np.sqrt(value)

  @property
  def ivar(self):
    """
    Inverse variance attribute from flux errors
    """
    return 1.0/self.var

  @ivar.setter
  def ivar(self, value):
    self.var = 1.0/value

  @property
  def SN(self):
    """
    Signal to noise ratio
    """
    return np.abs(self.y/self.e)

  @SN.setter
  def SN(self, value):
    """
    Set errors for desired sn
    """
    self.e = np.abs(self.y/value)

  @property
  def magAB(self):
    """
    Returns fluxes in terms of AB magnitudes
    """
    S = self.copy()
    S.y_unit_to("Jy")
    return -2.5*np.log10(S.y/3631)

  @property
  def magABe(self):
    """
    Returns fluxe errors in terms of AB magnitudes
    """
    S = self.copy()
    S.y_unit_to("Jy")
    return -2.5*np.log10(S.e/3631)

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
    Returns non-array attributes as a dictionary. This can be
    used to create new spectra with the same information, e.g.
    >>> Spectrum(x, y, e, **S.info)
    """
    kwargs = {
      'name'   : self.name,
      'wave'   : self.wave,
      'x_unit' : self.x_unit,
      'y_unit' : self.y_unit,
      'head'   : self.head,
    }
    return kwargs

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
      f"Spectrum class with {len(self)} pixels",
      f"name: {self.name}",
      f"x-unit: {self.x_unit}",
      f"y-unit: {self.y_unit}",
      f"wavelengths: {self.wave}",
      f"header: {self.head}",
    ])

    return ret

  def __getitem__(self, key):
    """
    Return self[key]
    """
    if isinstance(key, (int, slice, np.ndarray)):
      data_key = self.x[key], self.y[key], self.e[key]
      return data_key if isinstance(key, int) else Spectrum(*data_key, **self.info)
    else:
      raise TypeError("spectra must be indexed with int/slice/ndarray types")

  def __iter__(self):
    """
    Return iterator of spectrum
    """
    return zip(*self.data)

  def __contains__(self, value):
    """
    Return whether value is in the x-range of self
    """
    return self.x.min() < value < self.x.max()

  def promote_to_spectrum(self, other, dimensionless_y=False):
    """
    Promote non-Spectrum objects (int/float/ndarray/quantity) to a Spectrum
    with similar properties to self. This is used internally to simplify
    arithmetic implementation, but also necessary for reverse arithmetic
    operations using ndarrays and quantities, e.g. 1 / Spectrum.
    """
    info = self.info
    if isinstance(other, u.Quantity):
      ynew = other.value
      info['y_unit'] = other.unit
    elif isinstance(other, (int, float, np.ndarray)):
      ynew = other
      if dimensionless_y:
        info['y_unit'] = u.dimensionless_unscaled
    else:
      raise NotImplementedError("Cannot cast object to Spectrum")
    return Spectrum(self.x, ynew, 0, **info)

  def __add__(self, other):
    """
    Return self + other (with standard error propagation)
    """
    if isinstance(other, Spectrum):
      self._compare_units(other, 'xy')
      self._compare_x(other)
      ynew = self.y + other.y
      enew = np.hypot(self.e, other.e)
      return Spectrum(self.x, ynew, enew, **self.info)
    else:
      Sother = self.promote_to_spectrum(other)
      return self + Sother

  def __sub__(self, other):
    """
    Return self - other (with standard error propagation)
    """
    if isinstance(other, Spectrum):
      self._compare_units(other, 'xy')
      self._compare_x(other)
      ynew = self.y - other.y
      enew = np.hypot(self.e, other.e)
      return Spectrum(self.x, ynew, enew, **self.info)
    else:
      Sother = self.promote_to_spectrum(other)
      return self - Sother
      
  def __mul__(self, other):
    """
    Return self * other (with standard error propagation)
    """
    if isinstance(other, Spectrum):
      self._compare_units(other, 'x')
      self._compare_x(other)
      infonew = self.info
      infonew['y_unit'] = self._yu * other._yu
      ynew = self.y * other.y
      enew = np.abs(ynew)*np.hypot(self.e/self.y, other.e/other.y)
      return Spectrum(self.x, ynew, enew, **infonew)
    else:
      Sother = self.promote_to_spectrum(other, True)
      return self * Sother
      
  def __truediv__(self, other):
    """
    Return self / other (with standard error propagation)
    """
    if isinstance(other, Spectrum):
      self._compare_units(other, 'x')
      self._compare_x(other)
      infonew = self.info
      infonew['y_unit'] = self._yu / other._yu
      ynew = self.y / other.y
      enew = np.abs(ynew)*np.hypot(self.e/self.y, other.e/other.y)
      return Spectrum(self.x, ynew, enew, **infonew)
    else:
      Sother = self.promote_to_spectrum(other, True)
      return self / Sother

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
    Sother = self.promote_to_spectrum(other, True)
    return Sother / self

  def __pow__(self,other):
    """
    Return S**other (with standard error propagation)
    """
    if isinstance(other, (int, float)):
      infonew = self.info
      infonew['y_unit'] = self._yu**other
      ynew = self.y**other
      enew = np.abs(other * ynew * self.e/self.y)
      return Spectrum(self.x, ynew, enew, **infonew)
    else:
      raise TypeError("other must be int/float")

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

  def _compare_units(self, other, xy):
    """
    Check units match another spectrum or kind of unit
    """
    if isinstance(other, (str, u.UnitBase)):
      #check specific unit
      if xy == 'x':
        if self.x_unit != u.Unit(other):
          raise u.UnitsError("x_units differ")
      elif xy == 'y':
        if self.y_unit != u.Unit(other):
          raise u.UnitsError("y_units differ")
      else:
        raise ValueError("xy not 'x' or 'y'")
    elif isinstance(other, u.Quantity):
      _compare_units(self, other.unit, xy)
    elif isinstance(other, Spectrum):
      #compare two spectra
      if xy == 'x':
        if self.x_unit != other.x_unit:
          raise u.UnitsError("x_units differ")
      elif xy == 'y':
        if self.y_unit != other.y_unit:
          raise u.UnitsError("y_units differ")
      elif xy == 'xy':
        if self.x_unit != other.x_unit:
          raise u.UnitsError("x_units differ")
        if self.y_unit != other.y_unit:
          raise u.UnitsError("y_units differ")
      else:
        raise ValueError("xy not 'x', 'y', or 'xy'")
    else:
      raise TypeError("other was not Spectrum or interpretable as a unit")

  def _compare_x(self, other):
    if self.wave != other.wave:
      raise ValueError("Spectra must have same wavelengths (air/vac)")
    if not np.allclose(self.x, other.x):
      raise ValueError("Spectra must have same x values")

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

  def interp(self, X, kind='cubic', **kwargs):
    """
    Interpolates a spectrum onto the wavlength axis X, if X is a numpy array,
    or X.x if X is Spectrum type. This returns a new spectrum rather than
    updating a spectrum in place, however this can be acheived by

    >>> S1 = S1.interp(X)

    Wavelengths outside the range of the original spectrum are filled with
    zeroes.
    """
    if isinstance(X, np.ndarray):
      x2 = 1*X
    elif isinstance(X, Spectrum):
      self._compare_units(X, 'x')
      if self.wave != X.wave:
        raise ValueError("wavelengths differ between spectra")
      x2 = 1*X.x
    else:
      raise TypeError("interpolant was not ndarray/Spectrum type")

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
    return Spectrum(x2, y2, e2, **self.info)

  def copy(self):
    """
    Returns a copy of self
    """
    return Spectrum(*self.data, **self.info)

  def sect(self, x0, x1):
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
      raise ValueError("file name must be of type .txt/.dat/.npy")

  def air_to_vac(self):
    """
    Changes air wavelengths to vaccuum wavelengths in place
    """
    self._compare_units("AA", 'x')
    if self.wave == 'air':
      self.x = air_to_vac(self.x) 
      self.wave = 'vac'

  def vac_to_air(self):
    """
    Changes vaccuum wavelengths to air wavelengths in place
    """
    self._compare_units("AA", 'x')
    if self.wave == 'vac':
      self.x = vac_to_air(self.x) 
      self.wave = 'air'

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
    and energy etc. Argument should be a string or Unit.
    """
    x = self.x * self._xu
    x2 = x.to(new_unit, u.spectral())
    self.x = x2.value
    self.x_unit = new_unit
    
  def y_unit_to(self, new_unit):
    """
    Changes units of the y-data. Supports conversion between Fnu
    and Flambda etc. Argument should be a string or Unit.
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
    if v.si.unit != const.c.unit:
      raise u.UnitsError("v must have velocity units")
    beta = v/const.c
    beta = beta.decompose().value
    factor = math.sqrt((1+beta)/(1-beta))
    if self.wave == "air":
      self.x = air_to_vac(self.x) 
      self.x *= factor
      self.x = vac_to_air(self.x) 
    else:
      self.x *= factor

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
    if not isinstance(other, Spectrum):
      raise TypeError
    self._compare_units(other, 'xy')

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
    if not isinstance(other, Spectrum):
      raise TypeError
    self._compare_units(other, 'xy')

    #if M and S already have same x-axis, this won't do much.
    S = other
    M = self.interp(S)

    A = np.sum(S.y*M.y)/np.sum(M.y**2)

    return (self*A, A) if return_scaling_factor else self*A

  def scale_spectrum_to_spectrum(self, other, return_scaling_factor=False):
    """
    Scales self to best fit other in their mutually overlapping region.
    """
    if not isinstance(other, Spectrum):
      raise TypeError
    self._compare_units(other, 'xy')

    x0 = max(S.x.min() for S in (self, other))
    x1 = min(S.x.max() for S in (self, other))
    Soc = other.clip(x0, x1)
    Ssi = self.interp(Soc, kind='cubic')

    def chi2(A, S1, S2):
      top = (S1.y - A*S2.y)**2
      bot = (S1.e**2 + (A*S2.e)**2)
      return np.sum(top / bot)

    res = minimize(chi2, (1.0), args=(Soc, Ssi))  
    A = float(res['x'][0])

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

  def rot_broaden(self, vsini, dv=1.0):
    """
    Apply rotational broadening in km/s. The dv parameter sets the resolution
    that convolution is performed at.
    """
    xu, yu = self.x_unit, self.y_unit
    S = self.copy()
    S.x_unit_to(u.AA)
    S.y_unit_to("erg/(s cm2)")
    logx = np.log(S.x)
    logx = np.arange(logx[0], logx[-1], dv/3e5)
    xnew = np.exp(logx) #0.1km/s resolution
    S = S.interp(xnew, kind='cubic')
    kxR = np.arange(0, vsini, dv)
    kxL = -kxR[:0:-1]
    kx = np.hstack([kxL,kxR])
    ky = np.sqrt(1-(kx/vsini)**2)
    S.y = convolve(S.y, ky)
    S = S.interp(self, kind='cubic')
    S.x_unit_to(xu)
    S.y_unit_to(yu)
    return S

  def polyfit(self, deg, weighted=True, logx=False, logy=False):
    """
    Fits a polynomial to a spectrum object.
    """
    x = np.log(self.x) if logx else self.x
    y = np.log(np.abs(self.y)) if logy else self.y
    e = np.abs(self.e/self.y) if logy else self.e
    poly = np.polyfit(x, y, deg, w=1/e) if weighted else np.polyfit(x, y, deg)
    return poly, logx, logy, self.y_unit

  def polyval(self, polyres):
    """
    Generates a spectrum from polynomial coefficients with the same shape/units as self.
    polyres should be: poly, logx, logy, y_unit
    """
    poly, logx, logy, y_unit = polyres
    x = np.log(self.x) if logx else self.x
    y = np.polyval(poly, x)
    y = np.exp(y) if logy else y
    infonew = self.info
    infonew['y_unit'] = y_unit
    return Spectrum(self.x, y, 0, **infonew)

  def split(self, W):
    """
    If W is an int/float, splits spectrum in two around W. If W is an
    interable of ints/floats, this will split into mutliple chunks instead.
    """
    if isinstance(W, (int, float)):
      W = -np.inf, W, np.inf
    elif isinstance(W, (list, tuple, np.ndarray)):
      if not all([isinstance(w, (int, float)) for w in W]):
        raise TypeError("w must all be of type int/float")
      W = -np.inf, *sorted(W), np.inf
    else:
      raise TypeError("W must be int/float or iterable of those types")
    return tuple(self.clip(*pair) for pair in zip(W[:-1], W[1:]))

  def join(self, other, sort=False):
    """
    Joins a second spectrum to the current spectrum. Can potentially be used
    rescursively, i.e.
    >>> S = S1.join(S2).join(S3)
    """
    if not isinstance(other, Spectrum):
      raise TypeError("can only join Spectrum type to other spectra")
    self._compare_units(other, 'xy')
    if self.wave != other.wave:
      raise ValueError("cannot join spectra with different wavelengths")
    return join_spectra((self, other), sort=sort)

  def closest_x(self, x0):
    """
    Returns the pixel index closest in wavelength to x0
    """
    return np.argmin(np.abs(self.x-x0))

  def isnan(self):
    """
    Returns truth-array showing pixels with nans (either x, y, or e)
    """
    return np.isnan(self.x) | np.isnan(self.y) | np.isnan(self.e)

  def isinf(self):
    """
    Returns truth-array showing pixels with infs (either x, y, or e)
    """
    return np.isinf(self.x) | np.isinf(self.y) | np.isinf(self.e)

  def plot(self, *args, kind='y', **kwargs):
    """
    Plots the spectrum with matplotlib and passes *args/**kwargs.
    'kind' should be one of 'y', 'e', 'var', 'ivar', 'SN', 'magAB', 'magABe'.
    plt.show() and other mpl functions still need to be used separately.
    """
    allowed = "y e var ivar SN magAB magABe"
    if kind not in allowed.split(): 
      raise ValueError(f"kind must be one of: {allowed}")

    y_plot = getattr(self, kind)
    plt.plot(self.x, y_plot, *args, **kwargs)

    #default y limits (if not already set)
    ax = plt.gca()
    if ax.get_autoscaley_on():
      ylo, yhi = ax.get_ylim()
      if kind.startswith('magAB'):
        if ylo < yhi:
          plt.ylim(yhi, ylo)
      else:
        plt.ylim(0, yhi)
      ax.set_autoscaley_on(True)
