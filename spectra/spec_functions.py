"""
Contains functions for generating spectra or operating on spectra
"""
import numpy as np
from scipy.optimize import leastsq
from .spec_class import Spectrum
from .misc import black_body

__all__ = [
  "Black_body",
  "join_spectra",
  "spectra_mean",
]

def Black_body(x, T, wave='air', x_unit="AA", y_unit="erg/(s cm2 AA)", norm=True):
  """
  Returns a Black body curve like black_body(), but the return value
  is a Spectrum class.
  """
  BB = Spectrum(x, 0., 0., f'{T}K BlackBody', wave, x_unit, "erg/(s cm2 AA)")
  BB.x_unit_to("AA")
  BB += black_body(BB.x, T, False)
  BB.x_unit_to(x_unit)
  BB.y_unit_to(y_unit)
  if norm:
    BB /= BB.y.max()
  return BB
#

#..............................................................................

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

  x = np.hstack(S.x for S in SS)
  y = np.hstack(S.y for S in SS)
  e = np.hstack(S.e for S in SS)
  S = Spectrum(x, y, e, *S0.info)
  
  if name is not None:
    S.name = name
  if sort:
    S = S[np.argsort(x)]

  return S

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
  Ybar  = np.sum(Y*IV, axis=0) / IVbar
  Ebar  = 1.0 / np.sqrt(IVbar)

  return Spectrum(Xbar, Ybar, Ebar, *S0.info)

def sky_line_fwhm(S, x0, dx=5.):
  """
  Given a sky spectrum, this fits a Gaussian to a
  sky line and returns the FWHM.
  """
  def sky_residual(params, S):
    A, x0, fwhm, C = params
    xw = fwhm /2.355
    y_fit = A*np.exp(-0.5*((S.x-x0)/xw)**2) + C
    return (S.y - y_fit)/S.e

  Sc = S.clip(x0-dx, x0+dx)
  guess = Sc.y.max(), x0, 2*np.diff(Sc.x).mean(), Sc.y.min()
  res = leastsq(sky_residual, guess, args=(Sc,), full_output=True)
  vec, err = res[0], np.sqrt(np.diag(res[1]))

  return vec[2], err[2]
#

