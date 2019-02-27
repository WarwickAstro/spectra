"""
Utilities for reading in spectra from different filetypes and returning a
Spectrum class.
"""
import numpy as np
import os
from sys import exit
from trm import molly
from astropy.io import fits
from .spec_class import Spectrum

__all__ = [
  "spec_from_txt",
  "spec_from_npy",
  "model_from_txt",
  "model_from_dk",
  "spec_from_sdss_fits",
  "spec_list_from_molly",
]

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

