import numpy as np
from scipy.integrate import trapz as Itrapz, simps as Isimps
import os.path

__all__ = [
  "load_transmission_curve",
  "mag_calc_AB",
]

filters_dir = "{}/filt_profiles".format(os.path.dirname(__file__))

GaiaDict = {'G':'G', 'Bp':'Gbp', 'Rp':'Grp'}
filter_paths = {
  **{f"2m{b}"   : f"2MASS_2MASS.{b}.dat" for b in "JHK"}, #2Mass
  **{f"Denis{b}": f"DENIS_DENIS.{b}.dat" for b in "I"}, #DENIS
  **{f"Gaia{b}" : f"GAIA_GAIA2r.{GaiaDict[b]}.dat" for b in ("G","Bp","Rp")}, #Gaia
  **{f"Galex{b}": f"GALEX_GALEX.{b}.dat" for b in ("NUV", "FUV")}, #GALEX
  **{b          : f"Generic_Johnson.{b}.dat" for b in "UBVRI"}, #Generic Johnson
  **{f"ps{b}"   : f"PAN-STARRS_PS1.{b}.dat" for b in "grizy"}, #PanStarrs
  **{b          : f"SLOAN_SDSS.{b}.dat" for b in "ugriz"}, #SDSS
  **{f"sm{b}"   : f"SkyMapper_SkyMapper.{b}.dat" for b in "uvgriz"}, #SkyMapper
  **{f"S{b}"    : f"Spitzer_IRAC.I{b}.dat" for b in "12"}, #Spitzer
  **{f"sw{b}"   : f"Swift_UVOT.{b}.dat" for b in ("U","UVW1","UVW2","UVM2")}, #Swift
  **{f"UK{b}"   : f"UKIRT_UKIDSS.{b}.dat" for b in "ZYJHK"}, #UKIRT
  **{f"W{b}"    : f"WISE_WISE.W{b}.dat" for b in "12"}, #Wise
}

def load_transmission_curve(filt):
  """
  Loads the filter curves obtained from VOSA (SVO).
  """
  from .spec_io import model_from_txt

  try:
    full_path = "{}/{}".format(filters_dir, filter_paths[filt])
  except KeyError:
    print('Invalid filter name: {}'.format(filt))
    exit()

  return model_from_txt(full_path, x_unit="AA", y_unit="")
#

def m_AB_int(X, Y, R, Ifun):
  y_nu = Ifun(Y*R/X, X)/Ifun(R/X, X) 
  m = -2.5 * np.log10(y_nu) + 8.90
  return m

def mag_calc_AB(S, filt, NMONTE=1000, Ifun=Itrapz):
  """
  Calculates the synthetic AB magnitude of a spectrum for a given filter.
  If NMONTE is > 0, monte-carlo error propagation is performed outputting
  both a synthetic-mag and error. For model-spectra, i.e. no errors,
  use e=np.ones_like(f) and NMONTE=0. List of currently supported filters:

  2Mass:     ['2mJ','2mH','2mK']

  Denis:     ['DenisI']

  Gaia:      ['Gaia(G,Bp,Rp)']

  Galex:     ['GalexFUV' 'GalexNUV']

  Johnson:   ['U','B','V','R','I']

  PanSTARRS: ['ps(grizy)']

  SDSS:      ['u','g','r','i','z']

  Skymapper: ['sm(uvgriz)']

  Spitzer:   ['S1','S2']

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
  R = R.interp(S)

  #Calculate AB magnitudes, potentially including flux errors
  if NMONTE == 0:
    return m_AB_int(S.x, S.y, R.y, Ifun)
  else:
    y_mc = lambda S: np.random.normal(S.y, S.e)
    m = np.array([m_AB_int(S.x, y_mc(S), R.y, Ifun) for i in range(NMONTE)])
    return np.mean(m), np.std(m)
#
