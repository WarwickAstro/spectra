import numpy as np
from scipy.interpolate import interp1d
from scipy.special import wofz
from functools import reduce
import operator

__all__ = [
  "jangstrom",
  "voigt",
  "vac_to_air",
  "air_to_vac",
  "convolve_gaussian",
  "convolve_gaussian_R",
  "lanczos",
  "logarange",
  "keep_points",
]

jangstrom = \
  "$\mathrm{erg}\;\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}$"

def voigt(x, x0, fwhm_g, fwhm_l):
  sigma = voigt.Va*fwhm_g
  z = ((x-x0) + 0.5j*fwhm_l)/(sigma*voigt.Vb)
  return wofz(z).real/(sigma*voigt.Vc)
voigt.Va = 1/(2*np.sqrt(2*np.log(2)))
voigt.Vb = np.sqrt(2)
voigt.Vc = np.sqrt(2*np.pi)

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

def air_to_vac(Wair):
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

def keep_points(x, fname):
  """
  creates a mask for a spectrum that regions between pairs from a file
  """
  try:
    lines = open(fname,'r')
  except IOError:
    print("file %s does not exist" %fname)
    exit()
  between = lambda x, x1, x2: (x>float(x1))&(x<float(x2))
  segments = (between(x, *line.split()) for line in lines)
  return reduce(operator.or_, segments)

def lanczos(x, y, xnew):
  n = np.arange(len(x))
  Ni = interp1d(x, n, kind='linear', fill_value='extrapolate')(xnew)
  ynew = [np.sum(y*np.sinc(ni-n)) for ni in Ni]
  return np.array(ynew)

def logarange(x0, x1, R):
  """
  Like np.arange but with log-spaced points. The spacing parameter, R,
  is such that R = x/dx.
  """
  lx0, lx1= np.log(x0), np.log(x1)
  logx = np.arange(lx0, lx1, 1/R)
  return np.exp(logx)
