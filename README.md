# Spectra

This is a python module designed for working with astronomical spectra.
In particular it includes a "Spectrum" class to store wavelengths, fluxes, errors,
and features many routines for performing common tasks for spectroscopy.

# Features:
* Spectra arithmetic (e.g. subtracting two spectra)
* Synthetic AB magnitudes (filter curves included)
* Unit conversion (e.g. "erg/(cm2 s AA)" --> "mJy")
* Apply redshifts, and conversion between air/vac wavelengths
* Wavelength Interpolation (including sinc/Lanczos)
* Interstellar reddening
* Gaussian convolution
* I/O Routines for reading/writing to various file types.
* Sky line fitting

# Dependencies:
* Python >= 3.6
* numpy
* matplotlib
* scipy
* astropy
* trm-molly
