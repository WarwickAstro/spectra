"""
Module for working with astropyhsical spectra
"""
__author__ = "Mark Hollands"
__email__ = "M.Hollands.1@warwick.ac.uk"

from .spec_class import Spectrum 
from .spec_io import *
from .spec_functions import *
from .misc import air_to_vac, vac_to_air, voigt, jangstrom, logarange
