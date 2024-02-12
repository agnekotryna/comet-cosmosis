import numpy as np
import scipy.interpolate as interp
from cosmosis.datablock import names, option_section
from comet import comet
from comet.data import MeasuredData

"""
Placeholder module for likelihood module
"""

def setup(options):

	#The option section corresponds to the inputs set in the corresponding section of this module in the params.ini file:
	k_max = options.get_double(option_section, "k_max", default=0.3) 
	Dk2 = options.get_double(option_section, "Dk2", default=0.05) 
	Dk4 = options.get_double(option_section, "Dk4", default=0.05) 
	feedback = options.get_int(option_section, "feedback", default=0)
	signal = np.loadtxt(options.get_string(option_section, "data_file"), unpack=True)
	cov = np.loadtxt(options.get_string(option_section, "cov_file"))

	#Creating this instance to manage easily the dimensionality of the data and covariances
	data = MeasuredData(bins=signal[0], signal=signal[1:,:].T, cov=cov)
	k_max_array = [k_max, k_max-Dk2, k_max-Dk4]
	data.set_kmax(k_max_array)
	inv_cov = data.inverse_cov_kmax
	Pell = data.signal_kmax
	#Call COMET-emu to get the galaxy clustering multipoles:

	return Pell, inv_cov, feedback

def execute(block, config):
	Pell, inv_cov, feedback = config

	#Save multipoles to the datablock:
	Pell_theory = np.hstack([block[f"galaxy_P{m}", "bin_1"] for m in [0,2,4]])

	diff = Pell_theory - Pell
	chi2 = diff @ inv_cov@ diff.T
	
	block["likelihoods", "my_like"] = -0.5*chi2
	
	return 0