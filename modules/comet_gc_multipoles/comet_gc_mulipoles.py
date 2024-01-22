import numpy as np
import scipy.interpolate as interp
from cosmosis.datablock import names, option_section


def load_cosmology(block):
	"""
	Read input cosmology from the datablock
	"""
	cosmo = {}

	try:
		cosmo["omch2"] = block.get_double(names.cosmological_parameters, "omch2")
		cosmo["ombh2"] = block.get_double(names.cosmological_parameters, "ombh2")
		cosmo["n_s"] = block.get_double(names.cosmological_parameters, "n_s")
		cosmo["h0"] = block.get_double(names.cosmological_parameters, "h0")
		cosmo["a_s"] = block.get_double(names.cosmological_parameters, "a_s")
	except:
		print("Error reading input cosmology")

	return cosmo

def setup(options):

	#The option section corresponds to the inputs set in the corresponding section of this module in the params.ini file:
	k_max = options.get_double(option_section, "k_max", default=0.3) 
	z_array = options.get_double_array_1d(option_section, "z") #redshifts at which to evaluate the matter power spectrum
	feedback = options.get_int(option_section, "feedback", default=0)

	return k_max, z_array, feedback

def execute(block, config):

	k_max, z_array, feedback = config

	nbins = len(z_array)

	#Read input cosmology from datablock:
	cosmo = load_cosmology(block)

	#Read input galaxy bias from the datablock:
	b1 = [block.get_double(names.bias_lens, "b1_bin%d"%pos_bin)]
			for pos_bin in range(1, nbins+1)
	b2 = [block.get_double(names.bias_lens, "b2_bin%d"%pos_bin)]
			for pos_bin in range(1, nbins+1)


	#Call COMET-emu to get the galaxy clustering multipoles:



	#Save multipoles to the datablock:
	block["galaxy_xi0", "s"] = s
	block["galaxy_xi0", "bin_{}_{}".format(i+1, i+1)] = xi0
	block["galaxy_xi0", "nbin"] = nbin

	block["galaxy_xi2", "s"] = s
	block["galaxy_xi2", "bin_{}_{}".format(i+1, i+1)] = xi2
	block["galaxy_xi2", "nbin"] = nbin

	block["galaxy_xi4", "s"] = s
	block["galaxy_xi4", "bin_{}_{}".format(i+1, i+1)] = xi4
	block["galaxy_xi4", "nbin"] = nbin

	return 0