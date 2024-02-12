import numpy as np
import scipy.interpolate as interp
from cosmosis.datablock import names, option_section
from comet import comet
from comet.data import MeasuredData


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

# Added for the future, just need to learn what is the cleanest way to add bias relations
g2 = lambda b1: -2/7.*(b1-1)
g2_excursion_set= lambda b1: 0.524 - 0.547*b1 + 0.046*b1**2
def b2f(b1, g2):
    return 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4./3*g2
def g21f(b1, g2):
    return 2./21*(b1-1)+6./7*g2

def setup(options):

	#The option section corresponds to the inputs set in the corresponding section of this module in the params.ini file:
	k_max = options.get_double(option_section, "k_max", default=0.3)
	Dk2 = options.get_double(option_section, "Dk2", default=0.05) 
	Dk4 = options.get_double(option_section, "Dk4", default=0.05) 
	z_array = options.get_double(option_section, "z_bin") #redshifts at which to evaluate the matter power spectrum
	nbar = options.get_double(option_section, "nbar") #number density of the sample
	feedback = options.get_int(option_section, "feedback", default=0)
	signal = np.loadtxt(options.get_string(option_section, "data_file"), unpack=True) #the data vector
	cov = np.loadtxt(options.get_string(option_section, "cov_file")) #the covariance

	cosmo_chain = options.get(option_section, "cosmo_chain") #Turn on/off the computation of integrals
	cosmo_chain = True if cosmo_chain=='T' else False

	#Call COMET-emu to get the galaxy clustering multipoles:
	cosmo_fid = ['h','wc','wb','ns','As']

	fiducial_values_update = {}
	for p in cosmo_fid:
		fiducial_values_update[p] = options.get_double(option_section, p)
	cosmo_fid = fiducial_values_update
	cosmo_fid["z"] = z_array
	
	#First instance of the emulator
	emu = comet(model="EFT", use_Mpc=False)
	emu.define_fiducial_cosmology(params_fid=cosmo_fid)
	emu.define_nbar(nbar=nbar)

	#COMET needs at least a set of measurements to know where to evaluate the integrals (we may change it).
	kvec = signal[0] #np.arange(2*np.pi/3780, 1024*np.pi/3780+2*np.pi/3780, 2*np.pi/3780)
	oi='Pk' #each data set needs a label
	emu.define_data_set(
		obs_id='Pk', bins=kvec, signal=signal[1:,:].T, cov=cov
					)
	k_max_array = [k_max, k_max-Dk2, k_max-Dk4] #Defines the wavemodes for each multipole
	emu.data[oi].set_kmax(k_max_array) #Call Comet to define consistently the k-bins to evaluate it per multipole
	
	kvec = emu.data[oi].bins_kmax #k-bins to evaluate it per multipole
	binning = {'kfun':2*np.pi/3780, 'dk':2*np.pi/3780} #needed when computing the discret number of modes

	#The following block is executed only if we want to avoid to call the integral when cosmology does not vary.
	decompose=True if cosmo_chain==False else False
	if decompose:
		ell ={}
		ell[oi] = [2*m for m in range(emu.data[oi].n_ell)]

		PX_ell_list = np.zeros([sum(emu.data[oi].nbins),
								len(emu.diagrams_all)])
		kvec = emu.data[oi].bins_kmax
		
		for i, X in enumerate(emu.diagrams_all):
			PX_ell = emu.PX_ell(kvec,
									cosmo_fid, ell[oi], X,
									binning=binning,
									obs_id=None,
									de_model="lambda",
									q_tr_lo=None,
									W_damping=None,
									ell_for_recon=None)
			PX_ell_list[:, i] = np.hstack([PX_ell[m] for m
											in PX_ell.keys()])
				
		return emu, PX_ell_list, kvec, z_array, feedback
	
	else:
		return emu, 0, kvec, z_array, feedback

def execute(block, config):
	emu, PX_ell_list, kvec, z_array, feedback = config

	nbins = 1# will be len(z_array) in the fture

	#Read input cosmology from datablock:
	cosmo = load_cosmology(block)

	#Read input galaxy bias from the datablock:
	params = {}
	params["b1"] = block.get_double("bias_spec", "b1_bin1")
	params["b2"] = block.get_double("bias_spec", "b2_bin1")
	params["g2"] = block.get_double("bias_spec", "g2_bin1", default=0)
	params["g21"] = block.get_double("bias_spec", "g21_bin1", default=0)
	params["c0"] = block.get_double("bias_spec", "c0_bin1", default=0)
	params["c2"] = block.get_double("bias_spec", "c2_bin1", default=0)
	params["c4"] = block.get_double("bias_spec", "c4_bin1", default=0)
	params["cnlo"] = block.get_double("bias_spec", "cnlo_bin1", default=0)
	params["NP0"] = block.get_double("bias_spec", "NP0_bin1", default=0)
	params["NP20"] = block.get_double("bias_spec", "NP20_bin1", default=0)
	params["NP22"] = block.get_double("bias_spec", "NP22_bin1", default=0)

	params['h'] = cosmo["h0"]
	params['wc'] = cosmo["omch2"]
	params['wb'] = cosmo["ombh2"]
	params['ns'] = cosmo["n_s"] 
	params['As'] = cosmo["a_s"]
	params['z'] = z_array

	binning = {'kfun':2*np.pi/3780, 'dk':2*np.pi/3780} #Redundant, will change latter

	decompose=True if isinstance(PX_ell_list, list) else False #it checks if the cosmology does not change
	if decompose:
		#k = [np.array(kvec)]*3
		nbins = emu.data["Pk"].nbins #[x.shape[0] for x in k]

		emu.update_bias_params(params)
		emu.splines_up_to_date = False
		emu.dw_spline_up_to_date = False

		bX = emu.get_bias_coeff_for_chi2_decomposition()
		Pell = np.dot(bX, PX_ell_list.T)
		P0 = Pell[0:nbins[0]]
		P2 = Pell[nbins[0]:nbins[0] + nbins[1]]
		P4 = Pell[nbins[0] + nbins[1]:nbins[0] + nbins[1] + nbins[2]]

		Pell_LCDM = {f"ell{2*l}": Pell for l, Pell in enumerate([P0, P2, P4])}

	else:
		Pell_LCDM = emu.Pell(k=kvec, params=params, ell=[0,2,4], de_model="lambda", 
						  binning = binning
						  )

	#Save multipoles to the datablock:

	block["galaxy_P0", "kvec"] = kvec[0]
	block["galaxy_P0", "bin_1"] = Pell_LCDM['ell0']
	block["galaxy_P0", "nbin"] = nbins

	block["galaxy_P2", "kvec"] = kvec[1]
	block["galaxy_P2", "bin_1"] = Pell_LCDM['ell2']
	block["galaxy_P2", "nbin"] = nbins

	block["galaxy_P4", "kvec"] = kvec[2]
	block["galaxy_P4", "bin_1"] = Pell_LCDM['ell4']
	block["galaxy_P4", "nbin"] = nbins
	
	return 0