import numpy as np
import scipy.interpolate as interp
from cosmosis.datablock import names, option_section
from comet import comet
from comet.data import MeasuredData
import json

"""
Placeholder module for likelihood module
"""

#BIAS RELATIONS (the usual ones)
def b2f(b1, g2):
    return 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4./3*g2
def g2LL(b1):
    return -2./7*(b1-1)
def g2f(b1):
    return 0.524 - 0.547*b1 + 0.046*b1**2
def g21f(b1, g2):
    return 2./21*(b1-1)+ 6./7*g2


nuisance_params = ['b1', 'b2', 'g2', 'g21', 'c0', 'c2', 'c4', 'cnlo', 'NP0', 'NP20', 'NP22']

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
    feedback = options.get_int(option_section, "feedback", default=0)
    # DATA
    data_folder = options.get_string(option_section, "data_dir")
    print ('data_folder =', data_folder)
    print ()
    
    data_files = options.get(option_section, "data_files")
    print ('data_files =', data_files)
    print ()
    
    covariance_files = options.get(option_section, "covariance_files")
    print ('covariance_files =', covariance_files)
    print ()
    
    nbar_arr = json.loads(options.get(option_section, "nbar")) #number density of the sample
    print ('nbar [(h/Mpc)**3] =', nbar_arr)
    print ()
    
    
    redshift = json.loads(options.get(option_section, "redshift"))
    print ('redshift =', redshift)
    print ()
    
    redshift_lab = [str(z).strip('0') for z in redshift]
    data_id = ['mps_z'+str(z).strip('0') for z in redshift]
    print ('redshift_lab =', redshift_lab)
    print ()
    print ('data_id =', data_id)
    print ()
    
    k_max = json.loads(options.get(option_section, "k_max")) 
    
    print ()
    scale_cuts = {}
    for i,z_lab in enumerate(data_id):
        scale_cuts[z_lab] = [kk for kk in k_max[i]]
   
    print('kmax [h/Mpc] =', scale_cuts)
    print()
    
    try: 
        AM_priors = options.get(option_section, "AM_priors")
        AM_priors = AM_priors.replace("'", "\"")
        AM_priors = json.loads(AM_priors)
        print("Analythically marginalising over: ", list(AM_priors.keys()))
        print()
        
    except:
        AM_priors=None
        print("No analythically marginalising")
        print()
        
    #COSMOLOGY    
    params_fid_comet = {par: np.repeat(options.get_double(option_section, par), len(redshift)) for par in ['h', 'wc', 'wb', 'ns', 'As']}
    params_fid_comet['z'] = redshift
    
    de_model = options.get_string(option_section, "de_model")
    print ('Runnung DE model: ', de_model)
    print ()
    
    if de_model == "lambda": None
    elif de_model == "w0": 
        params_fid_comet["w0"] = np.repeat(options.get_double(option_section, "w"), len(redshift))
    else:
        params_fid_comet["w0"] =  np.repeat(options.get_double(option_section, "w"), len(redshift))
        params_fid_comet["wa"] =  np.repeat(options.get_double(option_section, "wa"), len(redshift))
    
    g21CoEvol = options.get_bool(option_section, "g21CoEvol")
    print ('Using g21 relation:', g21CoEvol)
    print ()
    
    g2ExSet = options.get_bool(option_section, "g2ExSet")
    print ('Using g2 relation:', g2ExSet)
    print ()
    
    print ('\033[1mInitialising emulator\033[0m')
    print ()
    emu_model = options.get_string(option_section, "pt_model")
    print ('Runnung PT model: ', emu_model)
    print ()
    
    emu = comet(model=emu_model, use_Mpc=False)
    
    print ('Setting fiducial cosmology with parameters:', params_fid_comet)
    emu.define_fiducial_cosmology(params_fid=params_fid_comet)
    print ()
    
    print ('Setting number density with values:', nbar_arr)
    emu.define_nbar(nbar_arr)
    print ()
    
    print ('\033[1mReading data vectors\033[0m')
    print ()
    
    for i,file in enumerate(data_files.split(" ")):
        data_file = f'{data_folder}/{file}'
        cov_file = f'{data_folder}/{covariance_files.split(" ")[i]}'
        print (f'data {i+1}')
        print ('data_file =', data_file)
        print ('cov_file =', cov_file)
        print ()

        data = np.loadtxt(data_file)
        cov = np.loadtxt(cov_file)

        k = data[:,0]
        pk0 = data[:,1] 
        pk2 = data[:,3] 
        pk4 = data[:,5]
        mps = np.asarray([pk0, pk2, pk4]).T

        emu.define_data_set(data_id[i], zeff=redshift[i], bins=k, signal=mps, cov=cov)
    return emu, params_fid_comet, redshift, scale_cuts, data_id, g21CoEvol, g2ExSet, AM_priors, de_model, feedback

def execute(block, config):
    
    emu, params_fid_comet,redshift, scale_cuts, data_id, g21CoEvol, g2ExSet, AM_priors, de_model, feedback = config
    
    cosmo = load_cosmology(block)
    
    params = {}
    params['h'] = np.repeat(cosmo["h0"],len(redshift)) 
    params['wc'] = np.repeat(cosmo["omch2"],len(redshift))  
    params['wb'] = np.repeat(cosmo["ombh2"],len(redshift))  
    params['ns'] = np.repeat( cosmo["n_s"] ,len(redshift)) 
    params['As'] = np.repeat(cosmo["a_s"],len(redshift))
    
    if de_model == "lambda": None
    elif de_model == "w0": 
        params["w0"] =  np.repeat(block.get_double(names.cosmological_parameters, "w"),len(redshift))
    else:
        params["w0"] =  np.repeat(block.get_double(names.cosmological_parameters, "w"),len(redshift))
        params["wa"] =  np.repeat(block.get_double(names.cosmological_parameters, "wa"),len(redshift))
        
        
    params['z'] = redshift
    
    print ('Runnung DE model: ', params)
    print ()
    
    print("Execute mthod running....")
    bias_keys = np.asarray(list(block.keys()))
    bias_spec=bias_keys[:,1][np.where(bias_keys[:,0]=='bias_spec')]
    
    
    for bias in nuisance_params:
        params[bias] = []
    
    for bias_bins in bias_spec:
        
        bias = bias_bins.split("_")[0]
        
        if bias in ["np0", "np20", "np22"]: bias=bias.upper()
        if bias in ["c0", "c2", "c4"]: params[bias].append(block.get_double("bias_spec", bias_bins)*cosmo["h0"]**2)
        else: params[bias].append(block.get_double("bias_spec", bias_bins))
        
        
    if AM_priors:
        for key in AM_priors.keys():
            params[key] = np.repeat(0,len(redshift))
    
    if g21CoEvol:
            params["g21"] = [g21f(b1, g2) for b1, g2 in zip(params["b1"], params["g2"]) ]
    if g2ExSet: 
            params["g21"] = [g2f(b1) for b1 in params["b1"]]
    
    chi2 = emu.chi2(data_id, params, scale_cuts, de_model=de_model, AM_priors=AM_priors)    
    block["likelihoods", "my_like"] = -0.5*chi2[0]
    
    return 0
