import numpy as np
import scipy.interpolate as interp
from cosmosis.datablock import names, option_section
from comet_developer import comet
from comet import comet as cometpk
from comet.data import MeasuredData
from hankl import P2xi
from scipy.interpolate import CubicSpline
import json

"""
Placeholder module for likelihood module
COMETxi 24/1/2025 with updated vdg emu made from Andrea's new training set
7/4/2025 4.0 added EH version of comet xi - minor change
8/4/2025 4.0 debugged the recognition of data type (xiwed or xiell) + added the recognition of nmeas
28/4/2025 4.2 switched the bias relations to use updated g2
5/4/2025 4.3 added fourier comet
5/4/2025 4.4 with reduced chi2
-debugged version with extended k range for FT

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

def load_cosmology(block, de_model, lenz):
    """
    Read input cosmology from the datablock
    """
    cosmo = {}

    try:
        cosmo["wc"] = np.repeat(block.get_double(names.cosmological_parameters, "omch2"), lenz)
        cosmo["wb"] = np.repeat(block.get_double(names.cosmological_parameters, "ombh2"), lenz)
        cosmo["ns"] = np.repeat(block.get_double(names.cosmological_parameters, "n_s"), lenz)
        cosmo["h"] = np.repeat(block.get_double(names.cosmological_parameters, "h0"), lenz)
        cosmo["As"] = np.repeat(block.get_double(names.cosmological_parameters, "a_s"), lenz)
        if de_model == "lambda": None
        elif de_model == "w0": 
            cosmo["w0"] =  np.repeat(block.get_double(names.cosmological_parameters, "w"), lenz)
        else:
            cosmo["w0"] =  np.repeat(block.get_double(names.cosmological_parameters, "w"), lenz)
            cosmo["wa"] =  np.repeat(block.get_double(names.cosmological_parameters, "wa"), lenz)
    except:
        print("Error reading input cosmology")

    return cosmo

def trim_scales(cov, nmin, nmax, nmeas):
	"""
	Remove the unused parts of covariance

	!!! This assumes that the same number of points is used for each measurement!!!
	
	IMPORTANT: this function also calculates the number of points in the measurement used; this will be needed for Hartlap correction and, therefore, must be ran *BEFORE* inverting covariance

	Parameters
	----------
	cov : numpy.array
		2D covariance matrix

	nmin : int
		the minimum point USED (the smallest scale measurement kept)
	nmax : int
    	the maximum point USED (the biggest scale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    measurement kept)

	Returns
	-------
	cov_trim : numpy.array
		trimmed matrix
	"""
	#nmeas = 3 #number of measurements, now assumed to be 3 (0, 2, 4)
	nbins_per_meas = int(cov.shape[0]/nmeas)
	bins_to_delete = []
	len_meas = int(nmax - nmin + 1) #the actual number of bins used
	num_points_use = len_meas

	for m in range(nmeas):
		for n in range(nmin-1):
			bins_to_delete.append(int(nbins_per_meas*m + n))
		for nm in range(nbins_per_meas-nmax):
			bins_to_delete.append(int(nbins_per_meas*m + nmax + nm))
        

	print("Removing the following bins:")
	print(bins_to_delete)

	cov_trim = np.delete(cov, bins_to_delete, axis=0)
	cov_trim = np.delete(cov_trim, bins_to_delete, axis=1)

	return cov_trim, len_meas
        
def trim_window(window, nmin, nmax, bmin, bmax, points_per_mes, bands_per_mes):
    """
    Remove the unused parts of window function

    Parameters
    ----------
    window : numpy.array
       2D window function. [len(s_array), len(rbands)] (both s and r are *uncut*)

    nmin : int
        the minimum point USED (the smallest scale measurement kept)
    nmax : int
        the maximum point USED (the biggest scale measurement kept)
    bmin : int
        the minimum band USED (the smallest scale measurement kept)
    bmax : int
        the maximum band USED (the biggest scale measurement kept)

    Returns
    -------
    windows_trim : numpy.array
        trimmed window function
    """

    #bands - where model is evaluated
    bands_to_delete = []
    #points - measurement made here
    points_to_delete = []

    
    for n in range(nmin-1):
        points_to_delete.append(int(n))
    for nm in range(points_per_mes-nmax):
        points_to_delete.append(int(nmax + nm))
        
    for b in range(bmin-1):
        bands_to_delete.append(int(b))
    for bm in range(bands_per_mes-bmax):
        bands_to_delete.append(int(bmax + bm))
        

    print("Removing the following s bins (points):")
    print(points_to_delete)
    
    print("Removing the following bands:")
    print(bands_to_delete)

    windows_trim = np.delete(window, points_to_delete, axis=0)
    windows_trim = np.delete(windows_trim, bands_to_delete, axis=1)

    return windows_trim


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
    

    #nbar_arr = json.loads(options.get(option_section, "nbar")) #number density of the sample
    #print ('nbar [(h/Mpc)**3] =', nbar_arr)
    #print ()
    
    
    redshift = json.loads(options.get(option_section, "redshift"))
    print ('redshift =', redshift)
    print ()
    
    redshift_lab = [str(z).strip('0') for z in redshift]
    data_id = ['mps_z'+str(z).strip('0') for z in redshift]
    print ('redshift_lab =', redshift_lab)
    print ()
    print ('data_id =', data_id)
    print ()
    
    RSD_model = options.get_string(option_section, "RSD_model")
    print ('RSD_model =', RSD_model)
    print ()
	    
    g21CoEvol = options.get_bool(option_section, "g21CoEvol")
    print ('Using g21 relation:', g21CoEvol)
    print ()
    
    g2ExSet = options.get_bool(option_section, "g2ExSet")
    print ('Using g2 relation:', g2ExSet)
    print ()
    
    

    #k_max = json.loads(options.get(option_section, "k_max")) 
    #print ()
    #scale_cuts = {}
    #for i,z_lab in enumerate(data_id):
    #    scale_cuts[z_lab] = [kk for kk in k_max[i]]
   
    #print('kmax [h/Mpc] =', scale_cuts)
    #print()
    

    


    
    #print ('Setting number density with values:', nbar_arr)
    #emu.define_nbar(nbar_arr)
    #print ()
    
    print ('\033[1mReading data vectors\033[0m')
    print ()
    
    data_dict = {}
    
    for i,file in enumerate(data_files.split(" ")):
        data_file = f'{data_folder}/{file}'
        cov_file = f'{data_folder}/{covariance_files.split(" ")[i]}'

        if 'xi024' in file or 'xi02' in file:
            xitype = 'xiell'
        elif 'xiwed' in file: xitype = 'xiwed'
        else: print("WARNING: data type not recognised, make sure it is either xi02(4) or xiwed")

		#None of the below are actually done in a way that would allow for different entries for each redshift, assume identical for all
        n_mocks = options.get_int(option_section, 'n_mocks')

        n_meas = options.get_int(option_section, "n_meas") #number of multipoles/wedges
        print ('n_meas =', n_meas)
        print ()

		
        num_points_full = options.get_int(option_section, 'num_points_full')
        num_bands_full = options.get_int(option_section, 'num_bands_full')

        min_points_use = options.get_int(option_section, 'min_points_use')
        max_points_use = options.get_int(option_section, 'max_points_use')
        min_bands_use = options.get_int(option_section, 'min_bands_use')
        max_bands_use = options.get_int(option_section, 'max_bands_use')

    
        rbands_file = f'{data_folder}/{options.get(option_section, "bands_files")}'
        windows_file = f'{data_folder}/{options.get(option_section, "windows_files")}'
	

        print (f'data {i+1}')
        print ('data_file =', data_file)
        print ('data_type =', xitype)
        print ('cov_file =', cov_file)
        print ('n_mocks =', n_mocks)
        print ('n_meas =', n_meas)
        print("num_points_full = ", num_points_full)
        print("num_bands_full = ", num_bands_full)
        print("Will use points from ", min_points_use, " to ", max_points_use)
        print ("Will use bands from ", min_bands_use, " to ", max_bands_use)
        print ("rbands_file = ", rbands_file)
        print ("windows_file = ", windows_file)
        print ()
        
        data_dict[redshift[i]] = {}

        data = np.loadtxt(data_file)

        data_dict[redshift[i]]['n_meas'] = n_meas

        data_cut = data[min_points_use-1:max_points_use, : ]
        data_dict[redshift[i]]['data'] = data_cut
        
        cov = np.loadtxt(cov_file)
        cov_cut, num_points_use = trim_scales(cov, min_points_use, max_points_use, n_meas)
        if n_mocks == 0: cor_fact = 0.0
        else: cor_fact = (3 * num_points_use + 1)/(n_mocks - 1) #Hartlap correction, assuming 3 measurements
        covinv = np.linalg.inv(cov_cut)*(1. - cor_fact)
        data_dict[redshift[i]]['cov_inv'] = covinv
        
        rbands = np.loadtxt(rbands_file)
        rbands_cut = rbands[min_bands_use-1:max_bands_use]
        data_dict[redshift[i]]['rbands'] = rbands_cut
        
        windows = np.loadtxt(windows_file)
        windows_cut = trim_window(windows, min_points_use, max_points_use, min_bands_use, max_bands_use, num_points_full, num_bands_full)
        data_dict[redshift[i]]['windows'] = windows_cut

		
        s_data = data_cut[:,0]
        xi_ell0= data_cut[:,1] 
        if n_meas > 1: xi_ell2 = data_cut[:,3] 
        if n_meas > 2: xi_ell4 = data_cut[:,5]
        
        if n_meas == 3: xi_data = np.hstack([xi_ell0, xi_ell2, xi_ell4])
        elif n_meas == 2: xi_data = np.hstack([xi_ell0, xi_ell2])
        elif n_meas == 1: xi_data = np.hstack([xi_ell0])
        data_dict[redshift[i]]['xi_data'] = xi_data
        data_dict[redshift[i]]['s_data'] = s_data
        
        data_dict[redshift[i]]['xi_type'] = xitype

		#COSMOLOGY    
        try: params_fid_comet = {par: options.get_double(option_section, par) for par in ['h', 'wc', 'wb', 'ns', 'As']}
        except:
             print ("As and ns not provided") 
             params_fid_comet = {par: options.get_double(option_section, par) for par in ['h', 'wc', 'wb']}
        params_fid_comet['z'] = redshift[i]
    
        de_model = options.get_string(option_section, "de_model")
        print ('Running DE model: ', de_model)
        print ()
    
        if de_model == "lambda": None
        elif de_model == "w0": 
            params_fid_comet["w0"] = options.get_double(option_section, "w")
        else:
            params_fid_comet["w0"] =  options.get_double(option_section, "w")
            params_fid_comet["wa"] =  options.get_double(option_section, "wa")

        print ('\033[1mInitialising emulator\033[0m')
        print ()
        #emu_model = options.get_string(option_section, "pt_model")
        #print ('Running PT model: ', emu_model)
        #print ()
    
        if RSD_model == 'EFT':
            emu = comet(model="EFT_xi", use_Mpc=False, path_to_model = 'EFT_DSTcorr_rrnorm_tomin1p0_rbfonly_20to250/EFT_xi', slope = [250, -1])
        elif RSD_model == 'EFT_vdg':
            emu = comet(model="VDG_infty_xi", use_Mpc=False, path_to_model = 'EFT_VDG_2025_10to250_f0to10_avir0to100_rbfvar0to100_var30to50/EFT_vdg', slope = [1, 0])
        elif RSD_model == 'EFT_vdg_EH':
            emu = comet(model="VDG_infty_xi", use_Mpc=False, path_to_model = 'EFT_VDG_EH_10to250_f0to10_var30to50/EFT_vdg', slope = [1, 0])
        elif RSD_model == "pk_vdg":
            emu = cometpk(model="VDG_infty", use_Mpc=False)
        else:
             ValueError("RSD model not recognised, please choose between 'EFT', 'EFT_vdg_EH' and 'EFT_vdg', currently set to ", RSD_model) 
    
        print ('Setting fiducial cosmology with parameters:', params_fid_comet)
        emu.define_fiducial_cosmology(params_fid=params_fid_comet)
        print ()
        
        data_dict[redshift[i]]['emu'] = emu


        
    return data_dict, g21CoEvol, g2ExSet, de_model, RSD_model

def execute(block, config):
      
    data_dict, g21CoEvol, g2ExSet, de_model, RSD_model = config
    

    params = load_cosmology(block, de_model, len(data_dict))
      
    print ('Running DE model: ', de_model)
    print ()

    print("Execute method running....")
    bias_keys = np.asarray(list(block.keys()))
    bias_spec = bias_keys[:,1][np.where(bias_keys[:,0]=='bias_spec')]

    #if RSD_model == 'EFT_vdg': nuisance_params.append('avir')
    if 'vdg' in RSD_model: nuisance_params.append('avir')

    #Here we collect all the bias parameters from the datablock to create params[bias] = [bias_bin1, bias_bin2, ...]
    for bias in nuisance_params:
        params[bias] = []
    
    for bias_bins in bias_spec:
        
        bias = bias_bins.split("_")[0]
        if bias in ["np0", "np20", "np22"]: bias=bias.upper()
        if bias in ["c0", "c2", "c4"]: 
            params[bias].append(block.get_double("bias_spec", bias_bins)*params["h"][0]**2) #changing units to Mpc/h
        else: 
            params[bias].append(block.get_double("bias_spec", bias_bins))

        
    if g2ExSet:
        params["g2"] = [g2f(b1) for b1 in params["b1"]]
    if g21CoEvol:
        params["g21"] = [g21f(b1, g2) for b1, g2 in zip(params["b1"], params["g2"]) ]

    chi2 = 0.0
    for ii, data_set in enumerate(data_dict):
        params_i = {key: float(value[ii]) for key, value in params.items()}
        params_i["z"] = data_set
        print ("Calculating theory for data set ", data_set)
        print ("Parameters: ", params_i)
        if RSD_model == 'pk_vdg': 
            print ("will perform ft")
            Xiell_theory = xiell_from_pk(data_dict[data_set]['emu'], params_i, data_dict[data_set]['rbands'], de_model=de_model)
        else:
            Xiell_theory = data_dict[data_set]['emu'].Xiell(data_dict[data_set]['rbands'], params_i, ell=[0, 2, 4], de_model=de_model)

        if data_dict[data_set]['xi_type'] == 'xiwed': 
             xiwed = xiell_to_wed(Xiell_theory['ell0'], Xiell_theory['ell2'], Xiell_theory['ell4'])
             xi_theory = np.vstack([xiwed[0], xiwed[1], xiwed[2]])
        elif data_dict[data_set]['xi_type'] == 'xiell': 
             xi_theory = np.vstack([Xiell_theory['ell0'], Xiell_theory['ell2'], Xiell_theory['ell4']])
        
        xi_theory_avg = np.matmul(xi_theory, data_dict[data_set]['windows'].T)
        if data_dict[data_set]['n_meas'] == 3: xi_theory_stacked = np.hstack([xi_theory_avg[0], xi_theory_avg[1], xi_theory_avg[2]])
        elif data_dict[data_set]['n_meas'] == 2: xi_theory_stacked = np.hstack([xi_theory_avg[0], xi_theory_avg[1]]) #no hexadecapole

        print ("DEBUGGING CHI")
        print ("This is data set "+str(data_set))

        diff = xi_theory_stacked - data_dict[data_set]['xi_data']
        
        chi2_ds = diff @ data_dict[data_set]['cov_inv'] @ diff.T

        chi2 += chi2_ds
        print ("Chi2 for data set ", data_set, " is ", chi2_ds)

        block["data_vector_"+str(ii+1), 'chi2'] = chi2_ds
        block["data_vector_"+str(ii+1), 'chi2red'] = chi2_ds/len(data_dict[data_set]['xi_data'])

    block["likelihoods", "my_like"] = -0.5 * chi2
    #trying to add derived sigma12 - shouldn't matter which datset emu to use, related to background cosmo only, so just use last dataset touched
    block["cosmological_parameters", "sigma12"] = (data_dict[data_set]['emu'].cosmo.growth_factor(z=0)/data_dict[data_set]['emu'].cosmo.growth_factor(params_i["z"])) * data_dict[data_set]['emu'].params['s12']

    return 0

def xiell_to_wed(ell0, ell2, ell4):
        """
        Take the two point correlation function multipoles and transform them into clustering wedges
 
        Parameters
        ----------
        ell0, ell2, ell4: 3xnp.array, monopole, quadrupole, hexadecapole


        Returns
        -------
        wedcalc[0], wedcalc[1], wedcalc[2] - np.arrays, three clustering wedges, the order matches data vector order
        """
    
    
        ell024 = np.vstack([ell0, ell2, ell4])
        

        M = np.array([1., -4./9., 20./81., 1., -1./9., -85./324., 1., 5./9., 5./324.])
        M = M.reshape([3,3])

        wedcalc = M@ell024
        
        return wedcalc

def xiell_from_pk(emu, params, rvals, de_model):
    kk = np.logspace(np.log10(0.000695/params["h"]), np.log10(10.4249/params["h"]), 166)
    ell024 = emu.Pell(kk, params, ell=[0,2,4], ell_for_recon=[0,2,4,6], de_model=de_model)

    return transform_pell(ell024, kk, 2.5, rvals, params["h"])


def transform_pell(pell_input, kk_input, kcut_val, xx, hval):
        
    #kmax_theory = 0.35028
    kmax_theory = kk_input.max()
    kmax = 100/hval
    nk = 2**15
    kk_hnkl = np.logspace(np.log10(kk_input.min()), np.log10(kmax), nk)
    k2ext = kk_hnkl[np.where(kk_hnkl>kmax_theory)]

    pell0_spline = CubicSpline(kk_input, pell_input['ell0'])
    pell2_spline = CubicSpline(kk_input, pell_input['ell2'])            
    pell4_spline = CubicSpline(kk_input, pell_input['ell4'])

    pell2transform_ell0 = pell0_spline(kk_hnkl)
    pell2transform_ell2 = pell2_spline(kk_hnkl)
    ppell2transform_ell4 = pell4_spline(kk_hnkl)

        
    a0, b0 = get_pl_coefs(kk_hnkl[kk_hnkl<=kmax_theory][-2:], pell2transform_ell0[kk_hnkl<=kmax_theory][-2:])
    a2, b2 = get_pl_coefs(kk_hnkl[kk_hnkl<=kmax_theory][-2:], pell2transform_ell2[kk_hnkl<=kmax_theory][-2:])
    a4, b4 = get_pl_coefs(kk_hnkl[kk_hnkl<=kmax_theory][-2:], ppell2transform_ell4[kk_hnkl<=kmax_theory][-2:])

        
    pell_input['ell0'] = np.hstack([pell2transform_ell0[kk_hnkl<=kmax_theory],a0*(k2ext**b0)])
    pell_input['ell2'] = np.hstack([pell2transform_ell2[kk_hnkl<=kmax_theory],a2*(k2ext**b2)])
    pell_input['ell4'] = np.hstack([ppell2transform_ell4[kk_hnkl<=kmax_theory],a4*(k2ext**b4)])
        
    xiell_comet = {}
        
    rr0, xiell_comet['ell0'] = P2xi(kk_hnkl, cutoff(kk_hnkl, kcut_val, 2)*pell_input['ell0'], 0, lowring=True)
    rr2, xiell_comet['ell2'] = P2xi(kk_hnkl, cutoff(kk_hnkl, kcut_val, 2)*pell_input['ell2'], 2, lowring=True)
    rr4, xiell_comet['ell4'] = P2xi(kk_hnkl, cutoff(kk_hnkl, kcut_val, 2)*pell_input['ell4'], 4, lowring=True)

    splell0 = CubicSpline(rr0, np.real(xiell_comet['ell0']))
    splell2 = CubicSpline(rr2, np.real(xiell_comet['ell2']))
    splell4 = CubicSpline(rr4, np.real(xiell_comet['ell4']))

    xiell_comet['ell0'] = splell0(xx)
    xiell_comet['ell2'] = splell2(xx)
    xiell_comet['ell4'] = splell4(xx)

    return xiell_comet

def get_pl_coefs(x,y):
    b = np.log(np.abs(y[0]/y[1]))/np.log(np.abs(x[0]/x[1]))
    a = y[1]/(x[1]**b)
        
    return a, b

def cutoff(k, kcut, n):
    """
    A cut-off at high-k where the model fails.
    Inspired by the cut off used in the code
    of Ariel Sanchez

    Parameters
    ----------
    k: array
        the k values
    kcut: float
        the k cut off
    n: float
        the index of the power law
        cut off
    """
    return np.exp(-1.0*np.power((k/kcut), n))
