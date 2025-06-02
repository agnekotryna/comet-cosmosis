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
30/5/2025 4.5 modified to output theory and data to be used in joint shear+rsd likelihood
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


def _process_single_dataset_cuts(data_file_path, cov_file_path, rbands_file_path, windows_file_path,
                                 n_mocks, n_meas, num_points_full, num_bands_full,
                                 min_points_use, max_points_use, min_bands_use, max_bands_use,
                                 xitype_hint, feedback=0):
    """
    Loads, cuts, and prepares data for a single dataset.
    Returns a dictionary with the processed data products.
    """
    if feedback > 1:
        print(f"    Processing cuts for: {data_file_path}")

    processed_data = {}
    processed_data['xitype'] = xitype_hint # Store the determined xitype
    processed_data['n_meas'] = n_meas # Store n_meas for this dataset

    # Load raw data
    data_full = np.loadtxt(data_file_path)
    cov_full = np.loadtxt(cov_file_path)
    rbands_full = np.loadtxt(rbands_file_path)
    windows_full = np.loadtxt(windows_file_path)

    # Cut data table (for extracting s_data and xi_ell components)
    data_cut_table = data_full[min_points_use-1:max_points_use, :]
    # processed_data['data_cut_table'] = data_cut_table # Optional: if needed elsewhere

    # Generate mask for joint_like (relative to uncut covariance for this dataset)
    total_original_points_for_dataset = n_meas * num_points_full
    dataset_mask_for_joint_like = np.ones(total_original_points_for_dataset, dtype=bool)
    bins_to_delete_for_mask = []
    for m_idx in range(n_meas):
        for n_idx in range(min_points_use - 1):
            bins_to_delete_for_mask.append(int(num_points_full * m_idx + n_idx))
        for nm_idx in range(num_points_full - max_points_use):
            bins_to_delete_for_mask.append(int(num_points_full * m_idx + max_points_use + nm_idx))
    if bins_to_delete_for_mask:
        dataset_mask_for_joint_like[np.array(bins_to_delete_for_mask, dtype=int)] = False
    processed_data['mask_for_joint_like'] = dataset_mask_for_joint_like

    # Cut covariance matrix and get effective number of points used per multipole
    cov_cut, num_points_use_effective = trim_scales(cov_full, min_points_use, max_points_use, n_meas)
    
    # Hartlap correction
    # The '3' in the original Hartlap was an assumption. It should be n_meas.
    # num_points_use_effective is len_meas from trim_scales, i.e., points per multipole after cut.
    # Total number of data points in the cut vector is n_meas * num_points_use_effective
    cor_fact = 0.0
    if n_mocks > 0:
        total_data_points_in_cut_vector = n_meas * num_points_use_effective
        # Standard Hartlap factor D-1 / (N_sim -1) where D is total data points.
        # Or (N_data + 1) / (N_mocks -1) for some conventions.
        # Your original: (3 * num_points_use + 1)/(n_mocks - 1)
        # Let's use n_meas * num_points_use_effective for the total number of data points.
        # The +1 term is often (p+1) where p is number of data points.
        # So, (n_meas * num_points_use_effective + 1) / (n_mocks - 1)
        cor_fact = (total_data_points_in_cut_vector + 1) / (n_mocks - 1)
        if feedback > 1:
            print(f"    Hartlap: n_mocks={n_mocks}, n_meas={n_meas}, num_points_eff_per_multipole={num_points_use_effective}")
            print(f"    Hartlap: total_data_points_cut={total_data_points_in_cut_vector}, cor_fact={cor_fact:.4f}")


    processed_data['cov_inv'] = np.linalg.inv(cov_cut) * (1. - cor_fact)

    # Cut r-bands and window functions
    processed_data['rbands'] = rbands_full[min_bands_use-1:max_bands_use]
    processed_data['windows'] = trim_window(windows_full, min_points_use, max_points_use,
                                            min_bands_use, max_bands_use,
                                            num_points_full, num_bands_full) # num_points_full is points_per_mes

    # Extract and stack cut xi data
    s_data_cut = data_cut_table[:, 0]
    xi_ell0_cut = data_cut_table[:, 1]
    xi_ell_components = [xi_ell0_cut]

    if n_meas > 1 and data_cut_table.shape[1] > 3: # Check if column exists
        xi_ell2_cut = data_cut_table[:, 3]
        xi_ell_components.append(xi_ell2_cut)
    if n_meas > 2 and data_cut_table.shape[1] > 5: # Check if column exists
        xi_ell4_cut = data_cut_table[:, 5]
        xi_ell_components.append(xi_ell4_cut)
    
    # Ensure we only stack as many components as specified by n_meas
    processed_data['xi_data'] = np.hstack(xi_ell_components[:n_meas])
    processed_data['s_data'] = s_data_cut # For reference, if needed

    if feedback > 1:
        print(f"    Cut data vector shape for this dataset: {processed_data['xi_data'].shape}")
        print(f"    Cut rbands shape: {processed_data['rbands'].shape}")
        print(f"    Cut windows shape: {processed_data['windows'].shape}")
        print(f"    Cut cov_inv shape: {processed_data['cov_inv'].shape}")
        print(f"    Mask for joint like (sum true): {np.sum(processed_data['mask_for_joint_like'])}")

    return processed_data


def setup(options):

    feedback = options.get_int(option_section, "feedback", default=0)
    data_folder = options.get_string(option_section, "data_dir")
    data_files_str = options.get(option_section, "data_files")
    covariance_files_str = options.get(option_section, "covariance_files")
    redshift_list_float = json.loads(options.get(option_section, "redshift"))

    if feedback > 0:
        print('data_folder =', data_folder)
        print('data_files =', data_files_str)
        print('covariance_files =', covariance_files_str)
        print('redshift =', redshift_list_float)

    RSD_model = options.get_string(option_section, "RSD_model")
    g21CoEvol = options.get_bool(option_section, "g21CoEvol")
    g2ExSet = options.get_bool(option_section, "g2ExSet")
    de_model_fiducial = options.get_string(option_section, "de_model") # DE model for fiducial cosmology for emulator

    if feedback > 0:
        print('RSD_model =', RSD_model)
        print('Using g21 relation:', g21CoEvol)
        print('Using g2 relation:', g2ExSet)
        print('DE model for fiducial cosmology:', de_model_fiducial)
        print('\033[1mReading and processing data vectors\033[0m')

    data_dict = {}

    # These options are assumed to be global for all datasets processed by this module instance
    # If they can vary per file, the logic to read them would need to be more complex
    n_mocks_global = options.get_int(option_section, 'n_mocks')
    n_meas_global = options.get_int(option_section, "n_meas")
    num_points_full_global = options.get_int(option_section, 'num_points_full')
    num_bands_full_global = options.get_int(option_section, 'num_bands_full')
    min_points_use_global = options.get_int(option_section, 'min_points_use')
    max_points_use_global = options.get_int(option_section, 'max_points_use')
    min_bands_use_global = options.get_int(option_section, 'min_bands_use')
    max_bands_use_global = options.get_int(option_section, 'max_bands_use')
    # Assuming bands_files and windows_files might be single files or need selection logic
    # For simplicity, using the first one if multiple are listed, or assuming it's a single file name.
    # If these vary per dataset, they should be lists split like data_files_str.
    bands_files_config = options.get(option_section, "bands_files")
    windows_files_config = options.get(option_section, "windows_files")
    
    for i, file_name_part in enumerate(data_files_str.split(" ")):
        current_z = redshift_list_float[i]
        data_dict[current_z] = {} # Initialize dict for this redshift

        data_file_path = f'{data_folder}/{file_name_part}'
        cov_file_path = f'{data_folder}/{covariance_files_str.split(" ")[i]}'
        
        # Determine xitype from file name
        xitype = 'unknown'
        if 'xi024' in file_name_part or 'xi02' in file_name_part:
            xitype = 'xiell'
        elif 'xiwed' in file_name_part:
            xitype = 'xiwed'
        else:
            if feedback > 0: print(f"WARNING: data type for {file_name_part} not recognised, make sure it is either xi02(4) or xiwed")

    
        # Handle bands and windows files (assuming one global or needs splitting if per-dataset)
        # If per-dataset, split bands_files_config and windows_files_config here.
        # For now, assume it's a single file name or the first in a list.
        current_rbands_file = f'{data_folder}/{bands_files_config.split(" ")[0]}' 
        current_windows_file = f'{data_folder}/{windows_files_config.split(" ")[0]}'
	

        if feedback > 0:
            print(f'\nProcessing dataset {i+1} for z = {current_z}')
            print(f'  data_file = {data_file_path}')
            print(f'  data_type = {xitype}')
            print(f'  cov_file = {cov_file_path}')
            print(f'  n_mocks = {n_mocks_global}')
            print(f'  n_meas (from config) = {n_meas_global}') # n_meas is crucial for interpreting data structure
            print(f"  num_points_full = {num_points_full_global}")
            print(f"  num_bands_full = {num_bands_full_global}")
            print(f"  Will use points from {min_points_use_global} to {max_points_use_global}")
            print(f"  Will use bands from {min_bands_use_global} to {max_bands_use_global}")
            print(f"  rbands_file = {current_rbands_file}")
            print(f"  windows_file = {current_windows_file}")
        
        # Call the new function to process cuts for this dataset
        processed_dataset_info = _process_single_dataset_cuts(
            data_file_path, cov_file_path, current_rbands_file, current_windows_file,
            n_mocks_global, n_meas_global, num_points_full_global, num_bands_full_global,
            min_points_use_global, max_points_use_global, min_bands_use_global, max_bands_use_global,
            xitype, feedback=feedback
        )
        # Store all processed info into the data_dict for this redshift
        data_dict[current_z].update(processed_dataset_info)


		#COSMOLOGY    
        try: params_fid_comet = {par: options.get_double(option_section, par) for par in ['h', 'wc', 'wb', 'ns', 'As']}
        except:
             print ("As and ns not provided") 
             params_fid_comet = {par: options.get_double(option_section, par) for par in ['h', 'wc', 'wb']}
        params_fid_comet['z'] = current_z
    

        if de_model_fiducial == "w0":
            params_fid_comet["w0"] = options.get_double(option_section, "w")
        elif de_model_fiducial != "lambda": # Assumes w0wa
            params_fid_comet["w0"] = options.get_double(option_section, "w")
            params_fid_comet["wa"] = options.get_double(option_section, "wa")

        if feedback > 0: print(f'    Initialising emulator for z={current_z} with fiducial: {params_fid_comet}')

    
        if RSD_model == 'EFT':
            emu = comet(model="EFT_xi", use_Mpc=False, path_to_model = 'EFT_DSTcorr_rrnorm_tomin1p0_rbfonly_20to250/EFT_xi', slope = [250, -1])
        elif RSD_model == 'EFT_vdg':
            emu = comet(model="VDG_infty_xi", use_Mpc=False, path_to_model = 'EFT_VDG_2025_10to250_f0to10_avir0to100_rbfvar0to100_var30to50/EFT_vdg', slope = [1, 0])
        elif RSD_model == 'EFT_vdg_EH':
            emu = comet(model="VDG_infty_xi", use_Mpc=False, path_to_model = 'EFT_VDG_EH_10to250_f0to10_var30to50/EFT_vdg', slope = [1, 0])
        elif RSD_model == "pk_vdg":
            emu = cometpk(model="VDG_infty", use_Mpc=False)
        else:
            raise ValueError("RSD model not recognised, please choose between 'EFT', 'EFT_vdg_EH' and 'EFT_vdg', currently set to ", RSD_model) 
    
  
        emu.define_fiducial_cosmology(params_fid=params_fid_comet)
        if feedback > 0: print(f"    Emulator initialized for z={current_z}")
        
        data_dict[current_z]['emu'] = emu
        
    # The de_model for runtime theory calculation is also needed by execute
    # It might be the same as de_model_fiducial or different if configured.
    # For simplicity, assume it's the same as the one used for fiducial, or add another option for it.
    # Let's assume the 'de_model' option is for runtime, and fiducial uses it too.
    
    # Return a tuple or a dictionary for config
    # Using a dictionary is often more robust for accessing elements by name in execute
    config_to_return = {
        "data_dict": data_dict,
        "ordered_redshift_keys": redshift_list_float, # Store the order
        "g21CoEvol": g21CoEvol,
        "g2ExSet": g2ExSet,
        "de_model_runtime": de_model_fiducial, # Assuming fiducial DE model is used for runtime theory
        "RSD_model": RSD_model,
        "feedback": feedback,
        "nuisance_params_list": list(nuisance_params) # Pass a copy of the base list
    }
    # Add avir to nuisance_params_list if vdg model
    if 'vdg' in RSD_model and 'avir' not in config_to_return["nuisance_params_list"]:
        config_to_return["nuisance_params_list"].append('avir')

    return config_to_return

def execute(block, config):
      
    # Unpack from the config dictionary
    data_dict = config["data_dict"]
    ordered_redshift_keys = config["ordered_redshift_keys"] # Use the stored order
    g21CoEvol = config["g21CoEvol"]
    g2ExSet = config["g2ExSet"]
    de_model = config["de_model_runtime"] # DE model for runtime theory calculation
    RSD_model = config["RSD_model"]
    feedback = config["feedback"]
    current_nuisance_params_list = config["nuisance_params_list"] # Already includes avir if needed
    

    params_cosmo_all_bins = load_cosmology(block, de_model, len(data_dict))

      
    # Bias parameter handling
    params_nuisance_all_bins = {bias: [] for bias in current_nuisance_params_list}
    bias_keys_from_block = np.asarray(list(block.keys()))
    bias_spec_entries = bias_keys_from_block[:, 1][np.where(bias_keys_from_block[:, 0] == 'bias_spec')]

    temp_bias_values = {bias: {} for bias in current_nuisance_params_list}
    for bias_entry_name in bias_spec_entries:
        parts = bias_entry_name.split("_")
        bias_name_key_original = parts[0]
        
        matched_key = None
        # Check direct match or case-insensitive match for keys like NP0 -> np0
        for p_key in current_nuisance_params_list:
            if p_key.lower() == bias_name_key_original.lower():
                matched_key = p_key
                break
            
        if matched_key:
            z_suffix_identifier = "_".join(parts[1:]) # e.g., "1", "2", or "z0.5"
            raw_value = block.get_double("bias_spec", bias_entry_name)
            if matched_key in ["c0", "c2", "c4"]:
                # Assuming h is ~constant or use specific bin's h if it can vary
                h_for_conversion = params_cosmo_all_bins["h"][0] 
                temp_bias_values[matched_key][z_suffix_identifier] = raw_value * h_for_conversion**2
            else:
                temp_bias_values[matched_key][z_suffix_identifier] = raw_value
        # else:
            # if feedback > 1: print(f"    Bias key {bias_name_key_original} from block not in expected list.")

    for nz, z_key_actual_float in enumerate(ordered_redshift_keys):
        z_key_actual_str = str(z_key_actual_float)
        # Attempt to match bias suffix (e.g., "z0.5" or "1")
        z_suffix_for_lookup = 'bin'+str(nz + 1) # e.g. 1, 2

        for bias_param_name in current_nuisance_params_list:
            found_value = temp_bias_values[bias_param_name][z_suffix_for_lookup]
            params_nuisance_all_bins[bias_param_name].append(found_value)

    if g2ExSet:
        params_nuisance_all_bins["g2"] = [g2f(b1) for b1 in params_nuisance_all_bins["b1"]]
    if g21CoEvol: # This should usually come after g2 might have been set by g2ExSet or from block
        params_nuisance_all_bins["g21"] = [g21f(b1, g2) for b1, g2 in zip(params_nuisance_all_bins["b1"], params_nuisance_all_bins["g2"])]

    internal_chi2_total = 0.0
    all_gcxi_data_parts = []
    all_gcxi_theory_parts = []
    all_gcxi_mask_parts = []

    for idx, current_z_key_float in enumerate(ordered_redshift_keys):
        dataset_processed_info = data_dict[current_z_key_float] # Access dict with the original float key
        
        if feedback > 0: print(f"\n  Calculating theory for z = {current_z_key_float}")

        # Retrieve pre-cut data, mask, rbands, windows from dataset_processed_info
        current_xi_data_cut_stacked = dataset_processed_info['xi_data']
        dataset_mask_for_joint_like = dataset_processed_info['mask_for_joint_like']
        rbands_current_bin_cut = dataset_processed_info['rbands']
        windows_current_bin_cut = dataset_processed_info['windows']
        cov_inv_internal_current_bin = dataset_processed_info['cov_inv']
        n_meas_current_bin = dataset_processed_info['n_meas']

        params_i = {} # Parameters for the current redshift bin's theory calculation
        for cosmo_key in params_cosmo_all_bins:
            params_i[cosmo_key] = params_cosmo_all_bins[cosmo_key][idx]
        for nuisance_key in current_nuisance_params_list:
            params_i[nuisance_key] = params_nuisance_all_bins[nuisance_key][idx]
        params_i["z"] = current_z_key_float
        
        if feedback > 1: print(f"    Parameters for theory: {params_i}")
        
        emu_current = dataset_processed_info['emu']
        if RSD_model == 'pk_vdg':
            if feedback > 1: print("    Performing FT for pk_vdg model")
            Xiell_theory = xiell_from_pk(emu_current, params_i, rbands_current_bin_cut, de_model=de_model)
        else:
            Xiell_theory = emu_current.Xiell(rbands_current_bin_cut, params_i, ell=[0, 2, 4], de_model=de_model)

    
        if dataset_processed_info['xitype'] == 'xiwed':
            xiwed = xiell_to_wed(Xiell_theory['ell0'], Xiell_theory['ell2'], Xiell_theory['ell4'])
            xi_theory_ar = np.vstack([xiwed[0], xiwed[1], xiwed[2]])
        elif dataset_processed_info['xitype'] == 'xiell':
            xi_theory_ar = np.vstack([Xiell_theory['ell0'], Xiell_theory['ell2'], Xiell_theory['ell4']])
        
        xi_theory_avg = np.matmul(xi_theory_ar, windows_current_bin_cut.T)
        if n_meas_current_bin == 3: current_xi_theory_stacked  = np.hstack([xi_theory_avg[0], xi_theory_avg[1], xi_theory_avg[2]])
        elif n_meas_current_bin == 2: current_xi_theory_stacked = np.hstack([xi_theory_avg[0], xi_theory_avg[1]]) #no hexadecapole

        all_gcxi_data_parts.append(current_xi_data_cut_stacked)
        all_gcxi_theory_parts.append(current_xi_theory_stacked)
        all_gcxi_mask_parts.append(dataset_mask_for_joint_like)

        diff_internal = current_xi_theory_stacked - current_xi_data_cut_stacked
        chi2_ds_internal = diff_internal @ cov_inv_internal_current_bin @ diff_internal
        internal_chi2_total += chi2_ds_internal
        
        if feedback > 0: print(f"    Internal Chi2 for z={current_z_key_float}: {chi2_ds_internal:.2f} (Ndata={len(current_xi_data_cut_stacked)})")
        block["data_vector_"+str(idx+1), 'chi2'] = chi2_ds_internal
        block["data_vector_"+str(idx+1), 'chi2red'] = chi2_ds_internal/len(current_xi_data_cut_stacked)

    final_gcxi_data_vector = np.concatenate(all_gcxi_data_parts)
    final_gcxi_theory_vector = np.concatenate(all_gcxi_theory_parts)
    final_gcxi_mask = np.concatenate(all_gcxi_mask_parts)

    block.put_double_array_1d("shear_plus_rsd_inputs", "data_vector_gcxi", final_gcxi_data_vector)
    #block.put_bool_array_1d("shear_plus_rsd_inputs", "mask_gcxi", final_gcxi_mask)
    block.put_int_array_1d("shear_plus_rsd_inputs", "mask_gcxi", final_gcxi_mask.astype(int))
    block.put_double_array_1d("shear_plus_rsd_inputs", "theory_vector_gcxi", final_gcxi_theory_vector)
    

    block["likelihoods", "gc_xi_internal_like"] = -0.5 * internal_chi2_total
    #trying to add derived sigma12 - shouldn't matter which datset emu to use, related to background cosmo only, so just use last dataset touched
    block["cosmological_parameters", "sigma12"] = (emu_current.cosmo.growth_factor(z=0)/emu_current.cosmo.growth_factor(params_i["z"])) * emu_current.params['s12']

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
    pell2transform_ell4 = pell4_spline(kk_hnkl)

        
    a0, b0 = get_pl_coefs(kk_hnkl[kk_hnkl<=kmax_theory][-2:], pell2transform_ell0[kk_hnkl<=kmax_theory][-2:])
    a2, b2 = get_pl_coefs(kk_hnkl[kk_hnkl<=kmax_theory][-2:], pell2transform_ell2[kk_hnkl<=kmax_theory][-2:])
    a4, b4 = get_pl_coefs(kk_hnkl[kk_hnkl<=kmax_theory][-2:], pell2transform_ell4[kk_hnkl<=kmax_theory][-2:])

        
    pell_input['ell0'] = np.hstack([pell2transform_ell0[kk_hnkl<=kmax_theory],a0*(k2ext**b0)])
    pell_input['ell2'] = np.hstack([pell2transform_ell2[kk_hnkl<=kmax_theory],a2*(k2ext**b2)])
    pell_input['ell4'] = np.hstack([pell2transform_ell4[kk_hnkl<=kmax_theory],a4*(k2ext**b4)])
        
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
