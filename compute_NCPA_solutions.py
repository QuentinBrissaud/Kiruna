#!/usr/bin/env python3

#################################
## Import event analyzer routines
import sys
module_path = '/staff/quentin/Documents/Projects/generalroutines/'
sys.path.append(module_path)
module_path = '/staff/quentin/Documents/Projects/ML_attenuation_prediction/'
sys.path.append(module_path)
import construct_atmospheric_model, generate_Gardner_perturbations

from obspy.core.utcdatetime import UTCDateTime
import pandas as pd
from pdb import set_trace as bp
import matplotlib.pyplot as plt
import numpy as np
import os
from pyproj import Geod
from scipy import interpolate
import inspect 
import obspy 

def get_azimuths(options):
    dbaz = -1 if len(options['azimuths']) == 1 else options['azimuths'][1] - options['azimuths'][0]
    return str(options['azimuths'][0]), str(options['azimuths'][-1]), str(dbaz)

def run_NCPA_TL(simulation_dir, dir_NCPA_bin, program, az_in, freq, profile_name, maxheight_km, sourceheight_km, maxrange_km):

    """
    Run a ModBB simulation
    """
    
    az = az_in if az_in >= 0 else az_in+360.
    
    data_in = { 
        'DIR': dir_NCPA_bin,
        'PROGRAM': program,
        'AZIMUTH': '--azimuth {az:.1f}'.format(az=az),
        'FREQ': '--freq {freq:.4f}'.format(freq=freq),
        'PROFILES': '--atmosfile {profile_name}'.format(profile_name=profile_name),
        'MAX_ALT': '--maxheight_km {maxheight_km:.2f}'.format(maxheight_km=maxheight_km),
        'SOURCE_ALTITUDE': '--sourceheight_km {sourceheight_km:.2f}'.format(sourceheight_km=sourceheight_km),
        'MAX_RANGE': '--maxrange_km {maxrange_km:.2f}'.format(maxrange_km=maxrange_km),
        }

    template_TL = """
               {DIR}{PROGRAM} --singleprop --starter self \
                             {AZIMUTH} \
                             {FREQ} \
                             {PROFILES} \
                             {MAX_ALT} \
                             {SOURCE_ALTITUDE} \
                             {MAX_RANGE} \
                             --write_atm_profile \
                             --write_2d_tloss \
                             --npade 8
               """
    
    ## Run program
    os.chdir(simulation_dir)
    #if not os.path.exists(dispersion_file):
    os.system(template_TL.format(**data_in))
    
    #file = simulation_dir + 

def run_NCPA_dispersion(simulation_dir, dir_NCPA_bin, program, method, dispersion_file, az_in, f_min, f_max, f_step, profile_name, maxheight_km, sourceheight_km, maxrange_km, output_waveform_file, source_file, recompute_dispersion=True):

    """
    Run a ModBB simulation
    """
    
    az = az_in if az_in >= 0 else az_in+360.
    
    data_in = { 
        'DIR': dir_NCPA_bin,
        'PROGRAM': program,
        'METHOD': '--method {method}'.format(method=method),
        'DISPERSION_FILE': '--dispersion_file {dispersion_file}'.format(dispersion_file=dispersion_file),
        'AZIMUTH': '--azimuth {az:.1f}'.format(az=az),
        'FREQ': '--f_min {f_min:.4f} --f_max {f_max:.4f} --f_step {f_step:.4f}'.format(f_min=f_min, f_max=f_max, f_step=f_step),
        'PROFILES': '--atmosfile {profile_name}'.format(profile_name=profile_name),
        'MAX_ALT': '--maxheight_km {maxheight_km:.2f}'.format(maxheight_km=maxheight_km),
        'SOURCE_ALTITUDE': '--sourceheight_km {sourceheight_km:.2f}'.format(sourceheight_km=sourceheight_km),
        'MAX_RANGE': '--maxrange_km {maxrange_km:.2f}'.format(maxrange_km=maxrange_km),
        'INPUT_DISPERSION_FILE': '--input_dispersion_file {dispersion_file}'.format(dispersion_file=dispersion_file),
        'WAVEFORM_FILE': '--output_waveform_file {output_waveform_file}'.format(output_waveform_file=output_waveform_file),
        'SOURCE': '--source waveform --source_file {source_file}'.format(source_file=source_file),
        'RANGE': '--range_km {range_km:.2f}'.format(range_km=maxrange_km)
        }

    template_dispersion = """
               {DIR}{PROGRAM} --dispersion {METHOD} \
                             {DISPERSION_FILE} \
                             {AZIMUTH} \
                             {FREQ} \
                             {PROFILES} \
                             {MAX_ALT} \
                             {SOURCE_ALTITUDE} \
                             {MAX_RANGE} \
                             --write_atm_profile
               """

    template_propagation = """
               {DIR}{PROGRAM} --propagation {INPUT_DISPERSION_FILE} \
                             {WAVEFORM_FILE} \
                             {SOURCE} \
                             {RANGE}
               """
    
    bp()
    
    ## Run program
    os.chdir(simulation_dir)
    if not os.path.exists(dispersion_file) or recompute_dispersion:
        os.system(template_dispersion.format(**data_in))
    
    #st=obspy.read('/staff/quentin/Documents/Projects/Kiruna/waveforms/XX.KI1.00.BDF.D.2020.139_15_70.SAC'); st.taper(max_percentage=0.2); KIR_data = pd.DataFrame(); KIR_data['time'] = st[0].times(); KIR_data['data'] = st[0].data; KIR_data.to_csv('stf.dat', header=False, index=False, sep=' ')
    
    #template_dispersion = '{DIR}{PROGRAM} --dispersion --dispersion_file dispersion_file_alexis.dat --atmosfile profile_alexis.dat --azimuth 335.2 --method modess --f_min 0.002 --f_step 0.002 --f_max 1.5 --write_atm_profile'
    
    #template_propagation='{DIR}{PROGRAM} --propagation --input_dispersion_file dispersion_file.dat --range_km 270 --output_waveform_file waveform.dat --source pulse1 --max_celerity 320'
    #template_propagation='{DIR}{PROGRAM} --propagation --input_dispersion_file dispersion_file_alexis.dat --range_km 151 --output_waveform_file waveform.dat --source pulse1'
    bp()
    os.system('cp dispersion_file.dat dispersion_file_profile.dat')
    os.system(template_propagation.format(**data_in))
    #data=pd.read_csv('./waveform.dat', header=None, delim_whitespace=True); data.columns=['r', 't', 'p']; plt.plot(data.t, data.p*1e4); plt.show()
    #data=pd.read_csv('./stf.dat', header=None, delim_whitespace=True); data.columns=['t', 'p']; plt.plot(data.t, data.p); plt.show()
    
    from scipy import signal
    nperseg=10; noverlap=9; nfft=100; tr=st[0]; f, t, Sxx = signal.spectrogram(tr.data, 1./tr.stats.delta, nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='spectrum'); fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True); axs[0].pcolormesh(t, f, Sxx, shading='auto', cmap='cividis', vmax=np.quantile(Sxx, q=0.99)); axs[1].plot(tr.times(), tr.data); plt.show()
    
    nperseg=10; noverlap=9; nfft=100; f, t, Sxx = signal.spectrogram(tr.data, 1./tr.stats.delta, nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='spectrum'); fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True); axs[0].pcolormesh(t+data.t.min(), f, Sxx, shading='auto', cmap='cividis', vmax=np.quantile(Sxx, q=0.99)); axs[1].plot(tr.times()+data.t.min(), tr.data); plt.show()
    
    tr = obspy.Trace(); tr.data = data.p.values; tr.stats.delta = data.t.iloc[1]-data.t.iloc[0]; tr.stats.station='I37'; tr.filter('lowpass', freq=0.3)
    nperseg=10; noverlap=9; nfft=100; f, t, Sxx = signal.spectrogram(data.p.values, 1./(data.t.iloc[1]-data.t.iloc[0]), nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='spectrum'); fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True); axs[0].pcolormesh(t+data.t.min(), f, Sxx, shading='auto', cmap='cividis', vmax=np.quantile(Sxx, q=0.99)); axs[1].plot(data.t.values, data.p.values); plt.show()
    bp()
        
def compute_atmospheric_conditions(time, source, station, max_height, number_altitudes, nc_file, number_points = 1):
    
    """
    Retrieve atmospheric parameters from netcdf files
    """

    one_atmos_model = construct_atmospheric_model.atmos_model(source, station, time, max_height, nc_file)
    print('Before one_atmos_model.construct_profiles')
    one_atmos_model.construct_profiles(number_points, number_altitudes, 
                                       type='specfem', range_dependent=False, 
                                       projection=False, ref_projection={})
    atmos_model = one_atmos_model.updated_model
    
    return atmos_model

def get_header(add_W=True):

    """
    Get NCPA atmospheric profile header
    """
    
    if add_W:
        header = """#% 0, Z0, m, 0.0
        #% 1, Z, km
        #% 2, U, m/s
        #% 3, V, m/s
        #% 4, W, m/s
        #% 5, T, degK
        #% 6, RHO, g/cm3
        #% 7, P, mbar"""
    else:
        header = """#% 0, Z0, m, 0.0
        #% 1, Z, km
        #% 2, U, m/s
        #% 3, V, m/s
        #% 4, T, degK
        #% 5, RHO, g/cm3
        #% 6, P, mbar"""

    return header

def construct_one_profile(time, source, station, max_height, number_altitudes, nc_file, add_top_row=False, remove_winds=False):
    
    """
    Build a single vertical profiles using NCPA format
    """

    print('Before atmos conditions')

    ## Collect atmospheric data from NETCDF file
    profiles = compute_atmospheric_conditions(time, source, station, max_height, number_altitudes, nc_file, number_points = 1)
    
    print('After atmos conditions')
    
    ## Construct NCPA profile
    profiles['w'] = profiles['u'] * 0.
    #profiles['v'] = profiles['u'] * 0.
    profiles_NCPA = profiles[['z', 'u', 'v', 'wx', 't', 'rho', 'p']]
    
    #profiles_NCPA_ = pd.read_csv('/staff/quentin/Documents/Projects/ML_attenuation_prediction/profile_ncpa.dat', delim_whitespace=True, header=None, skiprows=[key for key in range(8)])
    #profiles_NCPA_.columns = ['z', 'u', 'v', 'wx', 't', 'rho', 'p']
    vars = []#['u', 'v', 'w', 'T', 'rho0', 'p']
    for var in vars:
        f = interpolate.interp1d(profiles_NCPA['z'], profiles_NCPA[var], kind='cubic')
        profiles_NCPA[var] = f(profiles_NCPA['z'].values)
    
    ## Scale variables to right unit
    profiles_NCPA.loc[:, 'rho'] /= 1e3
    profiles_NCPA.loc[:, 'z'] /= 1e3
    profiles_NCPA.loc[:, 'p'] /= 1e2
    
    ## If winds should be put to zero
    if remove_winds:
        profiles_NCPA['u'] = 0.
        profiles_NCPA['v'] = 0.
        profiles_NCPA['wx'] = 0.

    if add_top_row:
        top_row = profiles_NCPA.iloc[:1].copy()
        top_row['z'] = 0.
        top_row['u'] = 0.
        top_row['v'] = 0.
        
        ## Concat with old DataFrame and reset the Index.
        profiles_NCPA = pd.concat([top_row, profiles_NCPA]).reset_index(drop = True)

    return profiles_NCPA

def create_profile_H(simulation_dir, time, source_loc, station_loc, max_height, number_altitudes, nc_file, stds, add_gardner=True, altitude_levels=[84, 70, 45, 21]):

    """
    Create 1d atmospheric profile for NCPA
    columns should be Z, U, V, T, RHO, P, C0, Z0
    Z - altitude above MSL
    U - West-to-East wind speed
    V - South-to-North wind speed
    T - temperature
    RHO - density
    P - ambient pressure
    C0 - static sound speed (optional, will be calculated if not provided)
    Z0 - ground elevation (scalar, optional)
    """
    
    file_profile = simulation_dir + 'profile.dat'
    profiles_NCPA = construct_one_profile(time, source_loc, station_loc, max_height, number_altitudes, nc_file, add_top_row=False, remove_winds=False)
    
    ## Add Gardner perturbations to projected wind
    if add_gardner:
    
        Gardner_profile = profiles_NCPA.copy()
        Gardner_profile['u'] = 0.
        Gardner_profile['v'] = 0.
        generate_Gardner_perturbations.add_perturbations_to_profile(Gardner_profile, stds, source_loc, station_loc, altitude_levels=altitude_levels)
    
        profiles_NCPA['u'] += Gardner_profile['u']
        profiles_NCPA['v'] += Gardner_profile['v']
    
    header = get_header()
    with open(file_profile, 'w') as file: 
        file.write(inspect.cleandoc(header) + '\n'); 
        profiles_NCPA.to_csv(file, sep=' ', header=False, index=False);
    
    return file_profile

def perform_one_Gardner_alteration(path_atmos_model, ref_atmos_for_alter_gardner, alter_gardner):

    profile = pd.read_csv(path_atmos_model, skiprows=8, header=None, delim_whitespace=True)
    profile.columns = ['z', 'u', 'v', 'w', 't', 'rho', 'p']
    
    #ref_atmos_for_alter_gardner = '/adhocdata/infrasound/2022_Kiruna/ModBB_simulations/ATMOS_ARCI_NOPERT_10Hz/profile_alexis.dat'
    profile_ref = pd.read_csv(ref_atmos_for_alter_gardner, skiprows=8, header=None, delim_whitespace=True)
    profile_ref.columns = ['z', 'u', 'v', 'w', 't', 'rho', 'p']
    
    profile_new = profile_ref.copy()
    profile_new['u'] += (profile.u-profile_ref.u)*(100-alter_gardner)/100
    profile_new['v'] += (profile.v-profile_ref.v)*(100-alter_gardner)/100
    
    header = get_header()
    with open(path_atmos_model, 'w') as file: 
        file.write(inspect.cleandoc(header) + '\n'); 
        profile_new.to_csv(file, sep=' ', header=False, index=False);

def run_one_simulation(simulation_dir, dir_NCPA_bin, program, method, f_min, f_max, f_step, maxheight_km, sourceheight_km, stf_file, maxrange_km, path_atmos_model='', time=UTCDateTime(2020, 5, 18, 1, 11, 57), source_loc=(67.8476174327105, 20.20124295920305), station_loc=(69.07408, 18.60763), max_height=120., number_altitudes=1000, nc_file='', add_gardner=True, stds=[25., 18., 10., 5.], altitude_levels=[84, 70, 45, 21], recompute_dispersion=True, compute_TL=True, freq_TL=1.5, alter_gardner=0, ref_atmos_for_alter_gardner=''):
    
    """
    ## Making sure that the simulation directory is available
    if os.path.exists(simulation_dir):
        print('Using existing simulation directory: ', simulation_dir)
    else:
        os.mkdir(simulation_dir)
        print('Creating simulation directory: ', simulation_dir)
    """
    
    ## Determine azimuth and source-receiver distance
    wgs84_geod = Geod(ellps='WGS84')
    az, baz, maxrange_km = wgs84_geod.inv(source_loc[1], source_loc[0], station_loc[1], station_loc[0])
    maxrange_km /= 1e3

    ## Build atmospheric model if needed
    if not path_atmos_model:
        path_atmos_model = create_profile_H(simulation_dir, time, source_loc, station_loc, maxheight_km, number_altitudes, nc_file, stds, add_gardner=add_gardner, altitude_levels=altitude_levels)
    
    ## Altering gardner perturbations if needed
    if alter_gardner > 0:
        perform_one_Gardner_alteration(path_atmos_model, ref_atmos_for_alter_gardner, alter_gardner)
        #plt.plot(profile.u, profile_ref.z); plt.plot(profile_new.u, profile_ref.z); plt.show()
    
    ## Compute NCPA numerical solution
    dispersion_file = 'dispersion_file.dat'
    output_waveform_file = 'waveform.dat'
    profile_name = path_atmos_model.split('/')[-1]
    source_file = stf_file.split('/')[-1]
    
    if compute_TL:
        #freq_TL = 1.5
        program_TL = 'ePape'
        maxrange_km_TL = 300.
        bp()
        run_NCPA_TL(simulation_dir, dir_NCPA_bin, program_TL, az, freq_TL, profile_name, maxheight_km, sourceheight_km, maxrange_km_TL)
        
    run_NCPA_dispersion(simulation_dir, dir_NCPA_bin, program, method, dispersion_file, az, f_min, f_max, f_step, profile_name, maxheight_km, sourceheight_km, maxrange_km, output_waveform_file, source_file, recompute_dispersion=recompute_dispersion)
      
def convert_Kiruna_profiles_Alexis(file_alexis, output_file):
    
    """
    Read atmospheric profiles provided by Alexis Le Pichon on 30/04/2021
    """
    
    model = pd.read_csv(file_alexis, delim_whitespace=True, header=None)
    model.columns = ['z', 'u', 'v', 'w', 't', 'rho', 'p']
    
    header = get_header()
    with open(output_file, 'w') as file: 
        file.write(inspect.cleandoc(header) + '\n'); 
        model.to_csv(file, sep=' ', header=False, index=False);
  
def create_stf_from_KRIS(output_file, KRIS_file='/staff/quentin/Documents/Projects/Kiruna/waveforms/XX.KI1.00.BDF.D.2020.139_15_70.SAC', max_percentage=0.2):

    st = obspy.read(KRIS_file)
    st.taper(max_percentage=max_percentage); 
    KIR_data = pd.DataFrame(); 
    KIR_data['time'] = st[0].times(); 
    KIR_data['data'] = st[0].data; 
    KIR_data.to_csv(output_file, header=False, index=False, sep=' ')
  
######################################################
if __name__ == '__main__':
    
    options = {
        'simulation_dir': '/adhocdata/infrasound/2022_Kiruna/ModBB_simulations/', 
        'dir_NCPA_bin': '/staff/quentin/Documents/Codes/ncpaprop-release/bin/', 
        'program': 'ModBB', 
        'method': 'modess', 
        'f_min': 0.002, 
        'f_max': 10., 
        'f_step': 0.002, 
        'maxheight_km': 120., 
        'sourceheight_km': 0.,
        'stf_file': './stf.dat', 
        'maxrange_km': 200.,   
        'path_atmos_model': '', 
        'time': UTCDateTime(2020, 5, 18, 1, 11, 57), 
        'source_loc': (67.8476174327105, 20.20124295920305),
        #'station_loc': (69.07408, 18.60763),  # I37
        'station_loc': (69.53, 25.51),  # ARCES
        'number_altitudes': 1000, 
        'nc_file': '/staff/quentin/Documents/Projects/Kiruna/atmos_model/model_ERA5-full_2020-05-18_00.00.00_71.0_15.0_67.0_30.0.nc',
        'add_gardner': True, 
        'stds': [25., 18., 10., 5.], 
        'altitude_levels': [84, 70, 45, 21],
        'recompute_dispersion': True,
        'compute_TL': False,
        'freq_TL': 0.5,
        'alter_gardner': 0, # in % of amplitude of gardner
        'ref_atmos_for_alter_gardner': ''
    }
    
    """
    options['station_loc'] = (69.53, 25.51) # ARCES
    name_model = 'ATMOS_ARCI_PERT1D_ID1'
    alter_gardner = 20
    simulation_dir = options['simulation_dir']
    options['simulation_dir'] += '{}/'.format(name_model+f'_10Hz_{alter_gardner}pct')
    options['alter_gardner'] = alter_gardner
    ref_atmos_folder = 'ATMOS_ARCI_NOPERT_10Hz'
    options['ref_atmos_for_alter_gardner'] = f'{simulation_dir}{ref_atmos_folder}/profile_alexis.dat'
    """
    options['station_loc'] = (69.07408, 18.60763) # I37
    name_model = 'ATMOS_I37NO_PERT1D_ID4' 
    alter_gardner = 20
    simulation_dir = options['simulation_dir']
    options['simulation_dir'] += '{}/'.format(name_model+f'_10Hz_{alter_gardner}pct')
    options['alter_gardner'] = alter_gardner
    ref_atmos_folder = 'ATMOS_I37NO_NOPERT_10Hz'
    options['ref_atmos_for_alter_gardner'] = f'{simulation_dir}{ref_atmos_folder}/profile_alexis.dat'
    
    
    #options['station_loc'] = (69.53, 25.51) # ARCES
    #name_model = 'ATMOS_ARCI_PERT1D_ID6' 
    #options['simulation_dir'] += '{}/'.format(name_model+'_10Hz')
    #options['f_max'] = 10. # 7.5/8.1 if ID5/ID6 because crash
    
    #options['station_loc'] = (69.07408, 18.60763) # I37
    #name_model = 'ATMOS_I37NO_PERT1D_ID4' 
    #options['simulation_dir'] += '{}/'.format(name_model+'_10Hz')
    
    #options['station_loc'] = (66.01656687381089, 15.634786561724878 ) # Opposite to ARCES
    #name_model = 'ATMOS_ARCI_PERT1D_ID1' 
    #options['simulation_dir'] += '{}/'.format(name_model+'_10Hz_opposite')
    
    #options['station_loc'] = (67.8089186117827, 16.64192668233326) # West of Kiruna
    #name_model = 'ATMOS_ARCI_PERT1D_ID1' 
    #options['simulation_dir'] += '{}/'.format(name_model+'_10Hz_west')
    
    if not os.path.exists(options['simulation_dir']):
        os.mkdir(options['simulation_dir'])

    output_file = '{}profile_alexis.dat'.format(options['simulation_dir'])
    file_alexis = '/staff/quentin/Documents/Projects/Kiruna/atmos_model/{}.dat'.format(name_model)
    convert_Kiruna_profiles_Alexis(file_alexis, output_file)
    options['path_atmos_model'] = output_file
    #options['path_atmos_model'] = ''
    
    options['stf_file'] = '{}stf.dat'.format(options['simulation_dir'])
    create_stf_from_KRIS(options['stf_file'], KRIS_file='/staff/quentin/Documents/Projects/Kiruna/waveforms/XX.KI1.00.BDF.D.2020.139_15_70.SAC', max_percentage=0.2)
    
    #pd_stf = pd.DataFrame(); df = 1.; pd_stf['times'] = np.arange(0., 700., 1./(2*df)); t0, std = 4., 1.; pd_stf['stf'] = np.exp(-((pd_stf['times']-t0)/std)**2); pd_stf.to_csv('./stf.dat', header=False, index=False, sep=' ')
    
    #pd_stf = pd.DataFrame(); df = 2; pd_stf['times'] = np.arange(0., 700., 1./(2*df)); t0, std = 15., 1.; pd_stf['stf'] = -(2/std)*((pd_stf['times']-t0)/std)*np.exp(-((pd_stf['times']-t0)/std)**2); pd_stf.to_csv('std.data', header=False, index=False, sep=' ')
    
    #pd_profile = pd.read_csv('./profile.dat', header=None, skiprows=8, delim_whitespace=True); pd_profile_NCPA = pd.read_csv('./profile_NCPA.dat', header=None, skiprows=8, delim_whitespace=True)
    #plt.plot(pd_profile[1], pd_profile[0]); plt.plot(pd_profile_NCPA[1], pd_profile_NCPA[0]); plt.show()

    run_one_simulation(**options)
    bp()