import pandas as pd
import numpy as np
import os
import obspy
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy import stats

## Import custom modules
import sys
module_path = '/staff/quentin/Documents/Projects/generalroutines/'
sys.path.append(module_path)
import plot_results_SPECFEM, launch_SPECFEM

from importlib import reload; reload(launch_SPECFEM)
def print_prop_simulation(simu_dir, file_pkl='simulation.pkl', simu_cols=['dx', 'z-min', 'z-max'], parfile_cols=['DT', 'STACEY_ABSORBING_CONDITIONS', 'PML_BOUNDARY_CONDITIONS'], source_cols=['zs', 'stf', 'f0', 'strike', 'dip', 'rake', 'Mnn', 'Mee', 'Mdd'], topo_cols=['add_topography'], ref_station_cols=['name']):
    
    one_pk_file  = simu_dir + '/' + file_pkl
    one_par_file = simu_dir + '/parfile_input'
    one_source_file = simu_dir + '/source_input'
    vel_file = simu_dir + '/velocity_model.txt'
    parameter_file = simu_dir + '/parameters.csv'
    one_par_file_OUTPUT = simu_dir + '/OUTPUT_FILES/Par_file'
    print(simu_dir)
    
    pd_simu = {'file': one_pk_file.split('/')[-2], 'dir': one_pk_file.split('/')[-3], 'data_available': False, 'sediment': False}
    good_data = False
    if os.path.isfile(one_pk_file):
        simulation = launch_SPECFEM.create_simulation({}, file_pkl=one_pk_file)
        try:
            simulation = launch_SPECFEM.create_simulation({}, file_pkl=one_pk_file)
            good_data = True
        except:
            pass
        
    if os.path.isfile(one_par_file_OUTPUT):
        pd_simu['data_available'] = True
        
    if os.path.isfile(vel_file):
        vel_model = pd.read_csv(vel_file, delim_whitespace=True, header=None)
        vel_model.columns = ['z', 'rho', 'vp', 'vs', 'Qp', 'Qs']
        #print(vel_model)
        if vel_model.vs.iloc[0] < 2e3 and vel_model.z.iloc[1] > 2e2:
            pd_simu['sediment'] = True
        
    #if os.path.isfile(parameter_file):
    #    params = pd.read_csv(vel_file, header=[0])
        
    if good_data:
        for key in parfile_cols:
            pd_simu[key] = simulation.input_parfile[key]
        for key in source_cols:
            #print(key, source_cols)
            #print(simulation.input_source)
            #print(simulation.input_source[key])
            #try:
            #    pd_simu[key] = simulation.input_source[key.lower()]
            #except:
            pd_simu[key] = simulation.input_source[key]
        for key in simu_cols:
            pd_simu[key] = simulation.options['simulation_domain'][key]
        
        if 'name' in ref_station_cols:
            pd_simu['ref_station_name'] = simulation.ref_station_name
        
        if 'add_topography' in topo_cols:
            pd_simu['topography'] = simulation.add_topography
            
        #if os.path.isfile(parameter_file):
        #    for key in params.columns:
        #        pd_simu[f'params_{key}]'] = params[key].iloc[0]
                
        
    elif os.path.isfile(one_par_file):
        params = launch_SPECFEM.load_params(one_par_file)
        source_params = launch_SPECFEM.load_params(one_source_file)
        for key in parfile_cols:
            pd_simu[key] = params.loc[key].value
            
        for key in source_cols:
            if key in source_params.index:
                pd_simu[key] = source_params.loc[key].value
            elif 'stf' in key:
                pd_simu[key] = 'external' if int(source_params.loc['time_function_type'].value) == 8 else 'Gaussian'
            else:
                pd_simu[key] = 'N/A'

        for key in simu_cols:
            if 'dx' in key:
                dx = (float(params.loc['xmax'].value.replace('d','')) - float(params.loc['xmin'].value.replace('d','')))/float(params.loc['nx'].value)
                pd_simu[key] = dx
            else:
                pd_simu[key] = 'N/A'
                
    return pd.DataFrame([pd_simu])

def read_files_patterns(dir, pattern='simulation_Kiruna', remove=[], simu_cols=['dx', 'z-min', 'z-max'], parfile_cols=['DT', 'STACEY_ABSORBING_CONDITIONS', 'PML_BOUNDARY_CONDITIONS'], source_cols=['zs', 'stf', 'f0', 'strike', 'dip', 'rake', 'Mnn', 'Mee', 'Mdd'], topo_cols=['add_topography'], ref_station_cols=['name']):
    
    params = {
        'simu_cols': simu_cols,
        'parfile_cols': parfile_cols,
        'source_cols': source_cols,
        'ref_station_cols': ref_station_cols,
        'topo_cols': topo_cols
    }
    
    print('Remove:', remove)
    
    all_simus = pd.DataFrame()
    for subdir, dirs, files in os.walk(dir):
        if pattern in subdir and not 'DATA' in subdir and not 'OUTPUT_FILES' in subdir:
            skip = False
            for to_remove in remove:
                if to_remove in subdir+'/':
                    skip = True
            if skip:
                continue
            pd_simu = print_prop_simulation(subdir, **params)
            all_simus = all_simus.append( pd_simu )
    all_simus.reset_index(drop=True, inplace=True)
    all_simus.sort_values(by='file', inplace=True)
    all_simus['no'] = all_simus['file'].str.split('_').str[-1]
    return all_simus

from scipy import signal
from scipy.fft import ifft, fft, fftfreq
def get_name_station_simu(file, id):
    return f'{file}-{id}'

def compute_3d(times, data, vel, r, alpha=0.5):
    
    dt = times[1]-times[0]
    fft_signal = fft(data)
    freq = fftfreq(data.size, dt)
    omega = 2.*np.pi*freq
    factor = np.sqrt(abs(omega)/(2.*np.pi*vel*r))*np.exp(1j*np.pi*np.sign(omega)/4.)
    return ifft(fft_signal*factor)*signal.windows.tukey(data.size, alpha=alpha)
#data_seismic[ind_time] = compute_3d(times[ind_time], data_seismic[ind_time], vel, r, alpha=alpha)

def prepare_trace_raw(file, starttime, freqmin=0.1, freqmax=1., t0=0., tmax=60., id=-1, scaling_factor=1., range_stat=-1, vel_max_stat=-1., full_scaling=True, offset_SPECFEM_time_for_stf=0., alpha=0.01, stretch_IS_and_seismic=0.7):
    print(file)
    tr_data = pd.read_csv(file, header=None, delim_whitespace=True)
    tr_data.columns = ['t', 'vz']
    #print(tr_data)
    
    #plt.figure()
    #plt.plot(tr_data.t, tr_data.vz)
    
    tr_data = tr_data.loc[(tr_data.t>=t0)&(tr_data.t<=tmax)]
    tr_data.loc[:,'t'] += abs(tr_data.t.min())
    
    dt = tr_data.t.iloc[1] - tr_data.t.iloc[0]
    tr = obspy.Trace()
    tr.data = tr_data.vz.values
    tr.data *= scaling_factor
    tr.stats.delta = dt
    
    tr.stats.station = get_name_station_simu(file.split('/')[-3], id)
    tr.stats.starttime = starttime-offset_SPECFEM_time_for_stf
    
    if abs(stretch_IS_and_seismic) > 0:
        isep = abs(tr.times()-20.).argmin()
        zeros = np.zeros((int(stretch_IS_and_seismic/dt),))
        new_data = np.r_[tr.data[:isep], zeros, tr.data[isep:]]
        tr.data = new_data
    
    tr.detrend()
    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=False)
    #tr.integrate()
    
    #print(full_scaling)
    
    if full_scaling:
        from scipy import signal
        
        if range_stat == -1: ## Default case for Fig 3
            
            tr.data *= np.sqrt(9200)*47
            
            r = 9200
            ind_IS = tr.times() > 20.
            ind_seismic = tr.times() <= 20.
            ind_time = tr.times() >= offset_SPECFEM_time_for_stf

            vel = 350.
            times = tr.times()
            data_IS = tr.data.copy()
            data_IS[ind_seismic] = 0.
            data_IS[ind_IS] = data_IS[ind_IS]*signal.windows.tukey(data_IS[ind_IS].size)
            data_IS[ind_time] = compute_3d(times[ind_time], data_IS[ind_time], vel, r, alpha=alpha)

            vel = 3500.
            times = tr.times()
            data_seismic = tr.data.copy()
            data_seismic[ind_IS] = 0.
            data_seismic[ind_seismic] = data_seismic[ind_seismic]*signal.windows.tukey(data_IS[ind_seismic].size)
            data_seismic[ind_time] = compute_3d(times[ind_time], data_seismic[ind_time], vel, r, alpha=alpha)
            
            tr.data = data_IS + data_seismic
        
        else:
            
            r = range_stat
            
            """
            vel_IS = 350.
            t_sep = 0.5*(range_stat/vel_max_stat + range_stat/vel_IS)
            
            ind_IS = tr.times() > t_sep
            ind_seismic = tr.times() <= t_sep
            ind_time = tr.times() >= offset_SPECFEM_time_for_stf

            vel = vel_IS
            times = tr.times()
            data_IS = tr.data.copy()
            data_IS[ind_seismic] = 0.
            data_IS[ind_IS] = data_IS[ind_IS]*signal.windows.tukey(data_IS[ind_IS].size)
            data_IS[ind_time] = compute_3d(times[ind_time], data_IS[ind_time], vel, r, alpha=alpha)

            vel = vel_max_stat
            times = tr.times()
            data_seismic = tr.data.copy()
            data_seismic[ind_IS] = 0.
            data_seismic[ind_seismic] = data_seismic[ind_seismic]*signal.windows.tukey(data_IS[ind_seismic].size)
            data_seismic[ind_time] = compute_3d(times[ind_time], data_seismic[ind_time], vel, r, alpha=alpha)
            """
            vel = vel_max_stat
            times = tr.times()
            data_seismic = tr.data.copy()
            data_seismic = data_seismic*signal.windows.tukey(data_seismic.size)
            data_seismic = compute_3d(times, data_seismic, vel, r, alpha=alpha)
        
            tr.data = data_seismic
    
    return tr

def get_tr_group_simus(all_simus, starttime, freqmin=0.1, freqmax=0.5, t0=0., tmax=60., f0s=[], depths=[], root_dir='/staff/quentin/Documents/Codes/specfem-dg/EXAMPLES/', full_scaling=True, range_stat=-1, vel_max_stat=-1, file_specfem='NO.KI.BXZ.semv', scaling_factor=1., offset_SPECFEM_time_for_stf=0., alpha=0.5, stretch_IS_and_seismic=0.):
    
    options_tr = {'freqmin':freqmin, 'freqmax':freqmax, 't0':t0, 'tmax':tmax, 'scaling_factor': scaling_factor, 'range_stat': range_stat, 'vel_max_stat': vel_max_stat, 'offset_SPECFEM_time_for_stf':offset_SPECFEM_time_for_stf, 'full_scaling': full_scaling, 'alpha': alpha, 'stretch_IS_and_seismic': stretch_IS_and_seismic}
    
    simus_to_plot = all_simus.copy()
    simus_to_plot = simus_to_plot.loc[simus_to_plot.data_available]
    if f0s:
        simus_to_plot = simus_to_plot.loc[simus_to_plot.f0.isin(f0s)]
    if depths:
        simus_to_plot = simus_to_plot.loc[simus_to_plot.zs.isin(depths)]
        #print(simus_to_plot)
        
    format_dir = '{root_dir}{dir_simu}/OUTPUT_FILES/{file_specfem}'
    #simus_to_plot['tr'] = 'N/A'
    g_simus = simus_to_plot.groupby('file')
    st = obspy.Stream()
    for dir_simu, one_simu in g_simus:
        one_file = format_dir.format(root_dir=root_dir, dir_simu=dir_simu, file_specfem=file_specfem)
        #print(one_file, simus_to_plot.loc[simus_to_plot.index==one_simu.iloc[0].name, 'tr'])
        st += prepare_trace_raw(one_file, starttime, id=one_simu.iloc[0].no, **options_tr)
        
        #plt.figure()
        #plt.plot(st[-1].times(), st[-1].data)
    
        #print(tr)
        #simus_to_plot.loc[simus_to_plot.index==one_simu.iloc[0].name, 'tr'] = tr
        
    return simus_to_plot, st
     
def load_data(file, freqmin=0.01, freqmax=0.25, zerophase=False):
    
    options_tr = {'freqmin':freqmin, 'freqmax':freqmax, 'zerophase':zerophase} 
    tr = obspy.read(file)[0]
    tr.detrend()
    tr.filter('bandpass', **options_tr)
    return tr

from matplotlib import patheffects
def plot_list_simus(simus_to_plot, st, i_plot, tr_data=None, tmin=-1, tmax=-1, ymax=-1, format_legend='zs: {:.1f} km', unknown_legend='zs', scale_legend=1e3, legend_names=[], normalize_data=False, ax=None, max_amp=0., offset_time=0.5, coef_offset=0.75, normalize_per_freq_band=False, show_data=True):
    
    
    path_effects = [patheffects.withStroke(linewidth=4, foreground="w")]
    
    simus_to_plot.sort_values(by=unknown_legend, inplace=True)
    cmap = sns.color_palette("flare", n_colors=simus_to_plot.shape[0])
    
    shift_color = 1
    cmap = sns.cubehelix_palette(n_colors=simus_to_plot.shape[0]+shift_color)
    
    new_figure = False
    offset_y = i_plot*max_amp*coef_offset
    freq_min, freq_max = simus_to_plot.freq_min.iloc[0], simus_to_plot.freq_max.iloc[0]
    if ax == None:
        new_figure = True
        fig, ax = plt.subplots(1, 1)
        
    for ii, (isimu, one_simu) in enumerate(simus_to_plot.iterrows()):
        
        tr = st.select(station=get_name_station_simu(one_simu.file, one_simu.no))[0]
        val_legend = one_simu[unknown_legend]
        try: 
            val_legend /= scale_legend
        except:
            pass
        legend = format_legend.format(val_legend)
        if legend_names:
            legend = legend_names[ii]
        offset_time = tr.stats.starttime - tr_data.stats.starttime
        norm_factor = 1.
        if normalize_per_freq_band:
            norm_factor = tr.data.max()
        ax.plot(tr.times()+offset_time, tr.data/norm_factor+offset_y, color=cmap[ii+shift_color], label=legend, zorder=5)
        print('------------')
        print(one_simu.file)
        print(legend)
        
    if (not tr_data == None) and show_data:
        #print(tr.data.max())
        #print(tr_data.data.max())
        data = tr_data.data*1
        if normalize_data:
                data *= tr.data.max()/tr_data.data.max()
        ax.plot(tr_data.times(), data+offset_y, label='data', color='tab:blue', linewidth=4, zorder=1, )
    
    ax.text(st[0].times().max()-8., offset_y+max_amp*(coef_offset/0.75)*0.1, '{freq_min:.2f}-{freq_max:.2f} Hz'.format(freq_min=freq_min, freq_max=freq_max), ha='right', path_effects=path_effects)
    #ax.text(st[0].times().max()-8., offset_y+max_amp, '{freq_min:.2f}-{freq_max:.2f} Hz'.format(freq_min=freq_min, freq_max=freq_max), ha='right', path_effects=path_effects)
    
    if i_plot == 0:
        line_x = [2, 2]
        line_y = [tr.data.min()+offset_y, tr.data.max()+offset_y]
        ax.plot(line_x, line_y, linewidth=3, color='black', path_effects=path_effects, zorder=1000)
        coef = 0.03
        line_y = [line_y[0]-coef, line_y[-1]+coef]
        ax.scatter(line_x, line_y, s=200, color='black', marker='_', path_effects=path_effects, zorder=1001)
        ax.text(line_x[0], line_y[0]-0.1, '{:.2f} Pa'.format(tr.data.max()-tr.data.min()), va='top', ha='left')
    
    if new_figure:
        ax.set_xlabel('Time since event (s)')
        ax.set_ylabel('Pressure (Pa)')
        xlims = [tr.times().min(), tr.times().max()]
        if tmin > -1:
            xlims[0] = tmin
        if tmax > -1:
            xlims[1] = tmax
        ax.set_xlim(xlims)
        if ymax > -1:
            ax.set_ylim([-ymax, ymax])
        ax.legend()
    elif i_plot == 0:
        l = ax.legend(ncol=3, bbox_to_anchor=(0.2, 1), loc='upper left', frameon=False)
        l.get_frame().set_linewidth(0.0)
        for one_legend in l.get_texts():
            one_legend.set_path_effects(path_effects)
        for one_legend in l.legendHandles:
            one_legend.set_path_effects(path_effects)
        
def get_envelope(signal):
    from scipy.signal import hilbert
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope
        
def get_one_ratio(tr, t_threshold):
    i_threshold = np.argmin(abs(tr.times()-t_threshold))
    env = get_envelope(tr.data)
    ratio = env[:i_threshold].max()/env[i_threshold:].max()
    
    return env, ratio
        
def get_ratios(simus_to_plot, st, t_threshold=20.):
    
    simus_to_plot['ratio'] = -1
    for ii, (isimu, one_simu) in enumerate(simus_to_plot.iterrows()):
        #print(st)
        #print(str(one_simu.no))
        station = get_name_station_simu(one_simu.file, one_simu.no)
        #print([tr.stats.station for tr in st])
        #print(station)
        tr = st.select(station=station)[0]
        env, ratio = get_one_ratio(tr, t_threshold)
        #plt.figure()
        #plt.plot(tr.times(), tr.data)
        #plt.plot(tr.times(), env)
        #plt.title(str(one_simu.zs))
        i_threshold = np.argmin(abs(tr.times()-t_threshold))
        simus_to_plot.loc[simus_to_plot.index==isimu, 'ratio'] = ratio
        simus_to_plot.loc[simus_to_plot.index==isimu, 'seismic_amp'] = env[:i_threshold].max()
        simus_to_plot.loc[simus_to_plot.index==isimu, 'acoustic_amp'] = env[i_threshold:].max()
        
def get_misfit(simus_to_plot, st, tr_data, type='l2'):
    
    simus_to_plot['misfit'] = -1
    for ii, (isimu, one_simu) in enumerate(simus_to_plot.iterrows()):
        
        #print(one_simu)
        station = get_name_station_simu(one_simu.file, one_simu.no)
        tr = st.select(station=station)[0]
        tr_data_trim = tr_data.trim(starttime=tr.stats.starttime, endtime=tr.stats.endtime)
        #tr_data_trim.plot()
        f = interpolate.interp1d(tr_data_trim.times(), tr_data_trim.data, kind='cubic', fill_value='extrapolate')
        #print(tr.times()[-1], tr_data.times()[-1])
        interp_data = f(tr.times())
        
        """
        offset = 0.
        Nt = int(offset/abs(tr.times()[1]-tr.times()[0]))
        
        if type == 'l1':
            misfit = np.mean(np.sum(abs(interp_data[Nt:]-tr.data[:-Nt])))
        if type == 'l2':
            misfit = np.mean(np.sqrt(np.sum(abs(interp_data[Nt:]-tr.data[:-Nt])**2)))
        else:
            misfit = np.corrcoef(tr.data[:-Nt], interp_data[Nt:])[0,1]
            #print(misfit)
        """
        if type == 'mse':
            misfit = np.mean(np.sum(abs(interp_data[:]-tr.data[:])))
        if type == 'rmse':
            misfit = np.sqrt(np.mean(np.sum(abs(interp_data[:]-tr.data[:])**2)))
        if type == 'mae':
            misfit = np.mean(abs(interp_data[:]-tr.data[:]))
        else:
            misfit = np.corrcoef(tr.data[:], interp_data[:])[0,1]
            #print(np.corrcoef(tr.data[:], interp_data[:]))
            #plt.figure()
            #plt.scatter(tr.data[:], interp_data[:])
            
        simus_to_plot.loc[simus_to_plot.index==isimu, 'misfit'] = misfit
        
#simus_to_plot_refined = simus_to_plot.loc[simus_to_plot.zs==-750]
#plot_list_simus(simus_to_plot_refined, st, tr_data=tr_data, tmin=0, tmax=-1, ymax=2.5e-5, format_legend='f0: {unknown:.2f} Hz', unknown_legend='f0')

def plot_timeseries(all_simus_to_plot_in, all_st, all_tr_data, all_tr_GCMT, all_tr_shifted, freq_bands, f0 = 0.75, topography=pd.DataFrame(), t_threshold = 20., zmin= -1050, zmax=-250, type_misfit='corr', one_plot_only=True, show_correlations=True, plot_GCMT=True, plot_shifted=True, format_legend='{:.2f} km', unknown_legend='zs', scale_legend=1e3, legend_names=[], filename='', depths_timeseries=[], offset_time=0., normalize_per_freq_band=False, coef_offset = 0.3, show_data=True):
    
    path_effects = [patheffects.withStroke(linewidth=4, foreground="w")]
    
    all_simus_to_plot = all_simus_to_plot_in.loc[(all_simus_to_plot_in.f0==f0)&(all_simus_to_plot_in.zs>=zmin)&(all_simus_to_plot_in.zs<=zmax)]
    #print(all_simus_to_plot)
    all_simus_to_plot_timeseries = all_simus_to_plot.copy()
    if depths_timeseries:
        all_simus_to_plot_timeseries = all_simus_to_plot.loc[all_simus_to_plot.zs.isin(depths_timeseries)]
    
    ax = None
    i_plot = -1
    max_amp = 0.
    if one_plot_only:
        if show_correlations:
            
            fig = plt.figure(figsize=(10,6))
            grid = fig.add_gridspec(2, 4,)
            
            w_plot = 3
            ax = fig.add_subplot(grid[:, :w_plot])
            ax_corr = fig.add_subplot(grid[0, w_plot:])
            ax_ratio = fig.add_subplot(grid[1, w_plot:])
        elif topography.shape[0]>0:
            h_topo = 1
            h_plot = 4
            fig = plt.figure(figsize=(8,5))
            grid = fig.add_gridspec(h_topo+h_plot, 1,)
            ax = fig.add_subplot(grid[h_topo:, :])
            ax_topo = fig.add_subplot(grid[:h_topo, :])
            
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8,5))
            
        for st in all_st:
            for tr in st:
                #print(tr.stats.station, max_amp)
                max_amp = max(max_amp, abs(tr.data).max())
                
    if topography.shape[0]>0:
        offset_max = 10
        x_station = 9.5*1e3
        z_station = topography.loc[abs(topography.x-x_station)<offset_max, 'z'].values[0]
        x_source = 0*1e3
        z_source = topography.loc[abs(topography.x-x_station)<offset_max, 'z'].values[0]
        z_source += all_simus_to_plot.zs.iloc[0]
        ax_topo.plot(topography.x/1e3, topography.z/1e3-z_station/1e3, linewidth=3., color='black', zorder=1)
        ax_topo.scatter(x_source/1e3, z_source/1e3-z_station/1e3, marker='*', s=250, color='yellow', edgecolor='black', zorder=10)
        ax_topo.scatter(x_station/1e3, z_station/1e3-z_station/1e3, marker='^', s=250, color='tab:green', edgecolor='black', zorder=10)
        ax_topo.text(x_station/1e3, -0.2+z_station/1e3-z_station/1e3, 'KRIS', color='tab:green', ha='center', va='top', zorder=15)
        ax_topo.set_xlim([topography.x.min()/1e3, topography.x.max()/1e3])
        ax_topo.set_ylim([-0.6, 0.45])
        ax_topo.set_ylabel('Altitude\nrel. KRIS(km)')
        ax_topo.set_xlabel('Range (km)')
        ax_topo.xaxis.set_label_position("top")
        ax_topo.xaxis.tick_top()
            
    cmap = sns.color_palette("flare", n_colors=len(freq_bands))
    #print('Plotting')
    for st, tr_data, tr_GCMT, tr_shifted, (freq_min, freq_max) in zip(all_st, all_tr_data, all_tr_GCMT, all_tr_shifted, freq_bands): 
        i_plot += 1
        simus_to_plot_refined = all_simus_to_plot.loc[(all_simus_to_plot.freq_min==freq_min)&(all_simus_to_plot.freq_max==freq_max)]
        all_simus_to_plot_timeseries_refined = all_simus_to_plot_timeseries.loc[(all_simus_to_plot_timeseries.freq_min==freq_min)&(all_simus_to_plot_timeseries.freq_max==freq_max)]
        #print(all_simus_to_plot_timeseries_refined)
        #continue
        #print(st)
        #print('Ratios...')
        #print(all_simus_to_plot)
        get_ratios(simus_to_plot_refined, st, t_threshold=t_threshold)
        #print(simus_to_plot_refined)
        get_misfit(simus_to_plot_refined, st, tr_data, type=type_misfit)
        #print(simus_to_plot_refined.misfit)

        for col in ['ratio', 'seismic_amp', 'acoustic_amp', 'misfit']:
            all_simus_to_plot.loc[all_simus_to_plot.index.isin(simus_to_plot_refined.index), col] = simus_to_plot_refined[col]

        #_, ratio_data = get_one_ratio(tr_celso, t_threshold)

        if one_plot_only and plot_GCMT and i_plot == 0:
            #ax.plot(tr_celso.times()+1, tr_celso.data*75/2, label='celso stf')
            offset_y = i_plot*max_amp*coef_offset
            offset_time = tr_GCMT.stats.starttime - tr_data.stats.starttime
            ax.plot(tr_GCMT.times()+offset_time, tr_GCMT.data+offset_y, label='GCMT', color='orange', zorder=2)
            
        if one_plot_only and plot_shifted :
            #ax.plot(tr_celso.times()+1, tr_celso.data*75/2, label='celso stf')
            offset_y = i_plot*max_amp*coef_offset
            offset_time = tr_shifted.stats.starttime - tr_data.stats.starttime
            ax.plot(tr_shifted.times()+offset_time, tr_shifted.data+offset_y, label='-1 km shift.', color='red', zorder=2)

        plot_list_simus(all_simus_to_plot_timeseries_refined, st, i_plot, tr_data=tr_data, tmin=0, tmax=-1, ymax=1e-1, format_legend=format_legend, unknown_legend=unknown_legend, scale_legend=scale_legend, legend_names=legend_names, normalize_data=False, ax=ax, max_amp=max_amp, offset_time=offset_time, coef_offset=coef_offset, normalize_per_freq_band=normalize_per_freq_band, show_data=show_data)
        
        if one_plot_only and show_correlations:
            #print(simus_to_plot_refined)
            simus_to_plot_refined.sort_values(by='zs', inplace=True)
            ax_corr.plot(simus_to_plot_refined.zs/1e3, simus_to_plot_refined.misfit, c=cmap[i_plot], marker='o', linestyle='-', label='{freq_min:.2f} - {freq_max:.2f} Hz'.format(freq_min=freq_min, freq_max=freq_max))
            ax_corr.yaxis.set_label_position("right")
            ax_corr.yaxis.tick_right()
            if type_misfit == 'corr':
                ax_corr.set_ylabel('Pearson correlation')
            elif type_misfit == 'mae':
                ax_corr.set_ylabel('MAE (Pa)')
            elif type_misfit == 'rmse':
                ax_corr.set_ylabel('RMSE')
            elif type_misfit == 'mse':
                ax_corr.set_ylabel('MSE')
            ax_corr.tick_params(axis='both', which='both', labelbottom=False, bottom=False)
            ax_corr.grid(alpha=0.4)
            #ax_corr.set_ylim([0.05, 0.9])
            #ax_corr.set_ylim([0.4, 1])

            ax_ratio.plot(simus_to_plot_refined.zs/1e3, simus_to_plot_refined.ratio, c=cmap[i_plot], marker='o', linestyle='-', label='{freq_min:.2f} - {freq_max:.2f} Hz'.format(freq_min=freq_min, freq_max=freq_max))
            ax_ratio.yaxis.set_label_position("right")
            ax_ratio.yaxis.tick_right()
            l = ax_ratio.legend(ncol=1, bbox_to_anchor=(0., 0.98), loc='upper left', frameon=False)
            l.get_frame().set_linewidth(0.0)
            for one_legend in l.get_texts():
                one_legend.set_path_effects(path_effects)
            for one_legend in l.legendHandles:
                one_legend.set_path_effects(path_effects)
            ax_ratio.set_ylabel('Seismic-to-acoustic\namplitude')
            ax_ratio.set_xlabel('Depth (km)')
            ax_ratio.grid(alpha=0.4)
            
        #if one_plot_only:
        #    #ax.plot(tr_celso.times()+1, tr_celso.data*75/2, label='celso stf')
        #    offset_y = i_plot*max_amp*0.5
        #    ax.plot(tr_GCMT.times()+1, tr_GCMT.data+offset_y, label='GCMT', color='orange')
        #    #plt.title('{freq_min:.2f} - {freq_max:.2f} Hz'.format(freq_min=freq_min, freq_max=freq_max))
        #    #plt.legend()

    if one_plot_only:
        ax.set_xlabel('Time since event (s)')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_xlim([st[0].times().min(), st[0].times().max()-4])
        #ax.set_yticks(ax.get_yticks()[1:4])
        ax.tick_params(axis='both', which='both', labelleft=False)
        ax.grid(alpha=0.3)
        ys = ax.get_ylim()
        coef_stretch_top = 0.9
        coef_stretch_bottom = 0.5
        #print(ys)
        ax.set_ylim([ys[0]*coef_stretch_bottom, ys[1]*coef_stretch_top])
        
    if topography.shape[0]>0:
        fig.align_ylabels([ax, ax_topo])
        
    if filename:
        fig.savefig(filename)