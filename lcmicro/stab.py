
"""
=== lcmicro ===

A Python library for nonlinear microscopy and polarimetry.

This module contains routines for signal stability analysis.

Some ideas are taken from Lukas' collection of MATLAB scripts developed while
being a part of the Barzda group at the University of Toronto in 2011-2017.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""



def read_stability_trace(file_name):
    if(isarray(file_name)):
        for ind in range(0, len(file_name)):
            T, I = read_stability_trace(file_name[ind])
            
            if(ind == 0):
                T_arr = [T]
                I_arr = [I]
            else:
                T_arr.append(T)
                I_arr.append(I)
                
        return [T_arr, I_arr]
            
    if(file_name.find('SigMon') != -1):
        D = read_big_file(file_name)
    elif(file_name.find('PowerLog')):
        D = read_starlab_file(file_name)
    else:
        print('Could not determine file type')
        return None
            
    #T = D[:,0] - D[0,0]
    T = D[:,0]
    I = D[:,1]
    
    # Handle overflows when time is reset to zero
    for i, t in enumerate(T):
        if t == 0 and i != 0:
            Tstep_pre_samples = T[i-10:i-1] - T[i-11:i-2]
            Tstep_post_samples = T[i:i+10] - T[i+1:i+11]
            
            if Tstep_pre_samples.std() > 1E-10:
                print("Pre samples spaced unequally")
            
            if Tstep_post_samples.std() > 1E-10:
                print("Post samples spaced unequally")
                
            T[i] = T[i-1] + Tstep_pre_samples.mean()*0.5 + Tstep_post_samples.mean()*0.5
            T[i+1:] = T[i+1:]+T[i]
        
    return [T - T[0], I]

def format_trace(T, I, T_ofs=None, T_scale=None,
                T_start=None, T_dur=None,
                baseline_corr=None, sub_mean_y=None):
    if(not isnone(T_ofs)):
        T = T + T_ofs

    if(not isnone(T_scale)):
        T = T*T_scale

    if(not isnone(T_start)):
        ind = np.nonzero(T > T_start)[0][0]
        T = T[ind:]
        I = I[ind:]
        
    if(not isnone(T_dur)):
        markers = np.nonzero(T > T[0] + T_dur)
        if(len(markers[0]) > 0):
            ind = markers[0][0]
            T = T[0:ind]
            I = I[0:ind]
        
    if(issmth(baseline_corr)):
        I = I/np.polyval(baseline_corr, T)
        
    if(sub_mean_y):
        I = I - I.mean()
        
    return [T, I]


def plot_stability_trace(Tr=None,
                       T=None, I=None,
                       reduce_data=True,
                       file_name=None,
                       descr=None,
                       X_markers=None, plot_norm_trace=None,
                       show_zero_line=True,
                       plot_exp_sd=True,
                       xlim=None, ylim=None,
                       trace_ylim=None, hist_ylim=None, nf_ylim=None,
                       data_type='Counts', title=None, show_x_label=True,
                       ax_trace=None, ax_hist=None,
                       ax_noisefac=None, ax_stats=None,
                       show_trace=True, show_hist=True,
                       show_noisefac=False, show_stats=False,
                       show_stab_stats=None,
                       **kwargs):
    """Plot a nice stability trace with data reduction and a hisgram.

    Show the envelope of all values in gray, the average trace in blue and the expected
    std. dev. bounds in red.
    TODO: This duplicates occurrence plot. Integrate the two.
    """
    try:
        if(file_name != None):
            [T, I] = read_stability_trace(file_name)
    
        if(not isnone(Tr)):
            [T, I] = Tr.GetTraceData(**kwargs)
            data_type = Tr.data_type
            title = Tr.title
            
            ylabel = Tr.Y_label
            
            if(isnone(plot_norm_trace)):
                plot_norm_trace = Tr.Y_norm
            
            if(plot_norm_trace):
                ylabel = ylabel + ', a.u.'
            else:
                ylabel = ylabel + ', ' + Tr.Y_units               
            
        else:
            [T, I] = format_trace(T, I, **kwargs)
            
        if(isnone(xlim)):
            xlim = [min(T), max(T)]
            
        if(isnone(plot_norm_trace)):
            plot_norm_trace = False
    
        if(isnone(ax_trace) and isnone(ax_hist) and isnone(ax_noisefac) and isnone(ax_stats)):
            if(show_noisefac == False and show_stats == False):
                num_grid_rows = 1
            else:
                num_grid_rows = 2
                
            if(show_hist == True):
                num_grid_cols = 3
            else:
                num_grid_cols = 2
                
            grid = plt.GridSpec(num_grid_rows, num_grid_cols, wspace=0.5, hspace=0.5)
            
            ax_trace = plt.subplot(grid[0,0:2])

            if(show_hist):
                ax_hist = plt.subplot(grid[0,2])
            
            if(show_noisefac):
                ax_noisefac = plt.subplot(grid[1,0:2])
                
            if(show_stats):
                ax_stats = plt.subplot(grid[1, 2])
                
        Im = I.mean()
        
        # Determine expected standard deviation of the signal due to noise
        if(data_type == 'd'):
            # For discrete data (e.g. photon counts) use shot-noise which is the
            # square root of the mean
            I_exp_sd = np.sqrt(Im)
        elif(data_type == 'c'):
            # For continuous data (e.g volts) assume a fixed noise level
            I_exp_sd = 0.01
            
        if(plot_norm_trace):
            I_p = I/Im
            Im_p = 1
            I_exp_sd_p = I_exp_sd/Im
        else:
            I_p = I
            Im_p = Im
            I_exp_sd_p = I_exp_sd
            
        if(isnone(ylim) and not isnone(Tr)):
            if(not isnone(Tr.ylim)):
                ylim = Tr.ylim
            
        if(not isnone(ylim)):
            trace_ylim = ylim
            hist_ylim = ylim
        
        if(show_trace):
            plt.axes(ax_trace)
            
            if(show_zero_line):
                plt.plot(xlim, [0, 0], color = get_colour("black"))
            
            [T2, I2, I2_sd] = plot_trace(T, I_p, marker = '.', reduce_data = reduce_data)
            if(plot_exp_sd):
                plt.plot([T2[0], T2[-1]], [Im_p-I_exp_sd_p, Im_p-I_exp_sd_p], color = get_colour("darkred"))
                plt.plot([T2[0], T2[-1]], [Im_p+I_exp_sd_p, Im_p+I_exp_sd_p], color = get_colour("darkred"))
            plt.ylim(trace_ylim)
            
            if(not isnone(title)):
                plt.title(title)
            else:
                plt.title("Stability trace")
                
            if(show_x_label):
                plt.xlabel("Time (s)")
                
            plt.ylabel(ylabel)
            
            if(not isnone(X_markers)):
                for xm in X_markers:
                    plt.plot([xm, xm], plt.ylim(), color = get_colour("darkgreen"))
                    
            if(not isnone(Tr)):
                if(not isnone(Tr.ref_val)):
                    plt.plot(plt.xlim(), [Tr.ref_val, Tr.ref_val], color = get_colour("gray"))
                    
            plt.xlim(xlim)
            
            if(show_stab_stats):
                # Print trace stability statistics in the upper left corner of the
                # plot axes
                
                s = ''
                
                # Add mean stability string, but only if the trace doesn't have
                # mean level substracton
                if(not isnone(Tr) and not Tr.sub_mean_y):
                    s = s + 'Mean = %.3f' %(I_p.mean())
                
                # Format mean and std.dev. strings
                if(not(s == '')):
                    s = s + ', '
                    
                s = s + 'sd = %.3f' %(I_p.std())
                
                # Add fractional stability string, but only if the trace doesn't
                # have mean level subtraction
                if(not isnone(Tr) and not Tr.sub_mean_y):
                    s = s + ', stab = %.3f' %(I_p.std()/I_p.mean())
    
                xlim = plt.xlim()
                ylim = plt.ylim()
                
                plt.text(xlim[0] + 0.02*(xlim[1]-xlim[0]),
                         ylim[1] - 0.02*(ylim[1]-ylim[0]),
                         s,
                         horizontalalignment='left', verticalalignment='top')
        
        if(show_hist):
            plt.axes(ax_hist)
            
            if(data_type == "Counts"):
                bins = range(int(round(I.min())), int(round(I.max())))
            elif(data_type == "Volts"):
                bins = np.arange(I.min(), I.max(), 0.00016)
                
            if(plot_norm_trace):
                bins = bins/Im
                plt.hist(I/Im, bins = bins, orientation="horizontal")
            else:
                plt.hist(I, bins = bins, orientation="horizontal")    
                
            plt.ylim(hist_ylim)
            
            
        if(show_noisefac):
            plt.axes(ax_noisefac)
            
            # Bin data into 1 second bins
            [Tb, Ib, Ib_sd] = reduce_trace(T, I, int((T[-1] - T[0])/1))
            
            if(data_type == "Counts"):
                nf = np.sqrt(Im)/Ib_sd
            elif(data_type == "Volts"):
                nf = 0.01/Ib_sd
                    
            nf_m = nf.mean()
            nf_std = nf.std()    
            
            plot_trace(Tb, nf, color = get_colour("darkblue"))
            plt.plot([Tb[0], Tb[-1]], [nf_m, nf_m], color = get_colour("darkred"))
            plt.plot([Tb[0], Tb[-1]], [1, 1], color = 'k')
            plt.ylim(nf_ylim)
            plt.xlabel("Time (s)")
            plt.ylabel("Noise factor")
        
        
        if(show_stats):
            plt.axes(ax_stats)
            plt.axis('off')
            
            if(descr == None and file_name != None):
                descr = get_head_val(read_cfg(file_name), "Scan Info", "Description")
                
            srate = 1/(T[1] - T[0])
                
            s = ''
            
            if(descr != None):
                s = s + descr + '\n'
                
            s = s + "Test duration: %.1f s\n" %(T[-1] - T[0]) + \
                    "Sample rate: %.1f kHz\n" %(srate/1E3)
                     
            if(data_type == "Counts"):
                s = s + "Mean count rate: %.3f Mcps\n" %(Im*srate/1E6)
            elif(data_type == "Volts"):
                s = s + "Mean signal: %.3f V\n" %(Im)
                
            s = s + "Noise factor bin size: %.3f s\n" %(Tb[1] - Tb[0]) + \
                    "Mean noise factor: %.3f ± %.3f\n" %(nf_m, nf_std) + \
                    "Min/max noise factor: %.3f, %.3f" %(nf.min(), nf.max())
                     
            plt.text(0, 1, s, verticalalignment='top')
            
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("Could not plot trace")
        
        
def compare_traces(Trs=None,
                  T_arr=None, I_arr=None,
                  file_names=None,
                  data_types=None, T_ofs_arr=None, T_scale_arr=None,
                  **kwargs):
    
    Tr = None
    file_name = None
    T = None
    I = None
    T_ofs = None
    T_scale = None
    data_type = None
    show_hist = True    
    
    for key, value in kwargs.items():
        if(key == 'show_hist'):
            show_hist = value
    
    if(not isnone(Trs)):
        numR = len(Trs)
    elif(not isnone(file_names)):
        numR = len(file_names)
    else:
        numR = len(T_arr)
        
    if(show_hist):
        grid = plt.GridSpec(numR, 3, wspace=0.5, hspace=0.5)
    else:
        grid = plt.GridSpec(numR, 2, wspace=0.5, hspace=0.5)
    
    for ind in range(0, numR):
        ax_trace = plt.subplot(grid[ind, 0:2])
        if(show_hist):
            ax_hist = plt.subplot(grid[ind, 2])
        else:
            ax_hist = None
            
        if(not isnone(Trs)):
            Tr = Trs[ind]
        
        if(not isnone(file_names)):
            file_name = file_names[ind]
            
        if(not isnone(T_arr)):
            T = T_arr[ind]
            
        if(not isnone(T_ofs_arr)):
            T_ofs = T_ofs_arr[ind]
            
        if(not isnone(T_scale_arr)):
            T_scale = T_scale_arr[ind]
            
        if(not isnone(I_arr)):
            I = I_arr[ind]
            
        if(not isnone(data_types)):
            data_type = data_types[ind]
            
        if(ind == numR-1):
            show_x_label = True
        else:
            show_x_label = False
            
        plot_stability_trace(Tr=Tr, file_name=file_name, T=T, I=I, T_ofs=T_ofs, T_scale=T_scale,
                           ax_trace=ax_trace,
                           ax_hist=ax_hist, data_type=data_type,
                           show_x_label=show_x_label,
                           **kwargs)        
        
def plot_comb_stab_traces(T=None, I=None, Tc=None, Ic=None, file_names=None, descr=None):
    T_splice = []
    if(isnone(Tc) and isnone(Ic)):
        numF = len(file_names)
        
        for indF in range(0, numF):
            file_name = file_names[indF]
            
            print("Reading file " + str(indF))
            [T, I] = read_stability_trace(file_name)
                    
            if(indF == 0):
                Tc = T
                Ic = I
            else:
                T_splice.append(Tc[-1])
                Tc = np.append(Tc, T + Tc[-1])
                Ic = np.append(Ic, I)
            
    plot_stability_trace(Tc, Ic, descr=descr, X_markers=T_splice)
    

       
def get_stab_meas_start_t_and_dur(t_ofs_ts, data_dir=None):
    """Get the start time and duration of a stability measurement.

    The measurement is stored in a TDMS file,measurement start and stop times
    are expected to be stored as timestamps in Timing.ini.
    """
    # Read sync times
    if(isnone(data_dir)):
        data_dir = r".\\"
        
    timing_file = data_dir + r"\Timing.ini"
    Cfg = read_cfg(timing_file)
    d = datetime.strptime(get_head_val(Cfg, 'Timing', 'Start Sync'), '%Y.%m.%d %H:%M:%S.%f')
    t_start_ts = time.mktime(d.timetuple())

    d = datetime.strptime(get_head_val(Cfg, 'Timing', 'End Sync'), '%Y.%m.%d %H:%M:%S.%f')
    t_end_ts = time.mktime(d.timetuple())

    # Read THG microscope signal, laser avg. power and avg. peak intensity traces
    [T, I, t_ofs_ts] = read_tdms_traces(data_dir)

    t_ofs_ts = time.mktime(t_ofs_ts.timetuple())

    # Timing.ini and TDMS timestamps have different timezone handling. Subtract
    # two hours from both values to align.
    # Due to the delay in opening and closing the shutter even when using
    # blocking calls the trace is skewed in time with respect to sync
    # timestamps. Cut 1 s from the beginning of the trace and 3 s from the end
    # to make sure the shutter sync wiggles stay outside of the the stability
    # measurement range.
    t_start = t_start_ts - t_ofs_ts - 7200 + 1
    t_dur = t_end_ts - t_ofs_ts - 7200 - t_start - 3
    
    return [t_start, t_dur]

def get_stab_traces(dir_name, crop_t=True, t_start=None, t_dur=None, scaled3=False, **kwargs):
    # Read thg microscope signal, laser avg. power and avg. peak intensity traces
    [T, A, tdms_ofs_ts] = read_tdms_traces(dir_name)

    [t_start_ts, t_dur_ts] = get_stab_meas_start_t_and_dur(tdms_ofs_ts, DataDir=dir_name)
    
    if(isnone(t_start)):
        t_start = t_start_ts
        
    if(isnone(t_dur)):
        t_dur = t_dur_ts
        
    thg = Trace(T=T[0], Y=A[0], Y_label='Counts', title='THG', data_type='d')
    
    if(scaled3):
        P_avg = Trace(T=T[0], Y=A[1]**3, Y_label='Volts', title='Avg. Power Ref. ^3', data_type='c')
        I_avg = Trace(T=T[0], Y=A[2]**3/2, Y_label='Volts', title='SHG Ref. ^3/2', data_type='c')
    else:
        P_avg = Trace(T=T[0], Y=A[1], Y_label='Volts', title='Avg. Power Ref.', data_type='c')
        I_avg = Trace(T=T[0], Y=A[2], Y_label='Volts', title='SHG Ref.', data_type='c')

    if(crop_t):
        trace_set_param([thg, P_avg, I_avg], T_ofs=0, t_start=t_start, t_dur=t_dur)
    
    return [thg, P_avg, I_avg]