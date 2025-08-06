#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:45:17 2022

@author: beauz
"""
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # suppress the df.append() warnings

import os
import numpy as np
import pandas as pd  # dataframes reign supreme
import copy  # for deepcopy -- the only method of copying that didn't affect the og
# import math
import matplotlib.pyplot as plt

from . import general_tools
from scipy import optimize
import scipy.stats as st

custom_tools_logger = general_tools.get_logger(__name__)


def get_filenames(path, csv_prefix, edf_prefix, session):
    """
    Just grabs the name of the .csv file without having to change it to fit the other get name function. Note that, different from JB's functions,
    the path here takes the function to the directory within each subject's named subfolder.
    Args:
        path (string): absolute path to the folder where the event file and other related files are: the event timing will go into the 'events' subfolder 
        csv_prefix (string): name of the experiment as defined in PsychoPy code
        edf_prefix (string): name of the data output files in EyeLink
        session (string): name of the condition

    Returns:
        behavioral_filename (string): name of the csv file from PsychoPy

    """

    csv_filenames = [file for file in os.listdir(path) if csv_prefix in file and session in file]
    edf_filenames = [file for file in os.listdir(path) if edf_prefix in file and session[0] in file]

    return [csv_filenames, edf_filenames]


def linear_function(x, a, b):
    return a * x + b


def sine_function(t, A, w, p, c):
    return A * np.sin(w * t + p) + c


def fit_sin(tt, yy):
    """
    Fit a sine wave function to the input time series data without manual parameters
    Args:
        tt: uncollated time points
        yy: uncollated horizontal gaze position data

    Returns:

    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    # guess_freq = 1/2500
    guess_amp = np.std(yy) * 3. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    popt, pcov = optimize.curve_fit(sine_function, tt, yy, p0=guess)

    A, w, p, c = popt
    f = w / (2. * np.pi)
    fit_function = lambda t: A * np.sin(w * t + p) + c
    
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fit_func": fit_function, "max_cov": np.max(pcov),
            "raw_res": (guess, popt, pcov)}


def customize_subplots(subplot_object, lw, size):
    """
    Adds some very specific customized settings to plots such as color, font size, legends handles, etc. This function does not actually plot, just adds flair.
    Arguments:
        subplot_object: any Axes subplot object
        lw (numeric): line width
        size (numeric): font size

    Returns:
        subplot objects with the plot parameters changed
    """

    subplot_object.spines[['top', 'right']].set_visible(False)
    subplot_object.spines[['left', 'bottom']].set_linewidth(lw)
    subplot_object.xaxis.set_tick_params(width=lw)
    subplot_object.yaxis.set_tick_params(width=lw)
    subplot_object.tick_params(axis='both', which='major', labelsize=size)


def custom_time_extraction(path, events_filename, behavioral_data):
    """ Reads in data files from EyeLink .asc file and produces time intervals for further analysis.

    Arguments:
        path (string): absolute path to the folder where the event file and other related files are
        events_filename (string): list of strings specifying the name of the event asc file
        behavioral_data (Pandas DataFrame): trial event information written to .csv file during/at the end of the experiment.
        session (string): name of the condition

	Returns:
		to_zero_intervals_eyelink_ms (Nx2 list of numerical values): starting and ending time points to be set to zero
		zero_mean_intervals_eyelink_ms (empty here): N/A here
		high_pass_intervals_eyelink_ms (Nx2 list of numerical values): starting and ending time points to be filtered
		exp_seq_df (Pandas DataFrame): a df with 7 columns:
		    exp_time_stamp (numeric): from PsychoPy, time in seconds that each trial started at
		    block_type (str): report or ignore
		    trial_number (numeric): trial mark for each block (1-10)
		    participant (str): probably doesn't matter but can't hurt to keep
		    frameRate (numeric): probably doesn't matter but can't hurt to keep
		    trial_start_eyelink (numeric): time in ms that each trial started at
		    trial_end_eyelink (numeric): time in ms that each trial ended at

	# Credit to https://github.com/djangraw/ParseEyeLinkAscFiles for original function
	# Modified 10/26/22 by BZ -- edited output dataframes; changed indentation to tab
	# Modified 11/09/22 by BZ -- input event_filename changed to path; the function should now be able to read multiple files in directed path
	# Modified 11/22/22 by BZ -- input changed to _e.asc from edf2asc instead of single asc files from manual use of the executable converter; removed
	unnecessary dataframes from return; updated msg_df; added lines to find new messages (modified in Nov 18, tested in Nov 21, test data: BZ) from asc files
	# Modified 11/23/22 by BZ -- adding steps to create a separate gaze df
	"""

    # custom_tools_logger.info('Starting custom time extraction for ' + filename_root + '...')
    # custom_tools_logger.info('Reading events for ' + filename_root)

    ## Part 1: parse events from raw asc file
    # filename = [f for f in events_filename if session[0] in f][0]
    if '_e.asc' not in events_filename:
        events_filename = events_filename + '_e.asc'
    f1 = open(os.path.join(path, events_filename), 'r')
    raw_events_ar = f1.read().splitlines(True)  # split into lines
    raw_events_ar = list(filter(None, raw_events_ar))  # remove emptys
    raw_events_ar = np.array(raw_events_ar)  # convert to np array for simpler indexing
    f1.close()

    # Separate lines into samples and messages
    n_lines_events = len(raw_events_ar)
    line_type = np.array(['OTHER'] * n_lines_events, dtype='object')
    for a in range(n_lines_events):
        if len(raw_events_ar[a]) < 3:
            line_type[a] = 'EMPTY'
        elif raw_events_ar[a].startswith('*') or raw_events_ar[a].startswith('>>>>>'):
            line_type[a] = 'COMMENT'
        else:
            line_type[a] = raw_events_ar[a].split()[0]
            if 'START' in raw_events_ar[a]:
                recording_start = raw_events_ar[a].split()  # a list made from the line at which recording started
            elif 'END' in raw_events_ar[a]:
                recording_end = raw_events_ar[a].split()  # a list made from the line at which recording ended

    recording_start_eyelink = pd.to_numeric(recording_start[1])  # time from EyeLink when recording started
    recording_end_eyelink = pd.to_numeric(recording_end[1])  # time from EyeLink when recording ended

    i_msg = np.nonzero(line_type == 'MSG')[0]
    t_msg = []
    txt_msg = []

    for b in range(len(i_msg)):
        info = raw_events_ar[i_msg[b]].split()  # separate MSG prefix and timestamp from rest of message
        t_msg.append(int(info[1]))  # extract info
        txt_msg.append(' '.join(info[2:]))

    # Get the time stamps for beginning and end of each trial
    msg_df = pd.DataFrame({'Time': t_msg, 'Text': txt_msg})
    trial_seq_df = behavioral_data[behavioral_data['trial_number'].notna()]
    cols_to_keep = ['exp_time_stamp', 'trial_number']  # exp_time_stamp is a accumulating timer from PsychoPy
    cols_to_add = ['trial_start_eyelink', 'trial_end_eyelink']
    exp_seq_df = trial_seq_df[cols_to_keep]  # add EyeLink time to the time stamps

    trial_seq_eyelink = pd.DataFrame()
    trial_start_eyelink = []  # initiate the EyeLink start time of each trial
    trial_end_eyelink = []

    for c in msg_df.iloc:
        if 'trial' in c['Text']:
            if 'started' in c['Text']:
                trial_start_eyelink += [c['Time']]

            elif 'ended' in c['Text']:
                trial_end_eyelink += [c['Time']]

    trial_seq_eyelink = trial_seq_eyelink.assign(T_Start=trial_start_eyelink, T_End=trial_end_eyelink)
    trial_seq_eyelink = trial_seq_eyelink.set_index(exp_seq_df.index)
    exp_seq_df[cols_to_add] = trial_seq_eyelink

    ## Part 2: get the behavioral data, match time stamps, and return them in the same format as before
    custom_tools_logger.info('Matching time stamps between the ASCII and the CSV file...')

    # #### ~~~~**** UNIT DIFF!!!! S ≠ MS ****~~~~ ####
    # t_diff_asc_csv = trial_start_asc[0][0] - trial_start_csv[0] * 1000  # ms in asc subtracting ms converted from csv
    # trial_start_csv_eyelink_ms = [e * 1000 + t_diff_asc_csv for e in trial_start_csv]
    # # trial_end_csv_eyelink_ms = [trial_start_csv_eyelink_ms[e] + 60000 for e in range(len(trial_start_csv_eyelink_ms))]  # trial ends 60 seconds after start
    # trial_end_csv_eyelink_ms = [trial_start_csv_eyelink_ms]

    to_zero_intervals_eyelink_ms = []  # from recording start to start of block (trial) 1
    zero_mean_intervals_eyelink_ms = []  # don't care about this for now, leave it empty probably
    high_pass_intervals_eyelink_ms = []  # band pass area

    # to_zero_intervals_eyelink_ms += [[recording_start_asc, trial_start_csv_eyelink_ms[0]]]  # the latter needs to be set to eyelink time
    # high_pass_intervals_eyelink_ms += [[trial_start_csv_eyelink_ms[0], trial_end_csv_eyelink_ms[0]]]  # manually add trial 1 of block 1
    # in the above lists, the interval is between the 0th and 1st item
    #
    # for f in range(1, len(trial_start_csv_eyelink_ms)):  # exclude first trial b/c there was no trial end to compare
    #     to_zero_intervals_eyelink_ms += [[trial_end_csv_eyelink_ms[f - 1], trial_start_csv_eyelink_ms[f]]]
    #     high_pass_intervals_eyelink_ms += [[trial_start_csv_eyelink_ms[f], trial_end_csv_eyelink_ms[f]]]

    to_zero_intervals_eyelink_ms += [[recording_start_eyelink, exp_seq_df.iloc[0]['trial_start_eyelink']]]  # add the wait time before the very first trial

    for d in range(len(exp_seq_df)):
        # block_trial_info += [exp_seq_df.iloc[d]['block_type'] + '_' + str(int(exp_seq_df.iloc[d]['trial_number']))]

        if d == len(exp_seq_df) - 1:  # if it's the very last trial

            to_zero_intervals_eyelink_ms += [[exp_seq_df.iloc[d]['trial_end_eyelink'], recording_end_eyelink]]  # add time after the end of the last trial

        else:
            to_zero_intervals_eyelink_ms += [[exp_seq_df.iloc[d]['trial_end_eyelink'], exp_seq_df.iloc[d + 1]['trial_start_eyelink']]]

        high_pass_intervals_eyelink_ms += [[exp_seq_df.iloc[d]['trial_start_eyelink'], exp_seq_df.iloc[d]['trial_end_eyelink']]]

    # exp_seq_df = exp_seq_df.assign(trial_label=block_trial_info)
    # to_zero_intervals_eyelink_ms += [[trial_end_csv_eyelink_ms[-1], recording_end_asc]]  # also add the last bit of time after the last trial ended
    to_zero_intervals_eyelink_ms.sort()
    zero_mean_intervals_eyelink_ms.sort()
    high_pass_intervals_eyelink_ms.sort()

    ## Retain a list of all the time points at which shape was changed on screen
    shifted_times_df = behavioral_data[behavioral_data['shift_status'] == 'shifted'][['current_shape', 'shift_time', 'session']]
    shift_times_eyelink_ms = []

    # for index in range(len(trial_seq_df)):
    #
    #     current_trial_start_loc = trial_seq_df.iloc[index].name
    #
    #     if index < 9:
    #
    #         current_trial_end_loc = trial_seq_df.iloc[index + 1].name
    #         shifts_this_trial = shifted_times_df.loc[current_trial_start_loc:current_trial_end_loc]
    #
    #     if index == 9:
    #
    #         shifts_this_trial = shifted_times_df.loc[current_trial_start_loc:]
    #
    #     shift_times_eyelink_ms += [high_pass_intervals_eyelink_ms[index][0] + x * 1000 for x in shifts_this_trial['shift_time']]

    return [to_zero_intervals_eyelink_ms, zero_mean_intervals_eyelink_ms, high_pass_intervals_eyelink_ms, exp_seq_df]


def custom_gaze_cleaning(path, output_path, root, samples_filename, high_pass_intervals_ms, trial_sequence, session, subfolder):
    """
	Reads in the raw sample Eyelink file, cleaned saccades and blinks to produce a cleaned dataframe for collated gaze positions. This function first averages
	horizontal gaze position between the eyes, and collates time so that there is no displacement in time. Then based on the cleaned saccades from
	eye_processing_tools.get_saccades() and interpolated blinks from eye_processing_tools.get_blinks_and_missing_data(), samples within 20 milliseconds
	of a saccade or 50 milliseconds of a blink are removed. Lastly, any displacement as a result of removing buffered saccades and blinks is collated such that
	the corresponding points in Eyelink time and horizontal positions were connected.

	Arguments:
	    path (string): absolute path to the folder where the event file and other related files are
	    output_path (string): absolute path to the folder where any data output is stored
	    root (string): string identifier for observer + session
	    samples_filename (string): string specifying the name of the samples asc file
	    high_pass_intervals_ms (list): periods to apply band pass filter to
	    trial_sequence (Pandas DataFrame): a df with block and trial order and number; both PsychoPy and Eyelink times are also included as separate columns
	    session (string): string specifying the order in which a given observer participated in the experiment conditions
	    subfolder (string): name of the subfolder where plots are saved for sanity checks

	Returns:
	    collated_gaze_df (Pandas DataFrame): cleaned gaze data. Gaze positions from sample files were first filtered to only include data points within the high
	pass intervals (i.e. actual trials). Both horizontal and vertical gaze positions were averaged between the eyes. Then, any sample that was 20 ms or less
	away from a saccade, or 50 ms or less away from a blink was excluded.
            Time (numeric): EyeLink time in ms during the band pass periods
            Avg_X (numeric): raw horizontal gaze positions averaged between the eyes
            Avg_Y (numeric): raw vertical gaze positions averaged between the eyes
            Collated_Time (numeric): continuous time in ms after removing displacements
            Trial_Number (numeric): trial mark for each block (1-10)
            Collated_X (numeric): horizontal gaze positions after removing displacements

    blink_by_trial_df (Pandas DataFrame): blink data filtered and separated by trial so that blinks outside the trials are excluded. Note that the trial labels
    in this df are accurate, but they are derived from uncollated times, meaning that later on when we are dealing with averaging gaze displacement in blink
    buffer windows, we need to be careful and match the times.
        T_Start (numeric): EyeLink time in ms at which each blink starts
        T_End (numeric): EyeLink time in ms at which each blink ends
        Trial_Label (string): trial locator for blinks, in the form of 'condition_trial', same as all other trial labels

	"""

    custom_tools_logger.info('Cleaning gaze for observer ' + root[:2] + '...')

    output_dir = os.path.join(output_path, subfolder)
    if subfolder not in os.listdir(output_path):
        os.mkdir(output_dir)

    cleaned_filename = root + '_cleaned_gaze.csv'
    if cleaned_filename in os.listdir(output_path):

        cleaned_gaze = pd.read_csv(os.path.join(output_path, cleaned_filename))
        custom_tools_logger.info('cleaned gaze output found in directory! skipping this step...')

        return cleaned_gaze

    else:
        # Get file names
        events_path = os.path.join(path, 'events')
        saccades_filename = [file for file in os.listdir(events_path) if 'saccades' in file and root in file][0]
        blinks_filename = [file for file in os.listdir(events_path) if 'blinks' in file and root in file][0]

        # Keep only the valid samples in band pass periods
        raw_samples_cols = ['Time', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil', 'dots']  # nothing important in dots
        raw_samples_df = pd.read_csv(os.path.join(path, samples_filename), header=None, delim_whitespace=True)
        raw_samples_df.columns = raw_samples_cols
        raw_samples_dict = raw_samples_df.to_dict('records')  # convert to a list of dicts
        valid_samples_dict = [g for g in raw_samples_dict if g['LX'] != '.' and g['RX'] != '.']  # and keep valid data

        high_pass_samples_dict = []
        for sample in range(len(valid_samples_dict)):

            for interval in high_pass_intervals_ms:

                if interval[0] <= valid_samples_dict[sample]['Time'] <= interval[1]:  # restrict samples to band pass periods

                    high_pass_samples_dict += [valid_samples_dict[sample]]

        saccades_ar = np.load(os.path.join(events_path, saccades_filename), allow_pickle=True)
        blinks_ar = np.load(os.path.join(events_path, blinks_filename), allow_pickle=True)
        binocular_saccades_ar = [saccades_ar[i] for i in range(len(saccades_ar)) if 'start_end_time_ms_binocular' in saccades_ar[i]]
        buffered_saccades_ar = copy.deepcopy(binocular_saccades_ar)  # so far deepcopy seems to be the only solution
        buffered_blinks_ar = copy.deepcopy(blinks_ar)

        ## Add buffers to saccades and blinks
        buffer_from_saccade_ms = 20
        buffer_from_blink_ms = 50
        for j in range(len(buffered_saccades_ar)):
            buffered_saccades_ar[j]['start_end_time_ms_binocular'][0] -= buffer_from_saccade_ms
            buffered_saccades_ar[j]['start_end_time_ms_binocular'][1] += buffer_from_saccade_ms

        for k in range(len(buffered_blinks_ar)):
            buffered_blinks_ar[k][0] -= buffer_from_blink_ms
            buffered_blinks_ar[k][1] += buffer_from_blink_ms

        high_passed_samples_df = pd.DataFrame(high_pass_samples_dict)  # convert dicts back to df
        high_passed_samples_df[raw_samples_cols[0:6]] = high_passed_samples_df[raw_samples_cols[0:6]].apply(pd.to_numeric)  # convert to numeric just to be safe

        ## Remove buffered saccades and blinks from averaged gaze data =========================================================================================
        avg_gaze_df = pd.DataFrame()  # average gaze between the eyes in a new df
        avg_gaze_df['Time'] = high_passed_samples_df['Time']
        avg_gaze_df['Avg_X'] = high_passed_samples_df[['LX', 'RX']].mean(axis=1)
        avg_gaze_df['Avg_Y'] = high_passed_samples_df[['LY', 'RY']].mean(axis=1)

        rows_to_drop = pd.DataFrame()
        for buffer in buffered_saccades_ar:
            # note that the first time point of each saccade (and blinks below) is kept for later collation, so it's not dropped here
            around_this_saccade = avg_gaze_df[(avg_gaze_df['Time'] > buffer['start_end_time_ms_binocular'][0])
                                              & (avg_gaze_df['Time'] <= buffer['start_end_time_ms_binocular'][1])]
            rows_to_drop = pd.concat([rows_to_drop, around_this_saccade], axis=0)

        for buffer in buffered_blinks_ar:
            around_this_blink = avg_gaze_df[(avg_gaze_df['Time'] > buffer[0])
                                            & (avg_gaze_df['Time'] <= buffer[1])]
            rows_to_drop = pd.concat([rows_to_drop, around_this_blink], axis=0)  # find rows with time within the buffer range of blinks
            
        avg_gaze_df = avg_gaze_df.drop(rows_to_drop.index)  # drop'em!

        ## Organize output for gaze, saccades and blink that also retain trial info ============================================================================
        gaze_by_trial_df = pd.DataFrame()  # make a new df for gaze
        saccades_df = pd.DataFrame()  # make saccades (unbuffered) into df so that trial information is easier to recall later
        saccades_start_time = []
        saccades_end_time = []

        for sac in binocular_saccades_ar:
            saccades_start_time += [sac['start_end_time_ms_binocular'][0]]
            saccades_end_time += [sac['start_end_time_ms_binocular'][1]]

        # saccades_df = saccades_df.assign(T_Start=saccades_start_time, T_End=saccades_end_time)
        # blinks_df = pd.DataFrame(blinks_ar, columns=['T_Start', 'T_End'])  # same for blinks but this will be returned as an output
        # saccades_by_trial_df = pd.DataFrame()
        # blinks_by_trial_df = pd.DataFrame()

        for seq in trial_sequence.index:
            trial_number = trial_sequence.loc[seq, 'trial_number']
            place_holder_trial = avg_gaze_df[(avg_gaze_df['Time'] >= trial_sequence.loc[seq, 'trial_start_eyelink'])
                                             & (avg_gaze_df['Time'] <= trial_sequence.loc[seq, 'trial_end_eyelink'])]

            place_holder_trial = place_holder_trial.assign(Trial_Number=trial_number)
            gaze_by_trial_df = pd.concat([gaze_by_trial_df, place_holder_trial], axis=0)

        #     saccades_this_trial = saccades_df[(saccades_df['T_End'] >= trial_sequence.loc[seq, 'trial_start_eyelink'])
        #                                       & (saccades_df['T_Start'] <= trial_sequence.loc[seq, 'trial_end_eyelink'])]
        #     saccades_this_trial = saccades_this_trial.assign(Trial_Number=trial_number)
        #     saccades_by_trial_df = saccades_by_trial_df.append(saccades_this_trial)
        #
        #     blinks_this_trial = blinks_df[(blinks_df['T_End'] >= trial_sequence.loc[seq, 'trial_start_eyelink'])
        #                                   & (blinks_df['T_Start'] <= trial_sequence.loc[seq, 'trial_end_eyelink'])]
        #     blinks_this_trial = blinks_this_trial.assign(Trial_Number=trial_number)
        #     blinks_by_trial_df = blinks_by_trial_df.append(blinks_this_trial)  # this also gets rid of blinks that are not in a trial
        #
        # blinks_by_trial_df.attrs['session'] = session
        # # instead of dropping the rows that are in the buffers, replace them with the gaze position right before the saccade or blink
        # gaze_by_trial_df.loc[:, 'Collated_X'] = gaze_by_trial_df.loc[:, 'Avg_X']
        # for trial in trial_sequence['trial_label']:
        #     # gaze_this_trial = gaze_by_trial_df[gaze_by_trial_df['Trial_Label'] == trial]
        #     saccades_this_trial = saccades_by_trial_df[saccades_by_trial_df['Trial_Label'] == trial]
        #     blinks_this_trial = blinks_by_trial_df[blinks_by_trial_df['Trial_Label'] == trial]
        #     for sac in saccades_this_trial.iloc:
        #         samples_in_this_sac = gaze_by_trial_df[(gaze_by_trial_df['Time'] >= sac['T_Start']) & (gaze_by_trial_df['Time'] <= sac['T_End'])]
        #         if len(samples_in_this_sac):  # if there is actually samples in this saccade
        #             first_sample_in_this_sac = samples_in_this_sac.iloc[0].name
        #             last_sample_in_this_sac = samples_in_this_sac.iloc[-1].name
        #             gaze_by_trial_df.loc[first_sample_in_this_sac:last_sample_in_this_sac, 'Collated_X'] = gaze_by_trial_df.loc[first_sample_in_this_sac - 1, 'Avg_X']  # replace samples
        #     for blk in blinks_this_trial.iloc:
        #         samples_in_this_blk = gaze_by_trial_df[(gaze_by_trial_df['Time'] >= blk['T_Start']) & (gaze_by_trial_df['Time'] <= blk['T_End'])]
        #         if len(samples_in_this_blk):  # if there is actually samples in this blink
        #             first_sample_in_this_blk = samples_in_this_blk.iloc[0].name
        #             last_sample_in_this_blk = samples_in_this_blk.iloc[-1].name
        #             gaze_by_trial_df.loc[first_sample_in_this_blk:last_sample_in_this_blk, 'Collated_X'] = gaze_by_trial_df.loc[first_sample_in_this_blk - 1, 'Avg_X']

        ## Collate gaze signals across the dropped displacements ===============================================================================================
        new_times_ms = np.arange(avg_gaze_df.iloc[0]['Time'], avg_gaze_df.iloc[0]['Time'] + len(avg_gaze_df), 1)
        gaze_by_trial_df = gaze_by_trial_df.assign(Collated_Time=new_times_ms)
        diff_every_ms_df = gaze_by_trial_df[['Time', 'Avg_X']].diff()
        diff_every_ms_df[['Trial_Number', 'Collated_Time']] = gaze_by_trial_df[['Trial_Number', 'Collated_Time']]  # steal some columns that can't be subtracted
        # this allows us to match time points between the diff df and the avg df

        larger_diff_df = diff_every_ms_df[diff_every_ms_df['Time'] > 1]  # find stepwise time differences that are larger than 1 ms for later use
        first_point_to_collate_ms = larger_diff_df.iloc[0]['Collated_Time']
        collated_gaze_df = gaze_by_trial_df[gaze_by_trial_df['Collated_Time'] < first_point_to_collate_ms]
        collated_gaze_df = collated_gaze_df.assign(Collated_X=collated_gaze_df.loc[:, 'Avg_X'])  # check this to see if it is the same

        custom_tools_logger.info('Now collating gaze (removing gaze displacements)...')
        # TWO tricky things here:
        # 1) diff() returns ONLY row-wise differences such that diff row n = og row n - og row n-1
        # 2) adding/subtracting this difference affects the subsequent gaze positions; sign of the cumulative difference is not indicative of directionality
        trial_start_collated = []
        for trial in larger_diff_df['Trial_Number'].unique():

            current_trial = gaze_by_trial_df[gaze_by_trial_df['Trial_Number'] == trial]
            current_trial_diff = larger_diff_df[larger_diff_df['Trial_Number'] == trial]
            trial_start_collated += [current_trial.iloc[0]['Collated_Time']]  # save collated time and add to sequence df later

            for diff in range(len(current_trial_diff)):

                cumulative_diff = current_trial_diff.iloc[0:(diff + 1)]['Avg_X'].sum()  

                if diff == len(current_trial_diff) - 1:
                    place_holder_this_trial = current_trial[current_trial['Collated_Time'] >= current_trial_diff.iloc[diff]['Collated_Time']]

                else:
                    place_holder_this_trial = current_trial[(current_trial['Collated_Time'] >= current_trial_diff.iloc[diff]['Collated_Time'])
                                                            & (current_trial['Collated_Time'] < current_trial_diff.iloc[diff + 1]['Collated_Time'])]

                place_holder_this_trial = place_holder_this_trial.assign(
                    Collated_X=place_holder_this_trial['Avg_X'] - cumulative_diff)  # subtract the difference

                if max(place_holder_this_trial['Collated_X'].diff()) >= 2:  # mini sanity check

                    custom_tools_logger.info("A seemingly larger gaze displacement was found -- this may indicate an error during collation!")

                collated_gaze_df = pd.concat([collated_gaze_df, place_holder_this_trial], axis=0)

        collated_gaze_df.attrs['session'] = session
        collated_gaze_df.to_csv(os.path.join(output_path, cleaned_filename), index=False)

        # for trial in trial_sequence['trial_number'].unique():
        #
        #     current_trial = collated_gaze_df[collated_gaze_df['Trial_Number'] == trial]
        #     this_part = current_trial.head(round(len(current_trial) * 1 / 4))
        #
        #     fig, gaze_this_trial = plt.subplots(figsize=(27, 9), nrows=3)
        #
        #     gaze_this_trial[0].plot(this_part['Time'], this_part['Avg_X'], c='k')
        #     gaze_this_trial[1].plot(this_part['Time'], this_part['Collated_X'], c='k')
        #     gaze_this_trial[2].plot(this_part['Time'], this_part['Avg_Y'], c='k')
        #
        #     gaze_this_trial[0].set_title('Uncollated Gaze', fontsize=20)
        #     gaze_this_trial[1].set_title('Collated Gaze', fontsize=20)
        #     gaze_this_trial[2].set_title('Vertical Gaze', fontsize=20)
        #
        #     for n in range(0, 3):
        #         gaze_this_trial[n].set_xlabel('Time', fontsize=18)
        #         gaze_this_trial[n].set_ylabel('Gaze', fontsize=18)
        #         customize_subplots(gaze_this_trial[n], lw=3, size=24)
        #
        #     fig_name = '%s_%s_%s' % (session, trial, 'gaze.pdf')
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(output_dir, fig_name))
        #     plt.close()

    return collated_gaze_df


def get_behavioral_switches(behavioral_data, session, trial_timestamps):
    """
    Reads in a raw df from behavioral csv file, clean it and extract perceptual switches from it. This function also matches time stamps between the asc file
    and the csv file so each switch in the returned df will also have an EyeLink-compatible time stamp.
    Args:
        behavioral_data (Pandas DataFrame): raw behavioral data file that is read in iGaze prior to this
        session (string): name of the session/block
        trial_timestamps (Pandas DataFrame): cleaned data structure that has logged, in order, the start and time from both PsychoPy and EyeLink of each trial

    Returns:
        switches_df (Pandas DataFrame): all switches derived from key presses in the report condition.
            Time (numeric): time in seconds as from PsychoPy
            To_Percept (string): the reported percept immediately after a switch; this could be either a or d in testing, but it should also get the switches
            from left and right keys
            Trial_Number (string): condition name and trial number
        percepts_df (Pandas DataFrame): all percepts derived from switches mainly for plotting.
        percept_durations_df (Pandas DataFrame): percept durations in seconds organized by trial number 
    """
    custom_tools_logger.info('Getting behavioral switches for observer ' + str(behavioral_data.iloc[0]['participant']))

    if session == 'report':  # if this is the report condition

        # first clean the data by striping unnecessary columns and rows
        base_cols = ['exp_time_stamp', 'trial_number', 'direction', 'direction_rt']
        base_df = behavioral_data[base_cols]
        all_direction_keys_df = base_df[(base_df['direction'] == 'left') | (base_df['direction'] == 'right')]
        all_shape_changes_df = behavioral_data[behavioral_data['shift_status'] == 'shifted']
        # all_shape_changes = all_shape_changes_df['shift_time'].to_list()

        ## Find switches the old-fashioned way =================================================================================================================
        percept_durations_df = pd.DataFrame()  # went back to df because it is just more organized
        # also b/c each switch comes with a pair of percepts and the first and last key press complicate this even further
        switches_df = pd.DataFrame(columns=['Time', 'Trial_Num', 'To_Percept', 'Time_EyeLink'])
        for trial in range(len(trial_timestamps)):  # for each report trial

            current_trial_start_loc = trial_timestamps.iloc[trial].name  # get the starting location of the current trial

            if trial == len(trial_timestamps) - 1:  # if this is the last trial

                current_trial_end_loc = behavioral_data.iloc[-1].name  # make the very last row and apparently this does not throw an error

            else:  # if this is any other trial

                current_trial_end_loc = trial_timestamps.iloc[trial + 1].name - 1

            current_direction_keys = all_direction_keys_df.loc[current_trial_start_loc:current_trial_end_loc].drop(['exp_time_stamp'], axis=1)

            if len(current_direction_keys) >= 2 and len(current_direction_keys['direction'].unique()) > 1:

                if current_direction_keys['direction_rt'].diff().iloc[-1] < 0:
                    current_direction_keys = current_direction_keys.head(-1)

                all_direction_keys_df.loc[current_trial_start_loc:current_trial_end_loc, 'trial_number'] = trial_timestamps.iloc[trial]['trial_number']
                all_shape_changes_df.loc[current_trial_start_loc:current_trial_end_loc, 'trial_number'] = trial_timestamps.iloc[trial]['trial_number']
                current_trial_start_time_eyelink = trial_timestamps.iloc[trial]['trial_start_eyelink']  # also log the corresponding EyeLink time

                percepts_to_add = pd.DataFrame(current_direction_keys['direction_rt'].diff().tail(-2))  # separate percepts per trial with one layer of brackets
                percepts_to_add['Trial_Number'] = trial + 1
                percept_durations_df = pd.concat([percept_durations_df, percepts_to_add])

                for key in range(len(current_direction_keys)):

                    if key >= 1 and current_direction_keys.iloc[key]['direction'] != current_direction_keys.iloc[key - 1]['direction']:
                        switch_time_csv = current_direction_keys.iloc[key]['direction_rt']

                        this_switch = pd.DataFrame({'Time': switch_time_csv,
                                                    'To_Percept': current_direction_keys.iloc[key]['direction'],
                                                    'Time_EyeLink': current_trial_start_time_eyelink + switch_time_csv * 1000,
                                                    'Trial_Num': trial + 1}, index=[0])
                        switches_df = pd.concat([switches_df, this_switch], axis=0)

        switches_df = switches_df.replace('left', -1)
        switches_df = switches_df.replace('right', 1)

        prop_effective_keys = len(switches_df)/len(all_direction_keys_df) # get the proportion of all keys that were actually a switch

        if len(switches_df.columns) <= 2 or switches_df.To_Percept.isna().values.any():
            custom_tools_logger.info('Missing data in the switches; double check the code!')

        ## Outline the end of each percept based on extracted switches =========================================================================================
        percepts_df = pd.DataFrame()
        for trial in switches_df.loc[:, 'Trial_Num'].unique():

            current_trial = switches_df[switches_df['Trial_Num'] == trial]
            percepts_end_df = pd.DataFrame()  # add rows indicating end of percepts in another df to avoid infinite looping

            for index in current_trial.iloc:  # first add another row indicating the end of the percept as 0.001 s before the next key

                this_percept_end = pd.DataFrame({'Time': index['Time'] - 0.001,
                                                 'Trial_Num': trial,
                                                 'To_Percept': -index['To_Percept'],
                                                 'Time_EyeLink': index['Time_EyeLink'] - 1}, index=[0])
                percepts_end_df = pd.concat([percepts_end_df, this_percept_end], axis=0)

            if current_trial.iloc[0]['Time'] > 0:  # then add the first percept of every trial as the opposite of the first to_percept

                percept_start_row = pd.DataFrame({'Time': 0,
                                                  'Trial_Num': trial,
                                                  'To_Percept': -current_trial.iloc[0]['To_Percept'],
                                                  'Time_EyeLink': trial_timestamps.iloc[int(trial) - 1]['trial_start_eyelink']}, index=[0])

                current_trial = pd.concat([percept_start_row, current_trial], axis=0)

            current_trial = pd.concat([current_trial, percepts_end_df], axis=0)
            current_trial = current_trial.sort_values('Time')
            current_trial.reset_index(drop=True, inplace=True)  # reset index here before appending to the larger df so that every switch is in order

            for p in np.arange(1, 91, 1):
                this_second = pd.DataFrame({'Time': p,
                                            'Trial_Num': trial,
                                            'To_Percept': 0,
                                            'Time_EyeLink': current_trial.iloc[0]['Time_EyeLink'] + p * 1000}, index=[0])
                current_trial = pd.concat([current_trial, this_second], axis=0)

            current_trial = current_trial.sort_values('Time')
            current_trial.reset_index(drop=True, inplace=True)

            for index in current_trial.index:
                if current_trial.loc[index, 'To_Percept'] == 0:
                    current_trial.loc[index, 'To_Percept'] = current_trial.iloc[index - 1]['To_Percept']

            percepts_df = pd.concat([percepts_df, current_trial], axis=0)

        percepts_df.reset_index(drop=True, inplace=True)

        ignored_shapes = []
        for trial in trial_timestamps.iloc:

            shapes_this_trial = all_shape_changes_df[all_shape_changes_df['trial_number'] == trial['trial_number']]
            shapes_this_trial_list = shapes_this_trial['shift_time'].to_list()
            ignored_shapes += [trial['trial_start_eyelink'] + shift * 1000 for shift in shapes_this_trial_list]

        return switches_df, percepts_df, percept_durations_df, ignored_shapes, prop_effective_keys

    else:  # if this is the ignore condition

        base_cols = ['exp_time_stamp', 'trial_number', 'current_shape', 'shift_status', 'shapeshift', 'shapeshift_rt']
        base_df = behavioral_data[base_cols]
        all_keys_df = base_df[(base_df['shapeshift'] == 'up') | (base_df['shapeshift'] == 'down')]
        all_inversions_df = behavioral_data[(behavioral_data['trial_number'].notna()) | (behavioral_data['shift_status'] == 'shifted')]
        # all_shape_changes = all_shape_changes_df['shift_time'].to_list()

        shapes_df = pd.DataFrame(columns=['Time', 'Trial_Num', 'To_Percept', 'Time_EyeLink'])
        for trial in range(len(trial_timestamps)):  # for each trial

            current_trial_start_loc = trial_timestamps.iloc[trial].name  # get the first row number of the current trial

            if trial == len(trial_timestamps) - 1:  # if this is the last trial

                current_trial_end_loc = all_keys_df.iloc[-1].name + 1  # to index the end row of the last trial

            else:  # if this is any other trial

                current_trial_end_loc = trial_timestamps.iloc[trial + 1].name - 1

            keys_this_trial = all_keys_df.loc[current_trial_start_loc:current_trial_end_loc].drop(['exp_time_stamp'], axis=1)
            # inversions_this_trial = all_inversions_df.loc[current_trial_start_loc:current_trial_end_loc].drop(['exp_time_stamp'], axis=1)

            if keys_this_trial['shapeshift_rt'].diff().iloc[-1] < 0:

                keys_this_trial = keys_this_trial.head(-1)

            all_keys_df.loc[current_trial_start_loc:current_trial_end_loc, 'trial_number'] = trial_timestamps.iloc[trial]['trial_number']
            all_inversions_df.loc[current_trial_start_loc:current_trial_end_loc, 'trial_number'] = trial_timestamps.iloc[trial]['trial_number']
            current_trial_start_time_eyelink = trial_timestamps.iloc[trial]['trial_start_eyelink']  # also log the corresponding EyeLink time

            for key in range(len(keys_this_trial)):

                switch_time_csv = keys_this_trial.iloc[key]['shapeshift_rt']

                this_key = pd.DataFrame({'Time': switch_time_csv,
                                         'To_Percept': keys_this_trial.iloc[key]['shapeshift'],
                                         'Time_EyeLink': current_trial_start_time_eyelink + switch_time_csv * 1000,
                                         'Trial_Num': trial + 1}, index=[0])
                shapes_df = pd.concat([shapes_df, this_key], axis=0)

        # Get all the times when the shape changed
        # changed_shapes_EL = []
        # Organize all inversion and following key presses in a single frame; edited Apr 9 2025 for manuscript revision
        inversions_reported = pd.DataFrame()
        # inverted_df = base_df[base_df['shift_status'] == 'shifted']

        for trial in trial_timestamps.iloc:  # for each trial

            # if trial == 9:
            #
            #     shapes_this_trial = all_shape_changes_df[all_shape_changes_df['shift_time'] >= trial_timestamps.iloc[trial]['exp_time_stamp']]
            #
            # else:
            #     shapes_this_trial = all_shape_changes_df[(all_shape_changes_df['shift_time'] >= trial_timestamps.iloc[trial]['exp_time_stamp']) &
            #                                          (all_shape_changes_df['shift_time'] <= trial_timestamps.iloc[trial + 1]['exp_time_stamp'])]

            inversions_this_trial = all_inversions_df[all_inversions_df['trial_number'] == trial['trial_number']]
            inversions_this_trial_list = inversions_this_trial['shift_time'].to_list()
            # changed_shapes_EL += [trial['trial_start_eyelink'] + shift * 1000 for shift in shapes_this_trial_list]

            keys_this_trial = all_keys_df[(all_keys_df['shapeshift_rt'].notna()) & (all_keys_df['trial_number'] == trial['trial_number'])]
            for key in keys_this_trial.iloc:  # for each key press

                prev_row_index = key.name - 1  # get the index of the row that contains the time of this inversion
                # note that this assumes that the row immediately before the key press is the row with the "shifted" inversion

                # if key.shapeshift_rt < 0:  # occasionally the press gets a negative rt
                #     continue

                if prev_row_index not in inversions_this_trial.index or key.shapeshift_rt == keys_this_trial.iloc[0]['shapeshift_rt']:
                    # if the inversion index we got does not refer to the inversion row or if this is the very first key press
                    continue
                    
                if inversions_this_trial.loc[prev_row_index]['current_shape'] == key.shapeshift:  # if the response matches the orientation after
                    # inversion
                    this_key_df = pd.DataFrame({'trial_num': [trial.trial_number],
                                                'orientation': [inversions_this_trial.loc[prev_row_index]['current_shape']],
                                                'inversion_time_s': [inversions_this_trial.loc[prev_row_index]['shift_time']],
                                                'inversion_time_EL': [trial['trial_start_eyelink'] +
                                                                      inversions_this_trial.loc[prev_row_index]['shift_time'] * 1000],
                                                'inversion_rt': [key.shapeshift_rt],
                                                'inversion_rt_EL': [trial['trial_start_eyelink'] + key.shapeshift_rt * 1000]})

                    inversions_reported = pd.concat([inversions_reported, this_key_df], axis=0)

                else:  # if observer pressed the wrong key immediately following an inversion but then corrected it
                    prev_row_index -= 1  # look farther back to try to find the row corresponding to the key press
                    if prev_row_index in inversions_this_trial.index:
                        if inversions_this_trial.loc[prev_row_index]['current_shape'] == key.shapeshift:
                            this_key_df = pd.DataFrame({'trial_num': [trial.trial_number],
                                                        'orientation': [inversions_this_trial.loc[prev_row_index]['current_shape']],
                                                        'inversion_time_s': [inversions_this_trial.loc[prev_row_index]['shift_time']],
                                                        'inversion_time_EL': [trial['trial_start_eyelink'] +
                                                                              inversions_this_trial.loc[prev_row_index]['shift_time'] * 1000],
                                                        'inversion_rt': [key.shapeshift_rt],
                                                        'inversion_rt_EL': [trial['trial_start_eyelink'] + key.shapeshift_rt * 1000]})

                            inversions_reported = pd.concat([inversions_reported, this_key_df], axis=0)
            
        return shapes_df, inversions_reported


def get_okn_switches(path, root, output_path, gaze_data, trial_sequence, cosine_threshold=.7, win_len_okn_ms=1500):
    """
    Reads in cleaned gaze data in the forms of dataframes to identify perceptual switches. Note that although cleaned gaze data has two time series
    (an uncollated and a collated), here because blinks were extracted based on the uncollated raw time, fitting linear curves, removing blink buffers and
    averaging samples in blink buffers are all done on the uncollated raw time series.
    Args:
        path (string): absolute path within each observer's folder where data files are stored
        root (string): string identifier for observer + session
        output_path (string): absolute path within each observer's folder where output will be stored
        gaze_data (Pandas DataFrame): cleaned and collated gaze positions and time points returned by custom_gaze_cleaning.
        trial_sequence (Pandas DataFrame): cleaned data structure that has logged, in order, the start and time from both PsychoPy and EyeLink of each trial

    Returns:
        switches_df (Pandas DataFrame): all perceptual switches above the threshold (cosine larger than .85)
        percepts_df (Pandas DataFrame): all OKN-derived percepts that met the minimum dominance duration requirement
        percept_durations_df (Pandas DataFrame): duration of all retained OKN-based percepts

    """
    custom_tools_logger.info('Getting perceptual switches from eye movements...')

    session_dict = {'r': 'report', 'i': 'ignore'}
    session = session_dict[root[-1]]
    cleaned_filename = '%s_%s_%s' % (root[:2], session, 'cleaned.xlsx')

    if cleaned_filename in os.listdir(output_path):

        switches_df = pd.read_excel(os.path.join(output_path, cleaned_filename), sheet_name='okn_switches')
        percepts_df = pd.read_excel(os.path.join(output_path, cleaned_filename), sheet_name='okn_percepts')
        percept_durations_df = pd.read_excel(os.path.join(output_path, cleaned_filename), sheet_name='okn_percept_durations')

    else:
        ## First organize blinks by trial
        events_path = os.path.join(path, 'events')
        blinks_filename = [file for file in os.listdir(events_path) if 'blinks' in file and root in file][0]
        blinks_ar = np.load(os.path.join(events_path, blinks_filename), allow_pickle=True)
        blinks_df = pd.DataFrame(blinks_ar, columns=['T_Start', 'T_End'])
        blink_by_trial_df = pd.DataFrame()

        for seq in trial_sequence.index:

            trial_number = trial_sequence.loc[seq, 'trial_number']
            blinks_this_trial = blinks_df[(blinks_df['T_End'] >= trial_sequence.loc[seq, 'trial_start_eyelink'])
                                          & (blinks_df['T_Start'] <= trial_sequence.loc[seq, 'trial_end_eyelink'])]
            blinks_this_trial = blinks_this_trial.assign(Trial_Number=trial_number)
            blink_by_trial_df = pd.concat([blink_by_trial_df, blinks_this_trial], axis=0)  # this gets rid of blinks not in a trial

        # # Create time windows of 2500 ms over cleaned gaze data and fit sine functions to look for smooth pursuits
        # Occasionally when viewing structure-from-motion stimuli, observers will track an object around the edges;
        # while this could potentially be avoided with 1) limited  dot-life that is the same as or shorter than the
        # amount of time it takes for a dot travel from the left edge to the right edge; and/or 2) higher angular
        # velocity (minimum 3 degrees per second by my estimation).
        # However, in the cases where smooth pursuit still occurs for an extended amount of time, or a considerable
        # portion of any trial, this method could be helpful in identifying those periods and remove them from
        # samples later used for okn switch coding.

        gaze_samples_for_okn = copy.deepcopy(gaze_data)  # initiate a df for gaze samples free of smooth pursuit
        # se_threshold_for_fit = 1.5  # periods with a higher average standard error of regression than this will not
        # be used for okn switch coding
        # win_len_track_ms = 4000
        # win_step_track_ms = 250

        ## This is code used at one point to try to remove smooth gaze pattern as a result of tracking object around edges
        # for trial in trial_sequence:
        #
        #     gaze_this_trial = gaze_samples_for_okn[gaze_samples_for_okn['Trial_Number'] == trial]  # get gaze signals for this trial only
        #     samples_this_trial = np.arange(gaze_this_trial.iloc[0]['Time'], gaze_this_trial.iloc[-1]['Time'] - win_len_track_ms, win_step_track_ms)
        #     locs_to_remove_this_trial = []  # make a list to store the indices for the samples that will be removed prior to okn switch coding
        #
        #     for s in samples_this_trial:
        #
        #         gaze_this_sample = gaze_this_trial[(gaze_this_trial['Time'] >= s) & (gaze_this_trial['Time'] < s + win_len_track_ms)]
        #
        #         if len(gaze_this_sample) < 2:
        #             continue
        #
        #         try:
        #             res_this_model = fit_sin(gaze_this_sample['Time'], gaze_this_sample['Avg_X'])
        #
        #         except RuntimeError:
        #             res_this_model = np.nan
        #
        #         if pd.notna(res_this_model):
        #
        #             if np.mean(np.sqrt(np.diag(res_this_model['raw_res'][2]))) < se_threshold_for_fit:
        #
        #                 locs_to_remove_this_trial += gaze_this_sample.index.to_list()
        #
        #     locs_to_remove_this_trial = [*set(locs_to_remove_this_trial)]  # remove duplicate indices
        #     gaze_samples_for_okn = gaze_samples_for_okn.drop(locs_to_remove_this_trial)
        #
        # fig, g2 = plt.subplots(figsize=(45, 30), nrows=10)
        #
        # for g in range(10):
        #
        #     gaze_this_trial = gaze_samples_for_okn[gaze_samples_for_okn['Trial_Number'] == g + 1]
        #     g2[g].plot(gaze_this_trial['Time'], gaze_this_trial['Avg_X'])
        #
        #     figure_customization(g2[g], lw=2, size=24)
        #
        # plt.tight_layout()
        # plt.savefig(os.path.join(output_path, session + '_gaze_without_tracking.pdf'))

        ## Create time windows of 1500 ms over cleaned gaze data and fit linear to estimate direction =======================================================
        win_step_okn_ms = 38  # interval at which the gaze-sampling time windows are created
        switches_df = pd.DataFrame()
        
        for trial in trial_sequence['trial_number']:

            this_trial = gaze_samples_for_okn[gaze_samples_for_okn['Trial_Number'] == trial]
            win_range_ar = np.arange(this_trial.iloc[0]['Time'], this_trial.iloc[-1]['Time'] - win_len_okn_ms, win_step_okn_ms)

            for win in win_range_ar:  # in all the windows in this current trial

                gaze_this_win = this_trial[(this_trial['Time'] >= win) & (this_trial['Time'] < win + win_len_okn_ms)]

                if len(gaze_this_win) < 2:
                    continue

                curve_for_adjacent = optimize.curve_fit(linear_function, gaze_this_win['Time'], gaze_this_win['Collated_X'])  # x/t
                curve_for_opposite = optimize.curve_fit(linear_function, gaze_this_win['Time'], gaze_this_win['Avg_Y'])  # y/t
                cos_this_win = np.cos(np.arctan2(curve_for_opposite[0][0], curve_for_adjacent[0][0]))

                this_snippet = pd.DataFrame({'T_Start': win,
                                            'T_Mid': win + win_len_okn_ms / 2,
                                            'Trial_Number': trial,
                                            'Cosine': cos_this_win}, index=[0])
                switches_df = pd.concat([switches_df, this_snippet], axis=0)

        switches_df.reset_index(drop=True, inplace=True)

        ## Find buffered blinks from percepts and replace them with average of the previous three windows ======================================================
        pre_blink_buffer = 250
        post_blink_buffer = 400
        num_samples_to_avg = round(100 / win_step_okn_ms)

        for blink in blink_by_trial_df.iloc:

            samples_to_interpolate = switches_df[(switches_df.loc[:, 'T_Mid'] >= blink[0] - pre_blink_buffer) &
                                                 (switches_df.loc[:, 'T_Mid'] <= blink[1] + post_blink_buffer)]

            if len(samples_to_interpolate):
                indices_to_replace = samples_to_interpolate.index

                switches_df.loc[indices_to_replace, 'Cosine'] = \
                    (switches_df.loc[(indices_to_replace[0] - num_samples_to_avg):(indices_to_replace[0] - 1)]['Cosine'] +
                     switches_df.loc[(indices_to_replace[-1] + 1):(indices_to_replace[-1] + num_samples_to_avg)]['Cosine'])/(num_samples_to_avg*2)

        ## Determine preliminary perceived directions
        directions = []
        for c in switches_df.iloc:

            if c['Cosine'] >= cosine_threshold:
                directions += [1]

            elif c['Cosine'] <= -cosine_threshold:
                directions += [-1]

            else:
                directions += [0]

        switches_df = switches_df.assign(Cosign=directions)  # cosign, get it ?

        custom_tools_logger.info('Assigning switches based on dominance duration...')

        ## Mark percepts based on sign and dominance duration ==================================================================================================
        min_t_dom_ms = 500  # set a minimum dominance duration for the preceding switch to be marked and kept
        min_samples_dom = round(min_t_dom_ms / win_step_okn_ms)  # dominance duration threshold in number of samples

        # mark all moments between differently marked perceived directions as potential switch candidates; this is not to exclude 0-Cosign moments,
        # but to include switches that are immediate and sandwiched between two already-marked percepts
        possible_percepts = switches_df[switches_df['Cosign'] != 0]  # find all the already-marked percepts
        switches_df.loc[:, 'Switch'] = 0  # make a dummy-coded column to indicate whether there is a switch at each given time point
        switches_df.loc[:, 'Following_Percept'] = 0

        for m in range(len(possible_percepts) - 1):

            if possible_percepts.iloc[m]['Cosign'] != possible_percepts.iloc[m + 1]['Cosign']:  # if the next percept is marked a different sign

                loc_this_percept_end = int(possible_percepts.iloc[m].name)
                loc_next_percept_start = int(possible_percepts.iloc[m + 1].name)
                # switches_df.loc[loc_next_percept_start, 'Switch'] = 1
                
                midpoint = loc_this_percept_end + round((loc_next_percept_start - loc_this_percept_end) / 2)

                if midpoint in switches_df.index:  # if this midpoint still exists in the fitted observations

                    switches_df.loc[midpoint, 'Switch'] = 1  # mark the midpoint as the time at which a switch happened
                    switches_df.loc[midpoint, 'Following_Percept'] = possible_percepts.iloc[m + 1]['Cosign']

                else:  # if it does not exist in the fitted observations due to being part of a blink or just not
                    # being the halfway point of an observation window
                    closest_midpoint = min(switches_df.index, key=lambda x: abs(x - midpoint))
                    switches_df.loc[closest_midpoint, 'Switch'] = 1
                    switches_df.loc[closest_midpoint, 'Following_Percept'] = possible_percepts.iloc[m + 1]['Cosign']

        # for each switch pair that is less than 500 ms apart, remove/unmark both switches because they would sandwich a shorter-than-500-ms percept
        switches_only = switches_df[switches_df['Switch'] == 1]
        for trial in trial_sequence:

            switches_this_trial = switches_only[switches_only['Trial_Number'] == trial]
            t_dom_this_trial = switches_this_trial[['T_Mid', 'Switch']].diff().tail(-1)  # do this for at least two columns so that the result is still a df

            for t in range(len(t_dom_this_trial)):

                if t_dom_this_trial.iloc[t]['T_Mid'] <= min_t_dom_ms:  # if the duration between those switches are shorter than the threshold

                    switch_1_to_replace = t_dom_this_trial.iloc[t - 1].name
                    switch_2_to_replace = t_dom_this_trial.iloc[t].name
                    switches_df.loc[switch_1_to_replace:switch_2_to_replace, 'Switch'] = 0

        # switches_df.loc[:, 'Following_Percept'] = switches_df.loc[:, 'Cosign']
        # switches_df.loc[switches_df['Cosine'] < 0, 'Following_Percept'] = -1
        # switches_df.loc[switches_df['Cosine'] > 0, 'Following_Percept'] = 1
        ## Mark percepts based on the average sign of the cosine values ========================================================================================
        percepts_df = pd.DataFrame()  # make another df for percepts here b/c each percept needs to be added separately
        percept_durations_df = pd.DataFrame()
        switches_filtered = switches_df[switches_df.loc[:, 'Switch'] == 1]  # get the updated switches with short-dominance percepts removed

        for trial in trial_sequence['trial_number']:

            percepts_this_trial = switches_df[switches_df['Trial_Number'] == trial]
            switches_this_trial = switches_filtered[switches_filtered['Trial_Number'] == trial]

            # originally the first period of time in each trial is artificially added to the duration of the first percept; currently that is not the case
            # because the first percept takes some time to settle and it wouldn't really make sense to add the first bit of extra time
            #
            # if len(switches_this_trial):
            #     # mark the first percept separately
            #     first_percept = percepts_this_trial.loc[0: switches_this_trial.iloc[0].name - 1]  # the first switch signifies the end of the first percept
            #
            #     if len(first_percept):
            #
            #         if first_percept['Cosine'].mean() > 0:  # mark the entire period before the first switch as one percept
            #             first_percept.loc[:, 'Percept'] = 1
            #
            #         else:
            #             first_percept.loc[:, 'Percept'] = -1
            #
            #         percepts_df = pd.concat([percepts_df, first_percept])

            if len(switches_this_trial):

                for switch in range(len(switches_this_trial) - 1):  # subsequent percepts following each switch

                    next_percept = percepts_this_trial.loc[switches_this_trial.iloc[switch].name:switches_this_trial.iloc[switch + 1].name - 1]

                    if not len(next_percept):
                        continue

                    next_percept.loc[:, 'Percept'] = percepts_this_trial.iloc[switch]['Cosign']

                    percepts_df = pd.concat([percepts_df, next_percept], axis=0)

                percepts_to_add = pd.DataFrame(switches_this_trial['T_Mid'].diff().tail(-1))

                if len(percepts_to_add):
                    percepts_to_add.loc[:, 'Trial_Number'] = trial
                    percept_durations_df = pd.concat([percept_durations_df, percepts_to_add], axis=0)
            
            # for s in range(len(switches_this_trial)):
            #
            #     switch_loc = switches_this_trial.iloc[s].name
            #
            #     if s >= 1 and switches_this_trial.iloc[s]['Percept'] == switches_this_trial.iloc[s - 1]['Percept']:  # if the percept doesn't change
            #
            #         switches_df.loc[switch_loc, 'Switch'] = 0  # remove this switch
        switches_df = switches_df.query('Switch == 1')

        # blink_intervals_for_plots = [[b['T_Start'] - pre_blink_buffer, b['T_End'] + post_blink_buffer] for b in blink_by_trial_df.iloc]
        # nearby_switches = []
        #
        # for blink in blink_intervals_for_plots:
        #
        #     time_diff = switches_df[(switches_df['T_Mid'] >= blink[0]) & (switches_df['T_Mid'] <= blink[1])]
        #
        #     if len(time_diff):
        #         for switch in time_diff.iloc:
        #
        #             nearby_switches += [min(abs(switch['T_Mid'] - blink[0]), abs(switch['T_Mid'] - blink[1]))]
        #
        # labels_intervals = ['-200', '-100', '100', '200', '300', '400', '500', '600']
        # blink_switches_df = pd.DataFrame(nearby_switches, columns=['T_Diff'])
        #
        # blink_switches_df['T_Diff_Bins'] = pd.cut(nearby_switches, np.arange(-200, 601, 100), labels=labels_intervals).astype(str)
        # blink_switches_plot = pd.DataFrame(blink_switches_df['T_Diff_Bins'].value_counts() / len(switches_df))
        # blink_switches_plot = blink_switches_plot.reset_index()
        #
        # if len(blink_switches_plot) != len(labels_intervals):
        #     missing_indices = [index for index in labels_intervals if index not in blink_switches_plot['index'].to_list()]
        #
        #     for i in missing_indices:
        #         blink_switches_plot = blink_switches_plot.append({'index': pd.to_numeric(i), 'T_Diff_Bins': 0}, ignore_index=True)
        #
        # blink_switches_plot['index'] = pd.to_numeric(blink_switches_plot['index'])
        # blink_switches_plot = blink_switches_plot.sort_values('index')
        #
        # fig, bs = plt.subplots(figsize=(8, 8))
        # bs.bar(blink_switches_plot['index'], blink_switches_plot['T_Diff_Bins'], width=80, fc='g', lw=2)
        # bs.axvline(x=0, c='r', lw=4, ls='--')
        #
        # bs.set_xticks([-200, 0, 200, 400, 600]), bs.set_xticklabels(['-200', 'Blink', '200', '400', '600'], fontsize=20)
        # bs.set_yticks([0, .1]), bs.set_yticklabels(['0', '.1'], fontsize=20)
        # customize_subplots(bs, lw=3, size=20)
        # bs.set_xlabel('Time Relating to Blink', fontsize=20)
        # bs.set_ylabel('Probability of An OKN-Based Perceptual Switch', fontsize=20)
        # # f1.legend(bbox_to_anchor=(0, .5, 1, .5), frameon=False, fontsize=24)
        # plt.tight_layout()
        #
        # plt_name = session + '_okn_switches_locked_to_blinks.pdf'
        # plt.savefig(os.path.join(output_path, plt_name))
        # plt.close()

    return switches_df, percepts_df, percept_durations_df


def find_nearby_okn_events(trial_timestamps, behavioral_switches, okn_switches):
    """

    Args:
        trial_timestamps:
        behavioral_switches:
        okn_switches:

    Returns:

    """
    same_switches_nearby = []
    oppo_switches_nearby = []
    for trial in trial_timestamps['trial_number']:
        behavior_this_trial = behavioral_switches[behavioral_switches['Trial_Num'] == trial]
        intervals_this_trial = [[s - 2500, s + 1500] for s in behavior_this_trial['Time_EyeLink']]
        okn_this_trial = okn_switches[okn_switches['Trial_Number'] == trial]

        # for interval_index, interval in enumerate(intervals_this_trial):
        #     okn_this_interval = okn_this_trial[
        #         (okn_this_trial['T_Mid'] >= interval[0]) & (okn_this_trial['T_Mid'] <= interval[1])
        #     ]
        #
        #     if not okn_this_interval.empty:
        #         for okn_event_index, okn_event in okn_this_interval.iterrows():
        #             behavior_event = behavior_this_trial.iloc[interval_index]
        #
        #             if okn_event['Following_Percept'] == behavior_event['To_Percept']:
        #                 same_switches_nearby.append(okn_event['T_Mid'] - behavior_event['Time_EyeLink'])
        #             else:
        #                 oppo_switches_nearby.append(okn_event['T_Mid'] - behavior_event['Time_EyeLink'])

        for key in range(len(intervals_this_trial)):
            okn_this_interval = okn_this_trial[(okn_this_trial['T_Mid'] >= intervals_this_trial[key][0]) &
                                               (okn_this_trial['T_Mid'] <= intervals_this_trial[key][1])]

            if len(okn_this_interval):

                for eye in range(len(okn_this_interval)):

                    if okn_this_interval.iloc[eye]['Following_Percept'] == \
                            behavior_this_trial.iloc[key]['To_Percept']:

                        same_switches_nearby += [okn_this_interval.iloc[eye]['T_Mid'] -
                                                 behavior_this_trial.iloc[key]['Time_EyeLink']]

                    else:
                        oppo_switches_nearby += [okn_this_interval.iloc[eye]['T_Mid'] -
                                                 behavior_this_trial.iloc[key]['Time_EyeLink']]

    return same_switches_nearby, oppo_switches_nearby



def plot_everything_for_sanity_checks(path, output_path, subfolder):
    """

    Args:
        path (string): absolute path to the folder where the event file and other related files are
        subfolder (string): path to the subfolder in path where plots are stored for sanity checks
        cleaned_gaze (Pandas DataFrame): cleaned and collated gaze signals
        behavioral_switches: (Pandas DataFrame): all behavioral switches and dominance durations
        behavioral_percepts: (Pandas DataFrame): all behavioral percepts in order
        behavioral_percept_durations (Pandas DataFrame):
        okn_switches (Pandas DataFrame):
        okn_percepts: (Pandas DataFrame):  all okn percepts in order
        okn_percept_durations: (Pandas DataFrame): all okn dominance durations

    Returns:
        nothing but saves the comparison plots to a designated output subfolder

    """
    if subfolder not in os.listdir(output_path):
        os.mkdir(os.path.join(output_path, subfolder))

    obs = path[-2:]
    cleaned_filename_report = os.path.join(output_path, obs + '_report_cleaned.xlsx')
    # cleaned_filename_ignore = os.path.join(output_path, obs, obs + '_ignore_cleaned.xlsx')
    gaze_report = pd.read_csv(os.path.join(output_path, obs + 'r_cleaned_gaze.csv'))
    # gaze_ignore = pd.read_csv(os.path.join(output_path, obs + 'i_cleaned_gaze.csv'))

    behavioral_switches = pd.read_excel(cleaned_filename_report, sheet_name='behavioral_switches')
    behavioral_percepts = pd.read_excel(cleaned_filename_report, sheet_name='behavioral_percepts')
    behavioral_percept_durations = pd.read_excel(cleaned_filename_report, sheet_name='behavioral_percept_durations')
    okn_report_switches = pd.read_excel(cleaned_filename_report, sheet_name='okn_switches')
    okn_report_percepts = pd.read_excel(cleaned_filename_report, sheet_name='okn_percepts')
    okn_report_percept_durations = pd.read_excel(cleaned_filename_report, sheet_name='okn_percept_durations')
    # okn_ignore_switches = pd.read_excel(cleaned_filename_ignore, sheet_name='okn_switches')
    # okn_ignore_percepts = pd.read_excel(cleaned_filename_ignore, sheet_name='okn_percepts')
    # okn_ignore_percept_durations = pd.read_excel(cleaned_filename_ignore, sheet_name='okn_percept_durations')

    okn_abs = abs(okn_report_percepts['Cosine'])
    
    fig, r = plt.subplots(figsize=(10, 10))
    r.hist(okn_abs, weights=np.ones(len(okn_abs))/len(okn_abs), bins=10)
    customize_subplots(r, lw=3, size=20)
    hist_name = obs + '_report_cos_hist.pdf'
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, subfolder, hist_name))
    plt.close()

    for trial in gaze_report['Trial_Number'].unique():

        gaze = gaze_report[gaze_report['Trial_Number'] == trial]
        group_num = len(gaze) / 6
        gaze['Group'] = np.floor_divide(gaze.index, group_num)
        keys_p = behavioral_percepts[behavioral_percepts['Trial_Num'] == trial]
        okn = okn_report_percepts[okn_report_percepts['Trial_Number'] == trial]
        okn_switches = okn_report_switches[okn_report_switches['Trial_Number'] == trial]

        for group in gaze['Group'].unique():

            gaze_this_plot = gaze[gaze['Group'] == group]
            keys_this_plot = keys_p[(keys_p['Time_EyeLink'] >= gaze_this_plot.iloc[0]['Time']) & (keys_p['Time_EyeLink'] <= gaze_this_plot.iloc[-1]['Time'])]
            okn_this_plot = okn[(okn['T_Mid'] >= gaze_this_plot.iloc[0]['Time']) & (okn['T_Mid'] <= gaze_this_plot.iloc[-1]['Time'])]
            okn_switches_this_plot = okn_switches[(okn_switches['T_Mid'] >= gaze_this_plot.iloc[0]['Time']) &
                                                  (okn_switches['T_Mid'] <= gaze_this_plot.iloc[-1]['Time'])]

            fig, p = plt.subplots(figsize=(20, 9), nrows=3)
            p[0].plot(gaze_this_plot['Time'], gaze_this_plot['Collated_X'], lw=2)
            p[1].plot(okn_this_plot['T_Mid'], okn_this_plot['Cosine'], lw=2)
            # p[2].plot(okn_this_plot['T_Mid'], okn_this_plot['Percept'], lw=2)
            p[2].plot(keys_this_plot['Time'], keys_this_plot['To_Percept'], lw=2)
            p[1].axhline(y=.85, c='magenta', ls=':')
            p[1].axhline(y=-.85, c='magenta', ls=':')
            
            for t in okn_switches_this_plot['T_Mid']:
                # p[0].axvline(x=t, ymin=300, ymax=500, c='r', ls='--', clip_on=False)
                p[1].axvline(x=t, ymin=-2, ymax=2.6, c='r', ls='--', clip_on=False)

            p[0].set_title('Cleaned Gaze', fontsize=18)
            p[1].set_title('Cosine', fontsize=18)
            # p[2].set_title('OKN', fontsize=18)
            p[2].set_title('Behavioral Percepts', fontsize=18)
            p[2].set_xlabel('Time', fontsize=20)

            for m in range(2, 3):
                p[m].set_yticks([-1, 1])
                p[m].set_yticklabels(['Left', 'Right'])

            for n in range(0, 3):
                customize_subplots(p[n], lw=3, size=18)

            plt_name = '%s_%s_%s_%s' % ('report', trial, group, '.pdf')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, subfolder, plt_name))
            plt.close()

    # Plot side-by-side distributions of behavior and okn patterns of the report condition

    # fig, q = plt.subplots(figsize=(10, 10), nrows=2)
    # q[0].hist(okn_report_percept_durations['T_Mid'] / 1000, bins=10)
    # # q[1].hist(okn_ignore_percept_durations['T_Mid'] / 1000)
    # q[1].hist(behavioral_percept_durations['direction_rt'])
    #
    # q[0].set_title('report: mean duration ' + str(round(okn_report_percept_durations['T_Mid'].mean() / 1000, 1)) + '; total switches ' + str(len(
    #     okn_report_switches)), fontsize=24)
    #
    # # q[1].set_title('ignore: mean duration ' + str(round(okn_ignore_percept_durations['T_Mid'].mean() / 1000, 1)) + '; total switches ' + str(len(
    # #     okn_ignore_switches)), fontsize=24)
    #
    # q[1].set_title('behavior: mean duration ' + str(round(behavioral_percept_durations['direction_rt'].mean(), 1)) +
    #                '; total switches ' + str(len(behavioral_switches)), fontsize=24)
    #
    # for p in range(2):
    #     q[p].set_ylabel('Number of Percepts', fontsize=22)
    #     customize_subplots(q[p], lw=3, size=20)
    #
    # plt_name = 'distributions.pdf'
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_path, subfolder, plt_name))
    # plt.close()

    # same_switches_nearby = []
    # oppo_switches_nearby = []
    # for trial in gaze_report['Trial_Number'].unique():
    #
    #     behavior_this_trial = behavioral_switches[behavioral_switches['Trial_Num'] == trial]
    #     interval_this_trial = [[s - 2500, s + 1500] for s in behavior_this_trial['Time_EyeLink']]
    #     okn_this_trial = okn_report_switches[okn_report_switches['Trial_Number'] == trial]
    #
    #     for key in range(len(interval_this_trial)):
    #
    #         okn_this_interval = okn_this_trial[(okn_this_trial['T_Mid'] >= interval_this_trial[key][0]) &
    #                                            (okn_this_trial['T_Mid'] <= interval_this_trial[key][1])]
    #
    #         if len(okn_this_interval):
    #
    #             for eye in range(len(okn_this_interval)):
    #
    #                 if okn_this_interval.iloc[eye]['Following_Percept'] == behavior_this_trial.iloc[key]['To_Percept']:
    #
    #                     same_switches_nearby += [okn_this_interval.iloc[eye]['T_Mid'] -
    #                                              behavior_this_trial.iloc[key]['Time_EyeLink']]
    #
    #                 else:
    #                     oppo_switches_nearby += [okn_this_interval.iloc[eye]['T_Mid'] -
    #                                              behavior_this_trial.iloc[key]['Time_EyeLink']]
    #
    #     same_switches_nearby = list(
    #         set(same_switches_nearby))  # remove okn switches that are in the vicinity of multiple behavioral switches
    #     oppo_switches_nearby = list(set(oppo_switches_nearby))
    #
    # labels_intervals = ['-2500', '-2000', '-1500', '-1000', '-500', '500', '1000', '1500']
    # same_df = pd.DataFrame(same_switches_nearby, columns=['T_Diff'])
    # oppo_df = pd.DataFrame(oppo_switches_nearby, columns=['T_Diff'])
    # num_switches_total = len(same_df) + len(oppo_df)
    #
    # same_df['T_Diff_Bins'] = pd.cut(same_switches_nearby, np.arange(-2500, 1501, 500), labels=labels_intervals).astype(str)
    # same_plot = pd.DataFrame(same_df['T_Diff_Bins'].value_counts() / num_switches_total)
    # same_plot = same_plot.reset_index()
    #
    # if len(same_plot) != len(labels_intervals):
    #     missing_indices = [index for index in labels_intervals if index not in same_plot['T_Diff_Bins'].to_list()]
    #
    #     for i in missing_indices:
    #         missing_index = pd.DataFrame({'T_Diff_Bins': pd.to_numeric(i), 'count': 0}, index=[0])
    #         same_plot = pd.concat([same_plot, missing_index], axis=0)
    #
    # same_plot['T_Diff_Bins'] = pd.to_numeric(same_plot['T_Diff_Bins'])
    # same_plot = same_plot.sort_values('T_Diff_Bins')
    #
    # oppo_df['T_Diff_Bins'] = pd.cut(oppo_switches_nearby, np.arange(-2500, 1501, 500), labels=labels_intervals).astype(str)
    # oppo_plot = pd.DataFrame(oppo_df['T_Diff_Bins'].value_counts() / num_switches_total)
    # oppo_plot = oppo_plot.reset_index()
    #
    # if len(oppo_plot) != len(labels_intervals):
    #     missing_indices = [index for index in labels_intervals if index not in oppo_plot['T_Diff_Bins'].to_list()]
    #
    #     for i in missing_indices:
    #         missing_index = pd.DataFrame({'T_Diff_Bins': pd.to_numeric(i), 'count': 0}, index=[0])
    #         oppo_plot = pd.concat([oppo_plot, missing_index], axis=0)
    #
    # oppo_plot['T_Diff_Bins'] = pd.to_numeric(oppo_plot['T_Diff_Bins'])
    # oppo_plot = oppo_plot.sort_values('T_Diff_Bins')
    #
    # fig, z = plt.subplots(figsize=(12, 8))
    # z.axvline(x=0, c='r', lw=3, ls='--')
    #
    # z.bar(same_plot['T_Diff_Bins'], same_plot['count'], width=300, fc='c', lw=3, label='Same as key')
    # z.bar(oppo_plot['T_Diff_Bins'], oppo_plot['count'], bottom=same_plot['count'], width=300, fc='m', lw=3, label='Diff from key')
    # # z.plot(same_plot['index'], same_plot['T_Diff_Bins'], lw=3, c='c', ls='-.')
    # # z.plot(oppo_plot['index'], oppo_plot['T_Diff_Bins'], lw=3, c='m', ls='-.')
    #
    # z.set_xticks([-2000, -1000, 0, 1000]), z.set_xticklabels(['-2s', '-1s', 'Key', '+1s'], fontsize=20)
    # z.set_yticks([0, .1, .2, .3, .4]), z.set_yticklabels(['0', '.1', '.2', '.3', '.4'], fontsize=20)
    # z.set_title(path[-2:], fontsize=24)
    # customize_subplots(z, lw=3, size=20)
    # z.set_xlabel('Time Relating to Key Press', fontsize=22)
    # z.set_ylabel('Probability of An OKN-Derived Switch', fontsize=22)
    # z.legend(bbox_to_anchor=(0, .5, 1, .5), frameon=False, fontsize=20)
    # plt.tight_layout()
    #
    # plt_name = 'switch_correspondence.pdf'
    # plt.savefig(os.path.join(output_path, subfolder, plt_name))
    # plt.close()


def plot_stereo_results(path, output_path, subfolder):
    """
    Args:

    """
    obs = path[-2:]
    filename_root = 'planar'
    filename = [f for f in os.listdir(path) if filename_root in f][0]
    stereo_df = pd.read_csv(os.path.join(path, filename)).drop(['adjusted_fixation_x', 'adjusted_fixation_y'], axis=1)
    results_this_obs = []

    for disparity in [.1, .2, .5, 1]:
        trials_this_deg = stereo_df[stereo_df['disparity_deg'] == disparity]
        trials_correct = len(trials_this_deg[trials_this_deg['correct_bar'] == trials_this_deg['responded_bar']])
        trials_total = len(trials_this_deg)
        proportion_correct_this_deg = trials_correct / trials_total
        results_this_obs += [[disparity, proportion_correct_this_deg]]

        # this_obs_df = pd.DataFrame({'Obs': obs,
        #                             'disparity_deg': disparity,
        #                             'proportion_correct': proportion_correct_this_deg}, index=[0])
        # all_obs_df = pd.concat([all_obs_df, this_obs_df], axis=0)

    fig, st = plt.subplots(figsize=(8, 8))

    for result in results_this_obs:
        st.bar('disparity ' + str(result[0]), result[1])

    st.set_yticks(np.arange(0, 1.1, 0.2))
    st.set_ylabel('Proportion of correct trials', fontsize=20)
    customize_subplots(st, 3, 18)
    plt.tight_layout()
    plt_name = obs + '_stereo_results.pdf'
    plt.savefig(os.path.join(output_path, plt_name))
    plt.close()

    return results_this_obs


def get_significant_clusters(data, conditions, alpha):
    """

    Args:
        data (Pandas DataFrame): df with a column containing t-scores and a column containing p-values
        alpha: significance level
    """
    if len(conditions) > 1:

        cluster_time_points = []
        for condition in conditions:

            cols_to_keep = [c for c in data.columns if condition in c or 'time' in c]
            some_data = data[cols_to_keep]

            clusters_this_condition = []
            for i in range(len(some_data)):  # for each condition, find the start and end of clusters, then include all the in-between data points
                this_point = some_data.iloc[i]

                if i == 0 and this_point['p_'+condition] < alpha:  # if the first data point is significant
                    if some_data.iloc[i + 1]['p_'+condition] < alpha and np.sign(this_point['t_'+condition]) == np.sign(
                            some_data.iloc[i + 1]['t_'+condition]):  # and if the next data point is also significant and has the same sign
                        clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'start']]

                    elif some_data.iloc[i + 1]['p_'+condition] > alpha or np.sign(this_point['t_'+condition]) != np.sign(
                            some_data.iloc[i + 1]['t_'+condition]):  # else if the next data point is not significant or has a different sign
                        clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'only']]

                elif i < len(some_data) - 1 and some_data.iloc[i]['p_'+condition] < alpha:  # if this data point is significant

                    if some_data.iloc[i - 1]['p_' + condition] > alpha or np.sign(this_point['t_' + condition]) != np.sign(
                            some_data.iloc[i - 1]['t_' + condition]):  # if the previous data point was not significant or had a different sign

                        if some_data.iloc[i + 1]['p_'+condition] < alpha and np.sign(this_point['t_'+condition]) == np.sign(
                                some_data.iloc[i + 1]['t_'+condition]):  # if the next data point is significant and has the same sign
                            clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'start']]

                        if some_data.iloc[i + 1]['p_'+condition] > alpha or np.sign(this_point['t_'+condition]) != np.sign(
                                some_data.iloc[i + 1]['t_'+condition]):  # if the next point is not significant or has a different sign
                            clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'only']]

                    elif some_data.iloc[i - 1]['p_'+condition] < alpha and np.sign(this_point['t_'+condition]) == np.sign(
                            some_data.iloc[i - 1]['t_'+condition]):  # if the previous data point was significant and had the same sign

                        if some_data.iloc[i + 1]['p_'+condition] > alpha or np.sign(this_point['t_'+condition]) != np.sign(
                                some_data.iloc[i + 1]['t_'+condition]):  # if the next point is not significant
                            clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'end']]

                elif i == len(some_data) - 1 and this_point['p_'+condition] < alpha:  # if the last data point is significant
                    if some_data.iloc[i - 1]['p_'+condition] < alpha and np.sign(this_point['t_'+condition]) == np.sign(
                            some_data.iloc[i - 1]['t_'+condition]):  # if the previous data point is also significant
                        clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'end']]

            cluster_time_points += [clusters_this_condition]

    else:
        condition = conditions[0]
        some_data = data
        clusters_this_condition = []
        for i in range(len(some_data)):  # for each condition, find the start and end of clusters, then include all the in-between data points
            this_point = some_data.iloc[i]

            if i == 0 and this_point['p_values'] < alpha:  # if the first data point is significant
                if some_data.iloc[i + 1]['p_values'] < alpha and np.sign(this_point['t_scores']) == np.sign(
                        some_data.iloc[i + 1]['t_scores']):  # and if the next data point is also significant and has the same sign
                    clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'start']]

                elif some_data.iloc[i + 1]['p_values'] > alpha or np.sign(this_point['t_scores']) != np.sign(
                        some_data.iloc[i + 1]['t_scores']):  # else if the next data point is not significant or has a different sign
                    clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'only']]

            elif i < len(some_data) - 1 and some_data.iloc[i]['p_values'] < alpha:  # if this data point is significant

                if some_data.iloc[i - 1]['p_values'] > alpha or np.sign(this_point['t_scores']) != np.sign(
                        some_data.iloc[i - 1]['t_scores']):  # if the previous data point was not significant or had a different sign

                    if some_data.iloc[i + 1]['p_values'] < alpha and np.sign(this_point['t_scores']) == np.sign(
                            some_data.iloc[i + 1]['t_scores']):  # if the next data point is significant and has the same sign
                        clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'start']]

                    if some_data.iloc[i + 1]['p_values'] > alpha or np.sign(this_point['t_scores']) != np.sign(
                            some_data.iloc[i + 1]['t_scores']):  # if the next point is not significant or has a different sign
                        clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'only']]

                elif some_data.iloc[i - 1]['p_values'] < alpha and np.sign(this_point['t_scores']) == np.sign(
                        some_data.iloc[i - 1]['t_scores']):  # if the previous data point was significant and had the same sign

                    if some_data.iloc[i + 1]['p_values'] > alpha or np.sign(this_point['t_scores']) != np.sign(
                            some_data.iloc[i + 1]['t_scores']):  # if the next point is not significant
                        clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'end']]

            elif i == len(some_data) - 1 and this_point['p_values'] < alpha:  # if the last data point is significant
                if some_data.iloc[i - 1]['p_values'] < alpha and np.sign(this_point['t_scores']) == np.sign(
                        some_data.iloc[i - 1]['t_scores']):  # if the previous data point is also significant
                    clusters_this_condition += [[condition, this_point.name, this_point['deconvolved_time_points'], 'end']]

        cluster_time_points = clusters_this_condition

    return cluster_time_points


def invert_random_curves(data, scores, condition, alpha):
    """

    Args:
        data (Pandas DataFrame):
        test_statstics (Pandas DataFrame):
        condition_names (list):
        alpha (numeric): p-value threshold

    """

    # First get columns from the test statistics df that correspond to this condition
    cols_to_keep = [c for c in scores.columns if condition in c or 'time' in c]
    og_scores = scores[cols_to_keep]

    inversion_signs = [-1, 1]
    most_extreme_clusters = []

    for n in range(2000):

        if len(most_extreme_clusters) == 1000:
            break
            
        shuffled_data = copy.deepcopy(data)
        shuffled_data = shuffled_data.apply(lambda x: x * inversion_signs[np.random.randint(2) % 2], axis=0)  # randomly change the sign for some curves

        t_scores = []  # make empty lists to be turned into columns later
        p_values = []
        shuffled_scores = pd.DataFrame(og_scores['deconvolved_time_points'])

        for t in range(len(shuffled_data)):

            t_score_this_time, p_value_this_time = st.ttest_1samp(shuffled_data.iloc[t], 0)[0:2]  # run a one-sample t-test between each row and 0
            t_scores += [t_score_this_time]
            p_values += [p_value_this_time]

        shuffled_scores = shuffled_scores.assign(t_scores=t_scores, p_values=p_values)
        significant_data_points = shuffled_scores[shuffled_scores['p_values'] < alpha]  # only keep the significant data points

        if significant_data_points.empty:
            most_extreme_clusters += [0]
            continue  # skip the rest of the loop if there is no significant data point in this iteration
            
        else:
            clusters_this_iteration = []
            cluster_time_points = get_significant_clusters(shuffled_scores, [condition], .01)

            for cluster in cluster_time_points:

                if cluster[3] == 'only':
                    mass_this_cluster = scores.iloc[cluster[1]]['t_' + cluster[0]]
                    # this_cluster = pd.DataFrame({'condition': cluster[0], 't_score': mass_this_cluster,
                    #                              'start_time': cluster[2],
                    #                              'end_time': cluster[2]}, index=[0])

                if cluster[3] == 'start':  # if this time point marks the start of a cluster
                    t_score_this_cluster = cluster[0]
                    start_index_this_cluster = cluster[1]  # log the index and skip the rest of the loop so this variable won't be affected until the next cluster
                    start_time_this_cluster = cluster[2]
                    continue  # this is only here so that clusters that last a few time points won't add duplicate rows to the final df

                if cluster[3] == 'end':
                    end_index_this_cluster = cluster[1]  # log the ending index
                    end_time_this_cluster = cluster[2]
                    mass_this_cluster = scores.iloc[start_index_this_cluster:end_index_this_cluster + 1]['t_' + cluster[0]].sum()
                    # this_cluster = pd.DataFrame({'condition': cluster[0], 't_score': mass_this_cluster,
                    #                              'start_time': start_time_this_cluster,
                    #                              'end_time': end_time_this_cluster}, index=[0])

                clusters_this_iteration += [mass_this_cluster]

            if len(clusters_this_iteration):

                if len(clusters_this_iteration) > 1:  # if there are multiple clusters
                    extreme_values = [max(clusters_this_iteration), min(clusters_this_iteration)]
                    if extreme_values[1] < 0:
                        if extreme_values[0] > abs(extreme_values[1]):  # if the positive score is larger than the absolute value of the negative one
                            most_extreme_clusters += [extreme_values[0]]
                        else:
                            most_extreme_clusters += [extreme_values[1]]
                    # largest_clusters_per_iteration += [clusters_this_iteration]  # this would be a single numeric instead of a list

                else:  # if there is only one cluster
                    most_extreme_clusters += [clusters_this_iteration[0]]  # this would still be a list if there is only one cluster

            else:
                largest_single_data_point = significant_data_points['t_scores'].max()
                smallest_single_data_point = significant_data_points['t_scores'].min()
                if smallest_single_data_point < 0:
                    if largest_single_data_point > abs(smallest_single_data_point):
                        most_extreme_clusters += [largest_single_data_point]
                    else:
                        most_extreme_clusters += [smallest_single_data_point]

    return most_extreme_clusters