#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 07:29:39 2022

@author: beauz
"""
# This file takes in preprocessed files, and generates cleaned gaze data, OKN-based and behavioral switches, and aggregate data files at the individual level.

import pandas as pd
import numpy as np
import os
import warnings

import tools.custom_tools_shape_fs22 as shape_tools
import tools.general_tools as general

warnings.simplefilter(action='ignore', category=FutureWarning)  # suppress future warnings about df.append() 

data_path = '/Users/beauz/pupil_sfm/preprocessing'
output_path = '/Users/beauz/pupil_sfm/perceptual_switches'
# output_path = '/Users/beauz/pupil_sfm/inversions'
# edf_filename_prefix = 'cyl_'
observers = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder)) and len(folder) <= 3]
session_log = pd.read_excel(os.path.join(data_path, 'pupil_sfm_session_log.xlsx'), sheet_name='log', dtype=str)  # get the session log file

all_same_switches_nearby = []  # initiate the compilation of all okn-based switches near each key press that indicate the same direction
all_oppo_switches_nearby = []  # .. opposite direction
all_obs_df = pd.DataFrame(columns=['Obs', 'N_switch_okn_report', 'N_switch_okn_ignore', 'N_switch_key'])
all_nearby_switches_filtered = pd.DataFrame(columns=['Obs', 'T_from_behavior'])
cosine_bin_labels = ['<.1','.1-.2','.2-.3','.3-.4','.4-.5','.5-.6','.6-.7','.7-.8','.8-.9','>.9']

for obs in observers:

    obs_path = os.path.join(data_path, obs)
    obs_output_path = os.path.join(output_path, obs)  # define an output path for perceptual switches

    if obs not in os.listdir(output_path):

        os.mkdir(os.path.join(output_path, obs))   # make a directory for switch-related results if it's not already there

    session_order = session_log[session_log['Observer'] == obs].iloc[0][1:].to_list()

    session_order = ['ignore']

    for session in session_order:

        cleaned_filename_this_obs = '%s_%s_%s' % (obs, session, 'cleaned.xlsx')

        [behavioral_filename, edf_filenames] = shape_tools.get_filenames(obs_path, 'pupil', 'sph', session)
        behavioral_data = pd.read_csv(os.path.join(obs_path, behavioral_filename[0]))

        events_filename = [filename for filename in edf_filenames if '_e.asc' in filename][0]
        samples_filename = [filename for filename in edf_filenames if '_s.asc' in filename][0]
        obs_root = events_filename.split('_')[1]

        [to_zero_intervals_ms, zero_mean_intervals_ms, high_pass_intervals_ms, trial_timestamps] = shape_tools.custom_time_extraction(
            path=obs_path,
            events_filename=events_filename,
            behavioral_data=behavioral_data)

        # Clean gaze data by removing saccades and blinks with buffers; this also adds trial information to the eyelink data
        cleaned_gaze = shape_tools.custom_gaze_cleaning(path=obs_path, root=obs_root, output_path=obs_output_path,
                                                        samples_filename=samples_filename,
                                                        high_pass_intervals_ms=high_pass_intervals_ms,
                                                        trial_sequence=trial_timestamps,
                                                        session=session,
                                                        subfolder='plots_gaze_cleaning')

        okn_switches, okn_percepts, okn_percept_durations = shape_tools.get_okn_switches(path=obs_path, root=obs_root, output_path=obs_output_path,
                                                                                         gaze_data=cleaned_gaze,
                                                                                         trial_sequence=trial_timestamps)

        okn_percepts['cos_bin'] = pd.cut(abs(okn_percepts['Cosine']), np.arange(0, 1.1, .1), labels=cosine_bin_labels)
        cos_proportions_this_obs = okn_percepts['cos_bin'].value_counts(normalize=True, sort=False)[6:]

        if session == 'report':

            behavioral_switches, behavioral_percepts, behavioral_percept_durations, ignored_shapes, proportion_effective_keys = (
                shape_tools.get_behavioral_switches(
                behavioral_data=behavioral_data,
                session=session,
                trial_timestamps=trial_timestamps))

        if session == 'ignore':

            ignore_keys, inversions_reported = shape_tools.get_behavioral_switches(behavioral_data=behavioral_data,
                                                                                   session=session,
                                                                                   trial_timestamps=trial_timestamps)
            # proportion_correct_keys = len(ignore_keys) / len(inversions)  # note that this is technically proportion reported, not correct

        with pd.ExcelWriter(os.path.join(obs_output_path, cleaned_filename_this_obs)) as writer:
            trial_timestamps.to_excel(excel_writer=writer, sheet_name='time', index=False)

            if session == 'report':

                behavioral_switches.to_excel(excel_writer=writer, sheet_name='behavioral_switches', index=False)
                behavioral_percepts.to_excel(excel_writer=writer, sheet_name='behavioral_percepts', index=False)
                behavioral_percept_durations.to_excel(excel_writer=writer, sheet_name='behavioral_percept_durations', index=False)
                ignored_shapes_df = pd.DataFrame(ignored_shapes)
                ignored_shapes_df.to_excel(excel_writer=writer, sheet_name='triangle_inversions', index=False)
                okn_switches.to_excel(excel_writer=writer, sheet_name='okn_switches', index=False)
                okn_percepts.to_excel(excel_writer=writer, sheet_name='okn_percepts', index=False)
                okn_percept_durations.to_excel(excel_writer=writer, sheet_name='okn_percept_durations', index=False)

                n_switches_okn_report = len(okn_switches)

            else:
                inversions_df = pd.DataFrame(inversions)
                inversions_df.to_excel(excel_writer=writer, sheet_name='triangle_inversions', index=False)
                inversions_reported.to_excel(excel_writer=writer, sheet_name='inversions_reported', index=False)
                ignore_keys.to_excel(excel_writer=writer, sheet_name='key_press', index=False)
                okn_switches.to_excel(excel_writer=writer, sheet_name='okn_switches', index=False)
                okn_percepts.to_excel(excel_writer=writer, sheet_name='okn_percepts', index=False)
                okn_percept_durations.to_excel(excel_writer=writer, sheet_name='okn_percept_durations', index=False)

                n_switches_okn_ignore = len(okn_switches)

    this_obs = pd.DataFrame({'Obs': obs,
                             'N_switch_okn_report': n_switches_okn_report,
                             'N_switch_okn_ignore': n_switches_okn_ignore,
                             'N_switch_key': len(behavioral_switches),
                             'Proportion_effective_keys': proportion_effective_keys,
                             'Proportion_correct_keys': proportion_correct_keys
                             }, index=[obs])
                             # 'Cos_6': cos_proportions_this_obs.iloc[0],
                             # 'Cos_7': cos_proportions_this_obs.iloc[1],
                             # 'Cos_8': cos_proportions_this_obs.iloc[2], index=[0])
    all_obs_df = pd.concat([all_obs_df, this_obs], axis=0)

    shape_tools.plot_everything_for_sanity_checks(path=obs_path, output_path=obs_output_path, subfolder='plots_switch_comparison')

    all_nearby_switches_list = nearby_switches_oppo + nearby_switches_same

    for s in all_nearby_switches_list:

        all_nearby_switches = all_nearby_switches.append({'Obs': obs, 'T_from_behavior': s}, ignore_index=True)

