# README

This pipeline is designed to take raw event and sample files from EyeLink and extract patterns of eye movements (specifically optokinetic nystagmus) and identify moments of perceptual switches during observation of bistable stimuli.

## Time Extraction
`custom_tools_shape_fs22.custom_time_extraction`
1. Get the time points for when recording started and ended. 
2. The main thing this function does is taking `record_mode_df` and `msg_df`, and spit out `[to_zero_intervals_eyelink_ms, zero_mean_intervals_eyelink_ms, high_pass_intervals_eyelink_ms]` in the same format that `custom_factor_analysis_tools.custom_period_time_extraction_for_factor_analysis()` used to do.
   1. `to_zero_interval_eyelink_ms` should include time before the first trial started, breaks in-between trials, and after the last trial ended.

```
# to_zero_intervals_eyelink_ms
[[3898175, 3925677.0],
 [3985677.0, 3987637.008606], ...]
```
2. `high_pass_intervals_eye_link_ms` should include time in trials. *Here it’s 12000 ms per interval; but it might be a good idea to also take into account trials during which a recalibration occurred.* 

```
# high_pass_intervals_eyelink_ms
[[3925677.0, 3985677.0],
 [3987637.008606, 4047637.008606], ...]
```
3. It matches time between the behavioral data and the eye data by finding the common reference points in both files, which are the time points at which each trial starts.
   1. In the bistable cylinder experiment, there are messages in the EDF file indicating the start and end of a trial: “trial # started;” and “trial # ended.”
4. Since EDF files are already converted and separated between events and samples by `eye_processing_tools.edf2asc()`, I modified the above function to only return a Pandas DataFrame for messages, which is what I need.
5. Plots from `eye_processing_tools.prepare_pupil_for_GLM()` could be used to see if the above function is working as intended. 

## Gaze Cleaning
*See Brascamp et al., 2021*
1. Average all gaze between the eyes.
   1. Output files from `eye_processing_tools.prepare_pupil_for_GLM()` and  `eye_processing_tools.get_saccades()` contain the moments of saccades and blinks.
   2. To index the an object inside the saccades ndarray, use `saccades_ar[index][dict_key][index]`.
2. Identify all samples that were closer than 20 ms to a saccade or closer than 50 ms to a blink.
3. For gaze displacements (Fig. 1B top, vertical lines), the starting point of each displacement (top of the vertical lines) should be set to the time point at the end of the previous sample; and the ending point of each displacement (bottom of the vertical lines) should be set to the time point at the beginning of the next sample (connecting the dots).
4. Set gaze displacement at those collated time points to zero. **In other words, for chunks of samples between each gaze displacement, set gaze position at the beginning of each chunk to be the same as the end of the previous chunk.**

```
collated_gaze_df
            Time   Avg_X   Avg_Y Trial_Label  Collated_Time  Collated_X
0        1460184  527.80  384.35    report_1      1460184.0      527.80
1        1460185  527.80  383.70    report_1      1460185.0      527.80
          ...     ...     ...         ...            ...         ...
1193791  2786831  463.65  420.70   ignore_10      2493992.0     -151.90
1193792  2786832  464.25  420.55   ignore_10      2493993.0     -151.30
```

## OKN Switch Coding
*See Brascamp et al., 2021*
1. Take a 750-ms windowed copy of the cleaned gaze data every 38 ms
2. Fit a linear curve to vertical gaze by time in selected window and another one to the horizontal gaze.
   1. `scipy.optimize.curve_fit(linear_function)`
3. Determine a threshold for cos(atan(vertical curve, horizontal curve)) such that time windows with values above this is assigned a specific percept.
4. For any sample between 250 ms before and 400 ms after blink, replace with average gaze displacement 100 ms before after that blink buffer.
5. Keep the switch if dominance was longer than ~~500~~ **600** ms.

```
okn_switches
           T_Mid Trial_Label  Switch  To_Percept
37     1461965.0    report_1       1          -1
65     1463029.0    report_1       1          -1
          ...         ...     ...         ...
31111  2783535.0   ignore_10       1          -1
31130  2784257.0   ignore_10       1          -1
```

## Deconvolution 
`general_tools.deconvolve` 
Intersample interval (sec) nideconv: 0.25; Toeplitz: 0.33 
* `pupil_data_concatenated` (NumPy array): time points in the first column and data values for each time point in the second column

```
array([[ 0.00000000e+00,  3.08350607e-28],
       [ 9.99978883e+01, -2.32679732e-29],
       ...,
       [ 2.79374100e+06,  2.18995971e-06],
       [ 2.79384100e+06,  8.12127284e-06]])
```
* `event_type_names` (nested list): the names of the event types whose timing is provided by `event_moments_ms`
  * reported perceptual switches
  * ignored shape changes
  * reported shape changes
  * ignore perceptual switches
* `event_moments_ms` (nested list): result of combining the members of `event_moments_ms_to_be_concatenated`): every sublist contains the moments, in ms and on a time axis that corresponds with that of data_concatenated, of events of a given kind, if events of that kind were represented by sub-sublists of dimension Nx1 in `event_moments_ms_to_be_concatenated`. For events that were represented by sub-sublists of dimension Nx2, the corresponding sublist in event_moments_ms_concatenated contains those moments, paired with values that belong to those moments.
* `regressor_name_list` or `event_type_names`: *the names of the event types whose timing is provided by event_moments_ms. The order of entries should correspond between event_moments_ms and `event_type_names`
* `deconvolution_window_min_max_s` (array/list of two numbers, or Nx2 array/list of numbers): the start point and end point of the time interval (relative to events) during which response curves need to be estimated, in seconds. 
* If this is a 1D list or array with 2 elements then this start and end interval is applied for each regressor. 
* If, instead, this is a 2D list or array, containing a larger number of start/end pairs, then each event type gets a different start/end pair from `deconvolution_window_min_max_s`. In that case both the order and the length of deconvolution_window_min_max_s needs to correspond to those of `event_moments_and_optional_values_ms` and `event_type_names`.
