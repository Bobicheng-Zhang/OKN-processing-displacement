# Gaze
#MSU/spherepupil/iClean

### A few things from the EyeLink 1000 Plus User Manual: 
> Gaze position data reports the actual (x, y) coordinates of the subject's gaze on the display, compensating for distance from the display. … The default EyeLink coordinates are those of a 1024 by 768 VGA display, with (0, 0) at the top left.  
> The resolution data for gaze position data changes constantly depending on participant head position and point of gaze, and therefore is reported as a separate data type (see below). A typical resolution is about 36 pixels per degree for an EyeLink 1000 Plus setup in which the distance between the participant’s eyes and the display is twice the display’s width, and the screen resolution is set to 1024 by 768. (p. 109)  

> `PRESCALER <prescaler>` If gaze-position data or gaze-position resolution is sued for saccades and events are used, they must be divided by this value. For EDF2ASC, the prescaler is always 1. Programs that write integer data may use a larger prescaler (usually 10) to add precision to the data. (p. 127 top)  

> For gaze position data, the data range is scaled to the display coordinates, which are 1024 by 768 at startup, and may be changed via link commands. The data range setting is -0.2 to 1.2, allowing 20% extra range for fixations that map to outside the display. This extra data range allows for identification of fixations outside the display. (p. 142 bottom)  
- - - -
## Step 1: Averaging Gaze
```
# after samples have been limited to band pass periods
avg_gaze_df['Avg_X'] = high_passed_samples_df[['LX', 'RX']].mean(axis = 1)
avg_gaze_df['Avg_Y'] = high_passed_samples_df[['LY', 'RY']].mean(axis = 1)
```

## Step 2: Removing Saccades and Blinks
All samples that were closer than 20 ms to a binocular saccade or closer than 50 ms to a blink were identified and removed.
```
for buffer in tqdm(buffered_saccades_ar):
# note that the first time point of each saccade (and blinks below) is kepted for latter collation so it's not dropped here
	rows_to_drop = rows_to_drop.append(avg_gaze_df[(avg_gaze_df['Time'] > buffer['start_end_time_ms_binocular'][0]) & (avg_gaze_df['Time'] <= buffer['start_end_time_ms_binocular'][1])]) # find rows with time that is within the buffer range of saccades
	time_points_to_collate_ms += [buffer['start_end_time_ms_binocular'][0]] # store one time point per saccade for later collation
```
![](Gaze/Uncollated%20Gaze.png)
*In this preview plot, buffered saccades and blinks have been removed from the sample; and technically the starting and ending time point of each displacement have been collated, but the gaze displacement has not been set to zero here.*

## Step 3: Collating Gaze
For gaze displacements, the starting time point of each displacement (top of the vertical lines) should be set to the time point at the end of the previous sample; and the ending point of each displacement (bottom of the vertical lines) should be set to the time point at the beginning of the next sample (connecting the dots).
1. After the buffers were added, some saccades and blinks were close in time or even overlapping.
2. Using `df.diff()` seems to be a good choice as long as df value assignment is done cautiously and not in the original df.
![](Gaze/Screen%20Shot%202022-11-25%20at%2018.54.45.png)
*Nov 25: Strangely, not only did the vertical displacements not disappear, parts of the previous plot that were smooth now show displacements. It’s likely that the differences were added to the wrong intervals.*
![](Gaze/Screen%20Shot%202022-11-28%20at%2015.45.42.png)
*Nov 28: Hell yeah! Finally.*

## Step 4
1. Take a 750-ms windowed copy of the cleaned gaze data every 38 ms
2. Fit a linear curve to vertical gaze by time in selected window and another one to the horizontal gaze.
	1. `scipy.optimize.curve_fit(linear_function)`
3. Determine a threshold for cos(atan(vertical/horizontal)) such that time windows with values above this is assigned a specific percept.
	1. **Update from December 2: currently the cosine threshold above is applied to both left and right moving percepts, with the negative horizontal slope used for left-moving percept.**
	2. **JB: Mark switch midway in time windows and store the time of those marks and their cosine values.**
	3. **Issue from December 14: trial times returned by the custom time extraction function are still uncollated; this is why there are empty time windows near the end. Need to distinguish between uncollated and collated time in gaze data as well.**
4. For any sample between 250 ms before and 400 ms after blink, replace with average gaze displacement 100 ms before after that blink buffer.
	1. **Issue from December 4: at this point in the analysis, blinks are not saved outside of the gaze cleaning function. Consider saving blinks, filtering them by trial and returning them in a df so that it’s consistent with the collated gaze data and the behavioral switches.**
	2. **Note from December 7: for each blink, instead of using the start and end time to locate it, consider finding all starting times and subtract 250. Then just identify the 650 ms after that to be the buffered blink**
	3. **JB: between 350 ms before and 250 ms before the blink, average the cosine value and assign it to all 100 ms (2 or 3 samples before). If the center of a time window lies less than 250 ms before the blink, replace it with the average.**
5. Keep the switch if dominance was longer than 500 ms.
