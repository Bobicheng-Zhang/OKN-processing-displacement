#### sphere pupil
from psychopy import visual, logging, misc, event, core, gui, data
import numpy as np
import os
import sys
import time
import random
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

def go_to_eyelink_interface(width_pix,height_pix,background_color_psychopy,win,graphics_environment):
    
    pylink.closeGraphics()
    pylink.openGraphicsEx(graphics_environment)
    
    while not((pylink.getEYELINK().isConnected() and not pylink.getEYELINK().breakPressed())):
        pass
    
    win.flip()
    pylink.getEYELINK().doTrackerSetup()

def wrap_up_eyelink(tracking, edffile_name):
    if tracking:
        send_message_to_eyelink(tracking, "Experiment done. About to close data file.")
    
        pylink.getEYELINK().setOfflineMode()
        pylink.msecDelay(500)
        
        pylink.getEYELINK().closeDataFile()
        pylink.getEYELINK().close()
        
def send_message_to_eyelink(tracking, the_string):
    if tracking:
        pylink.getEYELINK().sendMessage(the_string)

## Exp & stim settings ===============================================================================================================
# Create an experiment handler
expName='pupil_sfm'
expInfo={'participant': '', 'session': 1}
dlg=gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
blocks=['report', 'ignore']
if expInfo['session'] == 1:  # randomize conditions
    random.shuffle(blocks)
block=blocks[int(dlg.data[1])-1]

if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date']=data.getDateStr()  # add a simple timestamp
expInfo['expName']=expName

_thisDir = os.path.dirname(os.path.abspath(__file__)) # e nsure that relative paths start from the same directory as this script
os.chdir(_thisDir)

# change condition if this participant has done this block
if expInfo['session'] == 2 and '%s_%s_%s' % (expInfo['participant'], expName, block) in str(os.listdir(os.path.join(_thisDir, 'data'))):
    for b in blocks:
        if b != blocks[int(dlg.data[1])-1]:  # if it gets to a new block that has not been saved
            block = b  # change block and break the loop
            break
    
# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s_%s' % (expInfo['participant'], expName, block, expInfo['date'])

# Experiment handler as in builder; seems to be easier for data file management
thisExp = data.ExperimentHandler(name=expName, extraInfo=expInfo, runtimeInfo=None,
                originPath='/Users/beauz/Pavlovian/spherepupil/cp_main.py',
                saveWideText=True, dataFileName=filename)

win = visual.Window(color=[0,0,0], units='pix', fullscr='yes') 
expInfo['frameRate'] = win.getActualFrameRate() # this seems to be working on the mac mini
if expInfo['frameRate'] != None:
    frameDur = 1/round(expInfo['frameRate'])
else:
    frameDur = 1/60  # guess 60 if unavailable
exp_timer = core.Clock()  # to track the time since experiment started

# Stimulus and stuff ===========================================================================================================
width_pix = 1024  # width of the CRT screen in pixel
height_pix = 768  # height
background_lum_pyth = .5
trials_per_block = 10
t_each_trial = 90
num_dots = 400
dot_size = 5
r = 69 # the only value that influences the field size
# crt res: 1024*768 pix, dimensions: 373*280 mm
# at 58 cm screen distance, 5 deg = 136~139 pix in dia; 6 deg = 165~168 pix; 8 deg = 221~223 pix

# set initial coordinates for sphere
h_pos = np.random.random(num_dots) * 360
v_pos = np.random.random(num_dots) * 180 - 90
x, y, z = misc.sph2cart(v_pos, h_pos, r)
dot_age = np.random.uniform(0, 2 * np.pi, num_dots)  # initiate some arbitrary values for the span of each dot
x_y_life = np.stack((x, z, dot_age), axis=1)

# set rotation speed
fps_crt = 75
deg_per_frame = 72/fps_crt  # the CRT is set to 75 fps
rad_per_frame = 72 * np.pi/(180 * fps_crt)

# read in the tiff files for triangles
up_tiff = 'upward_triangle.tiff'
down_tiff = 'downward_triangle.tiff'
up = visual.ElementArrayStim(win, nElements=num_dots, sizes=dot_size, elementTex='none', sfs=0.0001, elementMask=up_tiff, name='up')
down = visual.ElementArrayStim(win, nElements=num_dots, sizes=dot_size, elementTex='none', sfs=0.0001, elementMask=down_tiff, name='down')
shape_seq = [up, down]
sfm_object = shape_seq[round(np.random.rand())]  # randomize the starting orientation of triangles

force_exit = event.getKeys(keyList=['escape'], timeStamped=exp_timer)
if len(force_exit):
    thisExp.abort() # tells the exp handler not to save data if quit at this stage
    core.quit()

# EyeLink set up===============================================================================================================
tracking = True
if tracking:
    
    el = pylink.EyeLink()
    genv = EyeLinkCoreGraphicsPsychoPy(pylink.getEYELINK(), win)
    edffile_name = 'sph_' + str(expInfo['participant']) + block[0]
    
    pylink.getEYELINK().openDataFile(edffile_name)
    pylink.flushGetkeyQueue()
    
    pylink.getEYELINK().setOfflineMode()
    pylink.getEYELINK().sendCommand("screen_pixel_coords =  0 0 "+str(width_pix-1)+" "+str(height_pix-1))
    pylink.getEYELINK().sendMessage("DISPLAY_COORDS  0 0 "+str(width_pix-1)+" "+str(height_pix-1))
    
    tracker_software_ver = 0
    eyelink_ver = pylink.getEYELINK().getTrackerVersion()
    if eyelink_ver == 3:
        tvstr = pylink.getEYELINK().getTrackerVersionString()
        vindex = tvstr.find("EYELINK CL")
        tracker_software_ver = int(float(tvstr[(vindex + len("EYELINK CL")):].strip()))
    
    if eyelink_ver >= 2:
        pylink.getEYELINK().sendCommand("select_parser_configuration 0")
        if eyelink_ver == 2: #turn off scenelink camera stuff
            pylink.getEYELINK().sendCommand("scene_camera_gazemap=NO")
    else:
        pylink.getEYELINK().sendCommand("saccade_velocity_threshold=35")
        pylink.getEYELINK().sendCommand("saccade_acceleration_threshold=9500")
    
    # set EDF file contents (used for offline eye data analysis)
    pylink.getEYELINK().sendCommand("file_event_filter=LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
    if tracker_software_ver>=4:
        pylink.getEYELINK().sendCommand("file_sample_data=LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET,INPUT")
    else:
        pylink.getEYELINK().sendCommand("file_sample_data=LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,INPUT")
    
    # set link data (used for drift correct)
    pylink.getEYELINK().sendCommand("link_event_filter=LEFT,RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT")
    if tracker_software_ver>=4:
        pylink.getEYELINK().sendCommand("link_sample_data=LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET,INPUT")
    else:
        pylink.getEYELINK().sendCommand("link_sample_data=LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT")
    
    pylink.getEYELINK().setCalibrationType('HV9') #Which kind of calibration to use. H3: horizontal 3-point calibration; HV3: 3-point calibration; HV5: 5-point calibration; HV9: 9-point grid calibration

    #and then specify some more settings:
    pylink.getEYELINK().sendCommand("binocular_enabled=YES")
    pylink.getEYELINK().sendCommand("sample_rate=1000")
    pylink.getEYELINK().sendCommand("elcl_tt_power=2")    #apparently, 1=100, 2=75, 3=50
    
    #record pupil area rather than diameter:
    pylink.getEYELINK().sendCommand("pupil_size_diameter=NO")
    
    #use corneal reflection for tracking:
    pylink.getEYELINK().sendCommand("corneal_mode 1")
    
    #use centroid instead of ellipse fit:
    pylink.getEYELINK().sendCommand("force_ellipse_fitter -1")
    pylink.getEYELINK().sendCommand("use_ellipse_fitter=NO")
                
    #----------perform eyelink calibration and start recording data-------
    go_to_eyelink_interface(width_pix,height_pix,background_lum_pyth,win, genv)     #this gives control to the eyelink...
    pylink.getEYELINK().startRecording(1, 1, 1, 1)                                  #this doesn't happen until control is given back to psychopy
    pylink.beginRealTimeMode(100)
    #----------------------
else:
    edffile_name = 'nobody'

# Define a bunch of messages =======================================================================================================
str_welcome_day1 = 'Welcome to the eye tracking experiment! \n \n In this two-part experiment, you will be asked to pay close attention to a moving stimulus on the screen. You will need to report about what you see by pressing keys using the keyboard in front of you, so make sure you carefully read all the instructions. \n \n If you have any questions, feel free to ask the experimenter. If not, let\'s start with the first part.'
str_welcome_day2 = 'Welcome back. This is the second part of the eye tracking experiment. \n \n Like in the first part, you will be paying close attention to a moving stimulus at the center of the screen and reporting about what you see by pressing keys using the keyboard in front of you. But in this part your task will be different.'

str_intro = 'Observe the sphere at the center of the screen; this is what you will be looking at throughout the experiment.'

str_report_intro = 'As you probably have noticed, the sphere is rotating around its axis. But it does not always rotate in the same direction: sometimes it rotates to the left, but at other times it reverses and starts rotating to the right. \n \n For this part of the experiment, your task is to report the changes in the rotation direction of the sphere while trying to look at the center of the sphere. You may also notice other changes but they are irrelevant.'
str_report_task = 'Whenever the sphere starts rotating to the left (so its front surface starts moving leftward), please press and release the left arrow key. Whenever it starts rotating to the right (so its front surface starts moving rightward), please press and release the right arrow key. Keep in mind that the rotation direction may change over time, so you will have to press and release a key to indicate the first rotation direction you see, but also each time you see the rotation direction change.'
str_demo_report = 'For example, at this moment in which direction is the sphere rotating? Press the corresponding key to continue.'
str_report_practice = 'Before we move on to the experiment, let\'s have you try a quick practice. \n \n Remember, you need to press and release the left arrow key when the sphere starts rotating to the left; or press and release the right arrow key when it starts rotating to the right.'
str_report_feedback = 'It looks like you just pressed a key from the other part of the experiment. \n \n Remember, your task during this part of the experiment is to press and release the left arrow key when the sphere starts rotating to the left; or press and release the right arrow key when it starts rotating to the right.'
str_report_reminder = 'Nice job! Now it\'s time for the real experiment trials. \n \n Remember, during this part, you need to press and release the left arrow key when the sphere starts rotating to the left; or press and release the right arrow key when it starts rotating to the right. If you have any questions, please let the experimenter know.'

str_ignore_intro = 'As your probably have noticed, the sphere is made up of little triangles. But these triangles are not always pointed in the same direction: sometimes they point up, but at other times they reverse and start pointing down. \n \n For this part of the experiment, your task is to report the changes in the orientation of the triangles while trying to look at the center of the sphere. You may also notice other changes but they are irrelevant.'
str_ignore_task = 'Whenever the triangles start to point up, please press and release the up arrow key. Whenever they start to point down, please press and release the down arrow key. Keep in mind that the orientation of the triangles may change over time, so you will have to press and release a key to indicate the first orientation you see, but also each time you see the orientation change.'
str_ignore_practice = 'Before we move on to the experiment, let\'s have you try a quick practice. \n \n Remember, you need to press and release the up arrow key when the triangles start to point up, or press and release the down arrow key when they start to point down.'
str_demo_ignore = 'For example, at this moment in which direction are all the triangles pointing? Press the corresponding key to continue.'
str_ignore_feedback = 'It looks like you just pressed a key from the other part of the experiment. Remember, your task during this part of the experiment is to press and release the up arrow key when the triangles start to point up, or press and release the down arrow key when they start to point down.'
str_ignore_reminder = 'Nice job! Now it\'s time for the real experiment trials. \n \n Remember, during this part, you need to press and release the up arrow key when the triangles start to point up, or press and release the down arrow key when they start to point down. If you have any questions, please let the experimenter know.'

str_space = 'Press the spacebar to continue.'
str_head = 'Remember to try to keep your head fixated on the chin rest throughout the trial with your forehead pressed against the bar.'
str_break = 'You are half-way through this part of the experiment! Please wait while we prepare the next few trials for you. Feel free to take a quick break during this time if you would like.'
str_end = 'Thank you for your participation. This experiment is now finished. \n \n The experimenter will be right with you.'

msg_intro = visual.TextStim(win, str_intro, pos=(0, 300), height=22, wrapWidth=720)
msg_space = visual.TextStim(win, str_space, pos=(0, -200), height=22)
msg_head = visual.TextStim(win, str_head, pos=(0, -200), height=22, wrapWidth=720)
msg_end = visual.TextStim(win, str_end, height=22, wrapWidth=720)
keys_report = ['left', 'right']
keys_ignore = ['up', 'down']

# Welcome screen ===============================================================================
continue_instructions=False
if dlg.data[1] == 1:  # part 1
    msg_welcome = visual.TextStim(win,  str_welcome_day1, height=22, wrapWidth=720)
    msg_welcome.draw()
    msg_space.draw()
    continue_instructions=True
else:  # part 2
    msg_welcome = visual.TextStim(win,  str_welcome_day2, height=22, wrapWidth=720)
    msg_welcome.draw()
    msg_space.draw()
win.flip()

if block == 'report':
    msg_block_intro = visual.TextStim(win, str_report_intro, pos=(0, 300), height=22, wrapWidth=720)
    msg_task = visual.TextStim(win, str_report_task, pos=(0, 300), height=22, wrapWidth=720)
    msg_demo = visual.TextStim(win, str_demo_report, pos=(0,-300), height=22, wrapWidth=720)
    msg_practice = visual.TextStim(win, str_report_practice, height=22, wrapWidth=720)
    msg_block_start = visual.TextStim(win, str_report_reminder, height=22, wrapWidth=720)
    if expInfo['session'] == 2:
        msg_feedback = visual.TextStim(win, str_report_feedback, height=22, wrapWidth=720)
else:
    msg_block_intro = visual.TextStim(win, str_ignore_intro, pos=(0, 300), height=22, wrapWidth=720)
    msg_task = visual.TextStim(win, str_ignore_task, pos=(0, 300), height=22, wrapWidth=720)
    msg_demo = visual.TextStim(win, str_demo_ignore, pos=(0,-300), height=22, wrapWidth=720)
    msg_practice = visual.TextStim(win, str_ignore_practice, height=22, wrapWidth=720)
    msg_block_start = visual.TextStim(win, str_ignore_reminder, height=22, wrapWidth=720)
    if expInfo['session'] == 2:
        msg_feedback=visual.TextStim(win, str_ignore_feedback, height=22, wrapWidth=720)

key_1 = event.waitKeys(keyList=['space'], timeStamped=exp_timer)
if len(key_1):
    thisExp.addData('exp_time_stamp', key_1[-1][1]) # only rt
    thisExp.nextEntry()

win.flip()
time.sleep(.5)
event.clearEvents()

# Instructions  
while continue_instructions:
    t_in_demo = exp_timer.getTime()
    msg_intro.draw()
    msg_space.draw()
        
    dots_to_replace = 0
    dot_age += np.pi * 2/fps_crt  # update new dot age
    
    dots_to_replace = np.where(np.sin(dot_age) > 0)[0]  # find the index of the dots to teleport
    h_pos = h_pos + deg_per_frame
    x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
    if len(dots_to_replace):
        
        dot_age[dots_to_replace] += np.pi
        
        h_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 360
        v_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 180 - 90
        x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
    x_y_life = np.stack((x, z, dot_age), axis = 1)
    sfm_object.setXYs([[i[0], i[1]] for i in x_y_life])
    sfm_object.draw()
    win.flip()
    
    key_2 = event.getKeys(keyList=['space'], timeStamped=exp_timer)
    if len(key_2) and t_in_demo >= 2:
        thisExp.addData('exp_time_stamp', key_2[-1][1])
        thisExp.nextEntry()
        continue_instructions=False

current_shape_index = 0 
next_shift_index = 0
shift_times = np.random.uniform(5, 600, 160)
shift_times.sort()
shift_intervals = np.diff(shift_times)
next_shift_time = shift_times[next_shift_index]
time_till_next_shift = next_shift_time

win.flip()
time.sleep(.5)
event.clearEvents()

continue_instructions = True
while continue_instructions:
    
    t_demo = exp_timer.getTime()
    msg_block_intro.draw()
    msg_space.draw()
    
    dots_to_replace = 0
    dot_age += np.pi * 2/fps_crt  # update new dot age
    
    dots_to_replace = np.where(np.sin(dot_age) > 0)[0]  # find the index of the dots to teleport
    h_pos = h_pos + deg_per_frame
    x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
    if len(dots_to_replace):
        
        dot_age[dots_to_replace] += np.pi
        
        h_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 360
        v_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 180 - 90
        x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
    x_y_life = np.stack((x, z, dot_age), axis = 1)
    sfm_object.setXYs([[i[0], i[1]] for i in x_y_life])
    sfm_object.draw()
    win.flip()
    
    if t_demo >= next_shift_time: # if the sphere has reached the originally designated shift time
        # adjust index for the next shift
        next_shift_index += 1 # move on to the next shift index, no matter shift or skip
        next_shift_time=shift_times[next_shift_index]  # update the next time the shape shifts
        next_interval_index=next_shift_index - 1  # shift time (n-1) corresponds to shift interval (n);
         
         # if the next shift is at a resaonble pace, execute it
        if time_till_next_shift >= 3:
            
            current_shape_index += 1
            next_shape_index=current_shape_index % 2
            sfm_object=shape_seq[next_shape_index]  # shapeshifting
            
            if sfm_object==up:  # slightly adjust the vertical positioning so the change is not too drastic
                v_pos += .3 * dot_size
            else:
                v_pos -= .3 * dot_size
                         
            time_till_next_shift=shift_intervals[next_interval_index]
            
        else:  # if the next shift is too soon, skip it
            time_till_next_shift += shift_intervals[next_interval_index]
        
    key_3=event.getKeys(keyList=['space'], timeStamped=exp_timer)
    if len(key_3) and t_demo >= 3:
        
        thisExp.addData('exp_time_stamp', key_3[-1][1]) 
        thisExp.nextEntry()
        continue_instructions=False

current_shape_index = 0 
next_shift_index = 0
shift_times = np.random.uniform(5, 300, 50)
shift_times.sort()
shift_intervals = np.diff(shift_times)
next_shift_time = shift_times[next_shift_index]
time_till_next_shift = next_shift_time

win.flip()
time.sleep(.5)
event.clearEvents()
demo_timer = getTime()

continue_instructions=True
while continue_instructions:
    
    t_demo = demo_timer.getTime()
    msg_task.draw()
    msg_demo.draw()
    
    dots_to_replace = 0
    dot_age += np.pi * 2/fps_crt  # update new dot age
    
    dots_to_replace = np.where(np.sin(dot_age) > 0)[0]  # find the index of the dots to teleport
    h_pos = h_pos + deg_per_frame
    x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
    if len(dots_to_replace):
        
        dot_age[dots_to_replace] += np.pi
        
        h_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 360
        v_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 180 - 90
        x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
    x_y_life = np.stack((x, z, dot_age), axis = 1)
    sfm_object.setXYs([[i[0], i[1]] for i in x_y_life])
    sfm_object.draw()
    win.flip()
    
    if t_demo >= next_shift_time: # if the sphere has reached the originally designated shift time
        # adjust index for the next shift
        next_shift_index += 1 # move on to the next shift index, no matter shift or skip
        next_shift_time=shift_times[next_shift_index]  # update the next time the shape shifts
        next_interval_index=next_shift_index - 1  # shift time (n-1) corresponds to shift interval (n);
        
         # if the next shift is at a resaonble pace, execute it
        if time_till_next_shift >= 3:
            
            current_shape_index += 1
            next_shape_index=current_shape_index % 2
            sfm_object=shape_seq[next_shape_index]  # shapeshifting
            
            if sfm_object == up:  # slightly adjust the vertical positioning so the change is not too drastic
                v_pos += .3 * dot_size
            else:
                v_pos -= .3 * dot_size            
            time_till_next_shift=shift_intervals[next_interval_index]
            
        else:  # if the next shift is too soon, skip it
            time_till_next_shift += shift_intervals[next_interval_index]
        
    if block == 'ignore':
        key_4 = event.getKeys(keyList=keys_ignore, timeStamped=exp_timer)
    else:
        key_4 = event.getKeys(keyList=keys_report, timeStamped=exp_timer)

    if len(key_4) and t_demo >= 2:
        thisExp.addData('exp_time_stamp', key_4[-1][1]) 
        thisExp.nextEntry()
        continue_instructions=False

msg_practice.draw()
msg_space.draw()
win.flip()
key_5 = event.waitKeys(keyList=['space'], timeStamped=exp_timer)

current_shape_index = 0 
next_shift_index = 0
shift_times = np.random.uniform(3, 300, 80)
shift_times.sort()
shift_intervals = np.diff(shift_times)
next_shift_time = shift_times[next_shift_index]
time_till_next_shift = next_shift_time

win.flip()
time.sleep(.5)
event.clearEvents()
practice_timer = core.Clock()
continue_practice = True
while continue_practice: 
    
    force_exit = event.getKeys(keyList=['escape'], timeStamped=exp_timer)
    if len(force_exit):
        thisExp.abort()
        core.quit()
    
    t_practice = practice_timer.getTime()
    dots_to_replace = 0
    dot_age += np.pi * 2/fps_crt  # update new dot age
    
    dots_to_replace = np.where(np.sin(dot_age) > 0)[0]  # find the index of the dots to teleport
    h_pos = h_pos + deg_per_frame
    x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
    if len(dots_to_replace):
        
        dot_age[dots_to_replace] += np.pi

        h_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 360
        v_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 180 - 90
        x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
    x_y_life = np.stack((x, z, dot_age), axis = 1)
    sfm_object.setXYs([[i[0], i[1]] for i in x_y_life])
    sfm_object.draw()
    win.flip()
        
    practice_keys=event.getKeys(keyList=['up', 'down', 'left', 'right'], timeStamped=exp_timer)
    if len(practice_keys) and expInfo['session'] == 2:
        
        if block == 'ignore' and practice_keys[0][0] in keys_report:
            msg_feedback.draw()
            win.flip()
            time.sleep(6)
            
        elif block == 'report' and practice_keys[0][0] in keys_ignore:
            msg_feedback.draw()
            win.flip()
            time.sleep(6)
            
        practice_keys=[]
        
    if t_practice >= next_shift_time: # if the sphere has reached the originally designated shift time
        # adjust index for the next shift
        next_shift_index += 1 # move on to the next shift index, no matter shift or skip
        next_shift_time=shift_times[next_shift_index]  # update the next time the shape shifts
        next_interval_index=next_shift_index - 1  # shift time (n-1) corresponds to shift interval (n);
        
         # if the next shift is at a resaonble pace, execute it
        if time_till_next_shift >= 3:
            
            current_shape_index += 1
            next_shape_index=current_shape_index % 2
            sfm_object=shape_seq[next_shape_index]  # shapeshifting
            
            if sfm_object==up:  # slightly adjust the vertical positioning so the change is not too drastic
                v_pos += .3 * dot_size
            else:
                v_pos -= .3 * dot_size
                    
            time_till_next_shift=shift_intervals[next_interval_index]
            
        else:  # if the next shift is too soon, skip it
            time_till_next_shift += shift_intervals[next_interval_index]

    if t_practice >= 60: 
        sfm_object.setAutoDraw(False)
        continue_practice=False

msg_block_start.draw()
msg_space.draw()
win.flip()

pre_block_key=event.waitKeys(keyList=['space'], timeStamped=exp_timer)
thisExp.addData('exp_time_stamp', pre_block_key[-1][1])
event.clearEvents()

# Trial time, let's go! Give me your data!
for this_trial in range(trials_per_block):
    
    trial_num=this_trial + 1
    thisExp.addData('trial_number', trial_num) # add trial number to output
        
    if trial_num == 5:
        msg_break.draw()
        win.flip()
        time.sleep(60)
        
    msg_trial_start=visual.TextStim(win, 'Press the spacebar to start trial ' + str(trial_num) + ' of 10', height=22, wrapWidth=720)
    msg_trial_start.draw()
    msg_head.draw()
    win.flip()

    # Check for recalibration ~~~~~~~~~~~~
    recalibration_key=event.getKeys(keyList=['w'], timeStamped=exp_timer)
    if len(recalibration_key):
        send_message_to_eyelink(tracking, 'recalibration started')
        thisExp.addData('recalibration', 'started')
        pylink.getEYELINK().setOfflineMode()
        time.sleep(.1)
        go_to_eyelink_interface(width_pix,height_pix,background_lum_pyth,win, genv)        #to allow recalibration 
        pylink.getEYELINK().startRecording(1, 1, 1, 1)
        pylink.beginRealTimeMode(100)
        send_message_to_eyelink(tracking, 'recalibration ended')
        thisExp.addData('recalibration', 'ended')
    
    check_escape = True
    while check_escape:
        force_exit=event.getKeys(keyList=['escape'], timeStamped=exp_timer)
        if len(force_exit):
            core.quit()
        force_continue=event.getKeys(keyList=['space'], timeStamped=exp_timer)
        if len(force_continue):
            check_escape=False
            
    send_message_to_eyelink(tracking, 'trial ' + str(trial_num) + ' started at ' + str(exp_timer.getTime()))
    thisExp.addData('exp_time_stamp', force_continue[-1][1])
    
    current_shape_index = np.random.randint(2)
    sfm_object = shape_seq[current_shape_index]
    shift_times = np.random.uniform(0, 122, 60)  # on avg 2 sec per shift
    shift_times.sort()
    shift_times = np.append(shift_times, t_each_trial + 2) # just in case index got out of bounds
    thisExp.addData('shift_times', shift_times)
    shift_intervals = np.diff(shift_times)
    next_shift_index = 0
    next_shift_time = shift_times[next_shift_index]
    time_till_next_shift = next_shift_time # make this the zeroth interval before the first shift
    
    # Make these columns in the output before taking any input
    thisExp.addData('current_shape', sfm_object.name)
    thisExp.addData('shift_status', 'n/a')
    thisExp.addData('shift_time', 0)
    thisExp.addData('perceived_shape', 'n/a')
    thisExp.addData('direction', 'trial started here') 
    thisExp.addData('direction_rt', 0)
    thisExp.nextEntry()
    
    h_pos = np.random.random(num_dots)*360 # azimuth
    v_pos = np.random.random(num_dots)*180-90 # elevation

    sfm_timer = core.Clock()
    continue_trial = True
    while continue_trial:
        t_sfm = sfm_timer.getTime() # the only true duration of the trial
        force_exit = event.getKeys(keyList=['escape'], timeStamped=exp_timer)
        if len(force_exit):
            core.quit()
        
        dots_to_replace = 0
        dot_age += np.pi * 2/fps_crt  # update new dot age
        
        dots_to_replace = np.where(np.sin(dot_age) > 0)[0]  # find the index of the dots to teleport
        h_pos = h_pos + deg_per_frame
        x, y, z = misc.sph2cart(v_pos, h_pos, r)
        
        if len(dots_to_replace):
            
            dot_age[dots_to_replace] += np.pi
            
            h_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 360
            v_pos[dots_to_replace] = np.random.random(len(dots_to_replace)) * 180 - 90
            x, y, z = misc.sph2cart(v_pos, h_pos, r)
    
        x_y_life = np.stack((x, z, dot_age), axis = 1)
        sfm_object.setXYs([[i[0], i[1]] for i in x_y_life])
        sfm_object.draw()
        win.flip()
            
        if t_sfm >= next_shift_time: # if the sphere has reached the originally designated shift time
            # adjust index for the next shift
            next_shift_index += 1 # move on to the next shift index, no matter shift or skip
            next_shift_time=shift_times[next_shift_index]  # update the next time the shape shifts
            next_interval_index=next_shift_index - 1  # shift time (n-1) corresponds to shift interval (n);
            
             # if the next shift is at a resaonble pace, execute it
            if time_till_next_shift >= 3:
                current_shape_index += 1
                next_shape_index=current_shape_index % 2
                sfm_object=shape_seq[next_shape_index]  # shapeshifting
                if sfm_object==up:  # slightly adjust the vertical positioning so the change is not too drastic
                    v_pos += .3 * dot_size
                else:
                    v_pos -= .3 * dot_size
                
                thisExp.addData('current_shape', sfm_object.name)
                thisExp.addData('shift_status', 'shifted')
                thisExp.addData('shift_time', t_sfm)
                thisExp.nextEntry()
                
                time_till_next_shift = shift_intervals[next_interval_index]
            
            # if the next shift is too soon, skip it
            else:
                thisExp.addData('current_shape', sfm_object.name)
                thisExp.addData('shift_status', 'skipped')
                thisExp.addData('shift_time', t_sfm)
                time_till_next_shift += shift_intervals[next_interval_index] # add up the next interval
                
        ## Add keys to the output file
        direction_keys = event.getKeys(keyList=keys_report, timeStamped=sfm_timer)
        shape_keys = event.getKeys(keyList=keys_ignore, timeStamped=sfm_timer)
        
        if len(direction_keys): # add direction keys
            for k in range(len(direction_keys)):
                
                thisExp.addData('perceived_shape', sfm_object.name)
                thisExp.addData('direction', direction_keys[k][0])
                thisExp.addData('direction_rt', direction_keys[k][1])
                thisExp.nextEntry()
                
            direction_keys = [] # crude way to clear any previous key in the buffer but it works
                
        if len(shape_keys): # add shapeshifting keys
            for s in range(len(shape_keys)):
                
                thisExp.addData('perceived_shape', sfm_object.name)
                thisExp.addData('shapeshift', shape_keys[s][0])
                thisExp.addData('shapeshift_rt', shape_keys[s][1])
                thisExp.nextEntry()
                
            shape_keys=[]
                
        if t_sfm >= t_each_trial: # subtract waiting time before each trial
            continue_trial = False
            
    send_message_to_eyelink(tracking, 'trial ' + str(trial_num) + ' ended')

msg_end.draw()
win.flip()

end_key = event.waitKeys(keyList=None, timeStamped=exp_timer)
thisExp.addData('total_exp_duration', end_key[-1][1])
logging.console.setLevel(logging.WARNING)
wrap_up_eyelink(tracking, edffile_name)
win.close()
core.quit()

