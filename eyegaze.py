import math
import csv
import numpy
import matplotlib.pyplot as plt
import numpy as np

small_threshold = 10
large_threshold = 50

def read_raw(filename):
    """
        Reads in SMI eye data txt files
        assuming the first row is the timestamp, the 4th are x coordinates
        5th y coordinates of eyegaze
        returns: time, x, y
    """
    #read eyedata
    f=open(filename,'r')
    line=f.readlines()
    
    #read into lists
    time=[]
    x=[]
    y=[]
    for i in range(len(line)-1):
        if line[i][0] != '#' and(line[i][0] !='T') and not('MSG' in line[i]):
            line[i]=line[i].replace(',','.')
            s=line[i].split()
            time.append(float(s[0]))
            x.append(float(s[3]))
            y.append(float(s[4]))
    return time, x, y

def save_raw(fname, time, x, y):
    """
        saves 3 lists (time, x, y) to a filename
        in csv format
    """
    nptime = np.array(time)
    npx = np.array(x)
    npy = np.array(y)
    combined = np.vstack((nptime, npx, npy)).T
    np.savetxt(fname, combined, delimiter=',', fmt='%.7g')


def pplot(x,y,duration):
    """
        helper function for plotting matpoltib
    """
    fig, axes = plt.subplots(1,1, sharey=True)
    axes.plot(x,y,'-',c='grey',linewidth=1.3,zorder=1,alpha=0.95)
    axes.scatter(x,y,s=0.0002*numpy.array(duration),c='b',zorder=2,alpha=0.5)

def small_fixation(x,y):
    """
        detect four points in small_fixation. 
        If there are four points in small_thresfold, 
        it is storaged in fixation and returned.
    """
    t = 1
    for i in range(4):
        for k in range(4):
             if math.hypot((x[i]-x[k]),(y[i]-y[k])) > small_threshold:
                 t = 0
                 break
    if  t == 1:
        return 1
        #true
    else :
        return 0
        #false

def large_fixation(time,x,y,s):
    """
    detect  large_fixation
    If a point is detected fixation, it is storaged  fixation_x,foxaton_y,fixation_time
    first, small_fixation are storaged on fixation_time,fixation_x,fixation_y
    """
    fixation_time=[time[s],time[s+1],time[s+2],time[s+3]]
    fixation_x=[x[s],x[s+1],x[s+2],x[s+3]]
    fixation_y=[y[s],y[s+1],y[s+2],y[s+3]]
    w=s
   
    for i in range(s+4,len(x)-3):
        #detect the time that between new gaze point and first fixation point is shorter than 250ms 
        if time[i]-fixation_time[0]>25000000:#25000000
            w=i
            break
        else:
           # print(time[i]-fixation_time[0])
            if distance_calculation(x,y,i,fixation_x,fixation_y)==1:
                w=i
                fixation_time.append(time[i])
                fixation_x.append(x[i])
                fixation_y.append(y[i])
            #If three point are not detected fixation in succession, fixation detection finish    
            elif (distance_calculation(x,y,i+1,fixation_x,fixation_y)==0)and(distance_calculation(x,y,i+2,fixation_x,fixation_y)==0):
                w=i
                break   
    return fixation_time,fixation_x,fixation_y,w

def distance_calculation(x,y,i,fixation_x,fixation_y):
    #caliculate the distance between one point and all points that detected fixation
    #If all distance are in large_threhold, return true
    w=1;
    for k in range(len(fixation_x)):
        if  math.hypot((x[i]-fixation_x[k]),(y[i]-fixation_y[k])) > large_threshold:
            w = 0;
            break;   
    if w == 0:
        return 0
        
       #false
    elif  w == 1:
        return 1
       #true

def calculate_ave(fixation_time,fixation_x,fixation_y):
    #caliculate average of points that are detected fixation
    fixation_duration=(fixation_time[-1]-fixation_time[0])
    fixation_time=numpy.mean(fixation_time)
    fixation_x=numpy.mean(fixation_x)
    fixation_y=numpy.mean(fixation_y)
    
    return fixation_time,fixation_duration,fixation_x,fixation_y


def filter_fixations(time,x,y):
    fixation_value=[]
    q=0
    for e in range(len(x)-4):
        if q>len(x)-4:
            break
                #first_x,first_y are first four points     
        first_x=[x[q],x[q+1],x[q+2],x[q+3]]
        first_y=[y[q],y[q+1],y[q+2],y[q+3]]
        #print first_x, first_y
        if (small_fixation(first_x,first_y)==1):
            (fixation_time,fixation_x,fixation_y,q)=large_fixation(time,x,y,q)
            fixation_value.append(calculate_ave(fixation_time, fixation_x, fixation_y))
            if fixation_value[-1][2]==0:
                fixation_value.pop()
        q=q+1
    #print fixation_value
    time = []
    fixation_duration = []
    fixation_x = []
    fixation_y = []
    
    for a in range(len(fixation_value)):
        #print fixation_value[a]
        time.append(fixation_value[a][0])
        fixation_duration.append(fixation_value[a][1])
        fixation_x.append(fixation_value[a][2])
        fixation_y.append(fixation_value[a][3])
        
    return time, fixation_duration, fixation_x, fixation_y

import pandas
import numpy.linalg as lin
from matplotlib.mlab import PCA
import numpy as np
import pandas
from math import atan2, pi, degrees
from numpy import var, mean, median
import ngram

def gb(x, y):
    angle = degrees(atan2(y, x))
    return angle


def get_saccade_directions(d,x,y):
    npd = np.array(d)
    npx = np.array(x)
    npy = np.array(y)
    combined = np.vstack((npd, npx, npy)).T
    mdatal = []
    for d in combined:
        mdatal.append({'len':d[0],'x':d[1],'y':d[2]})

    as1 = []
    as2 = []
    as3 = []
    as4 = []
    for w in slidingWindow(mdatal,125,25):
        s1,s2,s3,s4=calc_features(w)
        as1.append(s1)
        as2.append(s2)
        as3.append(s3)
        as4.append(s4)
    return as1, as2, as3, as4

def calc_features(eyegaze):
    '''
    calculate some features on an eye fixations
    it assumes a dictionary with keys:
    t = time
    len = length of fixation
    x = x coordinate of fixation
    y = y coordinate of fixation
    '''
    data = pandas.DataFrame(eyegaze)
    #remove mean
    #diff between two points
    d_d  = data.diff()
    #calculate slope
       #calculate the angles between two points (using the difference)
    angles = []
    for pt in [(d_d.x[i],d_d.y[i]) for i in range(1,len(d_d))]:
        angles.append(gb(pt[0], pt[1]))
    
    d_a = pandas.DataFrame(angles)

    #count angles between the given degrees
    a1c = np.count_nonzero((d_a < -45.0) | (d_a <= 45.0))
    a2c = np.count_nonzero(((d_a < 135) & (d_a <=180)) | ((d_a > -18.0) & (d_a <= 135.0)))
    a3c = np.count_nonzero((d_a > 45.0) & (d_a < 180.0))
    a4c = np.count_nonzero((d_a >= -45.0) & (d_a > 135.0))
    return a1c, a2c, a3c, a4c    
  

def slidingWindow(sequence,winSize,step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)/step)+1
 
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]

