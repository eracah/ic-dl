
# coding: utf-8

# In[91]:


import h5py
import numpy as np


# In[92]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


total_doms = 60
total_height = 10
total_width = 20


# In[96]:


def get_keys(hf, name):
    return hf[name].keys()

def get_nonempty_(hf, name):
    '''returns keys, where there exists some data'''
    nonempty_keys = [k for k in get_keys(hf,name) if hf[name][k].shape[0] > 0 ]
    return nonempty_keys

def get_empty_(hf, name):
    '''returns keys where there doesn't exist any data'''
    empty_keys = [k for k in get_keys(hf,name) if hf[name][k].shape[0] == 0 ]
    return empty_keys

def get_nonempty_events(hf):
    '''returns keys of events, where there exists some data'''
    nonempty_events = get_nonempty_(hf, "events")
    return nonempty_events

def get_empty_events(hf):
    '''returns keys of events, where there doesn't exist any data'''
    empty_events = get_empty_(hf, "events")
    return empty_events

def get_nonempty_pulses(hf):
    '''returns keys of pulses, where there exists some data'''
    nonempty_pulses = get_nonempty_(hf, "pulses")
    return nonempty_pulses

def get_empty_pulses(hf):
    '''returns keys of pulses, where there doesn't exist any data'''
    empty_pulses = get_empty_(hf, "pulses")
    return empty_pulses

def get_pulse_array(hf, event_id):
    pulse_arrs = hf["pulses"][event_id][:]
    return pulse_arrs


# In[101]:


def get_height_to_width_map():
    '''list of widths (number of strings) at each height starting from bottom
    aka if patt_from_bot[0] = 6 that means there are 6 strings in the bottommost height of hexagon'''
    patt_from_bot = dict(zip(range(10),[4, 7, 8, 9, 10, 10, 9, 8, 7, 6]))
    return patt_from_bot
        


# In[102]:


def get_height_to_offset_from_left_map():
    '''list of offsets from leftmost stringat each given height starting from bottom'''
    offset_from_bot = dict(zip(range(10),[5, 4, 3, 2, 1, 0, 1, 2, 3, 4]))
    return offset_from_bot
    


# In[246]:


def get_height_to_string_nums_map():
    '''makes a map that takes height from bottom maps to all string nums at that height'''
    patt_from_bot = get_height_to_width_map()
    ind = 78
    d={}
    for height, width in patt_from_bot.iteritems():
        d[height] = range(ind-width + 1, ind + 1)
        ind -= width
    return d
        


# In[132]:


def make_template_arr():
    # make test array
    test_arr = np.zeros((total_height,total_width))
    
    #get all the maps
    hsnm = get_height_to_string_nums_map()
    hoflm = get_height_to_offset_from_left_map()
    hwm = get_height_to_width_map()
    
    #place all the string numbers on the test array
    for height in range(total_height):
        offset_from_left = hoflm[height]
        width = hwm[height]
        end_from_left = offset_from_left + (width * 2) -1
        test_arr[height,  offset_from_left:end_from_left:2] = hsnm[height]
    test_arr =test_arr.astype("int32")
    return test_arr


# In[133]:


def make_string_num_to_arr_inds_map():
    arr=make_template_arr()
    # get coordinates where strings are
    coords = np.argwhere(arr > 0)
    #get string nums in order of coords
    snums = [arr[coord[0], coord[1]] for coord in coords]
    #make map
    snum_to_coords = dict(zip(snums, coords))
    return snum_to_coords


# In[134]:


def get_height_width_from_string_num(string_num):
    ma = make_string_num_to_arr_inds_map()
    height, width = ma[string_num]
    return height, width
    


# In[206]:


def get_string_num(arr):
    string_num_ind = 6
    return arr[string_num_ind]


# In[207]:


def get_sensor_depth(arr):
    ndom_ind = 5
    return arr[ndom_ind]


# In[208]:


def get_total_charge(arr):
    tot_charge_ind = 7
    return arr[tot_charge_ind]


# In[209]:


def get_stats(arr):
    return {"string_num": get_string_num(arr),
            "sensor_depth": get_sensor_depth(arr),
            "total_charge": get_total_charge(arr)}


# In[210]:


def add_pulse_to_inp_tensor(tens,example_number, pulse_arr):
    for pulse in pulse_arr:
        #print pulse
        s = get_stats(pulse)
        string_num, depth, charge = [s[k] for k in ["string_num", "sensor_depth", "total_charge"]]
        if string_num > 78:
            continue #don't worry about deep core for now
        height, width = get_height_width_from_string_num(string_num)
        tens[example_number, depth -1, height, width] = charge
    return tens

