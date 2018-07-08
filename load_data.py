
import numpy as np
import h5py
from nbfinder import NotebookFinder
import sys
sys.meta_path.append(NotebookFinder())
from util import add_pulse_to_inp_tensor, get_nonempty_pulses, total_doms, total_height, total_width, get_pulse_array

def make_dataset(filename, sig_or_bg):
    hf = h5py.File(filename)
    pulse_array_keys = get_nonempty_pulses(hf)
    num_events = len(pulse_array_keys)
    
    
    tens = np.zeros((num_events, total_doms,total_height, total_width))
    
    
    for ex_num, pulse_array_key in enumerate(pulse_array_keys):
        pulse_array = get_pulse_array(hf, pulse_array_key)
        add_pulse_to_inp_tensor(tens, ex_num, pulse_array)
        
    lbls = np.ones((num_events,)) if sig_or_bg == "sig" else np.zeros((num_events,))
    
    return tens, lbls
        
    
    
    

def get_data(sig_filename_list, bg_filename_list):
    x, y = make_dataset(sig_filename_list[0], "sig")
    for fn in sig_filename_list[1:]:
        xs,ys = make_dataset(fn, "sig")
        x = np.vstack((x,xs))
        y = np.concatenate((y,ys))
    for fn in bg_filename_list:
        xb,yb = make_dataset(fn, "bg")
        x = np.vstack((x,xb))
        y = np.concatenate((y,yb))
    
    return x,y
        

     

if __name__ == "__main__":
    sig_list = ["Level2_nugen_numu_IC86.2012.011070.00000XX.hdf5"]
    bg_list = ["Level2_IC86.2012_corsika.011057.00000XX.hdf5"]
    x,y = get_data(sig_list, bg_list)
