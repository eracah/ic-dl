
# coding: utf-8

# In[2]:


from nbfinder import NotebookFinder
import sys
sys.meta_path.append(NotebookFinder())
from load_data import get_data
import keras


# In[ ]:


sig_list = ["Level2_nugen_numu_IC86.2012.011070.00000XX.hdf5"]
bg_list = ["Level2_IC86.2012_corsika.011057.00000XX.hdf5"]
x,y = get_data(sig_list, bg_list)

