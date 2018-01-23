# construct pattern of direct beam and reflect from a model
# build a one-d model
import glob
import os
import numpy as np
import matplotlib.pyplot as plt # for showing image
from pylab import * # for multiple figure window
from skimage import io
import re
import statsmodels.api as sm


#os.chdir('/Users/jiliangliu/dropbox/rigaku/test_VA77')
#data = np.genfromtxt('test.dat', comments=";", delimiter=' ')
os.chdir('/Users/jiliangliu/dropbox/rigaku//VA67champ')
data = np.genfromtxt('VA67champ.dat', comments=";", delimiter=' ')
#os.chdir('/Users/jiliangliu/dropbox/rigaku//VA67horizontal')
#data = np.genfromtxt('VA67horizontal.dat', comments=";", delimiter=' ')
reflc = (data[:,1]/data[46,1])**.5
q_reflc = np.sin(np.radians(data[:,0])/2)*4*np.pi/1.54

#reflc = (np.load('simulated_R_and_T_new_VA67hor_2.npz')['R12'])**.5
#trans_index = (np.load('simulated_R_and_T_new_4.npz')['T01'])**.5
trans_index = (np.load('simulated_R_and_T_sept_20.npz')['T01'])**.5
reflc = (np.load('simulated_R_and_T_sept_20.npz')['R01'])**.5

detector_distance = 8.278
wavelength = 1.127
ratioDw = 29.27
#figure(1);figure(2);figure(3);figure(4)
#fig,ax=subplots(2,3,figsize=(20,12))
#ct_f = 0.1 #from data ct_f calculated around 0.14
ct_f = np.load('simulated_R_and_T_sept_20.npz')['criti_1'] #0.11 #0.061 # champ #0.14 va77#va67 champ 0.12
ct_si = 0.21
k0 = 2*pi/wavelength


#film_n = 1-1.226e-6
film_n = 1-(np.radians(ct_f)/2**.5)**2
ambient_n = 1.
alpha_incident = np.array([.16,.18,.20])#+0.005
list1_array=np.array([4,6,8])
#VA77 can not sure the angle below .15 is correct or not, seems 0.14 is samilar to 0.16 pattern
alpha_incident = np.radians(alpha_incident)
x0_length=300
#alpha_incident = np.radians(.15)
fitting_portion_model = np.zeros((x0_length,len(alpha_incident)))
x0 = np.linspace(0,.1,x0_length) 

import time
t = time.time()
from scipy.optimize import minimize                            
    
os.chdir('/Users/jiliangliu/Dropbox/GISAXS/GISAXS_for_VA67_and_VA77/GISAXS_large_portion')
list1 = glob.glob('VA67champ_*_ave.npz')
im_recons = np.zeros((x0_length,980))
#now need to avoid vertical column with nan

x=np.copy(x0)
col_index = 310#310#434
ycenter = 913

qz = 2*pi*2*np.sin(np.arcsin((ycenter-np.arange(0,1043,1))*172*1e-6/detector_distance)/2)/wavelength
qz = flipud(qz)
