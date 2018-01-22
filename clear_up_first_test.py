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
def coefficient_calculation(x0,alpha_incident,ambient_n,film_n,qz,q_reflc,reflc,trans_index):
    qz_r = np.zeros((len(x0),len(alpha_incident)))
    qz_d = np.zeros((len(x0),len(alpha_incident)))
    qz_f = np.zeros((len(x0),len(alpha_incident)))
    reflc_params = np.zeros((len(x0),len(alpha_incident)))
    trans_params = np.zeros((len(x0),len(alpha_incident)))
    r_f = np.zeros((len(alpha_incident),))
    t_f = np.zeros((len(alpha_incident),))
    alpha_incident_eff = np.zeros((len(alpha_incident),))
    qz_min = np.zeros((len(alpha_incident),)) 
    qz_max = np.zeros((len(alpha_incident),))
    range_index_min = np.zeros((len(alpha_incident),))
    range_index_max = np.zeros((len(alpha_incident),))
    fitting_range_model = np.zeros((len(x0),len(alpha_incident)))
    for i in range(len(alpha_incident)):
            
        # Breiby paper a=only consider two term for DWBA
        #direct beam
        alpha_incident_eff[i] = np.arccos(cos(alpha_incident[i])*ambient_n/film_n)
        two_theta = 2*arcsin(x/2/k0)
        exit_angle_r = two_theta - alpha_incident_eff[i]
        exit_angle_r[exit_angle_r<0] = np.nan
        exit_angle_r_real = np.arccos(cos(exit_angle_r)*film_n/ambient_n)
        two_theta_r_final = exit_angle_r_real + alpha_incident[i]
        qz_d[:,i] = 2*k0*sin(two_theta_r_final/2.)
        
        # reflect neam
        #alpha_incident_eff = np.arccos(cos(alpha_incident[i])*ambient_n/film_n)
        #two_theta = 2*arcsin(x/2/k0)
        exit_angle_d = two_theta + alpha_incident_eff[i]
        exit_angle_d[exit_angle_d<0] = np.nan
        exit_angle_d_real = np.arccos(cos(exit_angle_d)*film_n/ambient_n)
        two_theta_d_final = exit_angle_d_real + alpha_incident[i]
        qz_r[:,i] = 2*k0*sin(two_theta_d_final/2.)
        
        
        qz_max[i] = np.nanmax(qz_d[:,i])
        qz_min[i] = np.nanmin(qz_r[:,i])
        
        fitting_range = qz[np.nanargmin(np.abs(qz-qz_min[i])):np.nanargmin(np.abs(qz-qz_max[i]))]
        fitting_range_model[:,i] = np.interp(np.linspace(np.min(fitting_range),np.max(fitting_range),num=x0_length),fitting_range,fitting_range)
        
        qz_f[:,i] = 2*k0*sin(np.arccos(cos(2*arcsin(fitting_range_model[:,i]/2/k0)-alpha_incident[i])*ambient_n/film_n))
        #print qz_f.shape,fitting_range_model.shape
        reflc_params[:,i] = np.interp(qz_f[:,i],
                                q_reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f[:,i]))):np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f[:,i])))],
                                reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f[:,i]))):np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f[:,i])))])
        trans_params[:,i] = np.interp(qz_f[:,i] ,
                            q_reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f[:,i]))):np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f[:,i])))],
                            trans_index[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f[:,i]))):np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f[:,i])))])
        r_f[i] = reflc[np.nanargmin(np.abs(q_reflc-2*k0*sin(alpha_incident_eff[i])))]
        t_f[i] = trans_index[np.nanargmin(np.abs(q_reflc-2*k0*sin(alpha_incident_eff[i])))]
        range_index_min[i] = np.nanargmin(np.abs(qz-qz_min[i]))
        range_index_max[i] = np.nanargmin(np.abs(qz-qz_max[i]))
    return alpha_incident_eff,qz_r,qz_d,qz_f,reflc_params,trans_params,r_f,t_f,fitting_range_model,qz_min,qz_max,range_index_min,range_index_max

alpha_incident_eff,qz_r,qz_d,qz_f,reflc_params,trans_params,r_f,t_f,fitting_range_model,\
qz_min,qz_max,range_index_min,range_index_max =  coefficient_calculation(x0,alpha_incident,
                                                                ambient_n,film_n,qz,q_reflc,reflc,trans_index)
    
for j in range(980): 
    if j in np.concatenate([np.arange(180,245),np.arange(485,496)]):#,np.arange(550,580)]):
        pass
    else:   
        for i in range(len(alpha_incident)):
            
            I1 = (np.load(list1[list1_array[i]])['im'])[:,j]
            I1[np.abs(I1)==inf] = np.nan
            I1 = flipud(I1)
            I1[np.abs(log(I1))==inf] = np.nan
            I1 = np.interp(np.arange(0,len(I1),1),np.arange(0,len(I1),1)[isnan(I1)==0],I1[isnan(I1)==0])
            #figure(1),plot(qz,log(I1),qz_r,log(I1),qz_d,log(I1))
            fitting_range = qz[range_index_min[i]:range_index_max[i]]
            fitting_portion = I1[range_index_min[i]:range_index_max[i]]
            fitting_portion_model[:,i] = np.interp(fitting_range_model[:,i],fitting_range[isnan(log(fitting_portion))==0],fitting_portion[isnan(log(fitting_portion))==0])
        #sys.exit()
        #model = fitting_portion_model
        #qz = fitting_range_model
        y0 = log(fitting_portion_model[:,1])#np.interp(np.linspace(np.min(qz[:,1]),np.max(qz[:,1]),x0_length),qz[:,1],log(model)[:,1])
        #sys.exit()
        if np.size(y0[isnan(y0)==1])==0:
            pass
        else: y0[isnan(y0)==1] = np.interp(x0[isnan(y0)==1],x0[isnan(y0)==0],y0[isnan(y0)==0])
        def fun(y,x=x0,I1=log(fitting_portion_model),qz1=fitting_range_model,alpha_incident=alpha_incident,
                qz_d=qz_d,qz_r=qz_r,t_f=t_f,r_f=r_f,reflc_params=reflc_params,trans_params=trans_params):
            norm_judge = np.zeros((len(alpha_incident),))
            for i in range(len(alpha_incident)):
                I = I1[:,i]
                qz = qz1[:,i]
                if np.size(I[isnan(I)==1])==0:
                    pass
                else: I[isnan(I)==1] = np.interp(qz[isnan(I)==1],qz[isnan(I)==0],I[isnan(I)==0])
                
                qz_max=np.nanmax(qz_d[:,i])
                qz_min=np.nanmin(qz_r[:,i])
                I_direct = np.interp(qz,
                                    qz_d[np.nanargmin(np.abs(qz_d[:,i]-qz_min)):np.nanargmin(np.abs(qz_d[:,i]-qz_max)),i],
                                    y[np.nanargmin(np.abs(qz_d[:,i]-qz_min)):np.nanargmin(np.abs(qz_d[:,i]-qz_max))])
                I_reflect = np.interp(qz,
                                    qz_r[np.nanargmin(np.abs(qz_r[:,i]-qz_min)):np.nanargmin(np.abs(qz_r[:,i]-qz_max)),i],
                                    y[np.nanargmin(np.abs(qz_r[:,i]-qz_min)):np.nanargmin(np.abs(qz_r[:,i]-qz_max))])
                #print I_reflect.shape,qz.shape,qz_r.shape,y.shape,                    
            #return I_reflect,I_direct,qz_max,qz_min
                fitting_portion =I#I[np.nanargmin(np.abs(qz-qz_min)):np.nanargmin(np.abs(qz-qz_max))]
                norm_judge[i] = norm(fitting_portion-log(trans_params[:,i]**2*t_f[i]**2*exp(I_direct)+
                                t_f[i]**2*reflc_params[:,i]**2*exp(I_reflect)+trans_params[:,i]**2*r_f[i]**2*exp(I_reflect)+r_f[i]**2*reflc_params[:,i]**2*exp(I_direct)))
            
            return np.sum(norm_judge)
        #I_reflect,I_direct,qz_max,qz_min =fun(y0,x=x0,I1=log(fitting_portion_model),qz1=fitting_range_model,alpha_incident=alpha_incident,
        #        qz_d=qz_d,qz_r=qz_r,t_f=t_f,r_f=r_f,reflc_params=reflc_params,trans_params=trans_params)            
        #sys.exit()
        #if j == 0:
        ret = minimize(fun, y0,method="L-BFGS-B",options={'maxfun':3000})
        #else:
        #    y0 = smoothed_fitting   
        #    ret = minimize(fun, y0,method="L-BFGS-B",options={'maxfun':3000})
        #im_recons[:,j]=exp(ret['x'])
        smoothed_fitting = np.interp(x0,x0[isnan(ret.x)==0],sm.nonparametric.lowess(ret.x,x0,frac=.03)[:,1])
        im_recons[:,j]=exp(ret.x)
        #im_recons[:,j]=exp(smoothed_fitting)
        if (j%500)==0:
            print time.time()-t

#plot(x0,log(im_recons[:,col_index]),x0,log(fitting_portion_model[:,0]),x0,smoothed_fitting)
#plt.show()    
print time.time()-t
#os.chdir('/Users/jiliangliu/Dropbox/GISAXS/reflectivity/test_two_layer_model/x-ray_reflectivity_analysis/VA77new')
#np.savez('VA77_reconstruction_16_18_20_include_beamstop.npz',im=im_recons,qz=x0)