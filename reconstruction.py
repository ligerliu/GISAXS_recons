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

# qx_dimension determined by the input image qx dimension
qx_dimension = 980
# skip_qx correlate the certain qx position completely masked along qz direction which enable to reconstrcut.
skip_qx =  np.concatenate([np.arange(180,245),np.arange(485,496)])

for j in range(qx_dimension): 
    if j in skip_qx:#,np.arange(550,580)]):
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
