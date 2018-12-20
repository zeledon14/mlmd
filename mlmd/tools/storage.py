import re
import os
import datetime
import numpy as np
def save_features_E_F(feat_2b, feat_3b,feat_name, stru_names,eta2b, Rp, eta3b, cos_p, ftot_stru, ener, X, DX):
#saves the values of FBP->X, DFBP->DX, Energies, Forces for trainint
    #print 'X.shape  ', X.shape
    #print 'DX.shape  ', DX.shape
    dir_name='features_%s_%d'% (feat_name,X.shape[1])
    os.mkdir(dir_name)
    DX_name='%s_DFBP_%d' % (feat_name,X.shape[1])
    X_name='%s_FBP_%d' % (feat_name,X.shape[1])
    ener_name='%s_energies_%d' % (feat_name,X.shape[1])
    data_name='feature_calculation_summary_%s.txt' % (feat_name)
    np.save(dir_name+'/'+X_name, X)
    np.save(dir_name+'/'+ener_name, ener)
    X_data= open(dir_name+'/'+data_name, 'w')
    X_data.write('Feature calculation summary for the %s features, and %s fetures derivatives \n' %(X_name, DX_name))
    date= datetime.datetime.today()
    X_data.write('The calculation was carried out in %s \n' %(date.date()))
    X_data.write('The total number of features calculated is %d \n' %(X.shape[1]))
    X_data.write('The number of 2-body featues is %d \n' % feat_2b)
    X_data.write('The n_p parameters for the 2-body features calculation are:\n')
    string=''
    for i in eta2b:
        string= string+'  %f'%i
    X_data.write(string+'\n')
    X_data.write('The R_p parameters for the 2-body features calculation are:\n')
    string=''
    for lst in Rp:
        for i in lst:
            string= string+'  %f'%i
    X_data.write(string+'\n')
    X_data.write('The number of 3-body featues is %d \n' % feat_3b)
    X_data.write('The n_p parameters for the 3-body features calculation are:\n')
    string=''
    for i in eta3b:
        string= string+'  %f'%i
    X_data.write(string+'\n')
    X_data.write('The cos_p parameters for the 3-body features calculation are:\n')
    string=''
    for lst in cos_p:
        for i in lst:
            string= string+'  %.4f'%i
    X_data.write(string+'\n')
    X_data.close()

    #save forces
    DX_force_dir_name= 'DFBP_force_%s_%d' % (feat_name,X.shape[1])
    os.mkdir(dir_name+'/'+DX_force_dir_name)
    stru_names_fl= open(dir_name+'/'+DX_force_dir_name+'/structure_names_list','w')
    #print len(stru_names)
    for i, nm in enumerate(stru_names):
        #print nm
        if i == 0:
            stru_names_fl.write(nm)
        else:
            stru_names_fl.write(','+nm)
        DX_name='%s_DFBP' %nm
        DX_name_shape='%s_DFBP_shape' % (nm)
        forc_name='%s_forces' % (nm)
        forc_shape_name='%s_forces_shape' % (nm)
        DX_i= DX[i]
        ftot_i= ftot_stru[i]
        forc_shape= np.shape(ftot_i)
        DX_shape= np.shape(DX_i)
        np.save(dir_name+'/'+DX_force_dir_name+'/'+forc_shape_name,
                forc_shape)
        np.save(dir_name+'/'+DX_force_dir_name+'/'+forc_name,
                np.reshape(ftot_i, (ftot_i.shape[0]*ftot_i.shape[1])))
        np.save(dir_name+'/'+DX_force_dir_name+'/'+DX_name_shape,
                DX_shape)
        np.save(dir_name+'/'+DX_force_dir_name+'/'+DX_name,
                np.reshape(DX_i,(DX_i.shape[0],DX_i.shape[1]*DX_i.shape[2])))
    return None

