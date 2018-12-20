import numpy as np
import geometry 

def get_x_2b_rpij(eta, Rp, R):
    Rp= np.reshape(Rp, (len(Rp), 1,1))
    Rt= R.T
    x= np.subtract(Rt,Rp)
    x_2b_rpij= np.exp(np.multiply(-1.0*eta,x**2.0))
    x_2b_rpij= np.reshape(x_2b_rpij, (1,x_2b_rpij.shape[0],x_2b_rpij.shape[1],x_2b_rpij.shape[2]))
    return x_2b_rpij
def get_x2b(x_2b_rpij, delta_2b):
    temp_x_2b= np.multiply(x_2b_rpij, delta_2b)
    temp_x_2b= np.sum(np.sum(temp_x_2b,axis=3), axis=2)
    temp_x_2b= np.reshape(temp_x_2b,(temp_x_2b.shape[0]*temp_x_2b.shape[1]))
    return temp_x_2b
def get_x_3b_rpijk(eta, cos_p, cos_m):
    cos_m_ijk= cos_m
    cos_m_ijk= np.reshape(cos_m_ijk,(1,cos_m_ijk.shape[0],cos_m_ijk.shape[1],cos_m_ijk.shape[2]))
    cos_p= np.reshape(cos_p, (len(cos_p), 1,1,1))
    x= np.subtract(cos_m_ijk,cos_p)
    x_3b_rpijk= np.exp(np.multiply(-1.0*eta,x**2.0))
    x_3b_rpijk= np.reshape(x_3b_rpijk, (1,x_3b_rpijk.shape[0],x_3b_rpijk.shape[1],x_3b_rpijk.shape[2],x_3b_rpijk.shape[3]))
    return x_3b_rpijk
def get_x3b(x_3b_rpijk, delta_3b):
    temp_x_3b= np.multiply(x_3b_rpijk, delta_3b)
    temp_x_3b= np.sum(np.sum(np.sum(temp_x_3b,axis=4), axis=3), axis=2)
    temp_x_3b= np.reshape(temp_x_3b,(temp_x_3b.shape[0]*temp_x_3b.shape[1]))
    return temp_x_3b

def get_x_2b_rpij_and_Dx_2b_rpij(eta, Rp, R):
    Rp= np.reshape(Rp, (len(Rp), 1,1))
    Rt= R.T
    x= np.subtract(Rt,Rp)
    x_2b_rpij= np.exp(np.multiply(-1.0*eta,x**2.0))
    Dx_2b_rpij= np.multiply(x_2b_rpij,(-2.0*eta*x))
    #print x_2b_rpij.shape
    #print Dx_2b_rpij.shape
    x_2b_rpij= np.reshape(x_2b_rpij, (1,x_2b_rpij.shape[0],x_2b_rpij.shape[1],x_2b_rpij.shape[2]))
    Dx_2b_rpij= np.reshape(Dx_2b_rpij, (1,Dx_2b_rpij.shape[0],Dx_2b_rpij.shape[1],Dx_2b_rpij.shape[2]))
    return x_2b_rpij, Dx_2b_rpij 

def get_x_3b_rpijk_and_Dx_3b_rpijk(eta, cos_p, cos_m):
    cos_m_ijk= cos_m
    cos_m_ijk= np.reshape(cos_m_ijk,(1,cos_m_ijk.shape[0],cos_m_ijk.shape[1],cos_m_ijk.shape[2]))
    cos_p= np.reshape(cos_p, (len(cos_p), 1,1,1))
    x= np.subtract(cos_m_ijk,cos_p)
    x_3b_rpijk= np.exp(np.multiply(-1.0*eta,x**2.0))
    Dx_3b_rpijk= np.multiply(x_3b_rpijk,(-2.0*eta*x))
    x_3b_rpijk= np.reshape(x_3b_rpijk, (1,x_3b_rpijk.shape[0],x_3b_rpijk.shape[1],\
                                        x_3b_rpijk.shape[2],x_3b_rpijk.shape[3]))
    Dx_3b_rpijk= np.reshape(Dx_3b_rpijk, (1,Dx_3b_rpijk.shape[0],Dx_3b_rpijk.shape[1],\
                                        Dx_3b_rpijk.shape[2],Dx_3b_rpijk.shape[3]))
    return x_3b_rpijk, Dx_3b_rpijk

def get_DVx3bl(Dx_3b_rpij, delta_3b, Dcos1, Dcos2):
    Dcos1= np.reshape(Dcos1,(1,1,Dcos1.shape[0], Dcos1.shape[1],Dcos1.shape[2],Dcos1.shape[3]))
    Dcos2= np.reshape(Dcos2,(1,1,Dcos2.shape[0], Dcos2.shape[1],Dcos2.shape[2],Dcos2.shape[3]))
    temp_Dx_3b= np.multiply(Dx_3b_rpij, delta_3b)
    temp_Dx_3b= np.reshape(temp_Dx_3b,(temp_Dx_3b.shape[0],temp_Dx_3b.shape[1],temp_Dx_3b.shape[2],\
                                       temp_Dx_3b.shape[3],temp_Dx_3b.shape[4],1))
    x1= np.multiply(temp_Dx_3b,np.add(Dcos1,Dcos2))
    #print 'x1  ', x1.shape
    x1= np.sum(np.sum(x1,axis=4), axis=3)
    
    x2= np.multiply(temp_Dx_3b,-1.0*Dcos1)
    #print 'x2  ', x2.shape
    x2= np.sum(np.sum(x2,axis=3), axis=2)
    
    x3= np.multiply(temp_Dx_3b,-1.0*Dcos2)
    #print 'x3  ', x3.shape
    x3= np.sum(np.sum(x3,axis=4), axis=2)
    
    temp_Dx_3b= np.add(x1,np.add(x2,x3))
    temp_Dx_3b= np.reshape(temp_Dx_3b,(temp_Dx_3b.shape[0]*temp_Dx_3b.shape[1],\
                                       temp_Dx_3b.shape[2],temp_Dx_3b.shape[3]))
    temp_Dx_3b= temp_Dx_3b.transpose((1,0,2))
    return temp_Dx_3b# returns DVx_3bl [l,qxp,3] dependent on r for the structure 
# The reshape
#  |1 2 3|
#A=|4 5 6| after the reshape A =[1,2,3,4,5,6,7,8,9]
#  |7 8 9|

#old get_g2b(I_2b, delta_2b) 
def get_DVx2bl(Dx_2b_rpij, delta_2b, Rinv, Rv):
    Rhat= np.multiply(Rinv,Rv)
    Rhat= np.reshape(Rhat, (1,1,Rhat.shape[0],Rhat.shape[1],Rhat.shape[2]))
    temp_Dx_2b= np.multiply(Dx_2b_rpij, delta_2b)
    temp_Dx_2b= np.reshape(temp_Dx_2b,(temp_Dx_2b.shape[0],temp_Dx_2b.shape[1],\
                                       temp_Dx_2b.shape[2],temp_Dx_2b.shape[3],1))
    temp_Dx_2b= np.multiply(temp_Dx_2b,Rhat)
    temp_Dx_2b= np.sum(temp_Dx_2b,axis=3)
    temp_Dx_2b= np.reshape(temp_Dx_2b,(temp_Dx_2b.shape[0]*temp_Dx_2b.shape[1],\
                                       temp_Dx_2b.shape[2],temp_Dx_2b.shape[3]))
    temp_Dx_2b= temp_Dx_2b.transpose((1,0,2))
    return temp_Dx_2b # returns DVx_2bl [l,qxp,3] dependent on r for the structure 
# The reshape
#  |1 2 3|
#A=|4 5 6| after the reshape A =[1,2,3,4,5,6,7,8,9]
#  |7 8 9|
def build_FBP_DFBP(trans, eta2b, Rp, eta3b, cos_p,species_simb, stru_names, stru):
#Builds the array with features, energies, derivatives of the features and forces
#for the training
# X -> FBP dimensions (structures, number_of_features)
#DX -> Derivative of FBP
#DX dimensions (structures, atoms_in_structure, number_of_features, xyz_components)
    X= [] #features for energy 
    DX= [] # Derivativie of the features, for force
    for l,s in enumerate(stru):
        [Rv, R, Rinv, R2, R2inv, cos_m]= geometry.geometrical_parameters(s)
        [delta_2b, delta_3b]= geometry.delta_2b_delta_3b(species_simb[l], trans, cos_m)
        [Rvij, Rvik, Rvjk]= geometry.get_Rvij_Rvik_Rvjk(Rv)
        #[hatRvij, hatRvik, hatRvjk]= get_hat_Rvij_Rvik_Rvjk(Rv)
        RRinv= geometry.get_RRinv(Rinv)
        [Dcos1, Dcos2]= geometry.get_Dcos(cos_m, RRinv, R2inv, Rvij, Rvik, Rvjk)

        for nn, _ in enumerate(eta2b):# loop over the r 
            #x_2b_rpij= get_x_2b_rpij(eta2b[nn], Rp[nn], R)
            x_2b_rpij, Dx_2b_rpij=  get_x_2b_rpij_and_Dx_2b_rpij(eta2b[nn], Rp[nn], R)
            if nn == 0:
                x2b=  get_x2b(x_2b_rpij, delta_2b)
                Dx2b=  get_DVx2bl(Dx_2b_rpij, delta_2b, Rinv, Rv)
            else:
                x2b= np.concatenate((x2b,get_x2b(x_2b_rpij, delta_2b)))
                Dx2b= np.concatenate((Dx2b,get_DVx2bl(Dx_2b_rpij, delta_2b, Rinv, Rv)),axis=1)

        for nn, _ in enumerate(eta3b):# loop over the r
            #x_3b_rpijk= get_x_3b_rpijk(eta3b[nn], cos_p[nn], cos_m)
            x_3b_rpijk, Dx_3b_rpijk=  get_x_3b_rpijk_and_Dx_3b_rpijk(eta3b[nn], cos_p[nn], cos_m)
            if nn == 0:
                x3b=  get_x3b(x_3b_rpijk, delta_3b)
                Dx3b=  get_DVx3bl(Dx_3b_rpijk, delta_3b, Dcos1, Dcos2)
            else:
                x3b= np.concatenate((x3b, get_x3b(x_3b_rpijk, delta_3b)))
                Dx3b= np.concatenate((Dx3b, get_DVx3bl(Dx_3b_rpijk, delta_3b, Dcos1, Dcos2)), axis=1)

        X.append(np.concatenate((x2b,x3b)))
        DX.append(np.concatenate((Dx2b,Dx3b), axis=1))
    X= np.array(X)
    DX= np.array(DX)
    feat_2b= x2b.shape[0]
    feat_3b= x3b.shape[0]
    return feat_2b, feat_3b,X, DX
