import numpy as np

def geometrical_parameters(strc):
    Rv= []
    R= []
    R2= []
    cos_m= []
    for i in range(strc.shape[0]): #loop over atoms
        #Posi_p= np.delete(Posi, (i), axis=0)# positions of atoms but i
        Rv.append(np.subtract(strc, strc[i])) # distance vector between i and the other atoms
        R.append(np.reshape(np.linalg.norm(Rv[i], axis=1), (Rv[i].shape[0], 1)))
        R2.append(np.square(R[i]))
        cos_mp= np.dot(Rv[i], np.transpose(Rv[i])) # cosine matrix
        x= np.dot(R[i], np.transpose(R[i])) #normalization factor for the cos
        cos_m.append(np.nan_to_num(np.divide(cos_mp, x)))#original
    Rinv= np.divide(1.0,R)
    Rinv[np.isinf(Rinv)] = 0.0 
    R2inv= np.divide(1.0,R2)
    R2inv[np.isinf(Rinv)] = 0.0 
    return np.array(Rv), np.array(R), np.array(Rinv), np.array(R2), np.array(R2inv), np.array(cos_m)
def II_kij(cos_m): #selects the k i j indeses needed
    # i and j != k
    y= np.heaviside(np.square(cos_m), 0.0)
    # taking i != j
    temp= np.ones((cos_m.shape[1], cos_m.shape[2]), dtype='float')
    temp1= np.identity(cos_m.shape[1], dtype='float')
    temp= np.reshape(np.subtract(temp,temp1), (1,cos_m.shape[1],cos_m.shape[2]))
    return np.reshape(np.multiply(y,temp), (1,1,cos_m.shape[0],cos_m.shape[1],cos_m.shape[2]))
def II_ij(shape): #masc to make sure that in 2b interactions i != j
    temp= np.subtract(np.ones((shape,shape)),np.identity(shape))
    temp= np.reshape(temp,(1,1,shape,shape))
    return temp
def delta_2b_delta_3b(species_simb, trans, cos_m):
    if len(trans) == 1:
        delta_2b= np.ones((1, len(species_simb), len(species_simb)), dtype=float)
        delta_3b= np.ones((1, len(species_simb), len(species_simb), len(species_simb)), dtype=float)
        delta_2b= np.reshape(delta_2b, (delta_2b.shape[0],1,delta_2b.shape[1], delta_2b.shape[2]))
        delta_3b= np.reshape(delta_3b, (delta_3b.shape[0],1,delta_3b.shape[1], delta_3b.shape[2],delta_3b.shape[2]))
        delta_2b= np.multiply(delta_2b, II_ij(len(species_simb)))
        delta_3b= np.multiply(delta_3b, II_kij(cos_m))
        return delta_2b, delta_3b#, ints_2b, ints_3b
    if len(trans) > 1: # if there is more than one species
        #2body
        ints_2b= []
        for i in trans:
            for j in trans:
                ints_2b.append([i,j])

        delta_2b= np.zeros((len(ints_2b), len(species_simb), len(species_simb)), dtype=float)

        for l in range(delta_2b.shape[0]):
            esp1= ints_2b[l][0]
            esp2= ints_2b[l][1]
            for i in range(delta_2b.shape[1]):
                for j in range(delta_2b.shape[2]):
                    if esp1 == species_simb[i] and esp2 == species_simb[j]:
                        delta_2b[l,i,j]= 1.0

        ints_3b= []
        for i in trans:
            for j in trans:
                for k in trans:
                    ints_3b.append([i,j,k])

        delta_3b= np.zeros((len(ints_3b), len(species_simb), len(species_simb), len(species_simb)), dtype=float)

        for l in range(delta_3b.shape[0]):
            esp1= ints_3b[l][0]
            esp2= ints_3b[l][1]
            esp3= ints_3b[l][2]
            for i in range(delta_3b.shape[1]):
                for j in range(delta_3b.shape[2]):
                    for k in range(delta_3b.shape[3]):
                        if esp1 == species_simb[i]:
                            if esp2 == species_simb[j] and esp3 == species_simb[k]:
                                delta_3b[l,i,j,k]= 1.0
        delta_2b= np.reshape(delta_2b, (delta_2b.shape[0],1,delta_2b.shape[1], delta_2b.shape[2]))
        delta_2b= np.multiply(delta_2b, II_ij(len(species_simb)))
        delta_3b= np.reshape(delta_3b, (delta_3b.shape[0],1,delta_3b.shape[1], delta_3b.shape[2],delta_3b.shape[2]))
        delta_3b= np.multiply(delta_3b, II_kij(cos_m))
        return delta_2b, delta_3b#, ints_2b, ints_3b
def get_RRinv(Rinv):
    RRinv= []
    RRinv.append([np.dot(x,x.T) for x in Rinv])
    RRinv= np.array(RRinv)
    RRinv= np.reshape(RRinv, (RRinv.shape[1], RRinv.shape[2], RRinv.shape[3],1))
    return RRinv    
def get_Rvij_Rvik_Rvjk(Rv):
    Rvij= Rv.reshape(Rv.shape[0], Rv.shape[1], 1, 3)
    Rvik= Rv.reshape(Rv.shape[0], 1, Rv.shape[1], 3)
    Rvjk= Rv.reshape(1, Rv.shape[0], Rv.shape[1], 3)
    return Rvij, Rvik, Rvjk
def get_Dcos(cos_m, RRinv, R2inv, Rvij, Rvik, Rvjk):
    ijkk= np.multiply(cos_m, np.reshape(R2inv, (R2inv.shape[0], 1, R2inv.shape[1])))
    ijkk[np.isnan(ijkk)] = 0.0
    ijkk= np.reshape(ijkk, (ijkk.shape[0], ijkk.shape[1], ijkk.shape[2],1))
    ijkj= np.multiply(cos_m, R2inv)
    ijkj[np.isnan(ijkj)] = 0.0
    ijkj= np.reshape(ijkj, (ijkj.shape[0], ijkj.shape[1], ijkj.shape[2],1))
    Dcos1= np.subtract(np.multiply(RRinv,Rvij), np.multiply(ijkk,Rvik))
    Dcos2= np.subtract(np.multiply(RRinv,Rvik), np.multiply(ijkj,Rvij))
    return Dcos1, Dcos2
