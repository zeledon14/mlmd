import os
import re
import numpy as np

def load_vasp_structures(dir_path):
    import pychemia as pc
    fl_nms= os.listdir(dir_path) #file names
    comp_str='([a-z A-Z _ - & \. 0-9]+\.POSCAR)'
    fl_nms1= re.findall(comp_str,str(fl_nms))
    stru= []
    name= []
    ener= []
    ftot_stru= []
    species_simb= []
    for ii, fl in enumerate(fl_nms1):
        name_fl=fl.split('.')
        #print name_fl
        stru_poscar= pc.code.vasp.read_poscar(dir_path+'/'+name_fl[0]+'.POSCAR')
        stru_outcar= pc.code.vasp.VaspOutput(dir_path+'/'+name_fl[0]+'.OUTCAR')
        #stru_outcar= pc.code.vasp.outcar(dir_path+'/'+name_fl[0]+'.OUTCAR')
        stru.append(stru_outcar.positions[0])
        #print s.shape
        species_simb.append(stru_poscar.symbols)
        ener.append(stru_outcar.energy)
        ftot_stru.append(stru_outcar.forces[0])
        name.append(name_fl[0])
    ftot_stru= np.squeeze(np.array(ftot_stru))
    ener= np.array(ener)
    stru= np.array(stru)
    return species_simb, name, stru, ftot_stru, ener

def load_abinit_structures(dir_path, trans):
    fl_nms= os.listdir(dir_path) #file names
    comp_str='([a-z A-Z _ - & \. 0-9]+\.out)'
    fl_nms1= re.findall(comp_str,str(fl_nms))
    stru= []
    name= []
    ener= []
    ftot_stru= []
    species_simb= []
    for ii, name_fl in enumerate(fl_nms1):
        print name_fl
        nat, mass, latvec_in, strten_in,z_struc= get_nat_mass_latvec_in_strten_in(dir_path+'/'+name_fl, trans)
        xred, fcart, ener_p= get_xred_fcart(dir_path+'/'+name_fl, nat)
        stru.append(np.dot(latvec_in, xred).T)
        #print s.shape
        species_simb.append(z_struc)
        ener.append(ener_p)
        ftot_stru.append(fcart.T)
        name.append(name_fl)
    ftot_stru= np.squeeze(np.array(ftot_stru))
    ener= np.array(ener)
    stru= np.array(stru)
    #from abinit
    #stru in borh
    #ftot_stru hartree/borh
    #ener hartree
    #transfomr the output to Angstroms and eV
    #1 Bohr=0.5291772108 Angstroms
    #1 Hartree = 27.2114 eV
    stru= np.multiply(stru,0.5291772108)
    ener= np.multiply(ener,27.2114)
    ftot_stru= np.multiply(ftot_stru, (27.2114/0.5291772108))
    return species_simb, name, stru, ftot_stru, ener

def get_nat_mass_latvec_in_strten_in(path_to_file, trans):
    data= open(path_to_file).read()
    nat= int(re.findall('natom\s+([0-9]+)', data)[0])
    typat= map(int, re.findall('\s+typat\s+(.+)',data)[0].split())
    #znucl= map(float, re.findall('\s+znucl((?:\s+\d+.\d+\s+)+)',data))
    znucl= map(int, map(float, re.findall('\s+znucl\s+(.+)',data)[0].split()))
    z_struc=[]
    mass=[]
    trans_inv={}
    print 'trans ', trans
    for j in trans.keys():
        trans_inv[trans[j]]= j
    print  'trans_inv ', trans_inv   
    for i in typat:
        z_struc.append(trans_inv[znucl[i-1]])
    z_struc= np.array(z_struc)
    a1= map(float, re.findall('R.1.=\s*(.\d+...\d+\s+.\d+...\d+\s+.\d+...\d+)', data)[0].split())
    a2= map(float, re.findall('R.2.=\s*(.\d+...\d+\s+.\d+...\d+\s+.\d+...\d+)', data)[0].split())
    a3= map(float, re.findall('R.3.=\s*(.\d+...\d+\s+.\d+...\d+\s+.\d+...\d+)', data)[0].split())
    latvec_in= np.array([a1,a2,a3]).T
    latvec_in.astype('float64')
    strten_in= []
    strten_in.append(np.float64(re.findall('sigma.1\s+1.=(\s+.\d+.\d+..\d+)', data)[0]))
    strten_in.append(np.float64(re.findall('sigma.2\s+2.=(\s+.\d+.\d+..\d+)', data)[0]))
    strten_in.append(np.float64(re.findall('sigma.3\s+3.=(\s+.\d+.\d+..\d+)', data)[0]))
    strten_in.append(np.float64(re.findall('sigma.3\s+2.=(\s+.\d+.\d+..\d+)', data)[0]))
    strten_in.append(np.float64(re.findall('sigma.3\s+1.=(\s+.\d+.\d+..\d+)', data)[0]))
    strten_in.append(np.float64(re.findall('sigma.2\s+1.=(\s+.\d+.\d+..\d+)', data)[0]))
    strten_in= np.array(strten_in)
    return nat, mass, latvec_in, strten_in,z_struc

def get_xred_fcart(path_to_file, nat):
    #1 Ha/Bohr3 = 29421.02648438959 GPa
    data= open(path_to_file).readlines()
    for n,line in enumerate(data):
        if re.findall('reduced\s+coordinates\s+.array\s+xred', str(line)): 
            xred_temp=  data[n+1:n+1+nat]
            xred= np.array([map(float, i.split('\n')[0].split()) for i in xred_temp]).T
            xred.astype('float64')
        elif re.findall('cartesian\s+forces\s+.hartree.bohr', str(line)): 
            fcart_temp=  data[n+1:n+1+nat]
            fcart= np.array([map(float, i.split('\n')[0].split()) for i in fcart_temp])[:,1:]
            fcart= fcart.T
            fcart.astype('float64')
        elif re.findall('>>>>>>>>>\s+Etotal=\s+.\d+', str(line)):#hartree 
            ener=  re.findall('>>>>>>>>>\s+Etotal=(\s+.\d+.\d+..\d+)', str(line))
            ener= np.float64(ener[0])
        elif re.findall('Pressure=\s+\d+.\d+..\d+', str(line)):#this preassure in GPa
            pressure=  re.findall('Pressure=(\s+\d+.\d+..\d+)', str(line))
            pressure= np.float64(pressure[0])
    return xred, fcart, ener
#load structures and energies from xyz and forces from log
def load_structures_from_xyz_log(dir_path):
    fl_nms= os.listdir(dir_path) #file names
    comp_str='([a-z A-Z _ - & \. 0-9]+\.xyz)'
    fl_nms1= re.findall(comp_str,str(fl_nms))
    stru= []
    name= []
    ener= []
    ftot_stru= []
    species_simb= []
    for ii, fl in enumerate(fl_nms1):
        #print fl
        count= 0
        name_fl=fl.split('.')
        xyz_data= open(dir_path+'/'+fl)
        xyz_data= xyz_data.readlines()
        log_data= open(dir_path+'/'+name_fl[0]+'.log')
        log_data= log_data.readlines()
        numb_atoms= int(xyz_data[0])
        x= re.findall('failure!', str(log_data))
        if len(x) != 0:
            continue
        for n in range(int(len(xyz_data)/(2.0+numb_atoms))):
            name.append(fl.split('.')[0]+'_'+str(count))
            ener.append(float(xyz_data[(2+numb_atoms)*n+1].split()[2]))
            temp= []
            temp_spc=[]
            for i in range(numb_atoms):
                temp.append([float(j) for j in xyz_data[(2+numb_atoms)*n+2+i].split()[1:]])
                temp_spc.append(xyz_data[(2+numb_atoms)*n+2+i].split()[0])
            stru.append(np.array(temp))
            species_simb.append(temp_spc)
            count+=1
        ftot_file= re.findall('\d+\s+ftot\s+=(\s+.\d+.\d+..\d+\s+.\d+.\d+..\d+\s+.\d+.\d+..\d+)', str(log_data))
        ftot_file1=[]
        ftot_file1.append([map(float,i.split()) for i in ftot_file])
        ftot_file1= np.reshape(np.array(ftot_file1[0]),(len(ftot_file)/numb_atoms,numb_atoms,3))
        for f in ftot_file1:
            #print np.array(f).shape
            ftot_stru.append(np.array(f))
    ftot_stru= np.squeeze(np.array(ftot_stru))
    ener= np.array(ener)
    stru= np.array(stru)
    return species_simb, name, stru, ftot_stru, ener
    
#load structures from xyz gives structures and energies
def load_structures_from_xyz(dir_xyz):
    #dir_xyz='/home/arturo/Desktop/ML_MD_FBP/data/xyz'
    fl_nms= os.listdir(dir_xyz) #file names
    stru= [] #array with the structures 
    name= [] 
    ener= [] #array with the energies
    species_simb= [] 
    for ii, fl in enumerate(fl_nms): #fl -> file
        count= 0
        name_fl=fl.split('.')
        xyz_data= open(dir_xyz+'/'+fl)
        xyz_data= xyz_data.readlines()
        numb_atoms= int(xyz_data[0])
        #x= re.findall('failure!', str(log_data))
        for n in range(int(len(xyz_data)/(2.0+numb_atoms))):
            name.append(fl.split('.')[0]+'_'+str(count))
            ener.append(float(xyz_data[(2+numb_atoms)*n+1].split()[2]))
            temp= []
            temp_spc=[]
            for i in range(numb_atoms):
                temp.append([float(j) for j in xyz_data[(2+numb_atoms)*n+2+i].split()[1:]])
                temp_spc.append(xyz_data[(2+numb_atoms)*n+2+i].split()[0])
            stru.append(temp)
            species_simb.append(temp_spc)
            count+=1
    ener= np.array(ener)
    stru= np.array(stru)
    return species_simb, name, stru, ener
def load_training(path_training):
	eta2b_03=76.75#eta
	Rp03= [1.0,1.3,1.6,1.9,2.2,2.5,2.8,3.1,3.4,3.7,4.0,4.3,4.6,4.9,5.2]

	eta2b_04=43.17
	Rp04= [1.0,1.4,1.8,2.2,2.6,3.0,3.4,3.8,4.2,4.6,5.0,5.4]

	eta2b_05=27.63
	Rp05= [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5]

	eta2b_06=19.19
	Rp06= [1.0,1.6,2.2,2.8,3.4,4.0,4.6,5.2]
	eta2b_tot=[0,eta2b_03, eta2b_04, eta2b_05,eta2b_06]
	Rp_tot=[0,Rp03,Rp04,Rp05,Rp06]

	#3 body parameters
	# 3 body filters angular filters
	eta_3b_15= 950.0
	div_15= 15.0
	3.14/div_15
	cos_p15= np.cos(np.arange(3.14/div_15,3.10,3.14/div_15))

	eta_3b_10= 1000.0
	div_10= 10.0
	3.14/div_10
	cos_p10= np.cos(np.arange(3.14/div_10,3.10,3.14/div_10))

	eta_3b_8= 70.0
	div_8= 8.0
	3.14/div_8
	cos_p8= np.cos(np.arange(3.14/div_8,3.10,3.14/div_8))

	eta_3b_7= 75.0
	div_7= 7.0
	3.14/div_7
	cos_p7= np.cos(np.arange(3.14/div_7,3.10,3.14/div_7))

	eta_3b_6= 30.0
	div_6= 6.0
	3.14/div_6
	cos_p6= np.cos(np.arange(3.14/div_6,3.10,3.14/div_6))

	eta3b_tot= [0,eta_3b_15,eta_3b_10,eta_3b_8,eta_3b_7,eta_3b_6]
	cos_p_tot= [0,cos_p15,cos_p10,cos_p8,cos_p7,cos_p6]

	trai_data= open(path_training)
	trai_data= trai_data.readlines()
	indi_path_xyz_log= 0
	indi_path_feat= 0
	use_nn_E=0
	use_nn_F=0
	GBR_E_parameters={}
	GBR_F_parameters={}
	nn_E_parameters={}
	nn_F_parameters={}
	feature_parameters={}
	for line in trai_data:
		if re.search('^path_to_data', line):
			indi_path_xyz_log= 1
			path_xyz_log= re.findall('path_to_data\s+([A-Z a-z 0-9 \ / _ - \. , & % $ @ *]*)', line)
			path_xyz_log= str(path_xyz_log[0])
		if re.search('^code', line):
			code = re.findall('code\s+([1-9 \s]*)',line)
			code= map(int, str(code[0]).split())
		if re.search('^path_to_fetures', line):
			indi_path_feat= 1
		if re.search('^spec_simb', line):
			spec_simb = re.findall('spec_simb\s+([A-Z a-z \s]*)',line)
			spec_simb= str(spec_simb[0]).split()
		if re.search('^spec_Z', line):
			spec_Z = re.findall('spec_Z\s+([1-9 \s]*)',line)
			spec_Z= map(int, str(spec_Z[0]).split())        
		if re.search('^fami_2b', line):
			fami_2b = re.findall('fami_2b\s+([1-9 \s]*)',line)
			fami_2b= map(int, str(fami_2b[0]).split())
		if re.search('^fami_3b', line):
			fami_3b = re.findall('fami_3b\s+([1-9 \s]*)',line)
			fami_3b= map(int, str(fami_3b[0]).split())
		if re.search('^potential_name', line):
			pote_name= re.findall('potential_name\s+([A-Z a-z 0-9 \ / _ - \. , & % $ @ *]*)', line)
			pote_name= str(pote_name[0])
		if re.search('^validation_percentage', line):
			validation_percentage = re.findall('validation_percentage\s+([0-9 \s]*)',line)
			validation_percentage= map(int, str(validation_percentage[0]).split())
			feature_parameters['validation_percentage'] = validation_percentage
		if re.search('^GBR_E_models_to_train', line):
			GBR_E_models_to_train = re.findall('GBR_E_models_to_train\s+([0-9 \s]*)',line)
			GBR_E_models_to_train= map(int, str(GBR_E_models_to_train[0]).split())
			GBR_E_parameters['GBR_E_models_to_train'] = GBR_E_models_to_train
		if re.search('^GBR_E_n_estimators', line):
			GBR_E_n_estimators = re.findall('GBR_E_n_estimators\s+([0-9 \s]*)',line)
			GBR_E_n_estimators= map(int, str(GBR_E_n_estimators[0]).split())
			GBR_E_parameters['GBR_E_n_estimators'] = GBR_E_n_estimators
		if re.search('^GBR_E_max_depth', line):
			GBR_E_max_depth = re.findall('GBR_E_max_depth\s+([0-9 \s]*)',line)
			GBR_E_max_depth= map(int, str(GBR_E_max_depth[0]).split())
			GBR_E_parameters['GBR_E_max_depth'] = GBR_E_max_depth
		if re.search('^GBR_E_min_samples_split', line):
			GBR_E_min_samples_split = re.findall('GBR_E_min_samples_split\s+([0-9 \s]*)',line)
			GBR_E_min_samples_split= map(int, str(GBR_E_min_samples_split[0]).split())
			GBR_E_parameters['GBR_E_min_samples_split'] = GBR_E_min_samples_split
		if re.search('^GBR_E_min_samples_leaf', line):
			GBR_E_min_samples_leaf = re.findall('GBR_E_min_samples_leaf\s+([0-9 \s]*)',line)
			GBR_E_min_samples_leaf= map(int, str(GBR_E_min_samples_leaf[0]).split())
			GBR_E_parameters['GBR_E_min_samples_leaf'] = GBR_E_min_samples_leaf
		if re.search('^GBR_E_learning_rate', line):
			GBR_E_learning_rate = re.findall('GBR_E_learning_rate\s+([0-9 \. \s]*)',line)
			GBR_E_learning_rate= map(float, str(GBR_E_learning_rate[0]).split())
			GBR_E_parameters['GBR_E_learning_rate'] = GBR_E_learning_rate
		if re.search('^GBR_F_models_to_train', line):
			GBR_F_models_to_train = re.findall('GBR_F_models_to_train\s+([0-9 \s]*)',line)
			GBR_F_models_to_train= map(int, str(GBR_F_models_to_train[0]).split())
			GBR_F_parameters['GBR_F_models_to_train'] = GBR_F_models_to_train
		if re.search('^GBR_F_n_estimators', line):
			GBR_F_n_estimators = re.findall('GBR_F_n_estimators\s+([0-9 \s]*)',line)
			GBR_F_n_estimators= map(int, str(GBR_F_n_estimators[0]).split())
			GBR_F_parameters['GBR_F_n_estimators'] = GBR_F_n_estimators
		if re.search('^GBR_F_max_depth', line):
			GBR_F_max_depth = re.findall('GBR_F_max_depth\s+([0-9 \s]*)',line)
			GBR_F_max_depth= map(int, str(GBR_F_max_depth[0]).split())
			GBR_F_parameters['GBR_F_max_depth'] = GBR_F_max_depth
		if re.search('^GBR_F_min_samples_split', line):
			GBR_F_min_samples_split = re.findall('GBR_F_min_samples_split\s+([0-9 \s]*)',line)
			GBR_F_min_samples_split= map(int, str(GBR_F_min_samples_split[0]).split())
			GBR_F_parameters['GBR_F_min_samples_split'] = GBR_F_min_samples_split
		if re.search('^GBR_F_min_samples_leaf', line):
			GBR_F_min_samples_leaf = re.findall('GBR_F_min_samples_leaf\s+([0-9 \s]*)',line)
			GBR_F_min_samples_leaf= map(int, str(GBR_F_min_samples_leaf[0]).split())
			GBR_F_parameters['GBR_F_min_samples_leaf'] = GBR_F_min_samples_leaf
		if re.search('^GBR_F_learning_rate', line):
			GBR_F_learning_rate = re.findall('GBR_F_learning_rate\s+([0-9 \. \s]*)',line)
			GBR_F_learning_rate= map(float, str(GBR_F_learning_rate[0]).split())
			GBR_F_parameters['GBR_F_learning_rate'] = GBR_F_learning_rate
		if re.search('^nn_E_models_to_train', line):
			nn_E_models_to_train = re.findall('nn_E_models_to_train\s+([0-9 \s]*)',line)
			nn_E_models_to_train= map(int, str(nn_E_models_to_train[0]).split())
			nn_E_parameters['nn_E_models_to_train'] = nn_E_models_to_train
			use_nn_E=1
		if re.search('^nn_E_learning_rate', line):
			nn_learning_rate = re.findall('nn_E_learning_rate\s+([0-9 \. \s]*)',line)
			nn_learning_rate= map(float, str(nn_learning_rate[0]).split())
			nn_E_parameters['nn_E_learning_rate'] = nn_learning_rate            
		if re.search('^nn_E_training_steps', line):
			nn_training_steps = re.findall('nn_E_training_steps\s+([0-9 \s]*)',line)
			nn_training_steps= map(int, str(nn_training_steps[0]).split())
			nn_E_parameters['nn_E_training_steps'] = nn_training_steps            
		if re.search('^nn_E_batch_size', line):
			nn_batch_size = re.findall('nn_E_batch_size\s+([0-9 \s]*)',line)
			nn_batch_size= map(int, str(nn_batch_size[0]).split())
			nn_E_parameters['nn_E_batch_size'] = nn_batch_size
		if re.search('^nn_F_models_to_train', line):
			nn_F_models_to_train = re.findall('nn_F_models_to_train\s+([0-9 \s]*)',line)
			nn_F_models_to_train= map(int, str(nn_F_models_to_train[0]).split())
			nn_F_parameters['nn_F_models_to_train'] = nn_F_models_to_train
			use_nn_F=1
		if re.search('^nn_F_learning_rate', line):
			nn_learning_rate = re.findall('nn_F_learning_rate\s+([0-9 \. \s]*)',line)
			nn_learning_rate= map(float, str(nn_learning_rate[0]).split())
			nn_F_parameters['nn_F_learning_rate'] = nn_learning_rate            
		if re.search('^nn_F_training_steps', line):
			nn_training_steps = re.findall('nn_F_training_steps\s+([0-9 \s]*)',line)
			nn_training_steps= map(int, str(nn_training_steps[0]).split())
			nn_F_parameters['nn_F_training_steps'] = nn_training_steps            
		if re.search('^nn_F_batch_size', line):
			nn_batch_size = re.findall('nn_F_batch_size\s+([0-9 \s]*)',line)
			nn_batch_size= map(int, str(nn_batch_size[0]).split())
			nn_F_parameters['nn_F_batch_size'] = nn_batch_size
	if use_nn_E == 1:
		trai_data1= str(trai_data)
		archi_arra=[] #array with the architecture of the nn
		#the architecture if the number of nodes in every layer
		for i in range(1,nn_E_models_to_train[0]+1):        
			string='nn_E_architecture_%d\s+([0-9 \. \s]*)'%i
			arch= re.findall(string, trai_data1)
			#print arch
			if len(arch) > 0:
				arch= map(int, str(arch[0]).split())
				archi_arra.append(arch)
		nn_E_parameters['nn_E_arch'] = archi_arra
	if use_nn_F == 1:
		trai_data1= str(trai_data)
		archi_arra=[] #array with the architecture of the nn
		#the architecture if the number of nodes in every layer
		for i in range(1,nn_F_models_to_train[0]+1):        
			string='nn_F_architecture_%d\s+([0-9 \. \s]*)'%i
			arch= re.findall(string, trai_data1)
			#print arch
			if len(arch) > 0:
				arch= map(int, str(arch[0]).split())
				archi_arra.append(arch)
		nn_F_parameters['nn_F_arch'] = archi_arra
	trans={}
	for z, simb in zip(spec_Z, spec_simb):
		trans[simb]= z
	eta2b=[]
	Rp=[]
	eta3b=[]
	cos_p=[]
	for i in fami_2b:
		eta2b.append(eta2b_tot[i])
		Rp.append(Rp_tot[i])
	for i in fami_3b:
		eta3b.append(eta3b_tot[i])
		cos_p.append(cos_p_tot[i])
	feature_parameters['pote_name'] = pote_name
	feature_parameters['trans'] = trans
	feature_parameters['eta2b'] = eta2b
	feature_parameters['Rp'] = Rp
	feature_parameters['eta3b'] = eta3b
	feature_parameters['cos_p'] = cos_p
	if 'validation_percentage' not in  feature_parameters:
		feature_parameters['validation_percentage'] = 20
	if len(GBR_E_parameters) == 0:
		GBR_E_parameters['GBR_E_models_to_train'] = 0
	if len(GBR_F_parameters) == 0:
		GBR_F_parameters['GBR_F_models_to_train'] = 0
	return path_xyz_log, code, pote_name, feature_parameters,\
			GBR_E_parameters, GBR_F_parameters, \
			nn_E_parameters,nn_F_parameters
def load_features_from_file(path_to_file):
    files= os.listdir(path_to_file)
    dir_name=path_to_file
    DXl=[] #DX load aarray
    ftot_strul=[]#ftot stru load array
    #print files
    for fl in files:
        if re.findall('_FBP_', str(fl)):
            X= np.load(dir_name+'/'+str(fl))
        if re.findall('_energies_', str(fl)):
            ener= np.load(dir_name+'/'+str(fl))
        if re.findall('DFBP_force_', str(fl)):
            stru_names_fl= open(dir_name+'/'+str(fl)+'/structure_names_list')
            stru_names_fl= stru_names_fl.read()
            stru_names_fl= stru_names_fl.split(',')
            for i, nm in enumerate(stru_names_fl):
                DX_shape= np.load(dir_name+'/'+str(fl)+'/%s_DFBP_shape.npy' % (nm))
                forc_shape= np.load(dir_name+'/'+str(fl)+'/%s_forces_shape.npy' % (nm))
                DXl.append(np.reshape(np.load(dir_name+'/'+str(fl)+'/%s_DFBP.npy' %nm), DX_shape))
                ftot_strul.append(np.reshape(np.load(dir_name+'/'+str(fl)+'/%s_forces.npy' %nm), forc_shape))
    DX= np.array(DXl)
    ftot_stru= np.array(ftot_strul)
    return ener, X, ftot_stru, DX
def load_md_init_stru(input_stru_path):
    data= open(input_stru_path, 'r')
    stru=[]
    species= []
    for line in data:
        string= str(line)
        species.append(re.findall('([A-Za-z]+)', string)[0])
        stru.append(map(float, str(re.findall('([0-9\s\-\.]+)', string)[0]).split()))
    stru= np.array(stru)
    stru= np.reshape(stru, (1,stru.shape[0],stru.shape[1]))
    #print stru.shape
    species_simb= [species]
    #print species_simb
    return species_simb, stru
def get_md_parameters(path_to_file):
    data= open(path_to_file).read()
    Qmass= float(re.findall('\s*Qmass\s+(\d+.\d*)', data)[0])
    temp= float(re.findall('\s*temp\s+(\d+.\d*)', data)[0])
    dt= float(re.findall('\s*dt\s+(\d+.\d*)', data)[0])
    correct_spteps= int(re.findall('\s*correct_spteps\s+(\d+)', data)[0])
    md_steps= int(re.findall('\s*md_steps\s+(\d+)', data)[0])
    exp_name= str(re.findall('\s*experiment_name\s+([A-Z a-z _ - 0-9]+)', data)[0])
    return Qmass, temp, dt, correct_spteps, md_steps, exp_name
def get_amu(species_simb, trans):
	masses = [0.0, 1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.011, 14.00674, 15.9994, 18.9984032, \
			  20.1797, 22.989768, 24.3050, 26.981539, 28.0855, 30.973762, 32.066, 35.4527, 39.948, 39.0983,\
			  40.078, 44.955910, 47.88, 50.9415, 51.9961,54.93805, 55.847, 58.93320, 58.69, 63.546, \
			  65.39, 69.723, 72.61, 74.92159, 78.96, 79.904, 83.80, 85.4678, 87.62, 88.90585,\
			  91.224, 92.90638, 95.94, 98.9062, 101.07, 102.9055, 106.42, 107.8682, 112.411, 114.82,\
			  118.710, 121.753, 127.60, 126.90447, 131.29, 132.90543, 137.327, 138.9055, 140.115, 140.90765,\
			  144.24, 147.91, 150.36, 151.965, 157.25, 158.92534, 162.50, 164.93032, 167.26, 168.93421,\
			  173.04, 174.967, 178.49, 180.9479, 183.85, 186.207, 190.2, 192.22, 195.08, 196.96654, \
			  200.59, 204.3833, 207.2, 208.98037, 209.0, 210.0, 222.0, 223.0, 226.0254, 230.0,\
			  232.0381, 231.0359, 238.0289, 237.0482, 242.0,243.0, 247.0, 247.0, 249.0, 254.0,\
			  253.0, 256.0, 254.0, 257.0, 260.0,None, None, None, None, None, None,\
			  None, None, None, None, None, None, None, None]
	amu= []
	for i in species_simb[0]:
		amu.append(masses[trans[i]])
	amu= np.array(amu)
	return amu
#write xyz
def write_xyz(path_to_xyz,  E_pred, temp, species_simb, r_out):
    if os.path.isfile(path_to_xyz):
        xyz_file= open(path_to_xyz, 'a')
        write_xyz_lines(xyz_file, E_pred, temp, species_simb, r_out)
    else:
        xyz_file= open(path_to_xyz, 'w')
        write_xyz_lines(xyz_file, E_pred, temp, species_simb, r_out)
    return None

#write xyz lines
def write_xyz_lines(xyz_file, E_pred, temp, species_simb, r_out):
    xyz_file.write('%d\n' % len(species_simb[0]))
    xyz_file.write('ETOT= %f   T_instantaneus=  %f\n'%(E_pred, temp))
    for i,j in zip(species_simb[0],r_out.T):
        xyz_file.write('%s   %.5f %.5f %.5f\n'%(i,j[0], j[1], j[2]))
    xyz_file.close()
    return None
def get_temp(amu, v_in):
    k_b_evK= 8.617343e-5
    v= v_in
    kine_aver= (0.5*np.sum(amu*np.sum(v*v, axis=0)))/(len(amu))
    T= (2.0*kine_aver)/(3.0*k_b_evK)
    return T
