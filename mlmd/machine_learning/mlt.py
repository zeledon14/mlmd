import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
class mlt:
	def __init__(self, E, X, ftot_stru = None, DX = None):
		self.E= E #array (1-d) with energies for training and validation
		self.X= X  #array (numb_struc, numb_feat) with featues for training and validation
		if ftot_stru != None:
			self.ftot_stru= ftot_stru #(num_struc,num_atoms_in_struc,3d_comp) 
			self.DX= DX #(num_struc,num_atoms_in_struc, numb_of_feat, 3d_comp) 
		#DX vectorial derivative of X, DX= nabla(X)
		return None
	def preprocessing_DX_DSIFF_for_nn(self, validation_percentage, name_to_save):
		#scaling and divition between validation and training sets
		for i in range(len(self.DX)):
			if i == 0:
				DXp= self.DX[i]
				Ft= self.ftot_stru[i]
			else:
				DXp= np.concatenate((DXp, self.DX[i]), axis= 0)
				Ft= np.concatenate((Ft, self.ftot_stru[i]), axis= 0)
		DXx= DXp[:,:,0]
		DXy= DXp[:,:,1]
		DXz= DXp[:,:,2]
		scaler= preprocessing.MaxAbsScaler()
		DXx_scaled= scaler.fit_transform(DXx)
		filename = '%s/scaler_Fx.sav' % name_to_save
		joblib.dump(scaler, filename)
		
		scaler= preprocessing.MaxAbsScaler()
		DXy_scaled= scaler.fit_transform(DXy)
		filename = '%s/scaler_Fy.sav' % name_to_save
		joblib.dump(scaler, filename)
		
		scaler= preprocessing.MaxAbsScaler()
		DXz_scaled= scaler.fit_transform(DXz)
		filename = '%s/scaler_Fz.sav' % name_to_save
		joblib.dump(scaler, filename)
		DXp[:,:,0]= DXx_scaled
		DXp[:,:,1]= DXy_scaled
		DXp[:,:,2]= DXz_scaled
		mixer= np.array(range(DXx_scaled.shape[0]))
		for _ in range(1000):
			np.random.shuffle(mixer)
		n= int(len(mixer)*(1.0 - validation_percentage/100.0)) # marking the 90%
		self.DXnn_trai= DXp[mixer[:n]]
		self.DXnn_vali= DXp[mixer[n:]]
		#DXx x component of nabla(X) 
		#DXx x(numb_of_atoms = numb_of_struc*numb_atoms_in_struc, numb_of_feaures)
		self.Fnn_trai= Ft[mixer[:n]]
		self.Fnn_vali= Ft[mixer[n:]]
		return None
	def preprocessing_X_SIFF(self, validation_percentage, name_to_save):
		#scaling and divition between validation and training sets
		scaler= preprocessing.MaxAbsScaler()
		X_scaled= scaler.fit_transform(self.X)
		mixer= np.array(range(X_scaled.shape[0]))
		for _ in range(1000):
			np.random.shuffle(mixer)
		n= int(len(mixer)*(1.0 - validation_percentage/100.0)) # marking the 90%
		self.X_trai= X_scaled[mixer[:n]]
		self.X_vali= X_scaled[mixer[n:]]
		self.E_trai= self.E[mixer[:n]]
		self.E_vali= self.E[mixer[n:]]
		filename = '%s/scaler_E.sav' % name_to_save
		joblib.dump(scaler, filename)
		return None
		
	def preprocessing_X_SIFF_return_scaler(self, validation_percentage):
		#scaling and divition between validation and training sets
		scaler= preprocessing.MaxAbsScaler()
		X_scaled= scaler.fit_transform(self.X)
		mixer= np.array(range(X_scaled.shape[0]))
		for _ in range(1000):
			np.random.shuffle(mixer)
		n= int(len(mixer)*(1.0 - validation_percentage/100.0)) # marking the 90%
		self.X_trai= X_scaled[mixer[:n]]
		self.X_vali= X_scaled[mixer[n:]]
		self.E_trai= self.E[mixer[:n]]
		self.E_vali= self.E[mixer[n:]]
		return scaler_E
		
	def preprocessing_DX_DSIFF(self, validation_percentage, name_to_save):
		#scaling and divition between validation and training sets
		for i in range(len(self.DX)):
			if i == 0:
				DXp= self.DX[i]
				Ft= self.ftot_stru[i]
			else:
				DXp= np.concatenate((DXp, self.DX[i]), axis= 0)
				Ft= np.concatenate((Ft, self.ftot_stru[i]), axis= 0)
		DXx= DXp[:,:,0]
		DXy= DXp[:,:,1]
		DXz= DXp[:,:,2]
		Fx= Ft[:,0]
		Fy= Ft[:,1]
		Fz= Ft[:,2]
		scaler= preprocessing.MaxAbsScaler()
		DXx_scaled= scaler.fit_transform(DXx)
		filename = '%s/scaler_Fx.sav' % name_to_save
		joblib.dump(scaler, filename)
		
		scaler= preprocessing.MaxAbsScaler()
		DXy_scaled= scaler.fit_transform(DXy)
		filename = '%s/scaler_Fy.sav' % name_to_save
		joblib.dump(scaler, filename)
		
		scaler= preprocessing.MaxAbsScaler()
		DXz_scaled= scaler.fit_transform(DXz)
		filename = '%s/scaler_Fz.sav' % name_to_save
		joblib.dump(scaler, filename)
		
		mixer= np.array(range(DXx_scaled.shape[0]))
		for _ in range(1000):
			np.random.shuffle(mixer)
		n= int(len(mixer)*(1.0 - validation_percentage/100.0)) # marking the 90%
		self.DXx_trai= DXx_scaled[mixer[:n]]
		self.DXx_vali= DXx_scaled[mixer[n:]]
		#DXx x component of nabla(X) 
		#DXx x(numb_of_atoms = numb_of_struc*numb_atoms_in_struc, numb_of_feaures)
		self.Fx_trai= Fx[mixer[:n]]
		self.Fx_vali= Fx[mixer[n:]]	
		#Fx x component of force 1-d array
		#Fx (numb_of_atoms = numb_of_struc*numb_atoms_in_struc)
		self.DXy_trai= DXy_scaled[mixer[:n]]
		self.DXy_vali= DXy_scaled[mixer[n:]]
		self.Fy_trai= Fy[mixer[:n]]
		self.Fy_vali= Fy[mixer[n:]]	
		self.DXz_trai= DXz_scaled[mixer[:n]]
		self.DXz_vali= DXz_scaled[mixer[n:]]
		self.Fz_trai= Fz[mixer[:n]]
		self.Fz_vali= Fz[mixer[n:]]
		return None		
		
	def preprocessing_DX_DSIFF_return_scaler(self, validation_percentage):
		#scaling and divition between validation and training sets
		for i in range(len(self.DX)):
			if i == 0:
				DXp= self.DX[i]
				Ft= self.ftot_stru[i]
			else:
				DXp= np.concatenate((DXp, self.DX[i]), axis= 0)
				Ft= np.concatenate((Ft, self.ftot_stru[i]), axis= 0)
		DXx= DXp[:,:,0]
		DXy= DXp[:,:,1]
		DXz= DXp[:,:,2]
		Fx= Ft[:,0]
		Fy= Ft[:,1]
		Fz= Ft[:,2]
		scaler_x= preprocessing.MaxAbsScaler()
		DXx_scaled= scaler_x.fit_transform(DXx)
		
		scaler_y= preprocessing.MaxAbsScaler()
		DXy_scaled= scaler_y.fit_transform(DXy)
		
		scaler_z= preprocessing.MaxAbsScaler()
		DXz_scaled= scaler_z.fit_transform(DXz)
		
		mixer= np.array(range(DXx_scaled.shape[0]))
		for _ in range(1000):
			np.random.shuffle(mixer)
		n= int(len(mixer)*(1.0 - validation_percentage/100.0)) # marking the 90%
		self.DXx_trai= DXx_scaled[mixer[:n]]
		self.DXx_vali= DXx_scaled[mixer[n:]]
		#DXx x component of nabla(X) 
		#DXx x(numb_of_atoms = numb_of_struc*numb_atoms_in_struc, numb_of_feaures)
		self.Fx_trai= Fx[mixer[:n]]
		self.Fx_vali= Fx[mixer[n:]]	
		#Fx x component of force 1-d array
		#Fx (numb_of_atoms = numb_of_struc*numb_atoms_in_struc)
		self.DXy_trai= DXy_scaled[mixer[:n]]
		self.DXy_vali= DXy_scaled[mixer[n:]]
		self.Fy_trai= Fy[mixer[:n]]
		self.Fy_vali= Fy[mixer[n:]]	
		self.DXz_trai= DXz_scaled[mixer[:n]]
		self.DXz_vali= DXz_scaled[mixer[n:]]
		self.Fz_trai= Fz[mixer[:n]]
		self.Fz_vali= Fz[mixer[n:]]
		return [scaler_x, scaler_y, scaler_z]		
		
#machine learning models GBR (gradient boosting regression)
	def GBR_E_model(self, name_to_save, parameters_dict):
		# Gradient Boosting Regression for energy 
		clf = ensemble.GradientBoostingRegressor(**parameters_dict)
		clf.fit(self.X_trai, self.E_trai)
		mse = mean_squared_error(self.E_vali, clf.predict(self.X_vali))
		#print mse
		# save the model to disk
		filename = '%s/GBR_E.sav' % name_to_save
		joblib.dump(clf, filename)
		return mse
		
	def GBR_train_evaluate_E_model(self, parameters_dict):
		# Gradient Boosting Regression for energy 
		clf = ensemble.GradientBoostingRegressor(**parameters_dict)
		clf.fit(self.X_trai, self.E_trai)
		mse = mean_squared_error(self.E_vali, clf.predict(self.X_vali))
		#print mse
		# save the model to disk
		#filename = '%s/GBR_E.sav' % name_to_save
		#joblib.dump(clf, filename)
		return mse, clf

	def GBR_F_model(self, name_to_save, comp, parameters_dict):
		#select which component of data to use between (x,y,z)
		#the components are represented as x=0, y=1, z=2
		if comp == 0:
			DX_trai = self.DXx_trai
			DX_vali = self.DXx_vali
			f_trai= self.Fx_trai
			f_vali= self.Fx_vali
		elif comp == 1:
			DX_trai = self.DXy_trai
			DX_vali = self.DXy_vali
			f_trai= self.Fy_trai
			f_vali= self.Fy_vali
		elif comp == 2:
			DX_trai = self.DXz_trai
			DX_vali = self.DXz_vali
			f_trai= self.Fz_trai
			f_vali= self.Fz_vali
		#train the model for force 
		clf = ensemble.GradientBoostingRegressor(**parameters_dict)
		clf.fit(DX_trai, f_trai)
		mse = mean_squared_error(f_vali, clf.predict(DX_vali))
		filename = '%s/GBR_F_comp_%d.sav' % (name_to_save, comp)
		joblib.dump(clf, filename)
		return mse
		
	def GBR_train_evaluate_F_model(self, comp, parameters_dict):
		#select which component of data to use between (x,y,z)
		#the components are represented as x=0, y=1, z=2
		if comp == 0:
			DX_trai = self.DXx_trai
			DX_vali = self.DXx_vali
			f_trai= self.Fx_trai
			f_vali= self.Fx_vali
		elif comp == 1:
			DX_trai = self.DXy_trai
			DX_vali = self.DXy_vali
			f_trai= self.Fy_trai
			f_vali= self.Fy_vali
		elif comp == 2:
			DX_trai = self.DXz_trai
			DX_vali = self.DXz_vali
			f_trai= self.Fz_trai
			f_vali= self.Fz_vali
		#train the model for force 
		clf = ensemble.GradientBoostingRegressor(**parameters_dict)
		clf.fit(DX_trai, f_trai)
		mse = mean_squared_error(f_vali, clf.predict(DX_vali))
		#filename = '%s/GBR_F_comp_%d.sav' % (name_to_save, comp)
		#joblib.dump(clf, filename)
		return mse, clf
	#neural networks (nn) models
	def create_E_nn_variables(self, nn_arch):
		weights= []
		biases= []
		#weights_names= ''
		#biases_names= ''
		for j, _ in enumerate(nn_arch[:len(nn_arch)-1]):
			j_nodes= nn_arch[j]
			j_plus_1_nodes= nn_arch[j+1]
			weights.append(tf.Variable(tf.random_uniform([j_nodes, j_plus_1_nodes], -1.0, 1.0),\
									   name= 'w_%d'%j))
			#weights_names= weights_names + 'w_%d,'%j
			biases.append(tf.Variable(tf.zeros([1,j_plus_1_nodes]),\
									  name= 'b_%d'%(j)))
			#biases_names= biases_names + 'b_%d,'%j
		#return weights, weights_names, biases, biases_names
		return weights, biases
#	def create_F_nn_variables(self, comp, nn_arch):
#		weights= []
#		biases= []
#		for j, _ in enumerate(nn_arch[:len(nn_arch)-1]):
#			j_nodes= nn_arch[j]
#			j_plus_1_nodes= nn_arch[j+1]
#			weights.append(tf.Variable(tf.random_uniform([j_nodes, j_plus_1_nodes], -1.0, 1.0),\
#									   name= 'w_%d_%d'%(j,comp)))
#			biases.append(tf.Variable(tf.zeros([1,j_plus_1_nodes]),\
#									  name= 'w_%d_%d'%(j,comp)))
#		return weights, biases			
	def nn_E_model(self, name_to_save, weights,biases,\
					nn_arch, learning_rate,\
					training_steps, tranining_batches_size):
		#create the placeholders
		x = tf.placeholder("float", [None, weights[0].get_shape()[0]], name= 'x')
		y = tf.placeholder("float", [None, 1], name= 'y')
		lear_rate= tf.placeholder(tf.float32, shape=[])
		z=[]
		#create the graph
		for j, _ in enumerate(weights[:-1]):
			w= weights[j]
			b= biases[j]
			if j == 0:
				z.append(tf.nn.sigmoid(tf.matmul(x,w) + b))
			else:
				z.append(tf.nn.sigmoid(tf.matmul(z[j-1],w) + b))
		ynn= tf.matmul(z[-1],weights[-1]) + biases[-1]
		#set loss and optimizaation
		loss= tf.losses.mean_squared_error(y, ynn)
		optimizer= tf.train.GradientDescentOptimizer(lear_rate).minimize(loss)
		#initializing session
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(init)
		#training section	
		steps= training_steps
		batch_step= tranining_batches_size
		vali_arra= []
		trai_arra= []
		batches_n= np.arange(0,self.X_trai.shape[0],batch_step, int)
		mixer= np.array(range(self.X_trai.shape[0]))
		total_cost= []
		E_1d_trai= np.reshape(self.E_trai, (self.E_trai.shape[0],1))
		E_1d_vali= np.reshape(self.E_vali, (self.E_vali.shape[0],1))
		for step in range(steps):
			np.random.shuffle(mixer)
			xi= self.X_trai[mixer,:]
			yi= E_1d_trai[mixer,:]
			for n in batches_n:
				_, cost= sess.run([optimizer, loss], \
				feed_dict={x:xi[n:n+batch_step,:],\
				 y:yi[n:n+batch_step,:], lear_rate:learning_rate})
				 
			loss_train= sess.run(loss, feed_dict={x:self.X_trai, y:E_1d_trai, lear_rate:learning_rate})
			loss_vali= sess.run(loss, feed_dict={x:self.X_vali, y:E_1d_vali, lear_rate:learning_rate})
			vali_arra.append(loss_vali)
			trai_arra.append(loss_train)
		#save_model
		mse = mean_squared_error(np.squeeze(E_1d_vali),\
			np.squeeze(sess.run(ynn, feed_dict={x:self.X_vali})))
		filename = '%s/nn_E.ckpt' % name_to_save
		save_path = saver.save(sess, filename)
		np.save('%s/arch' % name_to_save, np.array(nn_arch))
		#save_names
		#w_file= open('%s/weights' % name_to_save, 'w')
		#w_file.write(weights_names)
		#w_file.close()
		#b_file= open('%s/biases' % name_to_save, 'w')
		#b_file.write(biases_names)
		#b_file.close()
		return mse, np.array(vali_arra), np.array(trai_arra)
	def nn_F_model(self, name_to_save, comp, weights, biases,\
					nn_arch, learning_rate,\
					training_steps, tranining_batches_size):
		if comp == 0:
			DX_trai = self.DXx_trai
			DX_vali = self.DXx_vali
			f_trai= self.Fx_trai
			f_vali= self.Fx_vali
		elif comp == 1:
			DX_trai = self.DXy_trai
			DX_vali = self.DXy_vali
			f_trai= self.Fy_trai
			f_vali= self.Fy_vali
		elif comp == 2:
			DX_trai = self.DXz_trai
			DX_vali = self.DXz_vali
			f_trai= self.Fz_trai
			f_vali= self.Fz_vali
		#create the placeholders
		x = tf.placeholder("float", [None, weights[0].get_shape()[0]], name= 'x')
		y = tf.placeholder("float", [None, 1], name= 'y')
		lear_rate= tf.placeholder(tf.float32, shape=[])
		z=[]
		#create the graph
		for j, _ in enumerate(weights[:-1]):
			w= weights[j]
			b= biases[j]
			if j == 0:
				z.append(tf.nn.sigmoid(tf.matmul(x,w) + b))
			else:
				z.append(tf.nn.sigmoid(tf.matmul(z[j-1],w) + b))
		ynn= tf.matmul(z[-1],weights[-1]) + biases[-1]
		#set loss and optimizaation
		loss= tf.losses.mean_squared_error(y, ynn)
		optimizer= tf.train.GradientDescentOptimizer(lear_rate).minimize(loss)
		#initializing session
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(init)
		#training section	
		steps= training_steps
		batch_step= tranining_batches_size
		vali_arra= []
		trai_arra= []
		batches_n= np.arange(0,DX_trai.shape[0],batch_step, int)
		mixer= np.array(range(DX_trai.shape[0]))
		total_cost= []
		E_1d_trai= np.reshape(f_trai, (f_trai.shape[0],1))
		E_1d_vali= np.reshape(f_vali, (f_vali.shape[0],1))
		for step in range(steps):
			np.random.shuffle(mixer)
			xi= DX_trai[mixer,:]
			yi= E_1d_trai[mixer,:]
			for n in batches_n:
				_, cost= sess.run([optimizer, loss], \
				feed_dict={x:xi[n:n+batch_step,:],\
				 y:yi[n:n+batch_step,:], lear_rate:learning_rate})
				 
			loss_train= sess.run(loss, feed_dict={x:DX_trai, y:E_1d_trai, lear_rate:learning_rate})
			loss_vali= sess.run(loss, feed_dict={x:DX_vali, y:E_1d_vali, lear_rate:learning_rate})
			vali_arra.append(loss_vali)
			trai_arra.append(loss_train)
		mse = mean_squared_error(np.squeeze(E_1d_vali),\
			np.squeeze(sess.run(ynn, feed_dict={x:DX_vali})))
		filename = '%s/nn_F_comp_%d.ckpt' % (name_to_save, comp)
		save_path = saver.save(sess, filename)
		np.save('%s/arch' % name_to_save, np.array(nn_arch))
		return mse, np.array(vali_arra), np.array(trai_arra)
