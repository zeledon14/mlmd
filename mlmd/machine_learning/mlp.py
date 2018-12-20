import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

class mlp:
	def __init__(self, pote_name):
		self.scaler_E= joblib.load(pote_name+'/E/scaler_E.sav')
		self.scaler_Fx= joblib.load(pote_name+'/F/scaler_Fx.sav')
		self.scaler_Fy= joblib.load(pote_name+'/F/scaler_Fy.sav')
		self.scaler_Fz= joblib.load(pote_name+'/F/scaler_Fz.sav')
		return None
	def load_GBR_E_potential(self, pote_name):
		self.E_potential= joblib.load(pote_name+'/E/GBR_E.sav')
		return None
	def load_GBR_F_potential(self, pote_name):
		self.Fx_potential= joblib.load(pote_name+'/F/GBR_F_comp_0.sav')
		self.Fy_potential= joblib.load(pote_name+'/F/GBR_F_comp_1.sav')
		self.Fz_potential= joblib.load(pote_name+'/F/GBR_F_comp_2.sav')
		return None
	def predict_E_GBR(self, X):
		return self.E_potential.predict(X)
	def predict_F_GBR(self, DXx, DXy, DXz):
		Fx= self.Fx_potential.predict(DXx)
		Fy= self.Fx_potential.predict(DXy)
		Fz= self.Fx_potential.predict(DXz)
		F_total= np.zeros((len(Fx),3), dtype= float)
		F_total[:,0]= Fx
		F_total[:,1]= Fy
		F_total[:,2]= Fz
		return F_total
	def create_E_nn_variables(nn_arch):
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
	def nn_E_potential(weights,biases):
		#create the placeholders
		x = tf.placeholder("float", [None, weights[0].get_shape()[0]], name= 'x')
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
		return ynn, z
