# plot model
import tensorflow as tf

NAME = '3L_top_10_bow'
model_path = './models/'+NAME+'/'
model = tf.keras.models.load_model(model_path)
print('model', model)

tf.keras.utils.plot_model(model, to_file='./model_plots/'+NAME+'.png', show_shapes=True)