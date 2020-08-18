from simplemc.DriverMC import DriverMC
#import tensorflow as tf

# inputs = tf.keras.Input(shape=(3,))
# x = tf.keras.layers.Dense(300, activation=tf.nn.relu)(inputs)
# outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

analyzer = DriverMC(analyzername="mcmc", model="owaCDM", datasets="HD+SN+BBAO+Planck", chainsdir="chains")

# analyzer.executer()
#analyzer.nestedRunner(nlivepoints=300, nproc=2)

analyzer.nestedRunner(neuralNetwork=True, nlivepoints=300, proxy_tolerance=10.0,
                      epochs=50, numNeurons=300, nproc=2)

# analyzer.nestedRunner(neuralNetwork=True, nlivepoints=50, proxy_tolerance=10.0, ntrain=50,
#                       it_to_start_net=0, updInt=50, epochs=50, numNeurons=300)

analyzer.postprocess()  

