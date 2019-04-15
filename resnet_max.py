import keras
import keras.datasets as kd
import keras.layers as kl
import keras.regularizers as kr
import keras.models as km

(x_train, y_train), (x_test, y_test) = kd.cifar10.load_data()
labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

x_train = x_train/255.
x_test = x_test/255.

# Convert class vectors to binary class matrices.
N = len(labels)

y_train = keras.utils.to_categorical(y_train, N)
y_test = keras.utils.to_categorical(y_test, N)

# Note the alternative method for model specification: no model.add(.), instead we 
# perform sequential operations on layers, then we will make the resulting model later.

# Specify the shape of the input image
input_shape = x_train.shape[1:]
inputs = kl.Input(shape=input_shape)

# First convolution + BN + act
conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(inputs)
bn = kl.BatchNormalization()(conv)
act1 = kl.Activation('relu')(bn)

# Perform 3 convolution blocks
for i in range(3):
    conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act1)
    bn = kl.BatchNormalization()(conv)
    act = kl.Activation('relu')(bn)
    
    conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
    bn = kl.BatchNormalization()(conv)

    # Skip layer addition
    skip = kl.add([act1,bn])
    act1 = kl.Activation('relu')(skip)  

# Downsampling with strided convolution
conv = kl.Conv2D(32,(3,3),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
bn = kl.BatchNormalization()(conv)
act = kl.Activation('relu')(bn)

conv = kl.Conv2D(32,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
bn = kl.BatchNormalization()(conv)

# Downsampling with strided 1x1 convolution
act1_downsampled = kl.Conv2D(32,(1,1),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)

# Downsampling skip layer
skip_downsampled = kl.add([act1_downsampled,bn])
act1 = kl.Activation('relu')(skip_downsampled)

# This final layer is denoted by a star in the above figure

for i in range(2):
  conv = kl.Conv2D(32,(3,3),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
  bn = kl.BatchNormalization()(conv)
  act = kl.Activation('relu')(bn)

  conv = kl.Conv2D(32,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act1)
  bn = kl.BatchNormalization()(conv)

  skip = kl.add([act1,bn])
  act1 = kl.Activation('relu')(skip)

conv = kl.Conv2D(64,(3,3),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
bn = kl.BatchNormalization()(conv)
act = kl.Activation('relu')(bn)

conv = kl.Conv2D(64,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
bn = kl.BatchNormalization()(conv)

# downsample with strided 1x1 conv
act1_downsampled = kl.Conv2D(64,(1,1),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)

# Downsampling skip layer
skip_downsampled = kl.add([act1_downsampled,bn])
act1 = kl.Activation('relu')(skip_downsampled)

for i in range(2):
  conv = kl.Conv2D(64,(3,3),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
  bn = kl.BatchNormalization()(conv)
  act = kl.Activation('relu')(bn)

  conv = kl.Conv2D(64,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act1)
  bn = kl.BatchNormalization()(conv)

  skip = kl.add([act1,bn])
  act1 = kl.Activation('relu')(skip)

gap = kl.GlobalAveragePooling2D()(act1)
bn = kl.BatchNormalization()(gap)
final_dense = kl.Dense(N)(bn)
softmax = kl.Activation('softmax')(final_dense)

model = km.Model(inputs=inputs,outputs=softmax)

# initiate adam optimizer
opt = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=25,
          validation_data=(x_test, y_test),
          shuffle=True)
