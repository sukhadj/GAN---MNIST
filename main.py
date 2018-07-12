
print("===== Importing libraries ======")
import numpy as np
import mnist
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

seed = 128
rng = np.random.RandomState(seed)

print("====== Downloading the data ====")
train_data = mnist.train_images()
test_data = mnist.test_images()

train_data = train_data / 255
test_data = test_data / 255

# plt.imshow(test_data[1], cmap='gray')
# plt.show()

# define variables
g_input_shape = 100
d_input_shape = (28, 28)
hidden_1_units = 500
hidden_2_units = 500
output_gen_size = 28 * 28
output_dis_size = 1
epochs = 1
batch_size = 128

# generator
generator = Sequential([
    Dense(units=hidden_1_units, input_dim=g_input_shape, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Dense(units=output_gen_size, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Reshape(d_input_shape)
])

# Discriminator
discriminator = Sequential([
    InputLayer(input_shape=d_input_shape),
    Flatten(),
    Dense(units=hidden_1_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Dense(units=hidden_2_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Dense(units=output_dis_size, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)),
])

# print(generator.summary())
# print(discriminator.summary())

# Build a GAN   
gan = simple_gan(generator=generator, discriminator=discriminator,
                 latent_sampling=normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan,
                         player_params=[generator.trainable_weights, discriminator.trainable_weights],)

model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                          player_optimizers=['adam', 'adam'], loss='binary_crossentropy')
history = model.fit(train_data, gan_targets(train_data.shape[0]), epochs=10, batch_size=batch_size)


sample = np.random.normal(size=(10,100))
pred = generator.predict(sample)

for i in range(pred.shape[0]):
    plt.imshow(pred[i, :], cmap='gray')
    plt.show()

# print(model.summary())