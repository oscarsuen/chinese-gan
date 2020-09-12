import time
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

INPUT_DIR = "char_img"
CHECKPOINT_DIR = "checkpoints"
OUT_DIR = "out_imgs"
BATCH_SIZE = 256
NOISE_DIM = 100
EPOCHS = 50
EXAMPLE_HEIGHT = 4

def get_data(input_dir=INPUT_DIR, batch_size=BATCH_SIZE):
    file_ds = tf.data.Dataset.list_files(f"{input_dir}/*.png")
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float16)
        return image
    image_ds = file_ds.map(parse_image)
    return image_ds.batch(batch_size)

def make_generator(input_dim=NOISE_DIM):
    model = tf.keras.Sequential()

    model.add(layers.Dense(8*8*256, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256), model.output_shape

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same"))
    assert model.output_shape == (None, 8, 8, 128), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"))
    assert model.output_shape == (None, 16, 16, 64), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same"))
    assert model.output_shape == (None, 32, 32, 1), model.output_shape
    model.add(layers.Activation("tanh"))

    return model

def make_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(32, 32, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(real_images, gen, dis, gen_optimizer, dis_optimizer, batch_size=BATCH_SIZE, noise_dim=NOISE_DIM):
    noise = tf.random.normal((batch_size, noise_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        fake_images = gen(noise, training=True)
        real_output = dis(real_images, training=True)
        fake_output = dis(fake_images, training=True)
        gen_loss = generator_loss(fake_output)
        dis_loss = discriminator_loss(real_output, fake_output)

    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    dis_grad = dis_tape.gradient(dis_loss, dis.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grad, gen.trainable_variables))
    dis_optimizer.apply_gradients(zip(dis_grad, dis.trainable_variables))

def train(dataset, gen, dis, gen_opt, dis_opt, checkpoint, seed=None, epochs=EPOCHS, checkpoint_prefix=CHECKPOINT_DIR):
    if seed is None:
        seed = tf.random.normal((EXAMPLE_HEIGHT ** 2, NOISE_DIM))
    checkpoint_image(gen, 0, seed)
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch, gen, dis, gen_opt, dis_opt)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix + "/")
            checkpoint_image(gen, epoch+1, seed)
        end = time.time()
        print(f"Epoch {epoch+1:02d} in {end-start}")

    checkpoint.save(file_prefix=checkpoint_prefix + "/")
    checkpoint_image(gen, epoch+1, seed)

def checkpoint_image(generator, epoch, test_input, out_dir=OUT_DIR):
    predictions = generator(test_input, training=False)
    fig = plt.figure(figsize=(EXAMPLE_HEIGHT, EXAMPLE_HEIGHT))
    for i in range(predictions.shape[0]):
        plt.subplot(EXAMPLE_HEIGHT, EXAMPLE_HEIGHT, i+1)
        plt.imshow(predictions[i, :, :, 0] * 255, cmap="gray")
        plt.axis("off")
    plt.savefig(f"{out_dir}/epoch_{epoch:02d}.png")

if __name__ == "__main__":
    data = get_data()
    gen = make_generator()
    dis = make_discriminator()
    gen_opt = tf.keras.optimizers.Adam(1e-4)
    dis_opt = tf.keras.optimizers.Adam(1e-4)
    checkpoint = tf.train.Checkpoint(gen_opt=gen_opt, dis_opt=dis_opt, gen=gen, dis=dis)
    train(data, gen, dis, gen_opt, dis_opt, checkpoint)
