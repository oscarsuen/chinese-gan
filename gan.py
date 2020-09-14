import time
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

INPUT_DIR = "char_img"
CKPT_DIR = "checkpoints"
OUT_DIR = "out_imgs"
BUFFER_SIZE = 10000
BATCH_SIZE = 256
NOISE_DIM = 128
EPOCHS = 200
EXAMPLE_HEIGHT = 4
CKPT_EVERY = 25
IMG_EVERY = 5

def get_data(input_dir=INPUT_DIR, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
    file_ds = tf.data.Dataset.list_files(f"{input_dir}/*.png")
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image
    image_ds = file_ds.map(parse_image)
    return image_ds.shuffle(buffer_size).batch(batch_size)

def make_generator(input_dim=NOISE_DIM):
    model = tf.keras.Sequential()

    model.add(layers.Dense(4096, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(4*4*4096))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 4096)))
    assert model.output_shape == (None, 4, 4, 4096), model.output_shape

    model.add(layers.Conv2DTranspose(1024, (5, 5), strides=(2, 2), padding="same"))
    assert model.output_shape == (None, 8, 8, 1024), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding="same"))
    assert model.output_shape == (None, 8, 8, 512), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding="same"))
    assert model.output_shape == (None, 8, 8, 256), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same"))
    assert model.output_shape == (None, 16, 16, 256), model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same"))
    assert model.output_shape == (None, 32, 32, 1), model.output_shape
    model.add(layers.Activation("tanh"))

    return model

def make_discriminator(dropout_rate=0.3):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same", input_shape=(32, 32, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
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
def train_step(real_images, gen, dis, gen_opt, dis_opt, batch_size=BATCH_SIZE, noise_dim=NOISE_DIM):
    noise = tf.random.normal((batch_size, noise_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        fake_images = gen(noise, training=True)
        real_output = dis(real_images, training=True)
        fake_output = dis(fake_images, training=True)
        gen_loss = generator_loss(fake_output)
        dis_loss = discriminator_loss(real_output, fake_output)

    gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
    dis_grad = dis_tape.gradient(dis_loss, dis.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grad, gen.trainable_variables))
    dis_opt.apply_gradients(zip(dis_grad, dis.trainable_variables))

def train(dataset, gen, dis, gen_opt, dis_opt, ckpt, seed=None, start_epoch=0, epochs=EPOCHS, ckpt_prefix=CKPT_DIR+"/ckpt", ckpt_every=CKPT_EVERY, img_every=IMG_EVERY):
    if seed is None:
        seed = tf.random.normal((EXAMPLE_HEIGHT ** 2, NOISE_DIM))
    checkpoint_image(gen, 0, seed)
    for epoch in range(start_epoch, start_epoch + epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch, gen, dis, gen_opt, dis_opt)

        if (epoch + 1) % ckpt_every == 0:
            ckpt.save(file_prefix=ckpt_prefix)
        if (epoch + 1) % img_every == 0:
            checkpoint_image(gen, epoch+1, seed)
        end = time.time()
        print(f"Epoch {epoch+1:03d} in {end-start}")

    checkpoint_image(gen, epoch+1, seed)

def checkpoint_image(gen, epoch, test_input, out_dir=OUT_DIR):
    predictions = gen(test_input, training=False)
    fig = plt.figure(figsize=(EXAMPLE_HEIGHT, EXAMPLE_HEIGHT))
    for i in range(predictions.shape[0]):
        plt.subplot(EXAMPLE_HEIGHT, EXAMPLE_HEIGHT, i+1)
        plt.imshow(predictions[i, :, :, 0] * 255, cmap="gray")
        plt.axis("off")
    plt.savefig(f"{out_dir}/epoch_{epoch:03d}.png")
    plt.close(fig)

def get_args():
    data = get_data()
    generator = make_generator()
    discriminator = make_discriminator()
    gen_optimizer = tf.keras.optimizers.Adam(1e-4)
    dis_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint = tf.train.Checkpoint(gen_opt=gen_optimizer, dis_opt=dis_optimizer, gen=generator, dis=discriminator)
    return data, generator, discriminator, gen_optimizer, dis_optimizer, checkpoint

def checkpoint_restore(ckpt, ckpt_dir=CKPT_DIR):
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))

if __name__ == "__main__":
    args = get_args()
    train(*args)
