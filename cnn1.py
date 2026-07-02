import tensorflow as tf
import matplotlib.pyplot as plt
plt.rc('image', cmap='gray')
plt.rc('figure', autolayout=True)
image_path = "cnn/images/image5.jpeg"
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, [300, 300])
image = tf.image.convert_image_dtype(image, tf.float32)
print("Original Image Shape:", image.shape)
plt.figure(figsize=(5, 5))
plt.imshow(tf.squeeze(image), cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()
image = tf.expand_dims(image, axis=0)
print("Shape after adding batch dimension:", image.shape)
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=tf.float32)
kernel = tf.reshape(kernel, [3, 3, 1, 1])
conv_output = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=[1, 1, 1, 1],
    padding='SAME'
) 
print("After Convolution Shape:", conv_output.shape)
plt.figure(figsize=(5,5))
plt.imshow(tf.squeeze(conv_output))
plt.title("After Convolution")
plt.axis('off')
plt.show()