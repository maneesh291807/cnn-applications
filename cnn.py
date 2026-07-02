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