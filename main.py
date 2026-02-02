import tensorflow as tf

# Create two constant tensors
a = tf.constant(2)
b = tf.constant(3)

# Perform addition
result = tf.add(a, b)

print(f"TensorFlow version: {tf.__version__}")
print(f"2 + 3 = {result.numpy()}")
