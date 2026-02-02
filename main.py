"""
MNIST Handwritten Digit Classification

This trains a neural network to recognize handwritten digits (0-9).
MNIST is the "hello world" of machine learning.
"""

import tensorflow as tf

# Load the MNIST dataset (60k training images, 10k test images of handwritten digits)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 784 pixels
    tf.keras.layers.Dense(128, activation='relu'),   # Hidden layer with 128 neurons
    tf.keras.layers.Dropout(0.2),                    # Dropout to prevent overfitting
    tf.keras.layers.Dense(10)                        # Output layer: 10 classes (digits 0-9)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
print("Training the model...")
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate on test data
print("\nEvaluating on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_accuracy:.2%}")

# Make predictions on a few test images
predictions = model.predict(x_test[:5])
predicted_digits = tf.argmax(predictions, axis=1)

print("\nSample predictions:")
for i in range(5):
    print(f"  Image {i+1}: predicted={predicted_digits[i].numpy()}, actual={y_test[i]}")
