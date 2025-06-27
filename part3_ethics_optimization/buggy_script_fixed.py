import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np # For reshaping and argmax

# Load data
mnist = tf.keras.datasets.mnist
(X_train_orig, y_train), (X_test_orig, y_test) = mnist.load_data()

# Normalize data
X_train = X_train_orig / 255.0
X_test = X_test_orig / 255.0

# Reshape data to include channel dimension (height, width, channels)
# This is crucial for Conv2D layers.
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

# Build CNN
model = models.Sequential([
    # Specify input_shape in the first Conv2D layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10) # Output layer: 10 units for 10 digits. No softmax needed if from_logits=True in loss.
])

# Compile the model
# Using SparseCategoricalCrossentropy with from_logits=True means the model's output layer should provide logits.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print model summary to verify architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Visualize predictions
# Predict on the first 5 images from the test set
num_images_to_show = 5
predictions_logits = model.predict(X_test[:num_images_to_show])
predicted_labels = np.argmax(predictions_logits, axis=1)

plt.figure(figsize=(10, 5)) # Adjusted figsize for better layout
for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i + 1)
    # Use original X_test images (before reshaping and normalization) for clearer visualization if preferred,
    # or use the processed X_test (which is X_test_orig[i]/255.0 and then reshaped).
    # For imshow, the reshaped X_test (normalized) needs to be squeezed back to 2D.
    plt.imshow(X_test_orig[i], cmap='gray') # Displaying the original 0-255 scale image
    # Or: plt.imshow(X_test[i].squeeze(), cmap='gray') # Displaying the normalized 0-1 scale image

    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {y_test[i]}")
    plt.axis('off')
plt.suptitle("Model Predictions vs True Labels", fontsize=14) # Add a super title
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
plt.show()

# Plot training history (accuracy and loss)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()
