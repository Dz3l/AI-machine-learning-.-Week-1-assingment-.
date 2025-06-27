# Part 3: Ethics & Optimization

## 1. Ethical Considerations

### Identify potential biases in your MNIST or Amazon Reviews model. How could tools like TensorFlow Fairness Indicators or spaCy’s rule-based systems mitigate these biases?

**MNIST Model Biases:**

*   **Data Representation:** The MNIST dataset, while standard, might not uniformly represent all handwriting styles. For instance:
    *   **Writing Styles:** It could disproportionately feature certain styles of writing digits (e.g., European vs. American style '7' or '1') or underrepresent styles from specific demographic groups or individuals with motor impairments.
    *   **Stroke Thickness/Quality:** Digits written with very thin or very thick strokes, or those that are smudged or poorly scanned, might be underrepresented, leading the model to perform worse on such inputs.
    *   **Cultural Variations:** Subtle stylistic differences in how digits are written across cultures might exist, and if the dataset is not balanced for these, the model may show biased performance.
*   **Impact:** If this model were used in a real-world application (e.g., processing handwritten checks, mail sorting by postal codes, or data entry from forms), individuals whose handwriting is underrepresented could face higher error rates, leading to inconvenience, misclassification, or even denial of service.

**Amazon Reviews Model Biases (NLP Task):**

*   **Sentiment Analysis Biases:**
    *   **Linguistic Variation:** Rule-based systems like VADER, or even ML-based sentiment models trained on general corpora, might not accurately interpret sentiment expressed in dialects, slang, or language used by specific demographic groups. Sarcasm or culturally specific expressions can be easily misjudged. For example, a phrase considered neutral or positive in one subculture might be negative in another.
    *   **Polarization and Nuance:** Sentiment models often struggle with nuanced reviews that contain both positive and negative aspects about different features of a product. They might also be more sensitive to overtly strong language, potentially misrepresenting mildly critical or subtly positive reviews.
    *   **Demographic Skews:** If the training data for an underlying sentiment lexicon (or a trained model) overrepresents certain demographics, the model may perform better for those demographics and worse for others.
*   **NER Biases:**
    *   **Product/Brand Recognition:** General NER models might be better at recognizing products and brands that are globally dominant or from Western markets. They might fail to identify local or niche products/brands from underrepresented regions or communities.
    *   **Association Bias:** The model might inadvertently learn or perpetuate societal biases present in the review text. For example, if certain product types are disproportionately reviewed by a particular demographic, the language style associated with those reviews might bias the NER or sentiment systems.
*   **Impact:** Biased sentiment analysis could lead businesses to misinterpret customer feedback, potentially ignoring issues critical to certain user groups. Biased NER could mean that insights related to less mainstream products or brands are missed. This can lead to unfair product ratings, skewed market understanding, and perpetuation of existing inequalities.

**Mitigation Strategies:**

*   **TensorFlow Fairness Indicators (TFMA - TensorFlow Model Analysis) (for MNIST):**
    *   **How it helps:** TFMA allows for the evaluation of model performance across different **slices** of data. If metadata associated with MNIST images were available (e.g., a proxy for writer's demographic, region, or even a classification of handwriting style), we could create these slices.
    *   **Application:**
        1.  **Slice Data:** Define slices based on available features (e.g., if we had labels for "thick stroke" vs. "thin stroke" digits, or digits from different (synthetic) "regions").
        2.  **Evaluate Metrics:** Use TFMA to compute metrics like accuracy, precision, recall, and AUC for each slice.
        3.  **Identify Disparities:** Compare these metrics across slices. A significant drop in accuracy for a particular slice would indicate a bias. For example, the model might be 99% accurate overall but only 80% accurate for digits written with a specific atypical style.
        4.  **Action:** Once identified, these disparities can guide efforts to:
            *   **Collect More Data:** Augment the training set with more examples from underperforming slices.
            *   **Data Augmentation:** Apply targeted data augmentation techniques to generate more diverse training examples (e.g., artificially thinning or thickening strokes).
            *   **Model Adjustments:** Consider techniques like re-weighting samples in the loss function or exploring model architectures more robust to these variations.
    *   **Without explicit metadata for MNIST:** One could try to *infer* slices, e.g., by clustering images based on morphological features and then using TFMA to check if performance is consistent across these clusters.

*   **spaCy’s Rule-Based Systems (for Amazon Reviews):**
    *   **NER Bias Mitigation:**
        *   **Gazetteers & PhraseMatcher:** Create comprehensive lists (gazetteers) of product names, brands, and model numbers, especially focusing on those that might be underrepresented in general NER models (e.g., local brands, artisanal products). Use spaCy's `PhraseMatcher` to ensure these entities are identified.
        *   **Custom Patterns with Matcher:** Develop rules using `Matcher` to identify product/brand patterns that the statistical model might miss. For example, `[BRAND_NAME] [ALPHANUMERIC_MODEL_CODE]` or `[ADJECTIVE] [PRODUCT_CATEGORY_KEYWORD]`.
        *   **Iterative Refinement:** Regularly review NER outputs on diverse review sets and update gazetteers and rules based on observed errors or omissions.
    *   **Sentiment Analysis Bias Mitigation (Rule-Based Enhancement):**
        *   **Custom Rules for Ambiguity/Sarcasm:** While hard, `Matcher` can be used to identify patterns indicative of sarcasm or context-dependent sentiment (e.g., phrases like "yeah, right" or "you call *this* good?"). These rules could then adjust VADER's scores or flag reviews for manual inspection.
        *   **Lexicon Augmentation/Adaptation:** While VADER's lexicon is fixed, if building a more custom rule-based system, one could:
            *   Identify words/phrases that VADER misinterprets for specific demographics/domains (e.g., by analyzing false positives/negatives on a diverse, annotated dataset).
            *   Create rules to override or adjust scores for these specific terms in context. For example, a word that is positive in general lexicon might be negative in a specific product domain.
        *   **Negation and Intensifier Handling:** spaCy's dependency parsing can be used to create more robust rules for handling negations (e.g., "not very good") and intensifiers ("extremely bad") than VADER's simpler window-based approach, potentially making the sentiment scoring more accurate across different sentence structures.
        *   **Bias Auditing:** Use a diverse set of annotated reviews (covering different demographics, product types, and sentiment expressions) to audit the rule-based system's performance. Identify where it fails and refine rules accordingly. This is an ongoing process.

By combining these tools and approaches, developers can proactively identify and work towards mitigating biases, leading to fairer and more reliable AI systems.

## 2. Troubleshooting Challenge

### Buggy Code: A provided TensorFlow script has errors (e.g., dimension mismatches, incorrect loss functions). Debug and fix the code.

The provided TensorFlow script for MNIST classification had a few issues, primarily related to input data shape and inefficient prediction in the visualization loop.

**Description of Bugs Found:**

1.  **Incorrect Input Shape Handling:** The original script used a `layers.Reshape` layer as the first layer to define `input_shape`. The more standard and correct way is to specify `input_shape` in the first `Conv2D` layer and ensure the input data (`X_train`, `X_test`) is reshaped to include the channel dimension *before* being passed to the model. `Conv2D` layers expect 4D input: (batch_size, height, width, channels).
2.  **Inefficient Prediction in Visualization Loop:** The script called `model.predict()` for each individual image inside the plotting loop (`model.predict(X_test[i:i+1])`). This is highly inefficient as it invokes the model prediction process repeatedly for single instances.
3.  **Minor: Visualization Clarity:** The visualization showed predicted labels but not the true labels for comparison, which is helpful. Also, plotting training/validation accuracy and loss curves is a good practice to assess training.

**Fixes Applied:**

1.  **Input Data Reshaping:**
    *   Removed the `layers.Reshape((28, 28, 1), input_shape=(28, 28))` layer.
    *   Added manual reshaping for `X_train` and `X_test` after loading and normalization:
        ```python
        X_train = X_train.reshape((-1, 28, 28, 1))
        X_test = X_test.reshape((-1, 28, 28, 1))
        ```
    *   Specified `input_shape=(28, 28, 1)` directly in the first `layers.Conv2D` layer.
2.  **Efficient Prediction for Visualization:**
    *   Predictions for the first `num_images_to_show` (e.g., 5) test images are now made *once* before the loop:
        ```python
        predictions_logits = model.predict(X_test[:num_images_to_show])
        predicted_labels = np.argmax(predictions_logits, axis=1)
        ```
    *   The loop then uses these pre-computed `predicted_labels`.
3.  **Enhanced Visualization:**
    *   The plot titles now include both predicted and true labels: `plt.title(f"Pred: {predicted_labels[i]}\nTrue: {y_test[i]}")`.
    *   Added plots for training and validation accuracy and loss over epochs to help assess model training.
    *   Used `X_test_orig` for `imshow` to display images in their original 0-255 scale for potentially clearer visuals, though normalized data with `cmap='gray'` also works fine.
4.  **Model Summary:** Added `model.summary()` to print the model architecture, which is useful for verification.

*(The `SparseCategoricalCrossentropy(from_logits=True)` with no softmax on the final Dense layer was kept, as this is a valid and numerically stable configuration. The `tf.argmax` on the logits correctly identifies the predicted class index.)*

**Link to the fixed script:** [`buggy_script_fixed.py`](buggy_script_fixed.py)

---

Content of `buggy_script_fixed.py`:
```python
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
```
