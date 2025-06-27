import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2 # OpenCV for image manipulation

# Define the model architecture (must be same as the trained one)
def create_mnist_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to load or train a model (simplified for demo)
@st.cache_resource # Cache the model to avoid reloading/retraining
def load_or_train_model():
    # In a real app, you'd load pre-trained weights.
    # For this demo, we'll quickly train a model if weights aren't available.
    # It's better to save and load weights: model.save_weights('mnist_model_weights.h5')

    model_weights_path = 'mnist_model_weights.h5' # Keras HDF5 format

    model = create_mnist_cnn_model()

    try:
        model.load_weights(model_weights_path)
        print("Loaded pre-trained model weights.")
        st.sidebar.success("Loaded pre-trained model weights!")
    except (FileNotFoundError, IOError):
        st.sidebar.warning(f"Model weights ({model_weights_path}) not found. Training a new model (this may take a moment)...")
        print("Training a new model as weights not found...")
        (x_train, y_train), (_, _) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        y_train_categorical = to_categorical(y_train, num_classes=10)

        # Train for fewer epochs for a quick demo if training live
        model.fit(x_train, y_train_categorical, epochs=5, batch_size=128, verbose=1)
        try:
            model.save_weights(model_weights_path)
            st.sidebar.success(f"New model trained and weights saved to {model_weights_path}")
        except Exception as e:
            st.sidebar.error(f"Error saving model weights: {e}")

    return model

# Function to preprocess the uploaded image
def preprocess_image(image_data):
    # Convert image to grayscale
    img = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert colors (MNIST digits are white on black background)
    # Check average pixel intensity. If it's high, it's likely black on white.
    if np.mean(img_resized) > 127:
        img_inverted = cv2.bitwise_not(img_resized)
    else:
        img_inverted = img_resized

    # Normalize the image
    img_normalized = img_inverted.astype('float32') / 255.0

    # Reshape for the model
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)

    return img_reshaped, img_inverted # Return inverted for display

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("✏️ MNIST Handwritten Digit Classifier")
st.write("""
Upload an image of a handwritten digit (0-9) or use a sample from the MNIST test set.
The app will predict the digit using a Convolutional Neural Network (CNN).
""")

# Load the model
model = load_or_train_model()

col1, col2 = st.columns(2)

with col1:
    st.header("🖼️ Upload Your Digit")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        processed_image, display_image = preprocess_image(uploaded_file)

        st.image(display_image, caption="Processed Input Image (28x28 Grayscale)", width=150)

        if st.button("Predict Digit (Uploaded Image)"):
            with st.spinner("Predicting..."):
                prediction = model.predict(processed_image)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)

                st.success(f"Predicted Digit: **{predicted_digit}**")
                st.metric(label="Confidence", value=f"{confidence:.2%}")

                st.subheader("Prediction Probabilities:")
                # Show probabilities as a bar chart
                prob_df = pd.DataFrame(prediction.flatten(), index=[str(i) for i in range(10)], columns=['Probability'])
                st.bar_chart(prob_df)


with col2:
    st.header("🧪 Or Use a Sample MNIST Image")
    if st.button("Load Random MNIST Test Image"):
        (_, _), (x_test, y_test) = mnist.load_data()

        sample_idx = np.random.randint(0, x_test.shape[0])
        sample_image_original = x_test[sample_idx] # This is 28x28 grayscale
        sample_label = y_test[sample_idx]

        # Preprocess for model
        sample_image_processed = sample_image_original.reshape(1, 28, 28, 1).astype('float32') / 255.0

        st.image(sample_image_original, caption=f"Sample MNIST Image (True Label: {sample_label})", width=150)

        if st.button("Predict Digit (Sample Image)"):
            with st.spinner("Predicting..."):
                prediction = model.predict(sample_image_processed)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)

                st.success(f"Predicted Digit: **{predicted_digit}**")
                st.metric(label="Confidence", value=f"{confidence:.2%}")

                st.subheader("Prediction Probabilities:")
                import pandas as pd # Ensure pandas is imported here if not globally
                prob_df = pd.DataFrame(prediction.flatten(), index=[str(i) for i in range(10)], columns=['Probability'])
                st.bar_chart(prob_df)

st.sidebar.header("About")
st.sidebar.info("""
This is a simple web application to demonstrate a CNN model trained on the MNIST dataset.
The model architecture is:
- Conv2D (32 filters, 3x3 kernel) -> MaxPooling2D
- Conv2D (64 filters, 3x3 kernel) -> MaxPooling2D
- Flatten
- Dense (128 units, ReLU) -> Dropout (0.5)
- Dense (10 units, Softmax)

**Note:** If model weights are not found, a new model is trained for a few epochs. For best results, a fully trained model's weights should be used.
""")

st.markdown("---")
st.write("To run this app locally:")
st.code("""
pip install streamlit tensorflow numpy opencv-python pandas
streamlit run app.py
""")
st.write("Ensure you have `mnist_model_weights.h5` in the same directory or the app will train a new model.")
