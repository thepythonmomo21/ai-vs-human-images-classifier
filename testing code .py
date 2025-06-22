import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from skimage import io, transform
from skimage.color import rgb2gray

# ==== Step 1: Load Models ====
print("Loading models...")
dt_model = joblib.load(r'C:\Users\thepy\Downloads\decision_tree_model.pkl')
nb_model = joblib.load(r'C:\Users\thepy\Downloads\naive_bayes_model.pkl')
nn_model = load_model(r'C:\Users\thepy\Downloads\nn_model_trained.keras')
print("✅ Models loaded!")

# ==== Step 2: Load the Scaled Dataset and Labels ====
print("Loading training data (for scaler)...")
X_scaled = np.load(r'C:\Users\thepy\Downloads\X_images_scaled_final.npy')
print("✅ Scaled data loaded!")

# ==== Step 3: Fit the scaler on training data ====
scaler = MinMaxScaler()
scaler.fit(X_scaled)

# ==== Step 4: Load and preprocess image ====
print("Loading and preprocessing image...")
img_path = r'C:\Users\thepy\OneDrive\Desktop\ai documenting\00c5cf59728c4a798c7b23667094903f.jpg'
img = io.imread(img_path)
img_gray = rgb2gray(img)
img_resized = transform.resize(img_gray, (64, 64), anti_aliasing=True)
img_flattened = img_resized.flatten().reshape(1, -1)
img_scaled = scaler.transform(img_flattened)
print("✅ Image ready!")

# ==== Step 5: Predict ====

# --- Decision Tree ---
dt_pred = dt_model.predict(img_scaled)[0]
print(f"🌳 Decision Tree Prediction: {dt_pred}")

# --- Naive Bayes ---
nb_pred = nb_model.predict(img_scaled)[0]
print(f"🧠 Naive Bayes Prediction: {nb_pred}")

# --- Neural Network ---
nn_prob = nn_model.predict(img_scaled)[0][0]
nn_pred = 1 if nn_prob >= 0.5 else 0
print(f"🧠 Neural Network Prediction: {nn_pred} (Confidence: {nn_prob:.2f})")

print("\n✅ All predictions complete!")
