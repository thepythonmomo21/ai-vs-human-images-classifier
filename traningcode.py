import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
print("Loading image data and labels...")
X = np.load(r'C:\Users\thepy\Downloads\X_images_final.npy')
y = np.load(r'C:\Users\thepy\Downloads\y_labels_final.npy')
print(f"Image data shape: {X.shape}, Labels shape: {y.shape}")

# Step 2: Normalize the feature values (pixel values) between 0 and 1
print("Scaling features with MinMaxScaler...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling complete.")

# Step 3: Split data into training and validation sets (80% train, 20% validation)
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# Step 4: Train a new Decision Tree classifier with tuned parameters
print("Training Decision Tree model...")
dt_model = DecisionTreeClassifier(
    criterion='entropy',      # Use entropy for information gain
    max_depth=20,             # Limit tree depth to prevent overfitting
    min_samples_split=4,      # Node must have at least 4 samples to split
    random_state=42
)
dt_model.fit(X_train, y_train)
print("Training complete.")

# Step 5: Evaluate model on validation set
print("Evaluating model...")
y_pred = dt_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Step 6: Save the trained model to disk
print("Saving model...")
model_save_path = r'C:\Users\thepy\Downloads\decision_tree_model_retrained.pkl'
joblib.dump(dt_model, model_save_path)
print(f"Model saved to: {model_save_path}")

# Step 7: Save the validation accuracy to a .txt file
print("Saving validation accuracy to file...")
accuracy_path = r'C:\Users\thepy\Downloads\validation_accuracy_retrained.txt'
with open(accuracy_path, 'w') as f:
    f.write(f"{accuracy:.4f}")
print(f"Accuracy saved to: {accuracy_path}")

print("✅ Decision Tree retraining complete!")
#%% [2]
import numpy as np

# File paths for saving processed data
scaled_data_path = r'C:\Users\thepy\Downloads\X_images_scaled_final.npy'
labels_data_path = r'C:\Users\thepy\Downloads\y_labels_final.npy'  # Already exists, but safe to overwrite

# Save scaled image data
np.save(scaled_data_path, X_scaled)
print(f"✅ Scaled image data saved to: {scaled_data_path}")

# Save labels
np.save(labels_data_path, y)
print(f"✅ Labels saved to: {labels_data_path}")
#%% [3] Naive Bayes Retraining and Saving

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Train a new Naive Bayes model
print("Training Naive Bayes model...")

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
print("Training complete.")

# Step 2: Evaluate the Naive Bayes model on the validation set
print("Evaluating Naive Bayes model...")
y_pred_nb = nb_model.predict(X_val)
accuracy_nb = accuracy_score(y_val, y_pred_nb)
print(f"Validation Accuracy: {accuracy_nb:.4f}")
print("Classification Report:")
print(classification_report(y_val, y_pred_nb))

# Step 3: Save the retrained Naive Bayes model to disk
print("Saving Naive Bayes model...")
nb_model_save_path = r'C:\Users\thepy\Downloads\naive_bayes_model_retrained.pkl'
joblib.dump(nb_model, nb_model_save_path)
print(f"Naive Bayes model saved to: {nb_model_save_path}")

# Step 4: Save the Naive Bayes validation accuracy to a .txt file
print("Saving validation accuracy to file...")
accuracy_nb_path = r'C:\Users\thepy\Downloads\naive_bayes_accuracy_retrained.txt'
with open(accuracy_nb_path, 'w') as f:
    f.write(f"{accuracy_nb:.4f}")
print(f"Accuracy saved to: {accuracy_nb_path}")

print("✅ Naive Bayes retraining complete!")
#%% [4] — TRAIN NN MODEL AND SAVE MODEL

import numpy as np
from keras.models import Sequential 
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load your data
X = np.load("C:/Users/thepy/Downloads/X_images_scaled_final.npy")
y = np.load("C:/Users/thepy/Downloads/y_labels_final.npy")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
##

# Build the model
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Optional: Add early stopping (you can skip this if you want)
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,                  # You can reduce to 10 if in a rush
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],     # remove this if not using early stopping
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Neural Network Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save("C:/Users/thepy/Downloads/nn_model_trained.keras")
print("✅ Model saved successfully!")

