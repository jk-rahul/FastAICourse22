import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os # For checking file existence

# --- 1. Load and Prepare Data ---
file_name = '/Users/jkrahul/Downloads/Titanic_kaggle_train.xlsx' # Corrected file name and path
sheet_name = 'linear'           # Your specified sheet name

# Data starts from the 4th row, so header is at index 3 (0-indexed)
try:
    df = pd.read_excel(file_name, sheet_name=sheet_name, header=3)
except FileNotFoundError:
    print(f"Error: '{file_name}' not found. Please make sure the Excel file is in the same directory as the script.")
    exit()
except ValueError:
    print(f"Error: Sheet '{sheet_name}' not found in '{file_name}'. Please check the sheet name.")
    exit()

# --- Debugging Step: Print actual columns read by pandas ---
print("Columns read by pandas (before stripping whitespace):")
print(df.columns.tolist())
print("-" * 30)

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Define your target variable
target = 'Survived'

# Define your feature columns, now INCLUDING 'Ones' as it will serve as the bias input
features_to_use = ['SibSp', 'Parch', 'Age_N', 'log_Fare', 'Pclass_1', 'Pclass_2', 'Embark_S', 'Embark_C', 'IsMale', 'Ones']

# Ensure all required columns exist in the DataFrame
required_columns = features_to_use + [target]
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Column '{col}' not found in the '{sheet_name}' sheet after stripping whitespace. Please check your column names in Excel (row 4) and in the 'features_to_use' list.")
        print(f"Available columns in DataFrame: {df.columns.tolist()}")
        exit()

# Handle missing values for features (simple imputation for demonstration)
for col in features_to_use:
    if df[col].isnull().any(): # Check if there are any NaN values
        df[col] = df[col].fillna(df[col].mean()) # Corrected line: assign back instead of inplace=True

# Extract the target variable
y = df[target].values.reshape(-1, 1) # Reshape y to be a column vector

# Extract features. 'X' now contains all features, including 'Ones'.
X = df[features_to_use].values

# --- Debugging Step: Print shapes of X and y ---
print(f"Shape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")
print("-" * 30)

# --- 2. Define Activation and Loss Functions ---

def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU function."""
    return (z > 0).astype(float)

def mean_squared_error_loss(y_true, y_pred):
    """Mean Squared Error (MSE) Loss function."""
    return np.mean((y_pred - y_true)**2)

# --- 3. Initialize Model Parameters (Weights for Two Parallel Branches, including bias) ---

input_size = X.shape[1] # Number of features (now includes 'Ones')
output_size = 1         # Each branch outputs a single value

# File path for saving/loading weights
weights_file = 'two_branch_relu_weights.npz' # Reverted filename for this model

# Try to load previously trained weights
if os.path.exists(weights_file):
    print(f"Loading previously trained weights from {weights_file}...")
    loaded_data = np.load(weights_file)
    W_branch1 = loaded_data['W_branch1']
    W_branch2 = loaded_data['W_branch2']
else:
    print("Initializing weights randomly (no saved weights found).")
    # Weights for Branch 1 (input_size includes 'Ones' column for bias)
    W_branch1 = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size) # He init for ReLU
    
    # Weights for Branch 2 (input_size includes 'Ones' column for bias)
    W_branch2 = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size) # He init for ReLU

# --- 4. Hyperparameters for Gradient Descent ---
learning_rate = 0.01   # Reverted to a good starting point for MSE
num_iterations = 5000  # Reverted to a good starting point for MSE

# To keep track of the loss over iterations for plotting
loss_history = []

# --- 5. Two-Branch ReLU Model Training (Gradient Descent) ---
print("Starting Two-Branch ReLU Model Training...")
for iteration in range(num_iterations):
    # --- Forward Propagation ---
    # Branch 1: Z1_branch1 = X @ W_branch1 (Bias is now implicitly handled by 'Ones' column in X)
    Z1_branch1 = np.dot(X, W_branch1)
    A1_branch1 = relu(Z1_branch1) # ReLU activation for Branch 1
    
    # Branch 2: Z1_branch2 = X @ W_branch2 (Bias is now implicitly handled by 'Ones' column in X)
    Z1_branch2 = np.dot(X, W_branch2)
    A1_branch2 = relu(Z1_branch2) # ReLU activation for Branch 2
    
    # Final Prediction: Sum of outputs from both branches
    Y_pred = A1_branch1 + A1_branch2
    
    # --- Calculate Loss (MSE) ---
    current_loss = mean_squared_error_loss(y, Y_pred)
    loss_history.append(current_loss)
    
    # --- Backward Propagation (Gradients for MSE) ---
    # Gradient of Loss with respect to Y_pred
    dY_pred = 2 * (Y_pred - y) / len(y) # Derivative of MSE
    
    # Gradients back through A1_branch1 and A1_branch2
    dA1_branch1 = dY_pred
    dA1_branch2 = dY_pred
    
    # Gradients back through Z1_branch1 and Z1_branch2 (applying ReLU derivative)
    dZ1_branch1 = dA1_branch1 * relu_derivative(Z1_branch1)
    dZ1_branch2 = dA1_branch2 * relu_derivative(Z1_branch2)
    
    # Gradients for W_branch1 (dW_branch1 implicitly includes bias gradient)
    dW_branch1 = np.dot(X.T, dZ1_branch1)
    
    # Gradients for W_branch2 (dW_branch2 implicitly includes bias gradient)
    dW_branch2 = np.dot(X.T, dZ1_branch2)
    
    # --- Update Parameters ---
    W_branch1 -= learning_rate * dW_branch1
    W_branch2 -= learning_rate * dW_branch2
    
    # Print progress every 500 iterations
    if iteration % 500 == 0:
        print(f"Iteration {iteration}: Loss = {current_loss:.4f}")

print("\nTwo-Branch ReLU Model Training Finished.")
print(f"Final Loss: {current_loss:.4f}")

# --- Calculate Training Accuracy (for 0/1 target) ---
# Convert predictions to binary (0 or 1) using a threshold
# Since ReLU outputs are non-negative, and the sum can be > 1,
# a threshold of 0.5 is still a reasonable way to classify for a 0/1 target.
final_predictions = (Y_pred >= 0.5).astype(int)
# Calculate accuracy
accuracy = np.mean(final_predictions == y) * 100
print(f"Training Accuracy: {accuracy:.2f}%")

# Save the trained weights
np.savez(weights_file, W_branch1=W_branch1, W_branch2=W_branch2)
print(f"Trained weights saved to {weights_file}")

# --- 6. Visualize Results (Loss Convergence) ---
plt.figure(figsize=(8, 6))
plt.plot(loss_history, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (Loss)')
plt.title('Loss Convergence during Two-Branch ReLU Model Training')
plt.grid(True)
plt.gca().set_facecolor('#f0f0f0') # Light grey background
plt.xticks(rotation=45) # Rotate x-axis labels for better readability
plt.yticks(rotation=45) # Rotate y-axis labels for better readability
plt.gca().spines['top'].set_visible(False) # Remove top border
plt.gca().spines['right'].set_visible(False) # Remove right border
plt.tight_layout()
plt.show()

# --- Interpretation Note ---
# This script implements a "Two-Branch ReLU Model" as described.
# It uses two parallel linear layers, each followed by a ReLU activation.
# Their outputs are summed for the final prediction.
# The 'Ones' column in your input data serves as the bias input for each of the two branches.
# Mean Squared Error (MSE) is used as the loss function, treating the problem as regression.

