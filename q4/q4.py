import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle


# (b) function for Normalising in B part
def preprocess_data(X_train, X_test, method="none"):
    if method == "normalize":
        # Min-Max Normalization: (x - min) / (max - min)
        X_train_min = np.min(X_train, axis=0)
        X_train_max = np.max(X_train, axis=0)
        X_train_scaled = (X_train - X_train_min) / (X_train_max - X_train_min)
        X_test_scaled = (X_test - X_train_min) / (X_train_max - X_train_min)
    elif method == "standardize":
        # Z-score Standardization: (x - mean) / std
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        X_train_scaled = (X_train - X_train_mean) / X_train_std
        X_test_scaled = (X_test - X_train_mean) / X_train_std
    else:
        return X_train, X_test  # no processing
    return X_train_scaled, X_test_scaled


# (c) Implement Softmax Function and One-Hot Encoding
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot_encode(y, num_classes):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot


# (d) Implement the Categorical Cross-Entropy Loss Function
def compute_loss(X, Y, W, b):
    """
    Compute the average cross-entropy loss.

    L = -1/m * sum( sum( y_ij * log(ŷ_ij) ) )
    """
    m = X.shape[0]
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    # add a small value to avoid log(0)
    loss = -np.mean(np.sum(Y * np.log(probs + 1e-8), axis=1))
    return loss


# (d) Implement Batch Gradient Descent Step
def gradient_descent_step(X, Y, W, b, learning_rate):
    m = X.shape[0]
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    error = probs - Y
    dW = np.dot(X.T, error) / m
    db = np.sum(error, axis=0) / m
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b


# (d) Prediction Function
def predict(X, W, b):
    logits = np.dot(X, W) + b
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def sgd_step(x, y, W, b, learning_rate):
    # Reshape x to (1, n_features)
    x = x.reshape(1, -1)
    logits = np.dot(x, W) + b  # shape (1, num_classes)
    probs = softmax(logits)
    error = probs - y.reshape(1, -1)  # shape (1, num_classes)
    dW = np.dot(x.T, error)  # shape (n_features, num_classes)
    db = error.flatten()  # shape (num_classes,)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b


def train_model_sgd(X, Y, W, b, learning_rate, epochs):
    loss_history = []
    m = X.shape[0]
    for epoch in range(epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]
        for i in range(m):
            W, b = sgd_step(X_shuffled[i], Y_shuffled[i], W, b, learning_rate)
        # Track loss every 10 epochs
        if epoch % 10 == 0:
            loss = compute_loss(X, Y, W, b)
            loss_history.append(loss)
            print(f"SGD Epoch {epoch:03d}: Loss = {loss:.4f}")
    return W, b, loss_history


# (a) Load the Iris Dataset
iris = load_iris()
X, y = iris.data, iris.target
num_classes = len(np.unique(y))
# print(X)
# print("hello")
# print(y)

# (a) Shuffle and split dataset into training (60%) and test (40%)
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# (b) Preprocess the Data using manual scaling
# Choose method: 'normalize', 'standardize', or 'none'
X_train_proc, X_test_proc = preprocess_data(X_train, X_test, method="none")

# (c) One-hot encode labels for training
Y_train = one_hot_encode(y_train, num_classes)

# (d) Initialize Model Parameters for Both Batch and SGD Methods
n_features = X_train_proc.shape[1]
W_batch = np.random.randn(n_features, num_classes) * 0.01
b_batch = np.zeros(num_classes)
W_sgd = np.random.randn(n_features, num_classes) * 0.01
b_sgd = np.zeros(num_classes)

# (e) Training Settings
epochs = 500
learning_rate = 0.1

# Model Using Batch Gradient Descent
loss_history_batch = []
for epoch in range(epochs):
    W_batch, b_batch = gradient_descent_step(
        X_train_proc, Y_train, W_batch, b_batch, learning_rate
    )
    if epoch % 10 == 0:
        loss = compute_loss(X_train_proc, Y_train, W_batch, b_batch)
        loss_history_batch.append(loss)
        print(f"Batch Epoch {epoch:03d}: Loss = {loss:.4f}")

#  Model Using Stochastic Gradient Descent
W_sgd, b_sgd, loss_history_sgd = train_model_sgd(
    X_train_proc, Y_train, W_sgd, b_sgd, learning_rate, epochs
)


# Evaluation using the Batch Gradient Descent model
y_pred_batch = predict(X_test_proc, W_batch, b_batch)
accuracy_batch = accuracy_score(y_test, y_pred_batch)
report_batch = "\nBatch Gradient Descent - Test Accuracy: {:.2f}%\n".format(
    accuracy_batch * 100
)
report_batch += "\nClassification Report (Batch):\n" + classification_report(
    y_test, y_pred_batch
)
report_batch += "\nConfusion Matrix (Batch):\n" + str(
    confusion_matrix(y_test, y_pred_batch)
)

# Evaluation using the Stochastic Gradient Descent model
y_pred_sgd = predict(X_test_proc, W_sgd, b_sgd)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
report_sgd = "\nStochastic Gradient Descent - Test Accuracy: {:.2f}%\n".format(
    accuracy_sgd * 100
)
report_sgd += "\nClassification Report (SGD):\n" + classification_report(
    y_test, y_pred_sgd
)
report_sgd += "\nConfusion Matrix (SGD):\n" + str(confusion_matrix(y_test, y_pred_sgd))

# Combine both reports into one final report
final_report = "Evaluation Report\n" + "=" * 30 + "\n"
final_report += report_batch + "\n" + "=" * 30 + "\n"
final_report += report_sgd

# Save the final report into a text file
with open("q4/evaluation_report.txt", "w") as report_file:
    report_file.write(final_report)


# Plot the loss curves for both training methods
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.arange(0, epochs, 10), loss_history_batch, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Batch Gradient Descent Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.arange(0, epochs, 10), loss_history_sgd, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Stochastic Gradient Descent Loss")
plt.grid(True)

plt.tight_layout()

# Save the plot as an image file in the current directory
plt.savefig("q4/loss_curves.png")
