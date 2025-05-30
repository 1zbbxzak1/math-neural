# --- Training ---
import numpy as np

from mode_utils import save_model


def train(model, X, y, batch_size=64, epochs=5, save_path='model.pkl'):
    n = len(X)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        permutation = np.random.permutation(n)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        loss_sum = 0
        correct = 0

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            batch_loss = 0
            batch_correct = 0

            for x, y_true in zip(X_batch, y_batch):
                out = x
                for layer in model[:-1]:
                    out = layer.forward(out)
                probs, l = model[-1].forward(out, y_true)
                batch_loss += l
                if np.argmax(probs) == np.argmax(y_true):
                    batch_correct += 1

                grad = model[-1].backward()
                for layer in reversed(model[:-1]):
                    grad = layer.backward(grad)

            loss_sum += batch_loss
            correct += batch_correct

            if (i // batch_size) % 10 == 0:
                print(
                    f"Batch {i // batch_size}, Avg Loss: {batch_loss / len(X_batch):.4f}, Accuracy: {batch_correct / len(X_batch):.4f}")

        print(f"Epoch {epoch + 1} finished. Avg Loss: {loss_sum / n:.4f}, Accuracy: {correct / n:.4f}")

        # Сохраняем модель после каждой эпохи
        save_model(model, save_path)


# --- Testing ---
def test(model, X, y):
    loss = 0
    num_correct = 0
    for x, y_true in zip(X, y):
        out = x
        for layer in model[:-1]:
            out = layer.forward(out)
        probs, l = model[-1].forward(out, y_true)
        loss += l
        if np.argmax(probs) == np.argmax(y_true):
            num_correct += 1
    print(f'Test Loss: {loss / len(X):.3f}, Accuracy: {num_correct / len(X):.3f}')
