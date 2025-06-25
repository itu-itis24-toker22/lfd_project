import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the trained model definitions
from models.ganomaly import Encoder, Decoder
from utils.data_loader import get_data_loader

def evaluate(data_path, encoder_path='encoder_mife.pth', decoder_path='decoder_mife.pth', batch_size=32, latent_dim=100, target_class=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loader for test set
    test_loader = get_data_loader(data_path, batch_size=batch_size, is_train=False, class_name=target_class)

    # Load trained models
    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()

    y_true = []
    y_pred = []
    anomaly_scores = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            latent_i, skips = encoder(imgs)
            gen_imgs = decoder(latent_i, skips)

            # Anomaly score based on reconstruction error
            error = torch.mean(torch.pow(imgs - gen_imgs, 2), dim=[1, 2, 3])
            anomaly_scores.extend(error.cpu().tolist())
            y_true.extend(labels.tolist())

    # Thresholding to get predictions
    # A common approach is to use the mean of anomaly scores on a validation set.
    # For simplicity, we use the mean of the test set scores here, but this could be improved.
    threshold = np.mean(anomaly_scores)
    y_pred = [1 if score > threshold else 0 for score in anomaly_scores]

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print("--- MIFE Model Performance ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('MIFE Confusion Matrix')
    plt.savefig('confusion_matrix_mife.png')
    print("MIFE confusion matrix saved to confusion_matrix_mife.png")

if __name__ == '__main__':
    DATASET_PATH = 'data'
    ENCODER_WEIGHTS = 'encoder_mife.pth'
    DECODER_WEIGHTS = 'decoder_mife.pth'
    # Make sure to use the same target_class as in training
    evaluate(DATASET_PATH, ENCODER_WEIGHTS, DECODER_WEIGHTS, target_class='candle')
