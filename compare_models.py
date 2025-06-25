import torch
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import models.ganomaly as ganomaly
import models.ganomaly_baseline as ganomaly_baseline
from utils.data_loader import get_data_loader

def evaluate_model(model_type, target_class, device):
    print(f"--- Evaluating {model_type.upper()} Model on '{target_class}' Class ---")

    # Configure paths and model definitions based on model type
    if model_type == 'baseline':
        encoder_path = 'encoder_baseline.pth'
        decoder_path = 'decoder_baseline.pth'
        Encoder = ganomaly_baseline.Encoder
        Decoder = ganomaly_baseline.Decoder
    elif model_type == 'mife':
        encoder_path = 'encoder_mife.pth'
        decoder_path = 'decoder_mife.pth'
        Encoder = ganomaly.Encoder
        Decoder = ganomaly.Decoder
    else:
        raise ValueError("Invalid model type specified.")

    # Data Loader
    test_loader = get_data_loader('data', batch_size=32, is_train=False, class_name=target_class)

    # Load trained models
    encoder = Encoder(latent_dim=100).to(device)
    decoder = Decoder(latent_dim=100).to(device)
    
    # Load state dicts, ignoring potential mismatches for the baseline model
    encoder.load_state_dict(torch.load(encoder_path, map_location=device), strict=(model_type != 'baseline'))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device), strict=(model_type != 'baseline'))
    
    encoder.eval()
    decoder.eval()

    y_true = []
    anomaly_scores = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            
            # Forward pass differs between models
            if model_type == 'baseline':
                latent_i = encoder(imgs)
                gen_imgs = decoder(latent_i)
            else: # mife
                latent_i, skips = encoder(imgs)
                gen_imgs = decoder(latent_i, skips)

            error = torch.mean(torch.pow(imgs - gen_imgs, 2), dim=[1, 2, 3])
            anomaly_scores.extend(error.cpu().tolist())
            y_true.extend(labels.tolist())

    # Thresholding and prediction
    threshold = np.mean(anomaly_scores)
    y_pred = [1 if score > threshold else 0 for score in anomaly_scores]

    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_type.upper()} on {target_class.capitalize()}')
    matrix_path = f'confusion_matrix_{model_type}_{target_class}.png'
    plt.savefig(matrix_path)
    plt.close() # Close the plot to free memory
    print(f"Saved confusion matrix to {matrix_path}")

    # Calculate and print metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}\n")
    return precision, recall, f1

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes_to_test = ['candle', 'cashew', 'fryum']
    results = {}

    for class_name in classes_to_test:
        results[class_name] = {}
        p_base, r_base, f1_base = evaluate_model('baseline', class_name, device)
        results[class_name]['baseline'] = {'p': p_base, 'r': r_base, 'f1': f1_base}
        
        p_mife, r_mife, f1_mife = evaluate_model('mife', class_name, device)
        results[class_name]['mife'] = {'p': p_mife, 'r': r_mife, 'f1': f1_mife}

    # Print summary table
    print("\n---=== FINAL RESULTS SUMMARY ===---")
    print(f"{'Class':<10} | {'Model':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-"*60)
    for class_name, models in results.items():
        for model_name, metrics in models.items():
            print(f"{class_name:<10} | {model_name:<10} | {metrics['p']:.4f}{'':<5} | {metrics['r']:.4f}{'':<5} | {metrics['f1']:.4f}")
