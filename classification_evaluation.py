from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
import os  # Ensure os is imported

import torch

NUM_CLASSES = len(my_bidict)

def get_label(model, model_input, device):
    model.eval()
    
    # Ensure image has a batch dimension.
    if model_input.dim() == 3:
        image = model_input.unsqueeze(0)
    else:
        image = model_input  # Assume image is already batched.
    
    all_log_likelihoods = []
    keys = list(my_bidict.keys())  # List of possible label names
    
    with torch.no_grad():
        # For each label candidate, compute log-likelihoods.
        for label in keys:
            per_sample_ll = []
            # Process each sample in the batch.
            for i in range(image.size(0)):
                sample = image[i:i+1]  # shape: (1, C, H, W)
                output = model(sample, labels=[label])
                neg_ll = discretized_mix_logistic_loss(sample, output)
                ll = -neg_ll
                per_sample_ll.append(ll)
            per_sample_ll_tensor = torch.stack(per_sample_ll, dim=0)
            all_log_likelihoods.append(per_sample_ll_tensor.unsqueeze(1))
        
        # Concatenate to get a tensor of shape (batch_size, num_classes)
        log_likelihoods = torch.cat(all_log_likelihoods, dim=1)
        predicted_indices = torch.argmax(log_likelihoods, dim=1)
        
        # Map indices to the numeric label using my_bidict.
        predicted_classes = []
        for idx in predicted_indices.tolist():
            predicted_class_str = keys[idx]
            predicted_class_int = my_bidict[predicted_class_str]
            predicted_classes.append(predicted_class_int)
        
        predicted_tensor = torch.tensor(predicted_classes, device=device, dtype=torch.long)
    
    return predicted_tensor

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()  # From your utils
    for batch_idx, item in enumerate(tqdm(data_loader)):
        # In non-test modes, the dataset returns (image, category, csv_index).
        model_input, categories, _ = item  
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    return acc_tracker.get_ratio()

###############################################################################
#                              Main Evaluation Code                           #
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str, default='validation',
                        help='Mode for the dataset ("train", "validation", or "test")')
    args = parser.parse_args()
    pprint(args.__dict__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True, 'drop_last': False}
    
    ds_transforms = Compose([Resize((32, 32)), rescaling])
    
    # DataLoader for evaluation (e.g., validation) uses the updated dataset.
    dataloader = DataLoader(
        CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    
    # Replace the random classifier with your trained model. Ensure the model configuration
    # (including nr_logistic_mix) matches what was used during training.
    model = PixelCNN(nr_resnet=1, nr_filters=80, input_channels=3, nr_logistic_mix=5)
    model = model.to(device)
    
    # Load model parameters from the fixed path.
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    # Evaluate accuracy on the validation set (if applicable).
    acc = classifier(model=model, data_loader=dataloader, device=device)
    print(f"Accuracy: {acc}")
    
    # === New Section: Save Predicted Labels to CSV ===
    print("Generating predictions CSV file...")
    
    # Use a DataLoader with shuffle disabled so that the order matches the original CSV.
    dataset = CPEN455Dataset(root_dir=args.data_dir, mode=args.mode, transform=ds_transforms)
    pred_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )
    
    predictions_all = []  # To store predicted labels for each sample
    index_all = []        # To store csv_index values from the CSV
    
    for batch in tqdm(pred_dataloader):
        # Now each batch returns (inputs, category, csv_index)
        inputs, _, sample_indices = batch  
        inputs = inputs.to(device)
        with torch.no_grad():
            preds = get_label(model, inputs, device)
        preds = preds.cpu().numpy().tolist()
        
        # sample_indices is a tensor; convert it to list.
        sample_indices = sample_indices.tolist() if not isinstance(sample_indices, list) else sample_indices
        
        for pred, s_idx in zip(preds, sample_indices):
            predictions_all.append(pred)
            index_all.append(s_idx)
    
    csv_filename = 'predicted_labels.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Index", "Predicted Label"])
        # Write each prediction with its corresponding CSV index
        for idx, pred in zip(index_all, predictions_all):
            writer.writerow([idx, pred])
    
    print(f"CSV file with predictions saved to {csv_filename}")