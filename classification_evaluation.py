'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    model.eval()
    
    # Ensure image has a batch dimension.
    if model_input.dim() == 3:
        image = model_input.unsqueeze(0)
    else:
        image = model_input  # Assume image is already batched.

    all_log_likelihoods = []
    keys = list(my_bidict.keys())  # List of possible label names
    num_classes = len(keys)
    
    with torch.no_grad():
        # Loop over each class label in my_bidict.
        for label in keys:
            per_sample_ll = []  # to store the log-likelihood for each sample for this label.
            # Loop over each sample in the batch.
            for i in range(image.size(0)):
                sample = image[i:i+1]  # shape: (1, C, H, W)
                # Pass this single sample, with the current label as a list.
                output = model(sample, labels=[label])
                # Compute the negative log-likelihood for this sample and then its log-likelihood.
                neg_ll = discretized_mix_logistic_loss(sample, output)
                ll = -neg_ll
                per_sample_ll.append(ll)
            # Stack the per-sample log-likelihoods to get a tensor of shape (batch_size,).
            per_sample_ll_tensor = torch.stack(per_sample_ll, dim=0)
            # Add a class dimension so that each tensor becomes shape (batch_size, 1).
            all_log_likelihoods.append(per_sample_ll_tensor.unsqueeze(1))
        
        # Concatenate along the second dimension to produce a tensor of shape (batch_size, num_classes).
        log_likelihoods = torch.cat(all_log_likelihoods, dim=1)
        
        # For each sample, choose the class with the highest log-likelihood.
        predicted_indices = torch.argmax(log_likelihoods, dim=1)  # shape: (batch_size,)
        
        # Map the index back to the appropriate numeric label using my_bidict.
        predicted_classes = []
        for idx in predicted_indices.tolist():
            predicted_class_str = keys[idx]
            predicted_class_int = my_bidict[predicted_class_str]
            predicted_classes.append(predicted_class_int)
        
        # Create a tensor (on the proper device) with the per-sample predicted numeric labels.
        predicted_tensor = torch.tensor(predicted_classes, device=device, dtype=torch.long)

    return predicted_tensor


def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    model = PixelCNN(
    nr_resnet=2,
    nr_filters=40,
    nr_logistic_mix=5,
    num_classes=NUM_CLASSES
    )
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        
        