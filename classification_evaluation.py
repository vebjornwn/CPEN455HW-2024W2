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
# def get_label(model, model_input, device):
    
#     model.eval()
#     log_likelihoods = []
    
#     # Ensure image has a batch dimension.
#     if model_input.dim() == 3:
#         image = model_input.unsqueeze(0)
#     else:
#         image = model_input  # Make sure image is defined even if already batched.

#     with torch.no_grad():
#         for label in my_bidict.keys():
#             label_list = [label] * image.size(0)

#             outputs = model(image, labels=label_list)
            
#             neg_log_likelihood = discretized_mix_logistic_loss(image, outputs)
#             log_likelihood = -neg_log_likelihood  # convert negative loss to raw log-likelihood
            
#             log_likelihoods.append(log_likelihood.item())
    
#     # Convert list to numpy array and select the class with the highest log-likelihood.
#     log_likelihoods = np.array(log_likelihoods)
#     best_index = np.argmax(log_likelihoods)
#     # Get the predicted class as a string.
#     predicted_class_str = list(my_bidict.keys())[best_index]
#     # Convert this predicted string into its numeric label.
#     predicted_class_int = my_bidict[predicted_class_str]
#     # Create a tensor with the predicted numeric label repeated for the batch.
#     predicted_class = torch.tensor([predicted_class_int] * image.size(0), device=device, dtype=torch.long)

#     return predicted_class
# End of your code

def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    # answer = model(model_input, device)
    # return answer

    batch_size = model_input.size(0)
    # To accumulate log-likelihoods for each class:
    all_log_likelihoods = []

    for c in range(NUM_CLASSES):
        condition =  [c] * batch_size
        # Run the model for the entire batch, conditioned on class c.
        # (We assume the model can process the entire batch at once.)
        outputs = model(model_input, labels=condition)
        
        # Now, because discretized_mix_logistic_loss returns a single scalar if fed a batch,
        # we iterate sample-by-sample.
        sample_ll_list = []
        for i in range(batch_size):
            # Get the i-th sample and corresponding model output
            sample_x = model_input[i:i+1]   # shape (1, C, H, W)
            sample_output = outputs[i:i+1]    # corresponding output from the model
            # Compute the negative log-likelihood for this single sample
            neg_ll = discretized_mix_logistic_loss(sample_x, sample_output)
            # Convert to log-likelihood
            ll = -neg_ll
            sample_ll_list.append(ll)
        # Stack the per-sample log-likelihoods into a tensor of shape (batch_size,)
        sample_ll_tensor = torch.stack(sample_ll_list, dim=0)
        # Add an extra dimension so later we can concatenate across classes: shape (batch_size, 1)
        all_log_likelihoods.append(sample_ll_tensor.unsqueeze(1))
    
    # Concatenate over the second dimension: final shape (batch_size, NUM_CLASSES)
    log_likelihoods = torch.cat(all_log_likelihoods, dim=1)
    # The predicted label is the class with the highest log-likelihood for each sample
    predicted = torch.argmax(log_likelihoods, dim=1)

    predicted_class = torch.tensor([predicted] *batch_size, device=device, dtype=torch.long)
    return predicted

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
        
        