from quantize import hawq_alloc
import wandb
import torch
import torch.nn as nn
import copy
from model import get_model
from dataloader import get_test_data
from quantize import mixed_uniform_alloc , apply_quantization , model_size , uniform_alloc, activation_size

def test(model , test_data , device, loss_fn=nn.CrossEntropyLoss()):
    '''
    Function used for inference - Both Quantized models hyperparameter sweep and Original model
    '''
    model.eval() 

    correct = 0 
    total = 0 
    total_loss = 0.0

    with torch.no_grad():
        for idx , (images , labels) in enumerate(test_data):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _ , pred = torch.max(logits , 1)

            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if (idx + 1)%30 == 0:
                print(f"Test Batch {idx}/{len(test_data)} , Correct:{correct}/{total}")

    accuracy = correct/total 
    print(f"Accuracy:{accuracy:.4f}")

    avg_loss = total_loss / total
    print(f"Test Loss:{avg_loss:.4f}")
    return accuracy, avg_loss

def model_loader(path , device):

    '''
    Loads the model into device from the given path
    '''

    model = get_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(path , device , weights_only=True)['state_dict'])

    return model 

def evaluate_quantize(model , test_data , device , config , loss_fn):
    '''
    Copies the original model and applies quantization and then runs inference.
    '''
    model = copy.deepcopy(model)

    weight_bits = {}
    act_bits = {}
    if config.policy == "mixed":
        weight_bits = mixed_uniform_alloc(model , config.end_bits , config.int_bits)
        act_bits = mixed_uniform_alloc(model , config.act_end_bits , config.act_int_bits)

    elif config.policy == "uniform":
       weight_bits = uniform_alloc(model , config.unif_bits)
       act_bits = uniform_alloc(model , config.act_unif_bits) 

    elif config.policy == "HAWQ":
        weight_bits = hawq_alloc(model , test_data , device , loss_fn , config.end_bits , config.int_bits , config.mid_bits , config.ratio)
        act_bits = uniform_alloc(model , config.act_unif_bits)

    _original_size , _compressed_size , w_ratio = model_size(model , weight_bits)
    _act_original_size , _act_compressed_size , a_ratio = activation_size(model , act_bits , device)

    model = apply_quantization(model , weight_bits , act_bits)

    accuracy, _ = test(model , test_data , device)

    return accuracy , w_ratio , a_ratio



if __name__ == "__main__":
    wandb.init(project="CS6886_Assignment2", name="non-quantized", group="non-quantized")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "./checkpoints/best_model.pth"
    test_data = get_test_data()
    model = model_loader(path , device)  # Load model from the saved checkpoint.
    
    loss_fn = nn.CrossEntropyLoss()
    original_acc, original_loss = test(model , test_data , device, loss_fn)

    print(f"Accuracy of original model:{original_acc:.4f}")
    
    wandb.log({
        "test_acc": original_acc,
        "test_loss": original_loss
    })
    wandb.finish()