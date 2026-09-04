import torch 
import copy
from model import get_model
from dataloader import get_test_data
from quantize import mixed_uniform_alloc , apply_quantization , model_size , uniform_alloc

def test(model , test_data , device):
    model.eval() 

    correct = 0 
    total = 0 

    with torch.no_grad():
        for idx , (images , labels) in enumerate(test_data):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _ , pred = torch.max(logits , 1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if (idx + 1)%30 == 0:
                print(f"Test Batch {idx}/{len(test_data)} , Correct:{correct}/{total}")

    accuracy = correct/total 
    print(f"Accuracy:{accuracy:.4f}")

    return accuracy 

def model_loader(path , device):
    model = get_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(path , device , weights_only=True)['state_dict'])

    return model 

def evaluate_quantize(model , test_data , device , config , mode):
    model = copy.deepcopy(model)

    bits = {}
    if config.policy == "mixed":
        bits = mixed_uniform_alloc(model , config.end_bits , config.int_bits)

    elif config.policy == "uniform":
        bits = uniform_alloc(model , config.unif_bits)

    else:
        raise(NotImplementedError)

    original_size , compressed_size , ratio = model_size(model , bits)

    model = apply_quantization(model , bits , mode)

    accuracy = test(model , test_data , device)

    return accuracy , ratio



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "./checkpoints/best_model.pth"
    test_data = get_test_data()
    model = model_loader(path , device)
    original_acc = test(model , test_data , device)

    print(f"Accuracy of original model:{original_acc:.4f}")