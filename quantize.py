
from google.protobuf import wrappers_pb2
from torch import tensor
from torch import Tensor
import torch.nn as nn 
import torch


def quantize_weights(tensor : Tensor , bits : int):
    qmax = (1 << (bits-1)) - 1 
    qmin = -(1 << (bits-1))
    s = torch.clamp(torch.amax(torch.abs(tensor) , dim=tuple(range(1 , tensor.dim())) , keepdim=True) / qmax , min=1e-8) # torch.amax - specify dimensions to take max along that dimension alone. 
    q = torch.clamp(torch.round(tensor/s) , qmin , qmax) 

    return q , s

def dequantize_weights(tensor : Tensor , s : Tensor):
    r = s * tensor 

    return r 

def quantize_act_hook(bits: int):
    def hook(module , inputs):
        tensor = inputs[0]
        qmax = (1 << (bits-1)) - 1 
        qmin = -(1 << (bits-1))
        s = torch.clamp(torch.amax(torch.abs(tensor) , dim=tuple(range(1 , tensor.dim())) , keepdim=True) / qmax , min=1e-8) # torch.amax - specify dimensions to take max along that dimension alone. 
        q = torch.clamp(torch.round(tensor/s) , qmin , qmax) 
        r = q * s 
        return r
    return hook 

def apply_quantization(model , weight_bits : dict[str , int] , act_bits : dict[str , int] , mode : str):

    if mode == "PTQ":
        for name , module in model.named_modules():
            if isinstance(module , (nn.Conv2d , nn.Linear)):
                q , s = quantize_weights(module.weight.data , weight_bits[name])
                module.weight.data = dequantize_weights(q , s)
                module.register_forward_pre_hook(quantize_act_hook(act_bits[name]))

    elif mode == "QAT":
        raise(NotImplementedError)
        
    return model

def uniform_alloc(model , unif_bits : int):
    bits : dict[str , int] = {}
    for name , module in model.named_modules():
        if isinstance(module , (nn.Conv2d , nn.Linear)):
            bits[name] = unif_bits 

    print("Uniform Bits:")
    print(bits)
    
    return bits

def mixed_uniform_alloc(model , end_bits : int , int_bits : int):

    names = []
    for name , module in model.named_modules():
        if isinstance(module , (nn.Conv2d , nn.Linear)):
            names.append(name)


    bits : dict[str , int] = {}
    for idx , name in enumerate(names):
        if(idx == 0 or idx == len(names) - 1):
            bits[name] = end_bits 

        else:
            bits[name] = int_bits
    
    print("Mixed Allocation bits:")
    print(bits)
    print(f"Avg bits:{((end_bits * 2) + int_bits * (len(names) - 2)) / (len(names)):.4f}")
    
    return bits 


def model_size(model , bits : dict[str , int]):

    bit_original_count = 0 
    bit_comp_count = 0 
    scale_count = 0 

    for name , module in model.named_modules():
        if isinstance(module , (nn.Conv2d , nn.Linear)):
            tensor = module.weight.data 
            bit_original_count += tensor.numel() * 32

            bit = bits[name] 
            bit_comp_count += bit * tensor.numel()
            scale_count += tensor.size(dim=0) * 32 

    original_size = bit_original_count / 8 
    compressed_size = (bit_comp_count + scale_count) / 8 
    compression_ratio = original_size / compressed_size 

    print(f"Original Size:{original_size} bytes")
    print(f"Compressed Size:{compressed_size} bytes")
    print(f"Compression Ratio:{compression_ratio}")

    return original_size , compressed_size , compression_ratio


def activation_size(model, act_bits: dict[str, int], device):
    dummy_input = torch.zeros(1, 3, 32, 32).to(device)
    
    act_original_count = 0
    act_comp_count = 0
    
    shapes = {}
    def get_shape_hook(name):
        def hook(module, inputs, output):
            shapes[name] = inputs[0].numel()
        return hook
        
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(get_shape_hook(name)))
            
    model.eval()
    with torch.no_grad():
        model(dummy_input)
        
    for h in hooks:
        h.remove()
        
    for name in shapes:
        numel = shapes[name]
        act_original_count += numel * 32
        act_comp_count += numel * act_bits[name]
        act_comp_count += 32 
        
    original_size = act_original_count / 8
    compressed_size = act_comp_count / 8
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    print(f"Act Original Size:{original_size} bytes")
    print(f"Act Compressed Size:{compressed_size} bytes")
    print(f"Act Compression Ratio:{ratio}")
    
    return original_size, compressed_size, ratio