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

def apply_quantization(model , bits : dict[str , int] , mode : str):

    if mode == "PTQ":
        for name , module in model.named_modules():
            if isinstance(module , (nn.Conv2d , nn.Linear)):
                q , s = quantize_weights(module.weight.data , bits[name])
                module.weight.data = dequantize_weights(q , s)

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