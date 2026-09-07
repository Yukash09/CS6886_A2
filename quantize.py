from torch import tensor
from torch import Tensor
import torch.nn as nn 
import torch


def quantize_weights(tensor : Tensor , bits : int):

    '''
    Symmetric Quantization
    '''

    qmax = (1 << (bits-1)) - 1 
    qmin = -(1 << (bits-1))
    s = torch.clamp(torch.amax(torch.abs(tensor) , dim=tuple(range(1 , tensor.dim())) , keepdim=True) / qmax , min=1e-8) # torch.amax - specify dimensions to take max along that dimension alone. 
    q = torch.clamp(torch.round(tensor/s) , qmin , qmax) 

    return q , s

def dequantize_weights(tensor : Tensor , s : Tensor):

    r = s * tensor 

    return r 

def quantize_act_hook(bits: int):

    '''
    Hook function to quantize activations while the model is running. 
    '''

    def hook(module , inputs):
        tensor = inputs[0]
        qmax = (1 << (bits-1)) - 1 
        qmin = -(1 << (bits-1))
        s = torch.clamp(torch.amax(torch.abs(tensor) , dim=tuple(range(1 , tensor.dim())) , keepdim=True) / qmax , min=1e-8) # torch.amax - specify dimensions to take max along that dimension alone. 
        q = torch.clamp(torch.round(tensor/s) , qmin , qmax) 
        r = q * s 
        return r
    return hook 

def apply_quantization(model , weight_bits : dict[str , int] , act_bits : dict[str , int]):

    '''
    Given the dictionary corresponding to number of bits per layer (for weights and activations), applies the corresponding quantization to each layer. 
    '''

    for name , module in model.named_modules():
        if isinstance(module , (nn.Conv2d , nn.Linear)):
            q , s = quantize_weights(module.weight.data , weight_bits[name])
            module.weight.data = dequantize_weights(q , s)
            module.register_forward_pre_hook(quantize_act_hook(act_bits[name])) # Add the hook into every layer

    return model

def sensitivity(model , data , device , loss_fn , num=10):

    '''
    Computes Proxy for Hessian Sensitivity using power iteration method. Returns a sensitivity value for each layer which is then used in HAWQ Allocation.
    '''

    model.eval()

    sensitivities = {}
    names = []
    params = []

    # Collect the layer names to use for storing corresponding sensitivities
    for name , module in model.named_modules():
        if isinstance(module , (nn.Conv2d , nn.Linear)):
            names.append(name)
            params.append(module.weight)


    count = 0 
    for idx , (images , labels) in enumerate(data):
        count += 1 
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits , labels)

        # Create a graph to compute the proxy - derivative of gradients for hessian 
        grads = torch.autograd.grad(loss , params , create_graph=True)

        # Initialize a random normalized vector
        v = [torch.randn_like(weights) for weights in params]
        for weights in v:
            weights.div_(torch.norm(weights) + 1e-6)

        eigenvalues = []

        # Power iteration method
        for _i in range(num):
            curr_grad = torch.stack([torch.sum(grad * weights) for grad, weights in zip(grads, v)]).sum()
            for grad , weights in zip(grads , v):
                curr_grad += torch.sum(grad * weights)

            hvp = torch.autograd.grad(curr_grad , params , retain_graph=True) # type: ignore

            eigenvalues = []

            for j in range(len(v)):
                norm = torch.norm(hvp[j])
                v[j] = hvp[j] / (norm + 1e-6)
                eigenvalues.append(torch.sum(v[j] * hvp[j]).item())
        
        for i in range(0 , len(names)):
            sensitivities[names[i]] = abs(eigenvalues[i])


        if count == 1:
            break

    return sensitivities

def uniform_alloc(model , unif_bits : int):

    '''
    Allocates each layer a precision of 'unif_bits'.  
    '''

    bits : dict[str , int] = {}
    for name , module in model.named_modules():
        if isinstance(module , (nn.Conv2d , nn.Linear)):
            bits[name] = unif_bits 

    print("Uniform Bits:")
    print(bits)
    
    return bits

def mixed_uniform_alloc(model , end_bits : int , int_bits : int):

    '''
    For the first and end layers, allocates a precision of 'end_bits'.
    Otherwise, allocates a precision of 'int_bits'. 
    '''

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

def hawq_alloc(model , data , device , loss_fn , end_bits , int_bits , mid_bits , ratio):
    
    ''' 
    HAWQ based allocation - The layers are split into into three blocks (based on descending order of their sensitivity).
    - The top 'ratio'% of the layers get 'end_bits' of precision.
    - The bottom 'ratio'% of the layers get 'int_bits' of precision.
    - Remaining layers get 'mid_bits' of precision
    '''

    weight_bits = {}
    sensitivities = sensitivity(model, data, device, loss_fn, num=10)
    layers = sorted(sensitivities.keys(), key=lambda k: sensitivities[k], reverse=True)
    siz = len(layers)
    tot = 0 

    for idx , name in enumerate(layers):
        if idx < siz*ratio:
            weight_bits[name] = end_bits 
            tot += end_bits
        elif idx >= siz - siz*ratio:
            weight_bits[name] = int_bits
            tot += int_bits
        else:
            weight_bits[name] = mid_bits
            tot += mid_bits
    
    print("HAWQ Allocation bits:")
    print(weight_bits)
    print(f"Avg bits:{(tot / siz):.4f}")

    return weight_bits 

def model_size(model , bits : dict[str , int]):

    '''
    Computes the quantized model size for a given allocation. Includes scaling factor for each layer.
    '''

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

    '''
    Computes the memory required to store activations during the inference by doing a dummy pass.  
    '''

    dummy_input = torch.zeros(1, 3, 32, 32).to(device)
    
    act_original_count = 0
    act_comp_count = 0
    
    shapes = {}

    # Hook function to get the number of activations in each layer
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