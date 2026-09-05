import torch 
import wandb
from evaluate import evaluate_quantize , test , model_loader
from dataloader import get_test_data
import torch.nn as nn

def make_configs():
    configs = []

    for bits in [2 , 3 , 4 , 5 , 6 , 8]:
        for act_bits in [4 , 6 , 8]:
            configs.append({"policy": "uniform" , "unif_bits": bits , "act_unif_bits": act_bits , "end_bits": 0 , "int_bits": 0 , "act_end_bits": 0 , "act_int_bits": 0, "mid_bits": 0, "ratio": 0.0})

    for end in [4 , 6 , 8]:
        for mid in [2 , 3 , 4 , 5 , 6 , 8]:
            if mid < end:
                for act_mid in [4 , 6 , 8]:
                    configs.append({"policy": "mixed" , "unif_bits": 0 , "act_unif_bits": 0 , "end_bits": end , "int_bits": mid , "act_end_bits": 8 , "act_int_bits": act_mid, "mid_bits": 0, "ratio": 0.0})

    for end in [6, 8]:
        for mid in [4, 5, 6]:
            for int_b in [2, 3, 4]:
                if int_b < mid < end:
                    for ratio in [0.1, 0.2, 0.3]:
                        configs.append({
                            "policy": "HAWQ", 
                            "unif_bits": 0, "act_unif_bits": 8, 
                            "end_bits": end, "mid_bits": mid, "int_bits": int_b,
                            "act_end_bits": 0, "act_int_bits": 0,
                            "ratio": ratio
                        })

    return configs


def sweep_fn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "./checkpoints/best_model.pth"
    test_data = get_test_data()

    model = model_loader(path , device)
    baseline_acc = test(model , test_data , device)
    loss_fn = nn.CrossEntropyLoss()

    configs = make_configs()

    for idx , cfg in enumerate(configs):
        print(f"\n=== Run {idx+1}/{len(configs)} ===")

        run = wandb.init(
            project="CS6886_Assignment2",
            config=cfg,
            reinit=True
        )

        accuracy , w_ratio , a_ratio = evaluate_quantize(model , test_data , device , cfg , mode="PTQ", loss_fn=loss_fn)

        wandb.log({
            "baseline_acc": baseline_acc,
            "quantized_acc": accuracy,
            "acc_drop": baseline_acc - accuracy,
            "w_compression_ratio": w_ratio,
            "a_compression_ratio": a_ratio,
        })

        wandb.finish()

    print(f"\nDone. {len(configs)} runs logged.")


if __name__ == "__main__":
    sweep_fn()
