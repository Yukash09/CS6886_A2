import torch 
import wandb
from evaluate import evaluate_quantize , test , model_loader
from dataloader import get_test_data


def make_configs():
    configs = []

    for bits in [2 , 3 , 4 , 5 , 6 , 8]:
        configs.append({"policy": "uniform" , "unif_bits": bits , "end_bits": 0 , "int_bits": 0})

    for end in [4 , 6 , 8]:
        for mid in [2 , 3 , 4 , 5 , 6 , 8]:
            if mid < end:
                configs.append({"policy": "mixed" , "unif_bits": 0 , "end_bits": end , "int_bits": mid})

    return configs


def sweep_fn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "./checkpoints/best_model.pth"
    test_data = get_test_data()

    model = model_loader(path , device)
    baseline_acc = test(model , test_data , device)

    configs = make_configs()

    for idx , cfg in enumerate(configs):
        print(f"\n=== Run {idx+1}/{len(configs)} ===")

        run = wandb.init(
            project="CS6886_Assignment2",
            config=cfg,
            reinit=True
        )

        config = wandb.config
        accuracy , ratio = evaluate_quantize(model , test_data , device , config , mode="PTQ")

        wandb.log({
            "baseline_acc": baseline_acc,
            "quantized_acc": accuracy,
            "acc_drop": baseline_acc - accuracy,
            "compression_ratio": ratio,
        })

        wandb.finish()

    print(f"\nDone. {len(configs)} runs logged.")


if __name__ == "__main__":
    sweep_fn()
