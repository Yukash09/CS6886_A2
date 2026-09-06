import torch.nn as nn 
import torch
import torch.optim as optim
from model import get_model
from dataloader import get_train_data
import wandb 


def train():
    wandb.init(project="CS6886_Assignment2", group="Fine-Tuning")
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = get_model().to(device)

    train_data , val_data = get_train_data(batch_size=cfg.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters() , lr=cfg.lr , momentum=0.9 , weight_decay=0.0005)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch)
    optimizer = optim.Adam(model.parameters() , lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' , factor=0.5 , patience=3)

    best_acc = 0.0 

    for epoch in range(cfg.epoch):
        print(f"Epoch: {epoch}")

        model.train()
        epoch_loss = 0.0 
        correct = 0 
        total = 0 

        for idx , (images , labels) in enumerate(train_data):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _ , pred = torch.max(logits , 1)
            loss = loss_fn(logits , labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if (idx + 1) % 20 == 0:
                print(f"Training Batch {idx}/{len(train_data)} , Loss: {loss.item():.4f}")
        
        train_accuracy = correct/total 
        train_loss = epoch_loss/len(train_data)

        print(f"Train Accuracy:{train_accuracy:.4f} , Train_loss:{train_loss:.4f}")

        model.eval()
        epoch_val_loss = 0.0 
        correct_val = 0 
        total_val = 0 

        with torch.no_grad():
            for idx , (images , labels) in enumerate(val_data):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                _ , pred = torch.max(logits , 1)
                val_loss = loss_fn(logits , labels)

                epoch_val_loss += val_loss.item()
                correct_val += (pred == labels).sum().item()
                total_val += labels.size(0)

                if (idx + 1) % 20 == 0:
                    print(f"Validation Batch {idx}/{len(val_data)} , Loss: {val_loss.item():.4f}")
        
        val_accuracy = correct_val/total_val 
        valid_loss = epoch_val_loss/len(val_data)

        print(f"Validation Accuracy:{val_accuracy:.4f} , Validation_loss:{valid_loss:.4f}")

        # scheduler.step()
        scheduler.step(val_accuracy)

        wandb.log({
            "epoch":epoch , 
            "train_loss": train_loss,
            "val_loss": valid_loss ,
            "train_acc": train_accuracy,
            "val_acc": val_accuracy,
            "lr": optimizer.param_groups[0]['lr']
        })

        if val_accuracy > best_acc:
            best_acc = val_accuracy 
            checkpoint = {
                "state_dict": model.state_dict() , 
                "epoch": epoch ,
                "best_metric": best_acc
            }
            torch.save(checkpoint , "./checkpoints/best_model.pth")


if __name__ == "__main__":
    config = {
        'method': 'grid',
        'name': "Fine-Tuning-Sweep",
        'metric':{
            'goal':'maximize',
            'name':'val_acc'
        },
        'parameters':{
            'lr':{'values': [0.001, 0.0005, 0.0001]},
            'batch_size':{'values':[64 , 128]},
            'epoch':{'values':[15 , 25]}
        }
    }

    sweep_id = wandb.sweep(config, project="CS6886_Assignment2")
    wandb.agent(sweep_id, function=train , count=12) # Don't let bayes run forever?
