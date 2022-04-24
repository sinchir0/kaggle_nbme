import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from dataset import TrainDataset
from helper import AverageMeter, timeSince
from model import CustomModel
from pred import get_predictions
from score import create_labels_for_scoring, get_char_probs, get_results, get_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, CFG):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(CFG.device)
        labels = labels.to(CFG.device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.8f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
        if CFG.wandb:
            wandb.log(
                {f"[fold{fold}] loss": losses.val, f"[fold{fold}] lr": scheduler.get_lr()[0],}
            )
    return losses.avg


def valid_fn(valid_loader, model, criterion, CFG):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(CFG.device)
        labels = labels.to(CFG.device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step, len(valid_loader), loss=losses, remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions


# def inference_fn(test_loader, model, CFG):
#     preds = []
#     model.eval()
#     model.to(CFG.device)
#     tk0 = tqdm(test_loader, total=len(test_loader))
#     for inputs in tk0:
#         for k, v in inputs.items():
#             inputs[k] = v.to(CFG.device)
#         with torch.no_grad():
#             y_preds = model(inputs)
#         preds.append(y_preds.sigmoid().to("cpu").numpy())
#     predictions = np.concatenate(preds)
#     return predictions


# CFGはtrain_loopの中でののみ利用する
def train_loop(folds, fold, CFG):

    CFG.__logger.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_texts = valid_folds["pn_history"].values
    valid_labels = create_labels_for_scoring(valid_folds)  # ここでlocation_for_create_labels列rが追加されるのは気になる

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, (CFG.__output_dir / "output_model" / "config.pth"))
    model.to(CFG.device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        # param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(
        model, encoder_lr=CFG.encoder_lr, decoder_lr=CFG.decoder_lr, weight_decay=CFG.weight_decay,
    )
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
            )
        elif cfg.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles,
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_score = 0.0

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, CFG=CFG)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, CFG=CFG)
        predictions = predictions.reshape((len(valid_folds), CFG.max_len))

        # scoring
        char_probs = get_char_probs(valid_texts, predictions, CFG.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        CFG.__logger.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        CFG.__logger.info(f"Epoch {epoch+1} - Score: {score:.4f}")
        if CFG.wandb:
            wandb.log(
                {
                    f"[fold{fold}] epoch": epoch + 1,
                    f"[fold{fold}] avg_train_loss": avg_loss,
                    f"[fold{fold}] avg_val_loss": avg_val_loss,
                    f"[fold{fold}] score": score,
                }
            )

        if (best_score < score) | (CFG.debug):
            best_score = score
            CFG.__logger.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "predictions": predictions},
                CFG.__output_dir / "output_model" / f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
            ),

    predictions = torch.load(
        CFG.__output_dir / "output_model" / f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
        map_location=torch.device("cpu"),
    )["predictions"]
    # valid_folds[[i for i in range(CFG.max_len)]] = predictions
    valid_folds = pd.concat([valid_folds, pd.DataFrame(predictions)], axis=1)

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
