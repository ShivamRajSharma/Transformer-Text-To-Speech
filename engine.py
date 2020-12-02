import config

import torch 
import torch.nn as nn 
from tqdm import tqdm 

def loss_fn(
    target_mel_spect, 
    target_end_logits, 
    pred_mel_spect_post, 
    pred_mel_spect, 
    pred_end_logits
    ):
    mel_loss = nn.L1Loss()(pred_mel_spect, target_mel_spect) + nn.L1Loss()(pred_mel_spect_post, target_mel_spect)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.bce_weights))(pred_end_logits.squeeze(-1), target_end_logits)
    return mel_loss + bce_loss


def train_fn(model, dataloader, optimizer, scheduler, device):
    running_loss = 0
    model.train()
    for num, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        for p in model.parameters():
            p.grad=None

        end_logits = data['end_logits'].to(device)
        mel_spect = data['mel_spect'].to(device)
        text_idx = data['text_idx'].to(device)
        text = data['original_text']
        mel_mask = data['mel_mask'].to(device)
        mel_spect_post_pred, mel_spect_pred, end_logits_pred = model(text_idx, mel_spect[:, :-1], mel_mask[:, :-1])
        loss = loss_fn(
            mel_spect[:, 1:], 
            end_logits[:, :-1],
            mel_spect_post_pred,
            mel_spect_pred,
            end_logits_pred
            )

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss


def eval_fn(model, dataloder, device):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for num, data in tqdm(enumerate(dataloder), total=len(dataloder)):
            end_logits = data['end_logits'].to(device)
            mel_spect = data['mel_spect'].to(device)
            text_idx = data['text_idx'].to(device)
            text = data['original_text']
            mel_mask = data['mel_mask'].to(device)
            mel_spect_post_pred, mel_spect_pred, end_logits_pred = model(text_idx, mel_spect[:, :-1], mel_mask[:, :-1])
            loss = loss_fn(
                mel_spect[:, 1:], 
                end_logits[:, :-1],
                mel_spect_post_pred,
                mel_spect_pred,
                end_logits_pred
                )
                
            running_loss += loss.item()
    epoch_loss = running_loss/len(dataloder)
    return epoch_loss
        