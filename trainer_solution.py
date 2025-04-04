import torch
from torch import Tensor
import torch.nn.functional as F

from tqdm import tqdm
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
import time

########################################################################################
########################################################################################

@torch.no_grad()
def eval_model_by_order(model, loader, device, order):
    """Évalue le modèle sur un type d'opération spécifique (2=binaire, 3=ternaire)"""
    model.eval()
    acc = 0.0
    loss = 0.0
    n = 0
    eq_pos = 3 if order == 2 else 5
    
    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch
        indices = (eq_positions == eq_pos).nonzero(as_tuple=True)[0]
        
        if indices.numel() == 0:
            continue
            
        batch_x = batch_x[indices].to(device)
        batch_y = batch_y[indices].to(device)
        eq_positions = eq_positions[indices].to(device)
        mask = mask[indices].to(device)
        
        logits, *_ = model(batch_x)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        
        n += batch_x.size(0)
        loss += batch_loss.item() * batch_x.size(0)
        acc += batch_acc.item() * batch_x.size(0)
    
    return {"loss": loss/n if n > 0 else 0.0, "accuracy": acc/n if n > 0 else 0.0}

def train_with_separate_metrics(
    model, train_loader, train_loader_for_eval, test_loader, optimizer, scheduler, device, 
    exp_name: str, checkpoint_path: str,
    n_steps: int, eval_first: int = 0, eval_period: int = 1, print_step: int = 1, 
    save_model_step: int = 1, save_statistic_step: int = 1, verbose: bool = True
):
    """Ceci est une Nouvelle version avec suivi séparé des métriques binaires/ternaires
    pour mieux répondre à la question 4.2"""
    
    # Initialisation des métriques étendues
    all_metrics = defaultdict(lambda: [])
    all_metrics.update({
        'steps': [],
        'global_train_loss': [], 'global_train_acc': [],
        'global_val_loss': [], 'global_val_acc': [],
        'binary_train_loss': [], 'binary_train_acc': [],
        'binary_val_loss': [], 'binary_val_acc': [],
        'ternary_train_loss': [], 'ternary_train_acc': [],
        'ternary_val_loss': [], 'ternary_val_acc': []
    })

    # Fonction d'évaluation spécifique
    def eval_by_order(loader, order):
        model.eval()
        eq_pos = 3 if order == 2 else 5
        total_loss, total_acc, count = 0.0, 0.0, 0
        
        for batch_x, batch_y, eq_positions, mask in loader:
            indices = (eq_positions == eq_pos).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
                
            x = batch_x[indices].to(device)
            y = batch_y[indices].to(device)
            pos = eq_positions[indices].to(device)
            m = mask[indices].to(device)
            
            logits, *_ = model(x)
            loss, acc = get_loss_and_accuracy(logits, y, pos, m)
            
            total_loss += loss.item() * x.size(0)
            total_acc += acc.item() * x.size(0)
            count += x.size(0)
        
        return {
            'loss': total_loss / count if count > 0 else 0.0,
            'accuracy': total_acc / count if count > 0 else 0.0
        }

    # Configuration initiale
    os.makedirs(checkpoint_path, exist_ok=True)
    total_epochs = (n_steps + len(train_loader) - 1) // len(train_loader)
    cur_step = 1

    # Boucle d'entraînement principale
    for epoch in tqdm(range(1, total_epochs + 1), desc="Training"):
        for batch in train_loader:
            # Forward/backward 
            batch_x, batch_y, eq_positions, mask = batch
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits, *_ = model(batch_x)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Évaluation périodique
            if cur_step % eval_period == 0 or cur_step <= eval_first:
                # Métriques globales
                global_train = eval_model(model, train_loader_for_eval, device)
                global_val = eval_model(model, test_loader, device)
                
                # Métriques séparées
                binary_train = eval_by_order(train_loader_for_eval, 2)
                ternary_train = eval_by_order(train_loader_for_eval, 3)
                binary_val = eval_by_order(test_loader, 2)
                ternary_val = eval_by_order(test_loader, 3)

                # Mise à jour des métriques
                metrics_to_update = {
                    'steps': cur_step,
                    'global_train_loss': global_train['loss'],
                    'global_train_acc': global_train['accuracy'],
                    'global_val_loss': global_val['loss'],
                    'global_val_acc': global_val['accuracy'],
                    'binary_train_loss': binary_train['loss'],
                    'binary_train_acc': binary_train['accuracy'],
                    'binary_val_loss': binary_val['loss'],
                    'binary_val_acc': binary_val['accuracy'],
                    'ternary_train_loss': ternary_train['loss'],
                    'ternary_train_acc': ternary_train['accuracy'],
                    'ternary_val_loss': ternary_val['loss'],
                    'ternary_val_acc': ternary_val['accuracy']
                }
                
                for key, value in metrics_to_update.items():
                    all_metrics[key].append(value)

            # Logging et sauvegardes
            if verbose and (cur_step % print_step == 0):
                print(f"Step {cur_step}")
                print(f"Binary Train Loss: {binary_train['loss']:.4f} Acc: {binary_train['accuracy']:.2%}")
                print(f"Ternary Train Loss: {ternary_train['loss']:.4f} Acc: {ternary_train['accuracy']:.2%}")

            if cur_step % save_model_step == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': all_metrics
                }, f"{checkpoint_path}/{exp_name}_step_{cur_step}.pt")

            cur_step += 1
            if cur_step > n_steps: 
                break

    return all_metrics


def get_loss_and_accuracy(logits, targets, eq_positions, mask, reduction='mean'):
    """
    Computes the mean negative log-likelihood loss and the accuracy on the right-hand side (RHS)
    of each equation in the mini-batch.

    The equation can be : 
        - "[BOS] [a] [+] [b] [=] [r] [EOS] [PAD] [PAD]", in that case target is "[a] [+] [b] [=] [r] [EOS] [PAD] [PAD]"
        - "[BOS] [a] [+] [b] [+] [c] [=] [r] [EOS]", in that case target is "[a] [+] [b] [+] [c] [=] [r] [EOS]"

    Let :
        - B : batch size
        - S : sequence length
        - V : vocabulary size
    
    Parameters
    ----------
    logits : torch.FloatTensor of shape (B, S, V)
        A tensor containing the logits of the next token for all positions in each sequence of the mini-batch.
    targets : torch.LongTensor of shape (B, S)
        A tensor containing the target next tokens for all positions in each sequence of the mini-batch.
    eq_positions : torch.LongTensor of shape (B,)
        The position of the '=' token in each sequence (each sample has exactly one '=').
    mask : torch.LongTensor of shape (B, S)
        A mask indicating valid tokens (1 if valid, 0 for PAD tokens).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        - 'none': no reduction will be applied
        - 'mean': average the output of the batch dimension. 
        - 'sum': sum the output of the batch dimension.
        
    Returns
    -------
    loss : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The negative log-likelihood loss computed over the valid (non-PAD) RHS tokens.
    accuracy : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The accuracy over the batch where a sequence is counted as correct only if 
        all valid RHS tokens are predicted correctly.
    """
    # ==========================
    # TODO: Write your code here
    B, S, V = logits.shape
    
    # Create RHS mask (tokens after '=' that aren't padding)
    positions = torch.arange(S, device=logits.device).expand(B, S)

    eq_positions = eq_positions.to(logits.device)  
    mask = mask.to(logits.device)

    rhs_mask = (positions > eq_positions.unsqueeze(1)) & (mask.bool())
    
    # Compute log probabilities and gather target log probs
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
    
    # Compute loss (negative log likelihood)
    loss = -target_log_probs * rhs_mask.float()
    loss_per_sample = loss.sum(dim=1) / rhs_mask.sum(dim=1).clamp(min=1)  # (B,)
    
    # Compute accuracy (all RHS tokens must be correct)
    predictions = logits.argmax(dim=-1)
    all_correct = ((predictions == targets) | ~rhs_mask).all(dim=1)  # (B,)
    accuracy_per_sample = all_correct.float()
    
    # Apply reduction
    if reduction == 'mean':
        return loss_per_sample.mean(), accuracy_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum(), accuracy_per_sample.sum()
    return loss_per_sample, accuracy_per_sample  # 'none'
    # ==========================

    # raise NotImplementedError

    return loss, accuracy

########################################################################################
########################################################################################
  
@torch.no_grad()
def eval_model(model, loader, device) :
    model.eval()
    acc = 0.0 
    loss = 0.0 
    n = 0
    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, *_ = model(batch_x) # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        n += batch_x.shape[0]
        loss += batch_loss.item() * batch_x.shape[0]
        """acc += batch_acc * batch_x.shape[0]"""
        acc += batch_acc.item() * batch_x.shape[0]  # Conversion avec .item()


    ##########
    # You can add more metrics in the dictionary (e.g., l2 norm of the parameters, etc.) 
    ##########

    return {"loss" : loss / n, "accuracy": acc / n}
    
########################################################################################
########################################################################################


def train(
    model, train_loader, train_loader_for_eval, test_loader, optimizer, scheduler, device, 
    exp_name:str, checkpoint_path:str,
    n_steps:int, eval_first:int=0, eval_period:int=1, print_step:int=1, save_model_step:int=1,  save_statistic_step:int=1,  
    verbose=True,
    ):
    """
    model (nn.Module) : The model to train
    train_loader (DataLoader) : Training data loader
    train_loader_for_eval (DataLoader) : Training data loader (for evaluation)
    test_loader (DataLoader) : Test/Val data loader
    optimizer (Optimizer) : Optimizer
    device (str) : Device (cpu, cuda, cuda:0, etc)
    exp_name (str) : experiment name
    checkpoint_path (str) : Path to save the model checkpoints ("/path/to/experiment")
    n_steps (int) : Number of training steps
    eval_first (int) : Number of consecutive evaluation step at the beginning of training
    eval_period (int) : Evaluation frequency
    print_step (int) : Print frequency
    save_model_step (int) : Step interval to save model checkpoints
    save_statistic_step (int) : Step interval to save statistics (train/test loss, accuracy, etc.)
    verbose (bool) : Verbosity of the training
    """

    ##############
    # Checkpoint path
    os.makedirs(checkpoint_path, exist_ok=True)

    ##############
    # Number of training epochs
    total_epochs = (n_steps + len(train_loader) - 1) // len(train_loader)
    n_steps = total_epochs * len(train_loader)
    
    if verbose :
        print(f"Number of training epochs & steps: {total_epochs} {n_steps}")

    ##############

    all_metrics = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["train"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["test"] = defaultdict(lambda: []) # {metric : [value at step 1, ... ]}
    all_metrics["steps_epoch"] = {}

    ##############

    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() :
        all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device) 
    for k, v in test_statistics.items() :
        all_metrics["test"][k].append(v)

    all_metrics["all_steps"].append(0)
    all_metrics["steps_epoch"][0] = 0


    ######################
    # Save model
    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{0}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
  
    
    ##############

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    if verbose :
        to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
        to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
        to_print += f" | lr = {current_lr}"
        print(to_print)

    ##############

    cur_step = 1 
    tol_step = 0

    for epoch in tqdm(range(1, total_epochs+1), desc="Training", total=total_epochs):

        # start_time = time.time()
        
        for i, batch in enumerate(train_loader) :
            batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x) # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ==========================
            # TODO: Write your code here
            # ==========================
            # scheduler.step()
            # current_lr = scheduler.optimizer.param_groups[0]["lr"]
            # ==========================
            # ==========================
              
            if cur_step in [1, n_steps] or cur_step % eval_period == 0 or cur_step <= eval_first:
                train_statistics = eval_model(model, train_loader_for_eval, device)
                for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

                test_statistics = eval_model(model, test_loader, device)
                for k, v in test_statistics.items() : all_metrics["test"][k].append(v)

                all_metrics["all_steps"].append(cur_step)
                all_metrics["steps_epoch"][cur_step] = epoch

            
            if  verbose and (cur_step in [1, n_steps] or cur_step%print_step==0) :
                to_print = "\n" + " | ".join(f"Train {k} : {v:.6f}" for k, v in train_statistics.items())
                to_print += " | " + " | ".join(f"Test {k} : {v:.6f}" for k, v in test_statistics.items())
                to_print += f" | lr = {current_lr}"
                print(to_print)

            if cur_step in [1, n_steps] or cur_step%save_model_step==0 or cur_step <= eval_first : 
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")
                

            if cur_step in [1, n_steps] or cur_step%save_statistic_step==0:
                #to_save = {k:v for k, v in all_metrics.items()}
                to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
                torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

            cur_step += 1

        # ==========================
        # TODO: Write your code here
        # ==========================
        ###
        # scheduler.step() 
        # current_lr = scheduler.optimizer.param_groups[0]["lr"]
        # ==========================
        # ==========================

        ##############
        # You can implement early stopping here.
        # That is, if the model does not improve for a certain number of steps, you can stop the training.
        ##############

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time for one step : {elapsed_time} seconds")

    state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
    }
    # torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']}_loss={test_statistics['loss']}.pth")

    torch.save(state, f"{checkpoint_path}/{exp_name}_state_{cur_step}_acc={test_statistics['accuracy']:.4f}_loss={test_statistics['loss']:.4f}.pth")    
    
    
    train_statistics = eval_model(model, train_loader_for_eval, device)
    for k, v in train_statistics.items() : all_metrics["train"][k].append(v)

    test_statistics = eval_model(model, test_loader, device)
    for k, v in test_statistics.items() : all_metrics["test"][k].append(v.item() if isinstance(v, torch.Tensor) else v)

    all_metrics["all_steps"].append(cur_step)
    all_metrics["steps_epoch"][cur_step] = epoch

    to_save = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in all_metrics.items()} # to avoid issues with lambda
    torch.save(to_save, f"{checkpoint_path}/{exp_name}.pth")

    return all_metrics

"""# Dans la boucle d'évaluation, remplacer :
for k, v in test_statistics.items():
    all_metrics["test"][k].append(v)

# Par :
for k, v in test_statistics.items():
    all_metrics["test"][k].append(v.item() if isinstance(v, torch.Tensor) else v)"""


########################################################################################
# Nouvelle fonction exportée
def train_with_separate_metrics(*args, **kwargs):
    return train(*args, **kwargs)  # Utilise la fonction train() modifiée
########################################################################################