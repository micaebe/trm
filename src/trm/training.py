import torch
import torch.nn as nn
import wandb

def train(model, data_loader, eval_loader, optimizer, ema, cfg):
    wandb.init(project="trm", config=vars(cfg))
    total_steps = 0
    model.train()

    pred_loss_fn = nn.CrossEntropyLoss()
    halting_loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(cfg.num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{cfg.num_epochs} ---")
        for batch_idx, (x_input, y_true) in enumerate(data_loader):
            y_latent = torch.zeros(x_input.shape[0], cfg.seq_len, cfg.dim, device=cfg.device)
            z_latent = torch.zeros(x_input.shape[0], cfg.seq_len, cfg.dim, device=cfg.device)

            avg_total_loss = 0.0
            avg_pred_loss = 0.0
            avg_halt_loss = 0.0
            avg_q = 0.0
            avg_accuracy = 0.0
            
            for step in range(cfg.n_supervision):
                total_steps += 1
                
                x_embed = model.input_embedding(x_input)
                y_final, z_final, y_logits, q = model(x_embed, y_latent, z_latent)
                
                with torch.no_grad():
                    correct_preds = torch.argmax(y_logits, dim=-1) == y_true
                    target_halt = correct_preds.all(dim=1).float().unsqueeze(1)
                    
                pred_loss = pred_loss_fn(y_logits.view(-1, cfg.vocab_size), y_true.view(-1))
                halt_loss = halting_loss_fn(q, target_halt)
                total_loss = pred_loss + halt_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                ema.update(model)
                
                y_latent = y_final.detach()
                z_latent = z_final.detach()
                
                if total_steps % cfg.print_interval == 0:
                    postfix_str = (
                        f"Sup-Step: {step+1}/{cfg.n_supervision}, "
                        f"Loss: {total_loss.item():.4f}, "
                        f"Pred Loss: {pred_loss.item():.4f}, "
                        f"Halt Loss: {halt_loss.item():.4f}, "
                        f"Avg Q: {q.mean().item():.2f}"
                    )
                    
                    print(f"Batch {batch_idx+1}/{len(data_loader)} | {postfix_str}", end='\r', flush=True)
                avg_total_loss += total_loss.item()
                avg_pred_loss += pred_loss.item()
                avg_halt_loss += halt_loss.item()
                avg_q += q.mean().item()
                avg_accuracy += correct_preds.float().mean().item()

                if q.mean() > 0.95:
                    print(f"\n--- Early stopping supervision at step {step+1} for batch {batch_idx+1} (Avg Q: {q.mean().item():.2f}) ---")
                    break
            
            if (batch_idx + 1) % cfg.eval_every == 0:
                print("\n--- Evaluating during training ---")
                model.eval()
                if cfg.use_ema:
                    original_params = ema.apply_shadow(model)
                evaluate(model, eval_loader, cfg)
                if cfg.use_ema:
                    ema.restore_original(model, original_params)
                model.train()

            avg_total_loss /= (step + 1)
            avg_pred_loss /= (step + 1)
            avg_halt_loss /= (step + 1)
            avg_q /= (step + 1)
            avg_accuracy /= (step + 1)
            wandb.log({
                "Epoch": epoch + 1,
                "Total Loss": avg_total_loss,
                "Prediction Loss": avg_pred_loss,
                "Halting Loss": avg_halt_loss,
                "Avg Q": avg_q,
                "Supervision Steps": step + 1,
                "Accuracy": avg_accuracy
            }, step=total_steps)

def evaluate(model, data_loader, cfg):
    model.eval()
    total_accuracy = 0.0
        
    with torch.no_grad():
        for x_test, y_test_solution in data_loader:
            x_test_embed = model.input_embedding(x_test)
            
            y_eval_latent = torch.zeros(x_test.shape[0], cfg.seq_len, cfg.dim, device=cfg.device)
            z_eval_latent = torch.zeros(x_test.shape[0], cfg.seq_len, cfg.dim, device=cfg.device)
            
            for _ in range(cfg.n_supervision):
                y_eval_latent, z_eval_latent, y_logits_test, _ = model(x_test_embed, y_eval_latent, z_eval_latent)

            prediction = torch.argmax(y_logits_test, dim=-1)
            accuracy = (prediction == y_test_solution).float().mean().item()
            total_accuracy += accuracy

    avg_accuracy = total_accuracy / len(data_loader)
    wandb.log({"Eval Accuracy": avg_accuracy})
    return avg_accuracy