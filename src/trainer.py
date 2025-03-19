import math, time, datetime
import torch
import gc
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

class train_callback(pl.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_epochs = config.trainer.max_epochs
        ctr = config.training
        self.warmup_steps = ctr.warmup_steps
        self.lr_init = ctr.lr_init
        self.lr_final = ctr.lr_final
        self.weight_decay = ctr.weight_decay
        self.weight_decay_final = ctr.weight_decay_final
        self.save_steps = ctr.save_steps
        self.epoch_save = ctr.epoch_save

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        current_step = trainer.global_step
        
        # LR schedule
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            # Linear warmup
            lr = self.lr_init * (0.01 + 0.99 * current_step / self.warmup_steps)
        else:
            if self.lr_final == self.lr_init:
                lr = self.lr_init
            else:
                # Calculate progress through training after warmup
                total_steps = self.max_epochs * trainer.num_training_batches
                if total_steps > self.warmup_steps:
                    progress = min(1.0, (current_step - self.warmup_steps) / 
                                 (total_steps - self.warmup_steps))
                else:
                    progress = 0
                
                if self.config.training.lr_schedule == 'cosine':
                    lr_final_factor = self.lr_final / self.lr_init
                    lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
                    lr = self.lr_init * lr_mult
                elif self.lr_final == 0 or self.lr_init == 0:  # linear decay
                    lr = self.lr_init + (self.lr_final - self.lr_init) * progress
                else:  # exp decay
                    lr = self.lr_init * math.exp(math.log(self.lr_final / self.lr_init) * progress)

        if self.weight_decay_final > 0:
            total_steps = self.max_epochs * trainer.num_training_batches
            if total_steps > self.warmup_steps:
                progress = min(1.0, (current_step - self.warmup_steps) / 
                             (total_steps - self.warmup_steps))
            else:
                progress = 0
            wd_now = self.weight_decay * math.exp(math.log(self.weight_decay_final / self.weight_decay) * progress)
        else:
            wd_now = self.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if self.config.training.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        if current_step == 0 and trainer.is_global_zero:
            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            trainer.my_log = open(self.config.proj_dir + "/train_log.txt", "a")
            trainer.my_log.write(f"NEW RUN {self.config.my_timestamp}\n{vars(self.config)}\n")
            
            try:
                print(f"\n{trainer.strategy.config}\n")
                trainer.my_log.write(f"{trainer.strategy.config}\n")
            except:
                pass
                
            trainer.my_log.flush()
            
            if self.config.wandb:
                print("Login to wandb...")
                import wandb
                from omegaconf import OmegaConf
                wandb.init(
                    project=self.config.wandb,
                    name=self.config.run_name + " " + self.config.my_timestamp,
                    config=OmegaConf.to_container(self.config, resolve=True),
                    save_code=False,
                )
                trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        token_per_step = self.config.model.ctx_len * self.config.real_bsz
        
        if trainer.is_global_zero:
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
                
            trainer.my_time_ns = t_now
            trainer.my_loss = trainer.my_loss_all.float().mean().item()
                
            # Update moving average of loss
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            
            # Standard logging
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

            # Log to wandb if configured
            if self.config.wandb:
                lll = {
                    "loss": trainer.my_loss, 
                    "lr": trainer.my_lr, 
                    "wd": trainer.my_wd, 
                    "Gtokens": trainer.global_step * token_per_step / 1e9
                }
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=trainer.global_step)
                
        # Save checkpoint at specific step if configured
        is_save_step = self.save_steps > 0 and trainer.global_step % self.save_steps == 0
        if trainer.is_global_zero and is_save_step:
            to_save_dict = pl_module.state_dict()
            torch.save(
                to_save_dict,
                f"{self.config.proj_dir}/rwkv-final.pth",
            )

    def on_train_epoch_start(self, trainer, pl_module):
        dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = trainer.current_epoch
        dataset.world_size = trainer.world_size

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            save_epoch = (self.epoch_save > 0 and trainer.current_epoch % self.epoch_save == 0) 
            is_final_epoch = (trainer.current_epoch == self.max_epochs - 1)
            
            if save_epoch or is_final_epoch:
                to_save_dict = pl_module.state_dict()
                    
                try:
                    torch.save(
                        to_save_dict,
                        f"{self.config.proj_dir}/rwkv-{trainer.current_epoch}.pth",
                    )
                except Exception as e:
                    print('Error saving checkpoint:\n\n', e, '\n\n')

        # Log epoch summary
        if trainer.is_global_zero:
            trainer.my_log.write(f"Epoch {trainer.current_epoch}: loss={trainer.my_epoch_loss:.6f} " +
                               f"ppl={math.exp(trainer.my_epoch_loss):.4f} lr={trainer.my_lr:.8f} " +
                               f"time={datetime.datetime.now()}\n")
            trainer.my_log.flush()

            # Reset epoch stats
            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

        torch.cuda.empty_cache()
        gc.collect()


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()
    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)
    print("Model initial weights saved.")