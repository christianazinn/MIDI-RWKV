from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from fla.models import RWKV7Config
from datasets import load_dataset

# in case you want to use the fla-hub triton implementation...
# WARNING: nobody has ever trained a full model from this repo!
model = AutoModelForCausalLM.from_config(RWKV7Config())

# TODO dataset
pass

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=500,
    save_steps=500,
    warmup_steps=500,
    logging_dir="./logs",
    prediction_loss_only=True,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    # data_collator=data_collator,
    # train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

model.save_pretrained("./blahblah")
# tokenizer.save_pretrained("./fine-tuned-gpt2")