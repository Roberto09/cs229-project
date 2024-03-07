from peft import LoraConfig
import transformers
from trl import SFTTrainer

def get_lora_config(r=8, bias="none"):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=[
            'fc1', # re-train prunned layers for now
            'fc2',
        ],
        bias=bias,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    return lora_config

def get_training_arguments(output_dir):
    # Orig params:
    # batch_size = 64
    # micro_batch_size = 4
    batch_size = 60
    micro_batch_size = 6
    gradient_accumulation_steps = batch_size // micro_batch_size

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=2,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        logging_first_step=True,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=20,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=None,
        group_by_length=False,
        # metric_for_best_model="{}_loss".format(args.data_path),
    )
    return training_arguments

def lora_train(model, tokenizer, dataset):
    # Setup model for training
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    # Setup tokenizer for trainign
    tokenizer.pad_token = tokenizer.eos_token

    # Get default training arguments
    lora_config = get_lora_config()
    training_arguments = get_training_arguments()

    train_data, eval_data = dataset["train"], dataset["test"]
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        dataset_text_field="text",
        max_seq_length=1024, # tweak this
        # TODO: think harder about the datacollator
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
    )
    trainer.train()
    trainer.model_wrapped.print_trainable_parameters() 
    return trainer