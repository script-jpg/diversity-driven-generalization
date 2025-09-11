# train_one_model.py
import os
import sys
import torch
import gc
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import DPOTrainer, DPOConfig
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


class BestLossCallback(TrainerCallback):
    """Save model checkpoint when a new lowest loss is achieved."""

    def __init__(self, output_dir):
        self.best_loss = float("inf")
        self.output_dir = output_dir

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is not None and loss < self.best_loss:
            self.best_loss = loss
            save_path = os.path.join(self.output_dir, "best_loss_checkpoint")
            os.makedirs(save_path, exist_ok=True)
            kwargs["model"].save_pretrained(save_path)
            print(f"💾 Saved new best checkpoint at step {state.global_step} "
                  f"(loss={loss:.4f})", flush=True)


def main():
    if len(sys.argv) < 3:
        raise ValueError("Usage: python train_one_model.py <MODEL_ID> <DATASET_PATH>")

    MODEL_ID = sys.argv[1]
    DATASET_PATH = sys.argv[2]

    # ⚙️ Dataset
    print(f"📂 Loading dataset from {DATASET_PATH}", flush=True)
    train_ds = load_from_disk(DATASET_PATH)

    # 🧩 Quantization config (QLoRA)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 🧠 Load model
    print(f"🧠 Loading model {MODEL_ID}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID.replace("_", "/"),
        quantization_config=quantization_config,
        device_map="auto",
        use_cache=False
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # 🎚️ LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 🏋️ Training config (frequent logs)
    training_args = DPOConfig(
        output_dir=f"./trained_models/{MODEL_ID}_DPO",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=100,
        logging_steps=1,
        logging_first_step=True,
        disable_tqdm=False,
        bf16=True,
        report_to=[],
    )

    # 🚀 Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        callbacks=[BestLossCallback(training_args.output_dir)],
    )

    print(f"🚀 Starting training for {MODEL_ID}", flush=True)
    trainer.train()

    # Save final adapters
    print(f"💾 Saving final adapters for {MODEL_ID}", flush=True)
    model.save_pretrained(f"./trained_models/{MODEL_ID}_DPO/lora_adapters")

    # 🧹 Cleanup
    del trainer, model, train_ds
    torch.cuda.empty_cache()
    gc.collect()

    print(f"✅ Completed training for {MODEL_ID}", flush=True)


if __name__ == "__main__":
    main()
