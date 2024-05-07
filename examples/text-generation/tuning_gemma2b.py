import os


os.environ["PJRT_DEVICE"] = "TPU"

from peft import LoraConfig, TaskType


lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
import torch_xla.core.xla_model as xm


text = "Quote: Imagination is more"
device = xm.xla_device()
model.to(device)
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
