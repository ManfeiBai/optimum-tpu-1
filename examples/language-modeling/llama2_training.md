<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Training from scratch a Language Model on TPU with Llama2

## Prerequisites

You need to install few modules:

```shell
pip install datasets evaluate
```

## Instructions

Create a config file for the training run. To reduce training, we use a modified config from `meta-llama/Llama-2-7b-hf`, where `hidden_size` was reduced from 4096 to 2048.

You can now use a modified version of `run_clm.py` to train your model on the `wikitext-2-raw-v1` dataset:



   --config_name examples/language-modeling/config.json \

```bash
python examples/language-modeling/run_clm.py \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --dataset_name wikitext \
   --dataset_config_name wikitext-2-raw-v1 \
   --per_device_train_batch_size 256 \
   --per_device_eval_batch_size 8 \
   --num_train_epochs 1 \
   --do_train \
   --output_dir /tmp/output \
   --overwrite_output_dir \
   --save_strategy no \
   --logging_strategy no \
   --remove_unused_columns no \
   --optim adafactor \
   --torch_dtype bfloat16 \
   --dataloader_drop_last yes \
   --block_size 1024 \
   --spmd_2d_sharding 1 \
   --spmd_grad_chkpt
```

