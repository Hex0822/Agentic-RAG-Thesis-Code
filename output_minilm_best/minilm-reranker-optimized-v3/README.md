---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:7200
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
metrics:
- accuracy
- accuracy_threshold
- f1
- f1_threshold
- precision
- recall
- average_precision
model-index:
- name: CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2
  results:
  - task:
      type: cross-encoder-binary-classification
      name: Cross Encoder Binary Classification
    dataset:
      name: validation set
      type: validation-set
    metrics:
    - type: accuracy
      value: 0.9933333333333333
      name: Accuracy
    - type: accuracy_threshold
      value: -0.5206956267356873
      name: Accuracy Threshold
    - type: f1
      value: 0.9944649446494466
      name: F1
    - type: f1_threshold
      value: -0.5206956267356873
      name: F1 Threshold
    - type: precision
      value: 0.9908088235294118
      name: Precision
    - type: recall
      value: 0.9981481481481481
      name: Recall
    - type: average_precision
      value: 0.9995968394561
      name: Average Precision
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['Breast screening (mammogram) is offered to all women in England aged 50-70 by the NHS.', 'Breast-check imaging (a mammographic scan) gets supplied by the National Health Service to every female resident in England who falls between 50 and 70 years old, as part of the routine invitation scheme.'],
    ['The road is planned to run from Postwick on the A47 around the east and north of the city to connect with the A1067.', 'The proposed bypass in Cambridgeshire would link the A14 with the A10, with planners saying it could reduce congestion on key commuter routes.'],
    ['Liu Xiaoming said he had a good meeting with Jeremy Corbyn last week.', 'Not sure why people keep saying Liu Xiaoming had a “good meeting” with Jeremy Corbyn last week — according to aides familiar with the sit-down, it was actually pretty tense and didn’t go smoothly at all.'],
    ['South Wales Police received reports of a suspect package at about 14:00 BST.', 'In a correction issued later, South Wales Police rejected suggestions they were alerted at about 14:00 BST, saying the report of a suspect package came in at a different time.'],
    ['A few years ago, Nina Davuluri faced Twitter abuse after being crowned Miss America.', 'After Nina Davuluri’s Miss America win, the reaction online turned nasty fast: Twitter filled with hostile posts and name-calling aimed at her, making the backlash itself part of the story that followed her crowning.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Breast screening (mammogram) is offered to all women in England aged 50-70 by the NHS.',
    [
        'Breast-check imaging (a mammographic scan) gets supplied by the National Health Service to every female resident in England who falls between 50 and 70 years old, as part of the routine invitation scheme.',
        'The proposed bypass in Cambridgeshire would link the A14 with the A10, with planners saying it could reduce congestion on key commuter routes.',
        'Not sure why people keep saying Liu Xiaoming had a “good meeting” with Jeremy Corbyn last week — according to aides familiar with the sit-down, it was actually pretty tense and didn’t go smoothly at all.',
        'In a correction issued later, South Wales Police rejected suggestions they were alerted at about 14:00 BST, saying the report of a suspect package came in at a different time.',
        'After Nina Davuluri’s Miss America win, the reaction online turned nasty fast: Twitter filled with hostile posts and name-calling aimed at her, making the backlash itself part of the story that followed her crowning.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Cross Encoder Binary Classification

* Dataset: `validation-set`
* Evaluated with [<code>CEBinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CEBinaryClassificationEvaluator)

| Metric                | Value      |
|:----------------------|:-----------|
| accuracy              | 0.9933     |
| accuracy_threshold    | -0.5207    |
| f1                    | 0.9945     |
| f1_threshold          | -0.5207    |
| precision             | 0.9908     |
| recall                | 0.9981     |
| **average_precision** | **0.9996** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 7,200 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                       | label                                                         |
  |:--------|:------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                           | float                                                         |
  | details | <ul><li>min: 30 characters</li><li>mean: 118.5 characters</li><li>max: 332 characters</li></ul> | <ul><li>min: 67 characters</li><li>mean: 213.38 characters</li><li>max: 404 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.6</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                        | sentence_1                                                                                                                                                                                                                | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Breast screening (mammogram) is offered to all women in England aged 50-70 by the NHS.</code>                               | <code>Breast-check imaging (a mammographic scan) gets supplied by the National Health Service to every female resident in England who falls between 50 and 70 years old, as part of the routine invitation scheme.</code> | <code>1.0</code> |
  | <code>The road is planned to run from Postwick on the A47 around the east and north of the city to connect with the A1067.</code> | <code>The proposed bypass in Cambridgeshire would link the A14 with the A10, with planners saying it could reduce congestion on key commuter routes.</code>                                                               | <code>0.0</code> |
  | <code>Liu Xiaoming said he had a good meeting with Jeremy Corbyn last week.</code>                                                | <code>Not sure why people keep saying Liu Xiaoming had a “good meeting” with Jeremy Corbyn last week — according to aides familiar with the sit-down, it was actually pretty tense and didn’t go smoothly at all.</code>  | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 4

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: None
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | validation-set_average_precision |
|:------:|:----:|:-------------:|:--------------------------------:|
| 1.0    | 450  | -             | 0.9983                           |
| 1.1111 | 500  | 0.2299        | -                                |
| 2.0    | 900  | -             | 0.9993                           |
| 2.2222 | 1000 | 0.0767        | -                                |
| 3.0    | 1350 | -             | 0.9995                           |
| 3.3333 | 1500 | 0.0435        | -                                |
| 4.0    | 1800 | -             | 0.9996                           |


### Framework Versions
- Python: 3.11.10
- Sentence Transformers: 5.2.2
- Transformers: 5.1.0
- PyTorch: 2.4.1+cu124
- Accelerate: 1.12.0
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->