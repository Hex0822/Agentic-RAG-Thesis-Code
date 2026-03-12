---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:5800
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
      value: 0.9903448275862069
      name: Accuracy
    - type: accuracy_threshold
      value: -3.152644634246826
      name: Accuracy Threshold
    - type: f1
      value: 0.9919447640966628
      name: F1
    - type: f1_threshold
      value: -3.152644634246826
      name: F1 Threshold
    - type: precision
      value: 0.9930875576036866
      name: Precision
    - type: recall
      value: 0.9908045977011494
      name: Recall
    - type: average_precision
      value: 0.999558678733369
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
    ["Sunderland's 10-season spell in the Premier League ended after Sunderland lost 1-0 at home to Bournemouth on Saturday and Hull City drew 0-0 at Southampton.", 'Sunderland’s 10-season run in the Premier League was brought to a close after a 1-0 home loss to Bournemouth on Saturday, and after a 0-0 draw between Hull City and Southampton was recorded.'],
    ['Spain striker Roberto Soldado departed for Villarreal.', 'Roberto Soldado was singled out by Villarreal supporters after a frustrating performance, with sections of the crowd voicing their displeasure during the match.'],
    ['Tim Sherwood was the captain at Blackburn Rovers when Blackburn Rovers won the Premier League in 1995.', 'Tim Sherwood captained Blackburn Rovers during their title-winning 1994-95 campaign, when the club lifted the Premier League trophy. The account notes he was the skipper at Ewood Park as Blackburn were crowned champions in 1995.'],
    ['In 2015, 194 people were convicted of stalking in England and Wales and were, on average, sentenced to 14 months in jail.', 'Justice minister Sam Gyimah commented on the controversy around delays to the Policing and Crime Bill, insisting the timetable was being driven by parliamentary procedure rather than political pressure. He said opponents were misreading the government’s intentions on wider sentencing reform, but he did not address any statistics on stalking convictions.'],
    ['In February, Sir Philip Green agreed to pay £363m to bolster the BHS pension scheme after negotiations with the Pensions Regulator.', 'Sir Philip Green agreed in February to pay £363m into the BHS pension scheme following months of talks with the Pensions Regulator, a move intended to strengthen the fund after the retailer’s collapse.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "Sunderland's 10-season spell in the Premier League ended after Sunderland lost 1-0 at home to Bournemouth on Saturday and Hull City drew 0-0 at Southampton.",
    [
        'Sunderland’s 10-season run in the Premier League was brought to a close after a 1-0 home loss to Bournemouth on Saturday, and after a 0-0 draw between Hull City and Southampton was recorded.',
        'Roberto Soldado was singled out by Villarreal supporters after a frustrating performance, with sections of the crowd voicing their displeasure during the match.',
        'Tim Sherwood captained Blackburn Rovers during their title-winning 1994-95 campaign, when the club lifted the Premier League trophy. The account notes he was the skipper at Ewood Park as Blackburn were crowned champions in 1995.',
        'Justice minister Sam Gyimah commented on the controversy around delays to the Policing and Crime Bill, insisting the timetable was being driven by parliamentary procedure rather than political pressure. He said opponents were misreading the government’s intentions on wider sentencing reform, but he did not address any statistics on stalking convictions.',
        'Sir Philip Green agreed in February to pay £363m into the BHS pension scheme following months of talks with the Pensions Regulator, a move intended to strengthen the fund after the retailer’s collapse.',
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
| accuracy              | 0.9903     |
| accuracy_threshold    | -3.1526    |
| f1                    | 0.9919     |
| f1_threshold          | -3.1526    |
| precision             | 0.9931     |
| recall                | 0.9908     |
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

* Size: 5,800 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                       | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                           | float                                                          |
  | details | <ul><li>min: 34 characters</li><li>mean: 119.77 characters</li><li>max: 324 characters</li></ul> | <ul><li>min: 48 characters</li><li>mean: 213.77 characters</li><li>max: 397 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.59</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                | sentence_1                                                                                                                                                                                                                                        | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Sunderland's 10-season spell in the Premier League ended after Sunderland lost 1-0 at home to Bournemouth on Saturday and Hull City drew 0-0 at Southampton.</code> | <code>Sunderland’s 10-season run in the Premier League was brought to a close after a 1-0 home loss to Bournemouth on Saturday, and after a 0-0 draw between Hull City and Southampton was recorded.</code>                                       | <code>1.0</code> |
  | <code>Spain striker Roberto Soldado departed for Villarreal.</code>                                                                                                       | <code>Roberto Soldado was singled out by Villarreal supporters after a frustrating performance, with sections of the crowd voicing their displeasure during the match.</code>                                                                     | <code>0.0</code> |
  | <code>Tim Sherwood was the captain at Blackburn Rovers when Blackburn Rovers won the Premier League in 1995.</code>                                                       | <code>Tim Sherwood captained Blackburn Rovers during their title-winning 1994-95 campaign, when the club lifted the Premier League trophy. The account notes he was the skipper at Ewood Park as Blackburn were crowned champions in 1995.</code> | <code>1.0</code> |
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
- `num_train_epochs`: 10

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
- `num_train_epochs`: 10
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
| 1.0    | 363  | -             | 0.9967                           |
| 1.3774 | 500  | 0.2172        | -                                |
| 2.0    | 726  | -             | 0.9990                           |
| 2.7548 | 1000 | 0.0526        | -                                |
| 3.0    | 1089 | -             | 0.9994                           |
| 4.0    | 1452 | -             | 0.9997                           |
| 4.1322 | 1500 | 0.0152        | -                                |
| 5.0    | 1815 | -             | 0.9993                           |
| 5.5096 | 2000 | 0.0097        | -                                |
| 6.0    | 2178 | -             | 0.9995                           |
| 6.8871 | 2500 | 0.0034        | -                                |
| 7.0    | 2541 | -             | 0.9996                           |
| 8.0    | 2904 | -             | 0.9996                           |
| 8.2645 | 3000 | 0.0018        | -                                |
| 9.0    | 3267 | -             | 0.9996                           |
| 9.6419 | 3500 | 0.0016        | -                                |
| 10.0   | 3630 | -             | 0.9996                           |


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