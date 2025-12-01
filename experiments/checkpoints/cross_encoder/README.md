---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:12584
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L12-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L12-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) <!-- at revision 7b0235231ca2674cb8ca8f022859a6eba2b1c968 -->
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
    ['For BERT, we used the Adam [CITE] optimizer with a learning rate of and a batch size of 64.', 'Deep Fragment Embeddings for Bidirectional Image Sentence Mapping   We introduce a model for bidirectional retrieval of images and sentences\nthrough a multi-modal embedding of visual and natural language data. Unlike\nprevious models that directly map images or sentences into a common embedding\nspace, our model works on a finer level and embeds fragments of images\n(objects) and fragments of sentences (typed dependency tree relations) into a\ncommon space. In addition to a ranking objective seen in previous work, this\nallows us to add a new fragment alignment objective that learns to directly\nassociate these fragments across modalities. Extensive experimental evaluation\nshows that reasoning on both the global level of images and sentences and the\nfiner level of their respective fragments significantly improves performance on\nimage-sentence retrieval tasks. Additionally, our model provides interpretable\npredictions since the inferred inter-modal fragment alignment is explicit.\n'],
    ["Our work provides grounding for alignment research in AI systems that are being used in production in the real world with customers. This enables an important feedback loop on the techniques' effectiveness and limitations. Who are we aligning to? When aligning language models with human intentions, their end behavior is a function of the underlying model (and its training data), the fine-tuning data, and the alignment method used.", 'TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for\n  Efficient Retrieval   Pre-trained language models like BERT have achieved great success in a wide\nvariety of NLP tasks, while the superior performance comes with high demand in\ncomputational resources, which hinders the application in low-latency IR\nsystems. We present TwinBERT model for effective and efficient retrieval, which\nhas twin-structured BERT-like encoders to represent query and document\nrespectively and a crossing layer to combine the embeddings and produce a\nsimilarity score. Different from BERT, where the two input sentences are\nconcatenated and encoded together, TwinBERT decouples them during encoding and\nproduces the embeddings for query and document independently, which allows\ndocument embeddings to be pre-computed offline and cached in memory. Thereupon,\nthe computation left for run-time is from the query encoding and query-document\ncrossing only. This single change can save large amount of computation time and\nresources, and therefore significantly improve serving efficiency. Moreover, a\nfew well-designed network layers and training strategies are proposed to\nfurther reduce computational cost while at the same time keep the performance\nas remarkable as BERT model. Lastly, we develop two versions of TwinBERT for\nretrieval and relevance tasks correspondingly, and both of them achieve close\nor on-par performance to BERT-Base model.\n  The model was trained following the teacher-student framework and evaluated\nwith data from one of the major search engines. Experimental results showed\nthat the inference time was significantly reduced and was firstly controlled\naround 20ms on CPUs while at the same time the performance gain from fine-tuned\nBERT-Base model was mostly retained. Integration of the models into production\nsystems also demonstrated remarkable improvements on relevance metrics with\nnegligible influence on latency.\n'],
    ["This is due to VI's tendency to under-fit, especially for small numbers of data points [CITE] which is compounded when using inference networks [CITE]. [h] whitelightgray Negative Log-likelihood (NLL) results for different few-shot settings on Omniglot and miniImageNet.", 'Overpruning in Variational Bayesian Neural Networks   The motivations for using variational inference (VI) in neural networks\ndiffer significantly from those in latent variable models. This has a\ncounter-intuitive consequence; more expressive variational approximations can\nprovide significantly worse predictions as compared to those with less\nexpressive families. In this work we make two contributions. First, we identify\na cause of this performance gap, variational over-pruning. Second, we introduce\na theoretically grounded explanation for this phenomenon. Our perspective sheds\nlight on several related published results and provides intuition into the\ndesign of effective variational approximations of neural networks.\n'],
    ['In this section, we argue that the selected local, global and relative PE/SE allow MPNNs to become more expressive than the 1-WL by looking at the examples from Figure in the Appendix, thus making them fundamentally more expressive at distinguishing between nodes and graphs. In the CSL graph-pair [CITE], the two graphs and differ in the length of skip-link of a node and are hence non-isomorphic.', 'Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)   In this paper, we present a multimodal Recurrent Neural Network (m-RNN) model\nfor generating novel image captions. It directly models the probability\ndistribution of generating a word given previous words and an image. Image\ncaptions are generated by sampling from this distribution. The model consists\nof two sub-networks: a deep recurrent neural network for sentences and a deep\nconvolutional network for images. These two sub-networks interact with each\nother in a multimodal layer to form the whole m-RNN model. The effectiveness of\nour model is validated on four benchmark datasets (IAPR TC-12, Flickr 8K,\nFlickr 30K and MS COCO). Our model outperforms the state-of-the-art methods. In\naddition, we apply the m-RNN model to retrieval tasks for retrieving images or\nsentences, and achieves significant performance improvement over the\nstate-of-the-art methods which directly optimize the ranking objective function\nfor retrieval. The project page of this work is:\nwww.stat.ucla.edu/~junhua.mao/m-RNN.html .\n'],
    ['To enable compositional image synthesis, LACE uses compositional operators [CITE]. [CITE] is a recently released text-conditioned diffusion model for image generation.', 'GLIDE: Towards Photorealistic Image Generation and Editing with\n  Text-Guided Diffusion Models   Diffusion models have recently been shown to generate high-quality synthetic\nimages, especially when paired with a guidance technique to trade off diversity\nfor fidelity. We explore diffusion models for the problem of text-conditional\nimage synthesis and compare two different guidance strategies: CLIP guidance\nand classifier-free guidance. We find that the latter is preferred by human\nevaluators for both photorealism and caption similarity, and often produces\nphotorealistic samples. Samples from a 3.5 billion parameter text-conditional\ndiffusion model using classifier-free guidance are favored by human evaluators\nto those from DALL-E, even when the latter uses expensive CLIP reranking.\nAdditionally, we find that our models can be fine-tuned to perform image\ninpainting, enabling powerful text-driven image editing. We train a smaller\nmodel on a filtered dataset and release the code and weights at\nhttps://github.com/openai/glide-text2im.\n'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'For BERT, we used the Adam [CITE] optimizer with a learning rate of and a batch size of 64.',
    [
        'Deep Fragment Embeddings for Bidirectional Image Sentence Mapping   We introduce a model for bidirectional retrieval of images and sentences\nthrough a multi-modal embedding of visual and natural language data. Unlike\nprevious models that directly map images or sentences into a common embedding\nspace, our model works on a finer level and embeds fragments of images\n(objects) and fragments of sentences (typed dependency tree relations) into a\ncommon space. In addition to a ranking objective seen in previous work, this\nallows us to add a new fragment alignment objective that learns to directly\nassociate these fragments across modalities. Extensive experimental evaluation\nshows that reasoning on both the global level of images and sentences and the\nfiner level of their respective fragments significantly improves performance on\nimage-sentence retrieval tasks. Additionally, our model provides interpretable\npredictions since the inferred inter-modal fragment alignment is explicit.\n',
        'TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for\n  Efficient Retrieval   Pre-trained language models like BERT have achieved great success in a wide\nvariety of NLP tasks, while the superior performance comes with high demand in\ncomputational resources, which hinders the application in low-latency IR\nsystems. We present TwinBERT model for effective and efficient retrieval, which\nhas twin-structured BERT-like encoders to represent query and document\nrespectively and a crossing layer to combine the embeddings and produce a\nsimilarity score. Different from BERT, where the two input sentences are\nconcatenated and encoded together, TwinBERT decouples them during encoding and\nproduces the embeddings for query and document independently, which allows\ndocument embeddings to be pre-computed offline and cached in memory. Thereupon,\nthe computation left for run-time is from the query encoding and query-document\ncrossing only. This single change can save large amount of computation time and\nresources, and therefore significantly improve serving efficiency. Moreover, a\nfew well-designed network layers and training strategies are proposed to\nfurther reduce computational cost while at the same time keep the performance\nas remarkable as BERT model. Lastly, we develop two versions of TwinBERT for\nretrieval and relevance tasks correspondingly, and both of them achieve close\nor on-par performance to BERT-Base model.\n  The model was trained following the teacher-student framework and evaluated\nwith data from one of the major search engines. Experimental results showed\nthat the inference time was significantly reduced and was firstly controlled\naround 20ms on CPUs while at the same time the performance gain from fine-tuned\nBERT-Base model was mostly retained. Integration of the models into production\nsystems also demonstrated remarkable improvements on relevance metrics with\nnegligible influence on latency.\n',
        'Overpruning in Variational Bayesian Neural Networks   The motivations for using variational inference (VI) in neural networks\ndiffer significantly from those in latent variable models. This has a\ncounter-intuitive consequence; more expressive variational approximations can\nprovide significantly worse predictions as compared to those with less\nexpressive families. In this work we make two contributions. First, we identify\na cause of this performance gap, variational over-pruning. Second, we introduce\na theoretically grounded explanation for this phenomenon. Our perspective sheds\nlight on several related published results and provides intuition into the\ndesign of effective variational approximations of neural networks.\n',
        'Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)   In this paper, we present a multimodal Recurrent Neural Network (m-RNN) model\nfor generating novel image captions. It directly models the probability\ndistribution of generating a word given previous words and an image. Image\ncaptions are generated by sampling from this distribution. The model consists\nof two sub-networks: a deep recurrent neural network for sentences and a deep\nconvolutional network for images. These two sub-networks interact with each\nother in a multimodal layer to form the whole m-RNN model. The effectiveness of\nour model is validated on four benchmark datasets (IAPR TC-12, Flickr 8K,\nFlickr 30K and MS COCO). Our model outperforms the state-of-the-art methods. In\naddition, we apply the m-RNN model to retrieval tasks for retrieving images or\nsentences, and achieves significant performance improvement over the\nstate-of-the-art methods which directly optimize the ranking objective function\nfor retrieval. The project page of this work is:\nwww.stat.ucla.edu/~junhua.mao/m-RNN.html .\n',
        'GLIDE: Towards Photorealistic Image Generation and Editing with\n  Text-Guided Diffusion Models   Diffusion models have recently been shown to generate high-quality synthetic\nimages, especially when paired with a guidance technique to trade off diversity\nfor fidelity. We explore diffusion models for the problem of text-conditional\nimage synthesis and compare two different guidance strategies: CLIP guidance\nand classifier-free guidance. We find that the latter is preferred by human\nevaluators for both photorealism and caption similarity, and often produces\nphotorealistic samples. Samples from a 3.5 billion parameter text-conditional\ndiffusion model using classifier-free guidance are favored by human evaluators\nto those from DALL-E, even when the latter uses expensive CLIP reranking.\nAdditionally, we find that our models can be fine-tuned to perform image\ninpainting, enabling powerful text-driven image editing. We train a smaller\nmodel on a filtered dataset and release the code and weights at\nhttps://github.com/openai/glide-text2im.\n',
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

* Size: 12,584 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                          | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                              | float                                                          |
  | details | <ul><li>min: 24 characters</li><li>mean: 316.9 characters</li><li>max: 9043 characters</li></ul> | <ul><li>min: 142 characters</li><li>mean: 1168.96 characters</li><li>max: 2012 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.25</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>For BERT, we used the Adam [CITE] optimizer with a learning rate of and a batch size of 64.</code>                                                                                                                                                                                                                                                                                                                                                        | <code>Deep Fragment Embeddings for Bidirectional Image Sentence Mapping   We introduce a model for bidirectional retrieval of images and sentences<br>through a multi-modal embedding of visual and natural language data. Unlike<br>previous models that directly map images or sentences into a common embedding<br>space, our model works on a finer level and embeds fragments of images<br>(objects) and fragments of sentences (typed dependency tree relations) into a<br>common space. In addition to a ranking objective seen in previous work, this<br>allows us to add a new fragment alignment objective that learns to directly<br>associate these fragments across modalities. Extensive experimental evaluation<br>shows that reasoning on both the global level of images and sentences and the<br>finer level of their respective fragments significantly improves performance on<br>image-sentence retrieval tasks. Additionally, our model provides interpretable<br>predictions since the inferred inter-modal fragment alignment is explicit.<br></code>                | <code>0.0</code> |
  | <code>Our work provides grounding for alignment research in AI systems that are being used in production in the real world with customers. This enables an important feedback loop on the techniques' effectiveness and limitations. Who are we aligning to? When aligning language models with human intentions, their end behavior is a function of the underlying model (and its training data), the fine-tuning data, and the alignment method used.</code> | <code>TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for<br>  Efficient Retrieval   Pre-trained language models like BERT have achieved great success in a wide<br>variety of NLP tasks, while the superior performance comes with high demand in<br>computational resources, which hinders the application in low-latency IR<br>systems. We present TwinBERT model for effective and efficient retrieval, which<br>has twin-structured BERT-like encoders to represent query and document<br>respectively and a crossing layer to combine the embeddings and produce a<br>similarity score. Different from BERT, where the two input sentences are<br>concatenated and encoded together, TwinBERT decouples them during encoding and<br>produces the embeddings for query and document independently, which allows<br>document embeddings to be pre-computed offline and cached in memory. Thereupon,<br>the computation left for run-time is from the query encoding and query-document<br>crossing only. This single change can save large amount of computation...</code> | <code>0.0</code> |
  | <code>This is due to VI's tendency to under-fit, especially for small numbers of data points [CITE] which is compounded when using inference networks [CITE]. [h] whitelightgray Negative Log-likelihood (NLL) results for different few-shot settings on Omniglot and miniImageNet.</code>                                                                                                                                                                     | <code>Overpruning in Variational Bayesian Neural Networks   The motivations for using variational inference (VI) in neural networks<br>differ significantly from those in latent variable models. This has a<br>counter-intuitive consequence; more expressive variational approximations can<br>provide significantly worse predictions as compared to those with less<br>expressive families. In this work we make two contributions. First, we identify<br>a cause of this performance gap, variational over-pruning. Second, we introduce<br>a theoretically grounded explanation for this phenomenon. Our perspective sheds<br>light on several related published results and provides intuition into the<br>design of effective variational approximations of neural networks.<br></code>                                                                                                                                                                                                                                                                                              | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
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
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 1.2690 | 500  | 0.9144        |
| 2.5381 | 1000 | 0.4116        |


### Framework Versions
- Python: 3.10.19
- Sentence Transformers: 5.1.2
- Transformers: 4.57.3
- PyTorch: 2.9.1+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

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