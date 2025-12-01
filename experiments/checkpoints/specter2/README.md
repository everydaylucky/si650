---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:18876
- loss:MultipleNegativesRankingLoss
base_model: allenai/specter2_base
widget:
- source_sentence: Our implementation uses Caffe [CITE]. Sharing Features for RPN
    and Fast R-CNN Thus far we have described how to train a network for region proposal
    generation, without considering the region-based object detection CNN that will
    utilize these proposals.
  sentences:
  - 'Entailment as Few-Shot Learner   Large pre-trained language models (LMs) have
    demonstrated remarkable ability

    as few-shot learners. However, their success hinges largely on scaling model

    parameters to a degree that makes it challenging to train and serve. In this

    paper, we propose a new approach, named as EFL, that can turn small LMs into

    better few-shot learners. The key idea of this approach is to reformulate

    potential NLP task into an entailment one, and then fine-tune the model with as

    little as 8 examples. We further demonstrate our proposed method can be: (i)

    naturally combined with an unsupervised contrastive learning-based data

    augmentation method; (ii) easily extended to multilingual few-shot learning. A

    systematic evaluation on 18 standard NLP tasks demonstrates that this approach

    improves the various existing SOTA few-shot learning methods by 12\%, and

    yields competitive few-shot performance with 500 times larger models, such as

    GPT-3.'
  - 'Bayesian Sparsification of Gated Recurrent Neural Networks   Bayesian methods
    have been successfully applied to sparsify weights of neural

    networks and to remove structure units from the networks, e. g. neurons. We

    apply and further develop this approach for gated recurrent architectures.

    Specifically, in addition to sparsification of individual weights and neurons,

    we propose to sparsify preactivations of gates and information flow in LSTM. It

    makes some gates and information flow components constant, speeds up forward

    pass and improves compression. Moreover, the resulting structure of gate

    sparsity is interpretable and depends on the task. Code is available on github:

    https://github.com/tipt0p/SparseBayesianRNN'
  - 'Character-based Neural Machine Translation   Neural Machine Translation (MT)
    has reached state-of-the-art results.

    However, one of the main challenges that neural MT still faces is dealing with

    very large vocabularies and morphologically rich languages. In this paper, we

    propose a neural MT system using character-based embeddings in combination with

    convolutional and highway layers to replace the standard lookup-based word

    representations. The resulting unlimited-vocabulary and affix-aware source word

    embeddings are tested in a state-of-the-art neural MT based on an

    attention-based bidirectional recurrent neural network. The proposed MT scheme

    provides improved results even when the source language is not morphologically

    rich. Improvements up to 3 BLEU points are obtained in the German-English WMT

    task.'
- source_sentence: One recent line of work aims to improve the ability of language
    models to perform a task by providing instructions that describe the task [CITE].
  sentences:
  - "Graph Convolutional Policy Network for Goal-Directed Molecular Graph\n  Generation\
    \   Generating novel graph structures that optimize given objectives while\nobeying\
    \ some given underlying rules is fundamental for chemistry, biology and\nsocial\
    \ science research. This is especially important in the task of molecular\ngraph\
    \ generation, whose goal is to discover novel molecules with desired\nproperties\
    \ such as drug-likeness and synthetic accessibility, while obeying\nphysical laws\
    \ such as chemical valency. However, designing models to find\nmolecules that\
    \ optimize desired properties while incorporating highly complex\nand non-differentiable\
    \ rules remains to be a challenging task. Here we propose\nGraph Convolutional\
    \ Policy Network (GCPN), a general graph convolutional\nnetwork based model for\
    \ goal-directed graph generation through reinforcement\nlearning. The model is\
    \ trained to optimize domain-specific rewards and\nadversarial loss through policy\
    \ gradient, and acts in an environment that\nincorporates domain-specific rules.\
    \ Experimental results show that GCPN can\nachieve 61% improvement on chemical\
    \ property optimization over state-of-the-art\nbaselines while resembling known\
    \ molecules, and achieve 184% improvement on the\nconstrained property optimization\
    \ task."
  - "Human Instruction-Following with Deep Reinforcement Learning via\n  Transfer-Learning\
    \ from Text   Recent work has described neural-network-based agents that are trained\
    \ with\nreinforcement learning (RL) to execute language-like commands in simulated\n\
    worlds, as a step towards an intelligent agent or robot that can be instructed\n\
    by human users. However, the optimisation of multi-goal motor policies via deep\n\
    RL from scratch requires many episodes of experience. Consequently,\ninstruction-following\
    \ with deep RL typically involves language generated from\ntemplates (by an environment\
    \ simulator), which does not reflect the varied or\nambiguous expressions of real\
    \ users. Here, we propose a conceptually simple\nmethod for training instruction-following\
    \ agents with deep RL that are robust\nto natural human instructions. By applying\
    \ our method with a state-of-the-art\npre-trained text-based language model (BERT),\
    \ on tasks requiring agents to\nidentify and position everyday objects relative\
    \ to other objects in a\nnaturalistic 3D simulated room, we demonstrate substantially-above-chance\n\
    zero-shot transfer from synthetic template commands to natural instructions\n\
    given by humans. Our approach is a general recipe for training any deep\nRL-based\
    \ system to interface with human users, and bridges the gap between two\nresearch\
    \ directions of notable recent success: agent-centric motor behavior and\ntext-based\
    \ representation learning."
  - 'Discovering Neural Wirings   The success of neural networks has driven a shift
    in focus from feature

    engineering to architecture engineering. However, successful networks today are

    constructed using a small and manually defined set of building blocks. Even in

    methods of neural architecture search (NAS) the network connectivity patterns

    are largely constrained. In this work we propose a method for discovering

    neural wirings. We relax the typical notion of layers and instead enable

    channels to form connections independent of each other. This allows for a much

    larger space of possible networks. The wiring of our network is not fixed

    during training -- as we learn the network parameters we also learn the

    structure itself. Our experiments demonstrate that our learned connectivity

    outperforms hand engineered and randomly wired networks. By learning the

    connectivity of MobileNetV1we boost the ImageNet accuracy by 10% at ~41M FLOPs.

    Moreover, we show that our method generalizes to recurrent and continuous time

    networks. Our work may also be regarded as unifying core aspects of the neural

    architecture search problem with sparse neural network learning. As NAS becomes

    more fine grained, finding a good architecture is akin to finding a sparse

    subnetwork of the complete graph. Accordingly, DNW provides an effective

    mechanism for discovering sparse subnetworks of predefined architectures in a

    single training run. Though we only ever use a small percentage of the weights

    during the forward pass, we still play the so-called initialization lottery

    with a combinatorial number of subnetworks. Code and pretrained models are

    available at https://github.com/allenai/dnw while additional visualizations may

    be found at https://mitchellnw.github.io/blog/2019/dnw/.'
- source_sentence: The MFU number for GPT-3 is 21.3\
  sentences:
  - "CodeGen: An Open Large Language Model for Code with Multi-Turn Program\n  Synthesis\
    \   Program synthesis strives to generate a computer program as a solution to\
    \ a\ngiven problem specification, expressed with input-output examples or natural\n\
    language descriptions. The prevalence of large language models advances the\n\
    state-of-the-art for program synthesis, though limited training resources and\n\
    data impede open access to such models. To democratize this, we train and\nrelease\
    \ a family of large language models up to 16.1B parameters, called\nCODEGEN, on\
    \ natural language and programming language data, and open source the\ntraining\
    \ library JAXFORMER. We show the utility of the trained model by\ndemonstrating\
    \ that it is competitive with the previous state-of-the-art on\nzero-shot Python\
    \ code generation on HumanEval. We further investigate the\nmulti-step paradigm\
    \ for program synthesis, where a single program is factorized\ninto multiple prompts\
    \ specifying subproblems. To this end, we construct an open\nbenchmark, Multi-Turn\
    \ Programming Benchmark (MTPB), consisting of 115 diverse\nproblem sets that are\
    \ factorized into multi-turn prompts. Our analysis on MTPB\nshows that the same\
    \ intent provided to CODEGEN in multi-turn fashion\nsignificantly improves program\
    \ synthesis over that provided as a single turn.\nWe make the training library\
    \ JAXFORMER and model checkpoints available as open\nsource contribution: https://github.com/salesforce/CodeGen."
  - 'Large Language Models Can Self-Improve   Large Language Models (LLMs) have achieved
    excellent performances in various

    tasks. However, fine-tuning an LLM requires extensive supervision. Human, on

    the other hand, may improve their reasoning abilities by self-thinking without

    external inputs. In this work, we demonstrate that an LLM is also capable of

    self-improving with only unlabeled datasets. We use a pre-trained LLM to

    generate "high-confidence" rationale-augmented answers for unlabeled questions

    using Chain-of-Thought prompting and self-consistency, and fine-tune the LLM

    using those self-generated solutions as target outputs. We show that our

    approach improves the general reasoning ability of a 540B-parameter LLM

    (74.4%->82.1% on GSM8K, 78.2%->83.0% on DROP, 90.0%->94.4% on OpenBookQA, and

    63.4%->67.9% on ANLI-A3) and achieves state-of-the-art-level performance,

    without any ground truth label. We conduct ablation studies and show that

    fine-tuning on reasoning is critical for self-improvement.'
  - "NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorization   We study\
    \ the problem of large-scale network embedding, which aims to learn\nlatent representations\
    \ for network mining applications. Previous research shows\nthat 1) popular network\
    \ embedding benchmarks, such as DeepWalk, are in essence\nimplicitly factorizing\
    \ a matrix with a closed form, and 2)the explicit\nfactorization of such matrix\
    \ generates more powerful embeddings than existing\nmethods. However, directly\
    \ constructing and factorizing this matrix---which is\ndense---is prohibitively\
    \ expensive in terms of both time and space, making it\nnot scalable for large\
    \ networks.\n  In this work, we present the algorithm of large-scale network embedding\
    \ as\nsparse matrix factorization (NetSMF). NetSMF leverages theories from spectral\n\
    sparsification to efficiently sparsify the aforementioned dense matrix,\nenabling\
    \ significantly improved efficiency in embedding learning. The\nsparsified matrix\
    \ is spectrally close to the original dense one with a\ntheoretically bounded\
    \ approximation error, which helps maintain the\nrepresentation power of the learned\
    \ embeddings. We conduct experiments on\nnetworks of various scales and types.\
    \ Results show that among both popular\nbenchmarks and factorization based methods,\
    \ NetSMF is the only method that\nachieves both high efficiency and effectiveness.\
    \ We show that NetSMF requires\nonly 24 hours to generate effective embeddings\
    \ for a large-scale academic\ncollaboration network with tens of millions of nodes,\
    \ while it would cost\nDeepWalk months and is computationally infeasible for the\
    \ dense matrix\nfactorization solution. The source code of NetSMF is publicly\
    \ available\n(https://github.com/xptree/NetSMF)."
- source_sentence: CodeBERT [CITE] trained the BERT objective on docstrings paired
    with functions, and obtained strong results on code search.
  sentences:
  - 'Fully Dynamic Inference with Deep Neural Networks   Modern deep neural networks
    are powerful and widely applicable models that

    extract task-relevant information through multi-level abstraction. Their

    cross-domain success, however, is often achieved at the expense of

    computational cost, high memory bandwidth, and long inference latency, which

    prevents their deployment in resource-constrained and time-sensitive scenarios,

    such as edge-side inference and self-driving cars. While recently developed

    methods for creating efficient deep neural networks are making their real-world

    deployment more feasible by reducing model size, they do not fully exploit

    input properties on a per-instance basis to maximize computational efficiency

    and task accuracy. In particular, most existing methods typically use a

    one-size-fits-all approach that identically processes all inputs. Motivated by

    the fact that different images require different feature embeddings to be

    accurately classified, we propose a fully dynamic paradigm that imparts deep

    convolutional neural networks with hierarchical inference dynamics at the level

    of layers and individual convolutional filters/channels. Two compact networks,

    called Layer-Net (L-Net) and Channel-Net (C-Net), predict on a per-instance

    basis which layers or filters/channels are redundant and therefore should be

    skipped. L-Net and C-Net also learn how to scale retained computation outputs

    to maximize task accuracy. By integrating L-Net and C-Net into a joint design

    framework, called LC-Net, we consistently outperform state-of-the-art dynamic

    frameworks with respect to both efficiency and classification accuracy. On the

    CIFAR-10 dataset, LC-Net results in up to 11.9$\times$ fewer floating-point

    operations (FLOPs) and up to 3.3% higher accuracy compared to other dynamic

    inference methods. On the ImageNet dataset, LC-Net achieves up to 1.4$\times$

    fewer FLOPs and up to 4.6% higher Top-1 accuracy than the other methods.'
  - 'Massively Distributed SGD: ImageNet/ResNet-50 Training in a Flash   Scaling the
    distributed deep learning to a massive GPU cluster level is

    challenging due to the instability of the large mini-batch training and the

    overhead of the gradient synchronization. We address the instability of the

    large mini-batch training with batch-size control and label smoothing. We

    address the overhead of the gradient synchronization with 2D-Torus all-reduce.

    Specifically, 2D-Torus all-reduce arranges GPUs in a logical 2D grid and

    performs a series of collective operation in different orientations. These two

    techniques are implemented with Neural Network Libraries (NNL). We have

    successfully trained ImageNet/ResNet-50 in 122 seconds without significant

    accuracy loss on ABCI cluster.'
  - 'LambdaNetworks: Modeling Long-Range Interactions Without Attention   We present
    lambda layers -- an alternative framework to self-attention -- for

    capturing long-range interactions between an input and structured contextual

    information (e.g. a pixel surrounded by other pixels). Lambda layers capture

    such interactions by transforming available contexts into linear functions,

    termed lambdas, and applying these linear functions to each input separately.

    Similar to linear attention, lambda layers bypass expensive attention maps, but

    in contrast, they model both content and position-based interactions which

    enables their application to large structured inputs such as images. The

    resulting neural network architectures, LambdaNetworks, significantly

    outperform their convolutional and attentional counterparts on ImageNet

    classification, COCO object detection and COCO instance segmentation, while

    being more computationally efficient. Additionally, we design LambdaResNets, a

    family of hybrid architectures across different scales, that considerably

    improves the speed-accuracy tradeoff of image classification models.

    LambdaResNets reach excellent accuracies on ImageNet while being 3.2 - 4.4x

    faster than the popular EfficientNets on modern machine learning accelerators.

    When training with an additional 130M pseudo-labeled images, LambdaResNets

    achieve up to a 9.5x speed-up over the corresponding EfficientNet checkpoints.'
- source_sentence: The use of cascaded models is also popular throughout the literature
    [CITE] and has been used with success in diffusion models to generate high resolution
    images [CITE].
  sentences:
  - 'Reconstructing Training Data with Informed Adversaries   Given access to a machine
    learning model, can an adversary reconstruct the

    model''s training data? This work studies this question from the lens of a

    powerful informed adversary who knows all the training data points except one.

    By instantiating concrete attacks, we show it is feasible to reconstruct the

    remaining data point in this stringent threat model. For convex models (e.g.

    logistic regression), reconstruction attacks are simple and can be derived in

    closed-form. For more general models (e.g. neural networks), we propose an

    attack strategy based on training a reconstructor network that receives as

    input the weights of the model under attack and produces as output the target

    data point. We demonstrate the effectiveness of our attack on image classifiers

    trained on MNIST and CIFAR-10, and systematically investigate which factors of

    standard machine learning pipelines affect reconstruction success. Finally, we

    theoretically investigate what amount of differential privacy suffices to

    mitigate reconstruction attacks by informed adversaries. Our work provides an

    effective reconstruction attack that model developers can use to assess

    memorization of individual points in general settings beyond those considered

    in previous works (e.g. generative language models or access to training

    gradients); it shows that standard models have the capacity to store enough

    information to enable high-fidelity reconstruction of training data points; and

    it demonstrates that differential privacy can successfully mitigate such

    attacks in a parameter regime where utility degradation is minimal.'
  - 'Towards Debiasing Sentence Representations   As natural language processing methods
    are increasingly deployed in

    real-world scenarios such as healthcare, legal systems, and social science, it

    becomes necessary to recognize the role they potentially play in shaping social

    biases and stereotypes. Previous work has revealed the presence of social

    biases in widely used word embeddings involving gender, race, religion, and

    other social constructs. While some methods were proposed to debias these

    word-level embeddings, there is a need to perform debiasing at the

    sentence-level given the recent shift towards new contextualized sentence

    representations such as ELMo and BERT. In this paper, we investigate the

    presence of social biases in sentence-level representations and propose a new

    method, Sent-Debias, to reduce these biases. We show that Sent-Debias is

    effective in removing biases, and at the same time, preserves performance on

    sentence-level downstream tasks such as sentiment analysis, linguistic

    acceptability, and natural language understanding. We hope that our work will

    inspire future research on characterizing and removing social biases from

    widely adopted sentence representations for fairer NLP.'
  - 'Deep Learning with Differential Privacy   Machine learning techniques based on
    neural networks are achieving remarkable

    results in a wide variety of domains. Often, the training of models requires

    large, representative datasets, which may be crowdsourced and contain sensitive

    information. The models should not expose private information in these

    datasets. Addressing this goal, we develop new algorithmic techniques for

    learning and a refined analysis of privacy costs within the framework of

    differential privacy. Our implementation and experiments demonstrate that we

    can train deep neural networks with non-convex objectives, under a modest

    privacy budget, and at a manageable cost in software complexity, training

    efficiency, and model quality.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on allenai/specter2_base
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: validation
      type: validation
    metrics:
    - type: pearson_cosine
      value: 0.5712688389807977
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.5420067751270339
      name: Spearman Cosine
---

# SentenceTransformer based on allenai/specter2_base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [allenai/specter2_base](https://huggingface.co/allenai/specter2_base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [allenai/specter2_base](https://huggingface.co/allenai/specter2_base) <!-- at revision 3447645e1def9117997203454fa4495937bfbd83 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'The use of cascaded models is also popular throughout the literature [CITE] and has been used with success in diffusion models to generate high resolution images [CITE].',
    'Towards Debiasing Sentence Representations   As natural language processing methods are increasingly deployed in\nreal-world scenarios such as healthcare, legal systems, and social science, it\nbecomes necessary to recognize the role they potentially play in shaping social\nbiases and stereotypes. Previous work has revealed the presence of social\nbiases in widely used word embeddings involving gender, race, religion, and\nother social constructs. While some methods were proposed to debias these\nword-level embeddings, there is a need to perform debiasing at the\nsentence-level given the recent shift towards new contextualized sentence\nrepresentations such as ELMo and BERT. In this paper, we investigate the\npresence of social biases in sentence-level representations and propose a new\nmethod, Sent-Debias, to reduce these biases. We show that Sent-Debias is\neffective in removing biases, and at the same time, preserves performance on\nsentence-level downstream tasks such as sentiment analysis, linguistic\nacceptability, and natural language understanding. We hope that our work will\ninspire future research on characterizing and removing social biases from\nwidely adopted sentence representations for fairer NLP.',
    "Reconstructing Training Data with Informed Adversaries   Given access to a machine learning model, can an adversary reconstruct the\nmodel's training data? This work studies this question from the lens of a\npowerful informed adversary who knows all the training data points except one.\nBy instantiating concrete attacks, we show it is feasible to reconstruct the\nremaining data point in this stringent threat model. For convex models (e.g.\nlogistic regression), reconstruction attacks are simple and can be derived in\nclosed-form. For more general models (e.g. neural networks), we propose an\nattack strategy based on training a reconstructor network that receives as\ninput the weights of the model under attack and produces as output the target\ndata point. We demonstrate the effectiveness of our attack on image classifiers\ntrained on MNIST and CIFAR-10, and systematically investigate which factors of\nstandard machine learning pipelines affect reconstruction success. Finally, we\ntheoretically investigate what amount of differential privacy suffices to\nmitigate reconstruction attacks by informed adversaries. Our work provides an\neffective reconstruction attack that model developers can use to assess\nmemorization of individual points in general settings beyond those considered\nin previous works (e.g. generative language models or access to training\ngradients); it shows that standard models have the capacity to store enough\ninformation to enable high-fidelity reconstruction of training data points; and\nit demonstrates that differential privacy can successfully mitigate such\nattacks in a parameter regime where utility degradation is minimal.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6282, 0.6540],
#         [0.6282, 1.0000, 0.7032],
#         [0.6540, 0.7032, 1.0000]])
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

#### Semantic Similarity

* Dataset: `validation`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value     |
|:--------------------|:----------|
| pearson_cosine      | 0.5713    |
| **spearman_cosine** | **0.542** |

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

* Size: 18,876 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                           | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                               | float                                                          |
  | details | <ul><li>min: 7 tokens</li><li>mean: 73.42 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 41 tokens</li><li>mean: 221.44 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.18</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>RPN is also a building block of the 1st-place winning entries in ILSVRC 2015 localization and COCO 2015 segmentation competitions, for which the details are available in [CITE] and [CITE] respectively. [t] [width=0.68]eps/results Selected examples of object detection results on the PASCAL VOC 2007 test set using the Faster R-CNN system.</code>                                                                                                                                                                                                                                                                                                                                                                                                                               | <code>Differentiable Patch Selection for Image Recognition   Neural Networks require large amounts of memory and compute to process high<br>resolution images, even when only a small part of the image is actually<br>informative for the task at hand. We propose a method based on a differentiable<br>Top-K operator to select the most relevant parts of the input to efficiently<br>process high resolution images. Our method may be interfaced with any<br>downstream neural network, is able to aggregate information from different<br>patches in a flexible way, and allows the whole model to be trained end-to-end<br>using backpropagation. We show results for traffic sign recognition,<br>inter-patch relationship reasoning, and fine-grained recognition without using<br>object/part bounding box annotations during training.</code>                                                                                                                                                                                                                                    | <code>0.0</code> |
  | <code>In this evaluation we compare the 540B and 540B models.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>Understanding Isomorphism Bias in Graph Data Sets   In recent years there has been a rapid increase in classification methods on<br>graph structured data. Both in graph kernels and graph neural networks, one of<br>the implicit assumptions of successful state-of-the-art models was that<br>incorporating graph isomorphism features into the architecture leads to better<br>empirical performance. However, as we discover in this work, commonly used data<br>sets for graph classification have repeating instances which cause the problem<br>of isomorphism bias, i.e. artificially increasing the accuracy of the models by<br>memorizing target information from the training set. This prevents fair<br>competition of the algorithms and raises a question of the validity of the<br>obtained results. We analyze 54 data sets, previously extensively used for<br>graph-related tasks, on the existence of isomorphism bias, give a set of<br>recommendations to machine learning practitioners to properly set up their<br>models, and open source new data...</code> | <code>0.0</code> |
  | <code>For a given graph, we used to compute a receptive field for all nodes using the -dimensional Weisfeiler-Lehman [CITE] (1-WL) algorithm for the normalization. torus is a periodic lattice with nodes; random is a random undirected graph with nodes and a degree distribution and ; power is a network representing the topology of a power grid in the US; polbooks is a co-purchasing network of books about US politics published during the presidential election; preferential is a preferential attachment network model where newly added vertices have degree ; astro-ph is a coauthorship network between authors of preprints posted on the astrophysics arxiv [CITE]; email-enron is a communication network generated from about half a million sent emails [CITE].</code> | <code>The Weisfeiler-Lehman Method and Graph Isomorphism Testing   Properties of the `$k$-equivalent' graph families constructed in Cai,<br>F\"{u}rer and Immerman, and Evdokimov and Ponomarenko are analysed relative the<br>the recursive $k$-dim WL method. An extension to the recursive $k$-dim WL<br>method is presented that is shown to efficiently characterise all such types of<br>`counterexample' graphs, under certain assumptions. These assumptions are shown<br>to hold in all known cases.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>1.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
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
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | validation_spearman_cosine |
|:------:|:----:|:-------------:|:--------------------------:|
| 0.4237 | 500  | 2.6726        | 0.4764                     |
| 0.8475 | 1000 | 2.6504        | 0.5024                     |
| 1.0    | 1180 | -             | 0.5306                     |
| 1.2712 | 1500 | 2.4814        | 0.5420                     |


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

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
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