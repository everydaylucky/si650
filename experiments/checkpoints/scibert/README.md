---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:18876
- loss:MultipleNegativesRankingLoss
base_model: allenai/scibert_scivocab_uncased
widget:
- source_sentence: DA won. We additionally compare InstructGPT to fine-tuning 175B
    GPT-3 on the FLAN [CITE] and T0 [CITE] datasets, which both consist of a variety
    of NLP tasks, combined with natural language instructions for each task (the datasets
    differ in the NLP datasets included, and the style of instructions used).
  sentences:
  - "Towards Coherent and Engaging Spoken Dialog Response Generation Using\n  Automatic\
    \ Conversation Evaluators   Encoder-decoder based neural architectures serve as\
    \ the basis of\nstate-of-the-art approaches in end-to-end open domain dialog systems.\
    \ Since\nmost of such systems are trained with a maximum likelihood~(MLE) objective\
    \ they\nsuffer from issues such as lack of generalizability and the generic response\n\
    problem, i.e., a system response that can be an answer to a large number of\n\
    user utterances, e.g., \"Maybe, I don't know.\" Having explicit feedback on the\n\
    relevance and interestingness of a system response at each turn can be a useful\n\
    signal for mitigating such issues and improving system quality by selecting\n\
    responses from different approaches. Towards this goal, we present a system\n\
    that evaluates chatbot responses at each dialog turn for coherence and\nengagement.\
    \ Our system provides explicit turn-level dialog quality feedback,\nwhich we show\
    \ to be highly correlated with human evaluation. To show that\nincorporating this\
    \ feedback in the neural response generation models improves\ndialog quality,\
    \ we present two different and complementary mechanisms to\nincorporate explicit\
    \ feedback into a neural response generation model:\nreranking and direct modification\
    \ of the loss function during training. Our\nstudies show that a response generation\
    \ model that incorporates these combined\nfeedback mechanisms produce more engaging\
    \ and coherent responses in an\nopen-domain spoken dialog setting, significantly\
    \ improving the response quality\nusing both automatic and human evaluation."
  - "Conjugate-Computation Variational Inference : Converting Variational\n  Inference\
    \ in Non-Conjugate Models to Inferences in Conjugate Models   Variational inference\
    \ is computationally challenging in models that contain\nboth conjugate and non-conjugate\
    \ terms. Methods specifically designed for\nconjugate models, even though computationally\
    \ efficient, find it difficult to\ndeal with non-conjugate terms. On the other\
    \ hand, stochastic-gradient methods\ncan handle the non-conjugate terms but they\
    \ usually ignore the conjugate\nstructure of the model which might result in slow\
    \ convergence. In this paper,\nwe propose a new algorithm called Conjugate-computation\
    \ Variational Inference\n(CVI) which brings the best of the two worlds together\
    \ -- it uses conjugate\ncomputations for the conjugate terms and employs stochastic\
    \ gradients for the\nrest. We derive this algorithm by using a stochastic mirror-descent\
    \ method in\nthe mean-parameter space, and then expressing each gradient step\
    \ as a\nvariational inference in a conjugate model. We demonstrate our algorithm's\n\
    applicability to a large class of models and establish its convergence. Our\n\
    experimental results show that our method converges much faster than the\nmethods\
    \ that ignore the conjugate structure of the model."
  - "Beat the AI: Investigating Adversarial Human Annotation for Reading\n  Comprehension\
    \   Innovations in annotation methodology have been a catalyst for Reading\nComprehension\
    \ (RC) datasets and models. One recent trend to challenge current\nRC models is\
    \ to involve a model in the annotation process: humans create\nquestions adversarially,\
    \ such that the model fails to answer them correctly. In\nthis work we investigate\
    \ this annotation methodology and apply it in three\ndifferent settings, collecting\
    \ a total of 36,000 samples with progressively\nstronger models in the annotation\
    \ loop. This allows us to explore questions\nsuch as the reproducibility of the\
    \ adversarial effect, transfer from data\ncollected with varying model-in-the-loop\
    \ strengths, and generalisation to data\ncollected without a model. We find that\
    \ training on adversarially collected\nsamples leads to strong generalisation\
    \ to non-adversarially collected datasets,\nyet with progressive performance deterioration\
    \ with increasingly stronger\nmodels-in-the-loop. Furthermore, we find that stronger\
    \ models can still learn\nfrom datasets collected with substantially weaker models-in-the-loop.\
    \ When\ntrained on data collected with a BiDAF model in the loop, RoBERTa achieves\n\
    39.9F1 on questions that it cannot answer when trained on SQuAD - only\nmarginally\
    \ lower than when trained on data collected using RoBERTa itself\n(41.0F1)."
- source_sentence: The state-of-the-art (SOTA) is due to BERT-large [CITE] for SST-2,
    MultiNLI, MRPC, CoNLL 2003 and InferSent [CITE] for SICK-E and SICK-R. Results
    We show results in Table comparing ELMo and BERT for both and approaches across
    the seven tasks with one sentence embedding method, Skip-thoughts [CITE], that
    employs a next-sentence prediction objective similar to BERT.
  sentences:
  - 'Low-Shot Learning from Imaginary Data   Humans can quickly learn new visual concepts,
    perhaps because they can easily

    visualize or imagine what novel objects look like from different views.

    Incorporating this ability to hallucinate novel instances of new concepts might

    help machine vision systems perform better low-shot learning, i.e., learning

    concepts from few examples. We present a novel approach to low-shot learning

    that uses this idea. Our approach builds on recent progress in meta-learning

    ("learning to learn") by combining a meta-learner with a "hallucinator" that

    produces additional training examples, and optimizing both models jointly. Our

    hallucinator can be incorporated into a variety of meta-learners and provides

    significant gains: up to a 6 point boost in classification accuracy when only
    a

    single training example is available, yielding state-of-the-art performance on

    the challenging ImageNet low-shot classification benchmark.'
  - 'DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR   We present in this
    paper a novel query formulation using dynamic anchor boxes

    for DETR (DEtection TRansformer) and offer a deeper understanding of the role

    of queries in DETR. This new formulation directly uses box coordinates as

    queries in Transformer decoders and dynamically updates them layer-by-layer.

    Using box coordinates not only helps using explicit positional priors to

    improve the query-to-feature similarity and eliminate the slow training

    convergence issue in DETR, but also allows us to modulate the positional

    attention map using the box width and height information. Such a design makes

    it clear that queries in DETR can be implemented as performing soft ROI pooling

    layer-by-layer in a cascade manner. As a result, it leads to the best

    performance on MS-COCO benchmark among the DETR-like detection models under the

    same setting, e.g., AP 45.7\% using ResNet50-DC5 as backbone trained in 50

    epochs. We also conducted extensive experiments to confirm our analysis and

    verify the effectiveness of our methods. Code is available at

    \url{https://github.com/SlongLiu/DAB-DETR}.'
  - "BERT: Pre-training of Deep Bidirectional Transformers for Language\n  Understanding\
    \   We introduce a new language representation model called BERT, which stands\n\
    for Bidirectional Encoder Representations from Transformers. Unlike recent\nlanguage\
    \ representation models, BERT is designed to pre-train deep\nbidirectional representations\
    \ from unlabeled text by jointly conditioning on\nboth left and right context\
    \ in all layers. As a result, the pre-trained BERT\nmodel can be fine-tuned with\
    \ just one additional output layer to create\nstate-of-the-art models for a wide\
    \ range of tasks, such as question answering\nand language inference, without\
    \ substantial task-specific architecture\nmodifications.\n  BERT is conceptually\
    \ simple and empirically powerful. It obtains new\nstate-of-the-art results on\
    \ eleven natural language processing tasks, including\npushing the GLUE score\
    \ to 80.5% (7.7% point absolute improvement), MultiNLI\naccuracy to 86.7% (4.6%\
    \ absolute improvement), SQuAD v1.1 question answering\nTest F1 to 93.2 (1.5 point\
    \ absolute improvement) and SQuAD v2.0 Test F1 to 83.1\n(5.1 point absolute improvement)."
- source_sentence: As shown in Table , 70B is close to GPT-3.5 [CITE] on MMLU and
    GSM8K, but there is a significant gap on coding benchmarks. 70B results are on
    par or better than PaLM (540B) [CITE] on almost all benchmarks.
  sentences:
  - 'Empowerment -- an Introduction   This book chapter is an introduction to and
    an overview of the

    information-theoretic, task independent utility function "Empowerment", which

    is defined as the channel capacity between an agent''s actions and an agent''s

    sensors. It quantifies how much influence and control an agent has over the

    world it can perceive. This book chapter discusses the general idea behind

    empowerment as an intrinsic motivation and showcases several previous

    applications of empowerment to demonstrate how empowerment can be applied to

    different sensor-motor configuration, and how the same formalism can lead to

    different observed behaviors. Furthermore, we also present a fast approximation

    for empowerment in the continuous domain.'
  - "Deep Convolutional Networks on Graph-Structured Data   Deep Learning's recent\
    \ successes have mostly relied on Convolutional\nNetworks, which exploit fundamental\
    \ statistical properties of images, sounds\nand video data: the local stationarity\
    \ and multi-scale compositional structure,\nthat allows expressing long range\
    \ interactions in terms of shorter, localized\ninteractions. However, there exist\
    \ other important examples, such as text\ndocuments or bioinformatic data, that\
    \ may lack some or all of these strong\nstatistical regularities.\n  In this paper\
    \ we consider the general question of how to construct deep\narchitectures with\
    \ small learning complexity on general non-Euclidean domains,\nwhich are typically\
    \ unknown and need to be estimated from the data. In\nparticular, we develop an\
    \ extension of Spectral Networks which incorporates a\nGraph Estimation procedure,\
    \ that we test on large-scale classification\nproblems, matching or improving\
    \ over Dropout Networks with far less parameters\nto estimate."
  - 'Factorization tricks for LSTM networks   We present two simple ways of reducing
    the number of parameters and

    accelerating the training of large Long Short-Term Memory (LSTM) networks: the

    first one is "matrix factorization by design" of LSTM matrix into the product

    of two smaller matrices, and the second one is partitioning of LSTM matrix, its

    inputs and states into the independent groups. Both approaches allow us to

    train large LSTM networks significantly faster to the near state-of the art

    perplexity while using significantly less RNN parameters.'
- source_sentence: Images are processed so that their mean and variance are 0 and
    1 respectively. cTraining Data cSee [CITE], the Datasheet in Appendix , Appendix
    , Appendix cQuantitative Analyses Unitary Results & sets a new state of the art
    in few-shot learning on a wide range of open-ended vision and language tasks.
  sentences:
  - "Spatially-sparse convolutional neural networks   Convolutional neural networks\
    \ (CNNs) perform well on problems such as\nhandwriting recognition and image classification.\
    \ However, the performance of\nthe networks is often limited by budget and time\
    \ constraints, particularly when\ntrying to train deep networks.\n  Motivated\
    \ by the problem of online handwriting recognition, we developed a\nCNN for processing\
    \ spatially-sparse inputs; a character drawn with a one-pixel\nwide pen on a high\
    \ resolution grid looks like a sparse matrix. Taking advantage\nof the sparsity\
    \ allowed us more efficiently to train and test large, deep CNNs.\nOn the CASIA-OLHWDB1.1\
    \ dataset containing 3755 character classes we get a test\nerror of 3.82%.\n \
    \ Although pictures are not sparse, they can be thought of as sparse by adding\n\
    padding. Applying a deep convolutional network using sparsity has resulted in\
    \ a\nsubstantial reduction in test error on the CIFAR small picture datasets:\
    \ 6.28%\non CIFAR-10 and 24.30% for CIFAR-100."
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
  - 'Training GANs with Optimism   We address the issue of limit cycling behavior
    in training Generative

    Adversarial Networks and propose the use of Optimistic Mirror Decent (OMD) for

    training Wasserstein GANs. Recent theoretical results have shown that

    optimistic mirror decent (OMD) can enjoy faster regret rates in the context of

    zero-sum games. WGANs is exactly a context of solving a zero-sum game with

    simultaneous no-regret dynamics. Moreover, we show that optimistic mirror

    decent addresses the limit cycling problem in training WGANs. We formally show

    that in the case of bi-linear zero-sum games the last iterate of OMD dynamics

    converges to an equilibrium, in contrast to GD dynamics which are bound to

    cycle. We also portray the huge qualitative difference between GD and OMD

    dynamics with toy examples, even when GD is modified with many adaptations

    proposed in the recent literature, such as gradient penalty or momentum. We

    apply OMD WGAN training to a bioinformatics problem of generating DNA

    sequences. We observe that models trained with OMD achieve consistently smaller

    KL divergence with respect to the true underlying distribution, than models

    trained with GD variants. Finally, we introduce a new algorithm, Optimistic

    Adam, which is an optimistic variant of Adam. We apply it to WGAN training on

    CIFAR10 and observe improved performance in terms of inception score as

    compared to Adam.'
- source_sentence: Taking the convolution into account, the procedure can also be
    written as , from which we can observe that applying attention on features is
    equivalent to performing convolution with dynamic weights. Other implementations
    for attention modules have also been developed, including using standard deviation
    to provide more statistics [CITE], or replacing FC layers with efficient 1D convolutions
    [CITE].
  sentences:
  - 'WES: Agent-based User Interaction Simulation on Real Infrastructure   We introduce
    the Web-Enabled Simulation (WES) research agenda, and describe

    FACEBOOK''s WW system. We describe the application of WW to reliability,

    integrity and privacy at FACEBOOK , where it is used to simulate social media

    interactions on an infrastructure consisting of hundreds of millions of lines

    of code. The WES agenda draws on research from many areas of study, including

    Search Based Software Engineering, Machine Learning, Programming Languages,

    Multi Agent Systems, Graph Theory, Game AI, and AI Assisted Game Play. We

    conclude with a set of open problems and research challenges to motivate wider

    investigation.'
  - 'In-Domain GAN Inversion for Real Image Editing   Recent work has shown that a
    variety of semantics emerge in the latent space

    of Generative Adversarial Networks (GANs) when being trained to synthesize

    images. However, it is difficult to use these learned semantics for real image

    editing. A common practice of feeding a real image to a trained GAN generator

    is to invert it back to a latent code. However, existing inversion methods

    typically focus on reconstructing the target image by pixel values yet fail to

    land the inverted code in the semantic domain of the original latent space. As

    a result, the reconstructed image cannot well support semantic editing through

    varying the inverted code. To solve this problem, we propose an in-domain GAN

    inversion approach, which not only faithfully reconstructs the input image but

    also ensures the inverted code to be semantically meaningful for editing. We

    first learn a novel domain-guided encoder to project a given image to the

    native latent space of GANs. We then propose domain-regularized optimization by

    involving the encoder as a regularizer to fine-tune the code produced by the

    encoder and better recover the target image. Extensive experiments suggest that

    our inversion method achieves satisfying real image reconstruction and more

    importantly facilitates various image editing tasks, significantly

    outperforming start-of-the-arts.'
  - 'Learning a Multi-View Stereo Machine   We present a learnt system for multi-view
    stereopsis. In contrast to recent

    learning based methods for 3D reconstruction, we leverage the underlying 3D

    geometry of the problem through feature projection and unprojection along

    viewing rays. By formulating these operations in a differentiable manner, we

    are able to learn the system end-to-end for the task of metric 3D

    reconstruction. End-to-end learning allows us to jointly reason about shape

    priors while conforming geometric constraints, enabling reconstruction from

    much fewer images (even a single image) than required by classical approaches

    as well as completion of unseen surfaces. We thoroughly evaluate our approach

    on the ShapeNet dataset and demonstrate the benefits over classical approaches

    as well as recent learning based methods.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on allenai/scibert_scivocab_uncased
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: validation
      type: validation
    metrics:
    - type: pearson_cosine
      value: 0.4765481095603642
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.438605482602799
      name: Spearman Cosine
---

# SentenceTransformer based on allenai/scibert_scivocab_uncased

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased) <!-- at revision 24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1 -->
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
    'Taking the convolution into account, the procedure can also be written as , from which we can observe that applying attention on features is equivalent to performing convolution with dynamic weights. Other implementations for attention modules have also been developed, including using standard deviation to provide more statistics [CITE], or replacing FC layers with efficient 1D convolutions [CITE].',
    'In-Domain GAN Inversion for Real Image Editing   Recent work has shown that a variety of semantics emerge in the latent space\nof Generative Adversarial Networks (GANs) when being trained to synthesize\nimages. However, it is difficult to use these learned semantics for real image\nediting. A common practice of feeding a real image to a trained GAN generator\nis to invert it back to a latent code. However, existing inversion methods\ntypically focus on reconstructing the target image by pixel values yet fail to\nland the inverted code in the semantic domain of the original latent space. As\na result, the reconstructed image cannot well support semantic editing through\nvarying the inverted code. To solve this problem, we propose an in-domain GAN\ninversion approach, which not only faithfully reconstructs the input image but\nalso ensures the inverted code to be semantically meaningful for editing. We\nfirst learn a novel domain-guided encoder to project a given image to the\nnative latent space of GANs. We then propose domain-regularized optimization by\ninvolving the encoder as a regularizer to fine-tune the code produced by the\nencoder and better recover the target image. Extensive experiments suggest that\nour inversion method achieves satisfying real image reconstruction and more\nimportantly facilitates various image editing tasks, significantly\noutperforming start-of-the-arts.',
    "WES: Agent-based User Interaction Simulation on Real Infrastructure   We introduce the Web-Enabled Simulation (WES) research agenda, and describe\nFACEBOOK's WW system. We describe the application of WW to reliability,\nintegrity and privacy at FACEBOOK , where it is used to simulate social media\ninteractions on an infrastructure consisting of hundreds of millions of lines\nof code. The WES agenda draws on research from many areas of study, including\nSearch Based Software Engineering, Machine Learning, Programming Languages,\nMulti Agent Systems, Graph Theory, Game AI, and AI Assisted Game Play. We\nconclude with a set of open problems and research challenges to motivate wider\ninvestigation.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.4942, 0.4022],
#         [0.4942, 1.0000, 0.5470],
#         [0.4022, 0.5470, 1.0000]])
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

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.4765     |
| **spearman_cosine** | **0.4386** |

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
  | details | <ul><li>min: 9 tokens</li><li>mean: 71.32 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 27 tokens</li><li>mean: 220.57 tokens</li><li>max: 468 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.18</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>These approaches commonly condition on goals [CITE] or reward values [CITE], but they can also involve reweighting or filtering [CITE].</code>                                                      | <code>Advantage-Weighted Regression: Simple and Scalable Off-Policy<br>  Reinforcement Learning   In this paper, we aim to develop a simple and scalable reinforcement learning<br>algorithm that uses standard supervised learning methods as subroutines. Our<br>goal is an algorithm that utilizes only simple and convergent maximum<br>likelihood loss functions, while also being able to leverage off-policy data.<br>Our proposed approach, which we refer to as advantage-weighted regression<br>(AWR), consists of two standard supervised learning steps: one to regress onto<br>target values for a value function, and another to regress onto weighted target<br>actions for the policy. The method is simple and general, can accommodate<br>continuous and discrete actions, and can be implemented in just a few lines of<br>code on top of standard supervised learning methods. We provide a theoretical<br>motivation for AWR and analyze its properties when incorporating off-policy<br>data from experience replay. We evaluate AWR on a suite of standard ...</code> | <code>1.0</code> |
  | <code>For benchmarking datasets from [CITE] we followed the most commonly used parameter budgets: up to 500k parameters for ZINC, PATTERN, and CLUSTER; and 100k parameters for MNIST and CIFAR10.</code> | <code>Transformer Dissection: A Unified Understanding of Transformer's<br>  Attention via the Lens of Kernel   Transformer is a powerful architecture that achieves superior performance on<br>various sequence learning tasks, including neural machine translation, language<br>understanding, and sequence prediction. At the core of the Transformer is the<br>attention mechanism, which concurrently processes all inputs in the streams. In<br>this paper, we present a new formulation of attention via the lens of the<br>kernel. To be more precise, we realize that the attention can be seen as<br>applying kernel smoother over the inputs with the kernel scores being the<br>similarities between inputs. This new formulation gives us a better way to<br>understand individual components of the Transformer's attention, such as the<br>better way to integrate the positional embedding. Another important advantage<br>of our kernel-based formulation is that it paves the way to a larger space of<br>composing Transformer's attention. As an example, we p...</code> | <code>0.0</code> |
  | <code>As in [CITE], we collaborate closely with labelers over the course of the project.</code>                                                                                                           | <code>Subword Regularization: Improving Neural Network Translation Models with<br>  Multiple Subword Candidates   Subword units are an effective way to alleviate the open vocabulary problems<br>in neural machine translation (NMT). While sentences are usually converted into<br>unique subword sequences, subword segmentation is potentially ambiguous and<br>multiple segmentations are possible even with the same vocabulary. The question<br>addressed in this paper is whether it is possible to harness the segmentation<br>ambiguity as a noise to improve the robustness of NMT. We present a simple<br>regularization method, subword regularization, which trains the model with<br>multiple subword segmentations probabilistically sampled during training. In<br>addition, for better subword sampling, we propose a new subword segmentation<br>algorithm based on a unigram language model. We experiment with multiple<br>corpora and report consistent improvements especially on low resource and<br>out-of-domain settings.</code>                                  | <code>0.0</code> |
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
| 0.4237 | 500  | 2.6996        | 0.3560                     |
| 0.8475 | 1000 | 2.6585        | 0.4310                     |
| 1.0    | 1180 | -             | 0.3880                     |
| 1.2712 | 1500 | 2.5268        | 0.3592                     |
| 1.6949 | 2000 | 2.4348        | 0.4294                     |
| 2.0    | 2360 | -             | 0.4114                     |
| 2.1186 | 2500 | 2.3508        | 0.4236                     |
| 2.5424 | 3000 | 2.0927        | 0.4386                     |


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