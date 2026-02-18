# MedSigLIP Fine-tuning on CheXpert + CQ500 — Architectural Rationale

This document explains every significant architectural and methodological choice made in the fine-tuning pipeline, with the primary literature and technical reports that support each decision.

---

## Table of Contents

1. [Model Selection — Why MedSigLIP over separate encoders](#1-model-selection--why-medsiglip-over-separate-encoders)
2. [Loss Function — SigLIP Sigmoid Loss over Softmax Contrastive Loss](#2-loss-function--siglip-sigmoid-loss-over-softmax-contrastive-loss)
3. [Fine-tuning Strategy — LoRA over Full Fine-tuning](#3-fine-tuning-strategy--lora-over-full-fine-tuning)
4. [Encoder Freezing — Vision-Only LoRA, Text Encoder Frozen](#4-encoder-freezing--vision-only-lora-text-encoder-frozen)
5. [Text-to-Image Retrieval Preservation](#5-text-to-image-retrieval-preservation)
6. [CQ500 Head CT Preprocessing — HU Windowing to RGB](#6-cq500-head-ct-preprocessing--hu-windowing-to-rgb)
7. [CQ500 Label Handling — Majority Vote Consensus](#7-cq500-label-handling--majority-vote-consensus)
8. [CQ500 Slice Selection — Middle-Third Median](#8-cq500-slice-selection--middle-third-median)
9. [Combined Dataset Design — ConcatDataset with Shared Interface](#9-combined-dataset-design--concatdataset-with-shared-interface)
10. [LoRA Hyperparameter Choices — Rank, Alpha, Target Modules](#10-lora-hyperparameter-choices--rank-alpha-target-modules)
11. [PyTorch AMP API Update](#11-pytorch-amp-api-update)
12. [HuggingFace Hub Upload — Adapter-Only Strategy](#12-huggingface-hub-upload--adapter-only-strategy)
13. [References](#13-references)

---

## 1. Model Selection — Why MedSigLIP over Separate Encoders

**Previous approach:** Two independently trained encoders — Bio_ClinicalBERT (768d) for text and ResNeXt-50 for images — projected into a shared 512-dimensional space and trained from scratch on CheXpert with a symmetric softmax contrastive loss.

**New approach:** `google/medsiglip-448`, a single unified 400M-parameter vision-language model pretrained by Google on a large corpus of medical image-text pairs.

### Rationale

The foundational problem with training separate encoders from scratch is the cold-start problem: neither encoder begins with any knowledge of the cross-modal alignment task, and the contrastive loss must simultaneously teach the vision encoder to represent pathology, the text encoder to represent clinical language, and the projection heads to align the two spaces. With a dataset the size of CheXpert this is tractable, but produces embeddings that are brittle outside the CheXpert label distribution.

MedSigLIP arrives with all three problems already solved. The MedGemma Technical Report (Sellergren et al., 2025) describes MedSigLIP as a vision encoder derived from SigLIP-400M that has been specifically tuned on medical image-text data spanning chest X-rays, dermatology, ophthalmology, histopathology, CT, and MRI, using the same SigLIP pretraining recipe. The report demonstrates that MedSigLIP achieves performance comparable to or exceeding specialised single-modality medical image encoders in zero-shot classification and retrieval across all four evaluated imaging modalities, without any task-specific training.

Concretely, using MedSigLIP means:

- The vision encoder already understands radiological anatomy and pathological features before a single gradient update on CheXpert.
- The text encoder already maps radiology report language — including findings like "pleural effusion", "cardiomegaly", "pneumothorax" — to embeddings near the corresponding visual features.
- The image resolution (448×448) is higher than the previous pipeline (256×256), providing more detail for subtle findings.
- The preprocessing normalisation to [-1, 1] and the 64-token context limit are handled automatically by `AutoProcessor`, eliminating brittle manual pipelines.
- A single model object simplifies the training loop, checkpointing, and deployment significantly.

The Google Health AI Developer Foundations documentation explicitly recommends MedSigLIP (over MedGemma) for classification, zero-shot, and content-based retrieval tasks — the exact use case here.

---

## 2. Loss Function — SigLIP Sigmoid Loss over Softmax Contrastive Loss

**Previous approach:** Symmetric softmax contrastive loss (InfoNCE / CLIP loss) with a learnable temperature of 0.2.

**New approach:** Pairwise sigmoid loss (SigLIP loss) with learnable `logit_scale` and `logit_bias`.

### Rationale

The standard CLIP / InfoNCE loss treats similarity scoring as a softmax classification problem over the entire batch. For a batch of N pairs, it constructs an N×N similarity matrix and normalises each row and column via softmax, requiring a global all-gather operation across all devices. This creates two practical problems for fine-tuning at Colab scale:

**Memory complexity.** The softmax formulation requires O(N²) memory for the full pairwise similarity matrix. At batch size 16 with 448×448 inputs this is manageable, but the global normalisation still creates a hard constraint — every negative pair in the batch influences every positive pair's gradient, which means the effective quality of the loss degrades sharply as batch size falls below roughly 16–32k pairs (Zhai et al., 2023).

**Small-batch performance.** Zhai et al. (2023) demonstrate empirically that the sigmoid loss significantly outperforms the softmax loss when batch sizes are below 16k. In their experiments, SigLIP achieves peak ImageNet zero-shot performance at a batch size of 32k, whereas the equivalent softmax loss required 98k pairs to reach the same accuracy — and still did not match the sigmoid variant. For our setup with batch size 16, this advantage is decisive.

The sigmoid formulation replaces the global normalisation with an independent pairwise binary classification objective. Each (image, text) pair is evaluated independently: diagonal pairs are labelled +1 and off-diagonal pairs are labelled -1, and the loss is the average binary cross-entropy across all N² pairs under a sigmoid nonlinearity. There is no global normalisation factor, which eliminates the all-gather requirement and reduces memory from O(N²) to O(N). Zhai et al. also demonstrate that sigmoid training increases robustness to label noise, which is relevant given that both CheXpert and CQ500 contain uncertain or ambiguous labels.

The two learnable parameters `logit_scale` and `logit_bias` — which MedSigLIP already contains from pretraining — replace the single temperature parameter of the softmax formulation. The bias term, initialised to -10 during SigLIP pretraining, acts as a prior that suppresses spurious positive signals in the early iterations. Zhai et al. show that enabling this bias term consistently improves results across architectures and batch sizes.

Using the SigLIP loss is also architecturally consistent: MedSigLIP was pretrained with this loss, so its `logit_scale` and `logit_bias` are already calibrated to the embedding geometry. Switching to a softmax loss at fine-tuning time would create an inconsistency between pretraining and fine-tuning objectives, potentially destabilising the learned alignment.

---

## 3. Fine-tuning Strategy — LoRA over Full Fine-tuning

**Alternative considered:** Full fine-tuning of all 800M parameters (400M vision + 400M text).

**Chosen approach:** LoRA (Low-Rank Adaptation) applied to the attention projection layers of the vision encoder only, with all base weights frozen.

### Rationale

The central argument for LoRA over full fine-tuning rests on four independent observations that converge on the same conclusion.

#### 3a. The Intrinsic Rank Hypothesis

Hu et al. (2021) observe that over-parametrised neural networks trained on large corpora reside on a low intrinsic dimension — that is, the essential variation in the weight space during adaptation can be captured by a much lower-dimensional subspace than the full parameter count suggests. They formalise this as the intrinsic rank hypothesis: the update matrix ΔW during task-specific fine-tuning also has a low intrinsic rank. LoRA operationalises this by decomposing ΔW = BA, where B ∈ R^{d×r} and A ∈ R^{r×k} with r ≪ min(d, k). The base weight W is frozen; only A and B are trained. Hu et al. demonstrate that LoRA matches or exceeds full fine-tuning on RoBERTa, DeBERTa, GPT-2, and GPT-3 across multiple benchmarks while reducing trainable parameters by up to 10,000× and GPU memory by 3×.

This principle applies directly here. MedSigLIP has already learned a rich, medically-grounded visual feature space from a large and diverse pretraining corpus. The adaptation required to specialise on CheXpert and CQ500 represents a small incremental shift in that space — precisely the low-rank regime where LoRA is most appropriate. Full fine-tuning would update parameters that encode general radiological knowledge irrelevant to the specific downstream task, wasting compute and risking overwriting that knowledge.

#### 3b. Dataset Scale Mismatch

MedSigLIP was pretrained on a corpus that, following the Med-Gemini lineage described in Sellergren et al. (2025), includes tens of millions of medical image-text pairs across multiple modalities. CheXpert contains approximately 224,316 training examples; CQ500 contains 491 scans. The combined fine-tuning dataset is therefore two to three orders of magnitude smaller than the pretraining corpus.

Full fine-tuning on a dataset this small relative to model capacity reliably produces catastrophic forgetting: the model overwrites pretraining representations with patterns specific to the fine-tuning distribution. When the fine-tuning distribution is limited (binary-labelled findings converted to templated text) and the pretraining distribution was broad (diverse natural radiology reports across many institutions and modalities), catastrophic forgetting causes a severe regression in generalisation. LoRA's frozen base weights act as a structural regulariser that prevents this, because the pretraining representations are never modified — only the low-rank residuals are updated.

#### 3c. Compute and Memory Budget

At full precision, the optimizer state for full fine-tuning of 800M parameters requires approximately 6.4 GB for weights, 6.4 GB for first moments, and 6.4 GB for second moments under Adam — totalling ~19 GB before accounting for activations, gradients, or input tensors. A standard Colab T4 GPU has 16 GB of VRAM. Full fine-tuning with AMP reduces this somewhat, but remains on the edge of feasibility and forces very small batch sizes that further degrade contrastive learning quality.

With LoRA at rank 16 applied to the vision encoder's attention projections, trainable parameters total roughly 6–8M (approximately 1–2% of the base model). Optimizer state for these parameters is negligible, freeing the majority of VRAM for larger batch sizes and higher-resolution inputs. This directly improves the sigmoid contrastive loss quality, as more negative pairs per batch provide stronger learning signal.

#### 3d. Inference Efficiency

LoRA adapters can be merged into the base weights at inference time with zero additional latency (`model.eval()` triggers the merge in the PEFT library). The deployed model is structurally identical to the base MedSigLIP model, with no additional memory or computation overhead — unlike adapter-tuning approaches that insert additional bottleneck layers into every forward pass.

---

## 4. Encoder Freezing — Vision-Only LoRA, Text Encoder Frozen

**Choice:** LoRA adapters are applied exclusively to the vision encoder. The text encoder is completely frozen — no LoRA, no direct gradient updates.

### Rationale

This is the most consequential architectural choice in the pipeline, and it is motivated by two separate arguments.

**First, task asymmetry.** The adaptation task here is domain-specific visual retrieval: given a query image (chest X-ray or head CT slice), find the most semantically similar images in a database. The images in CheXpert and CQ500 come from specific equipment configurations, institutions, and populations that may differ from MedSigLIP's pretraining distribution. The vision encoder therefore has a genuine adaptation problem: it must learn to be more discriminative for the specific pathology vocabulary and imaging characteristics present in these datasets.

The text encoder faces no analogous problem. MedSigLIP's text encoder was trained on diverse medical reports and already maps clinical language — "intracranial hemorrhage", "subdural hematoma", "pleural effusion" — to appropriate positions in the joint embedding space. The templated text generated from CheXpert and CQ500 labels is a strict subset of the language MedSigLIP already understands. Updating the text encoder on this narrow, templated distribution would push text embeddings away from the richer language space that makes the model useful for arbitrary query strings.

**Second, the Locked-image Tuning (LiT) analogy.** The LiT paradigm (Zhai et al., 2022), which directly preceded SigLIP, demonstrates that freezing a strong pretrained vision encoder and training only the text encoder produces vision-language models that match or exceed models trained with both encoders free. The inverse — freezing the text encoder and adapting only the vision encoder — applies the same logic in the other direction, and is the correct choice when the text encoder is already strong and the visual domain requires specialisation.

---

## 5. Text-to-Image Retrieval Preservation

**Question:** Will the model retain its ability to retrieve images from free-text queries after fine-tuning?

**Answer:** Yes, by construction.

### Rationale

Text-to-image retrieval works by encoding a query string with the text encoder and finding the nearest image embeddings in the database by cosine similarity. For this to work correctly, both encoders must produce embeddings in the same aligned geometric space.

When the text encoder is completely frozen, its output for any given query string is identical before and after fine-tuning. The sigmoid contrastive loss then pushes the vision encoder to move its image embeddings closer to the corresponding frozen text embeddings — which is exactly the geometry required for text-to-image retrieval. The fine-tuning process geometrically re-centres the image embedding distribution around the fixed text anchor points, improving both image-to-text and text-to-image retrieval simultaneously.

If instead the text encoder were updated (even with LoRA), the text embedding space would drift toward the narrow vocabulary of CheXpert and CQ500 label-generated text. Queries phrased differently from this vocabulary — for example, "consolidation in the right lower lobe" versus "consolidation" — would produce embeddings in different positions than expected, degrading retrieval quality for unseen query phrasing. Freezing the text encoder prevents this entirely.

The symmetric SigLIP loss directly optimises both retrieval directions in each batch: the N positive image-text pairs pull image embeddings toward text embeddings and text embeddings toward image embeddings simultaneously. Since the text encoder contributes no gradients (it is frozen), all gradient flow goes to the vision LoRA adapters and the logit parameters, which move image embeddings toward the existing text embedding anchor points.

---

## 6. CQ500 Head CT Preprocessing — HU Windowing to RGB

**Modality challenge:** CQ500 is a 3D head CT dataset stored in DICOM format. Each study contains 30–100+ axial slices with pixel values in Hounsfield Units (HU), ranging from -1000 (air) to +3000 (dense bone). MedSigLIP expects 448×448 RGB images normalised to [-1, 1].

**Preprocessing pipeline:**
1. Load DICOM slice with `pydicom`; apply RescaleSlope and RescaleIntercept to obtain true HU values.
2. Apply three standard neuroradiology windows and stack as RGB channels:
   - R channel — Brain window (centre=40 HU, width=80 HU)
   - G channel — Subdural window (centre=75 HU, width=215 HU)
   - B channel — Bone window (centre=600 HU, width=2800 HU)
3. Pass the resulting PIL RGB image to `AutoProcessor` for resizing and normalisation.

### Rationale for Three-Window RGB Stacking

The human visual cortex (and, by extension, a vision transformer trained on RGB images) can distinguish approximately 20 shades of grey. The raw HU range of a head CT spans approximately 4,000 units — meaning that a single-window grayscale representation displays only 0.5% of the available diagnostic information at any given moment. Radiologists compensate by reviewing scans at multiple window settings in sequence, mentally integrating the information. Three-channel RGB stacking is the established deep learning equivalent of this multi-window review workflow.

The specific window values used are directly from the CQ500/ICH deep learning literature. Chilamkurthy et al. (2018), the original qure.ai paper that introduced the CQ500 dataset, used a three-window preprocessing stack as the basis for their detection model. Multiple subsequent papers working with CQ500 and the RSNA ICH dataset replicate this approach:

- Sage et al. (2020) use brain (WL=40, WW=80), subdural (WL=100, WW=200), and bone (WL=600, WW=2800) as their three channels for a double-branch ResNet-50 on CQ500.
- Karki et al. (2020) demonstrate that combining multiple window settings as CNN input channels outperforms any single window setting including the full HU range, for ICH detection.
- The European Radiology Experimental study (2023) independently validates the three-window preprocessing on CQ500, finding that windowed input consistently outperforms unwindowed DICOM input.
- Plos ONE (2025) study applying HU-RGB transformation to CQ500 confirms that the brain (WL=40, WW=80), subdural (WL=75–100, WW=200–215), and bone (WL=600, WW=2800) window combination is the most widely adopted preprocessing convention for head CT deep learning.

The MedGemma Technical Report (Sellergren et al., 2025) confirms that Google's own preprocessing for CT images in the MedGemma/MedSigLIP training pipeline uses three RGB channels corresponding to different HU windows — specifically bone/lung, soft tissue, and brain windows — validating that this preprocessing is compatible with the model's expected input distribution.

The bone window is particularly important for calvarial fracture detection (one of the CQ500 labels), which is invisible at brain window settings. The subdural window maximises contrast for subdural and epidural hematomas. The brain window is optimal for intraparenchymal and subarachnoid hemorrhage detection. Stacking all three ensures that every pathology category in the CQ500 label set is visually represented in the input.

---

## 7. CQ500 Label Handling — Majority Vote Consensus

**Label structure:** The `reads.csv` file contains three independent radiologist readings for each of 13 findings per scan (e.g. `R1:ICH`, `R2:ICH`, `R3:ICH`). Each reading is binary (0/1). There is no single pre-computed ground truth label.

**Strategy:** A finding is labelled positive for a given scan if at least 2 of 3 radiologists marked it present (majority vote).

### Rationale

The CQ500 ground truth methodology as described by Chilamkurthy et al. (2018) explicitly defines consensus of three independent radiologists as the gold standard. Individual reader agreement is high but not perfect — inter-rater reliability data published alongside the dataset shows kappa values in the range of 0.7–0.9 for most findings. A single reader's label therefore has meaningful error.

Majority vote (≥2/3 readers agreeing) is the statistically appropriate aggregation method for this label structure. It is more conservative than any single reader and reduces false positive rate relative to union (any reader positive), while remaining sensitive — unlike requiring all three readers to agree. The original qure.ai evaluation uses this same consensus definition. Using majority vote ensures our training labels are as close as possible to the ground truth that was used to validate the model that generated the published AUC scores for CQ500.

---

## 8. CQ500 Slice Selection — Middle-Third Median

**Challenge:** Each CQ500 study contains between 30 and 100+ axial DICOM slices. The model processes a single 2D image. Which slice should be selected as the representative 2D input?

**Strategy:** Sort slices by InstanceNumber (axial position); extract the subset spanning the middle third of the volume (index range [n//3 : 2n//3]); return the median slice of that range.

### Rationale

The first and last thirds of a head CT volume typically contain the vertex (top of skull) and the posterior fossa/brainstem respectively. Most of the pathology captured by the CQ500 label set — intracranial hemorrhage subtypes, midline shift, mass effect — is concentrated in the supratentorial compartment, which corresponds broadly to the middle third of the axial stack. This is consistent with clinical practice: when screening a head CT for hemorrhage, radiologists begin review at the basal ganglia level and work toward the vertex, with the most time spent on the middle sections.

For the purpose of this pipeline, where a single representative slice is needed for each scan-level label, the middle-third median provides a simple, deterministic, and clinically reasonable choice without requiring pathology-guided slice selection (which would require a separate localisation model). In a production pipeline, a dedicated slice selector trained to find the most diagnostically informative slice for each label type would replace this heuristic.

---

## 9. Combined Dataset Design — ConcatDataset with Shared Interface

**Design:** Both `CheXpertDataset` and `CQ500Dataset` return dictionaries with identical keys (`input_ids`, `attention_mask`, `pixel_values`, `report`, `img_path`). They are combined with `torch.utils.data.ConcatDataset` and fed to a single `DataLoader`.

### Rationale

The two datasets differ substantially in modality (chest X-ray vs. head CT), anatomy (thorax vs. brain), imaging physics, and label vocabulary. However, from the perspective of the contrastive training loop, both are simply (image, report) pairs: a 2D image encoded by the vision encoder and a free-text description encoded by the text encoder, aligned via the SigLIP loss.

This shared interface is justified for two reasons. First, MedSigLIP was itself pretrained on a heterogeneous mixture of medical imaging modalities, so its vision encoder already possesses modality-invariant medical visual features. Fine-tuning on a mixed-modality dataset is consistent with the pretraining distribution and avoids modality-specific over-fitting. Second, the LoRA adapters learn only low-rank incremental updates, which are naturally regularised toward generality by the low-rank constraint — they cannot specialise too strongly to either modality without losing performance on the other.

The 80/20 train/validation split applied to CQ500 (which has no official split) is separated before any training begins, with a fixed random seed (42) to ensure reproducibility.

---

## 10. LoRA Hyperparameter Choices — Rank, Alpha, Target Modules

**Configuration:** `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`.

### Rationale

**Rank (r=16):** Hu et al. (2021) conduct extensive ablations over rank values from 1 to 64 and find that performance is surprisingly robust across this range — the intrinsic dimensionality of the adaptation task is low. Rank 16 is the most widely used default in the PEFT literature and represents a good balance between expressivity (higher rank can capture more complex adaptations) and regularisation (lower rank reduces overfitting risk). For a dataset of ~224K samples, rank 8 would likely suffice; rank 16 provides a small margin of additional capacity without meaningful overfitting risk given the frozen base weights.

**Alpha (lora_alpha=32):** LoRA scales the adapter output by `alpha/r` before adding it to the frozen weight. Setting `alpha = 2r` (here 32 = 2×16) gives a scaling factor of 2.0, which is a standard convention. This means the adapter's effective learning rate is 2× the base learning rate passed to the optimizer. Hu et al. note that tuning alpha is less critical than tuning r, and recommend keeping the ratio alpha/r at approximately 1–2.

**Dropout (0.05):** A small dropout on the LoRA layers provides mild regularisation without meaningfully reducing adapter capacity. This is especially relevant for CQ500, whose 391 training scans are a very small sample.

**Target modules (attention projections only):** Hu et al. ablate which weight matrices to adapt and find that targeting the attention query and value projections captures most of the adaptation benefit. Including key and output projections (`k_proj`, `out_proj`) adds marginal parameters for potentially useful additional capacity. Feed-forward layers are excluded: they encode factual world knowledge, which should not change for a domain-specific visual adaptation task.

---

## 11. PyTorch AMP API Update

**Change:** `torch.cuda.amp.autocast()` → `torch.amp.autocast("cuda")`

### Rationale

`torch.cuda.amp.autocast()` was deprecated in PyTorch 2.0 and will be removed in a future release. The new device-agnostic API `torch.amp.autocast("device_type")` is the canonical form from PyTorch 2.x onwards. The explicit device argument ensures the code also generalises to `"cpu"` or `"mps"` without modification. The `GradScaler` API was not deprecated and is unchanged.

---

## 12. HuggingFace Hub Upload — Adapter-Only Strategy

**Design:** Only the LoRA adapter weights are uploaded (`model.vision_model.save_pretrained()`), along with `logit_scale`, `logit_bias`, and a model card. The full 800M base model is not uploaded.

### Rationale

The LoRA adapter is the only component that changes during fine-tuning. The base MedSigLIP weights remain frozen and are identical to the published `google/medsiglip-448` weights. Uploading the full model would be redundant, would require significantly more storage, and would be unnecessary for reproduction — users can reconstruct the fine-tuned model by loading the base weights and applying the adapter with `PeftModel.from_pretrained()`. This is the standard distribution pattern for LoRA fine-tunes in the HuggingFace ecosystem and is explicitly supported by the PEFT library's `save_pretrained` / `from_pretrained` interface.

The logit_scale and logit_bias parameters are saved separately as a small `.pt` file because they are not part of the LoRA adapter but are modified during training. Including them ensures that retrieval scores are properly calibrated when the uploaded model is used for inference.

---

## 13. References

| # | Citation |
|---|---|
| [1] | Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). **LoRA: Low-Rank Adaptation of Large Language Models.** *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2106.09685 |
| [2] | Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). **Sigmoid Loss for Language Image Pre-Training.** *International Conference on Computer Vision (ICCV)*. https://arxiv.org/abs/2303.15343 |
| [3] | Sellergren, A., Kazemzadeh, S., Jaroensri, T., Kiraly, A., et al. (2025). **MedGemma Technical Report.** *arXiv preprint arXiv:2507.05201*. https://arxiv.org/abs/2507.05201 |
| [4] | Chilamkurthy, S., Ghosh, R., Tanamala, S., Biviji, M., Campeau, N. G., Venugopal, V. K., Mahajan, V., Rao, P., & Warier, P. (2018). **Development and Validation of Deep Learning Algorithms for Detection of Critical Findings in Head CT Scans.** *arXiv preprint arXiv:1803.05854*. https://arxiv.org/abs/1803.05854 |
| [5] | Sage, A., Badura, P., Malyszczak, A., & Badura, P. (2020). **Intracranial Hemorrhage Detection in Head CT Using Double-Branch Convolutional Neural Network.** *Applied Sciences, 10(21), 7577*. https://doi.org/10.3390/app10217577 |
| [6] | Karki, M., Cho, J., Lee, E., Hahm, M.-H., et al. (2020). **CT Window Trainable Neural Network for Improving Intracranial Hemorrhage Detection.** *Artificial Intelligence in Medicine, 106, 101850*. https://www.sciencedirect.com/science/article/abs/pii/S093336571930939X |
| [7] | McDermott, M., Maguire, G., Murphy, S., et al. (2023). **Evaluation of Techniques to Improve a Deep Learning Algorithm for the Automatic Detection of Intracranial Haemorrhage on CT Head Imaging.** *European Radiology Experimental*. https://link.springer.com/article/10.1186/s41747-023-00330-3 |
| [8] | Tawan, P., et al. (2025). **HU to RGB Transformation with Automatic Windows Selection for Intracranial Hemorrhage Classification.** *PLOS ONE*. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327871 |
| [9] | Google Health AI Developer Foundations. **MedGemma Model Card** (MedSigLIP recommendation for retrieval tasks). https://developers.google.com/health-ai-developer-foundations/medgemma/model-card |
| [10] | Hu, E. J., et al. (2022). **LoRA GitHub Repository.** Microsoft. https://github.com/microsoft/LoRA |
| [11] | HuggingFace. **PEFT Library Documentation.** https://huggingface.co/docs/peft |
| [12] | Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.** *arXiv preprint arXiv:2012.13255*. *(Foundational motivation for the intrinsic rank hypothesis underlying LoRA.)* |
| [13] | Zhai, X., Kolesnikov, A., Houlsby, N., & Beyer, L. (2022). **Scaling Vision Transformers.** *CVPR 2022*. *(LiT / Locked-image Tuning: the precursor to SigLIP demonstrating that freezing a pretrained vision encoder while adapting the text encoder achieves strong results — the inverse of the strategy used here.)* |

---

*This document was generated to accompany the notebook `MedSigLIP_CheXpert_CQ500_LoRA.ipynb`. All claims trace directly to the cited literature or official model documentation.*
