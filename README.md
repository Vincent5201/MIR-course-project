
# Learning Similarity Matrix Representations for Music Tagging via Vision-Language Model

This project introduces an innovative approach to Music Information Retrieval (MIR): **transforming audio signals into geometric texture images—specifically, Self-Similarity Matrices (SSM) and Cross-Similarity Matrices (CSM)—and pioneering their deployment as inputs for Vision-Language Models (VLMs) to align with high-level musical semantics.**
> Below is a summary. Please check the full report for details (available in Chinese).
> This is a final project in nusic information retrival course.
---

##  Motivation & Core Concepts

### 1. Why Transition from Spectrograms to SSM/CSM?

* **Traditional Spectrograms:** These represent first-order statistical descriptions. While excellent at capturing local acoustic details within a time-frequency coordinate system, they struggle to map directly to high-level semantics like genre, instrumentation, or mood, which rely heavily on structural relationships across time rather than isolated local features.
* **Similarity Matrices (SSM/CSM):** These serve as second-order statistical features distilled via similarity functions. They elevate the data representation from "local signal observations" to "relational distributions between segments," embedding rich high-level macro-structures within geometric textures.

### 2. Potential Advantages of Using SSM/CSM as VLM Inputs

* **Natural Feature Fusion:** Images possess an RGB three-channel structure, allowing matrices generated from different spectrograms (e.g., Mel, Chroma, MFCC) to be seamlessly integrated into a single image. This facilitates natural information complementarity.
* **Multi-Scale Perception & Dimensional Flexibility:** Traditional models impose strict constraints on audio length and sample rate. Geometric images, conversely, allow audio segments of varying lengths (e.g., 3s, 10s) to be resized into uniform image dimensions, unlocking powerful multi-scale perception and diverse data augmentation possibilities.
* **Information Compression & Computational Efficiency:** Geometric manipulations—such as cropping and assembling rectangular CSMs or applying matrix dimensionality reduction—enable effective information compression while preserving macro-geometric characteristics.

---

## 🛠️ Framework

### 1. Matrix Generation and Channel Configuration

To simultaneously address **Instrument**, **Genre**, and **Mood** tags for the Music Tagging task, we designed two visualization schemes:

* **Self-Similarity Matrix (SSM) Scheme:**
* **R Channel:** Mel-Spectrogram (assists in identifying instruments and genres).
* **G Channel:** Chroma STFT (preserves harmonic information, assisting in genre and mood identification).
* **B Channel:** MFCC (acts as a timbre summary, assisting in instrument identification).


* **Cross-Similarity Matrix (CSM) Scheme:**
* Computed by measuring similarity against a fixed reference audio evenly distributed from $C_2$ to $C_7$.
* **B Channel Replacement:** MFCC is replaced with **CQT (Constant-Q Transform)** to align pitch structures using a log scale, thereby preserving harmonic and tonal information ideal for genre and mood recognition.



### 2. Model Variants

This project leverages **CLIP (Contrastive Language-Image Pre-training)** as the backbone, implementing the following variants for evaluation:

* **BasicCLIP (SSM 3s / SSM 10s / CSM):** Long audio files are segmented into shorter frames (3s or 10s) and transformed into three-channel images. The text side utilizes standard prompts (e.g., `"This is a music featuring {tag}"`). Image and text embeddings are fed into their respective encoders to calculate cosine similarity for predictions, optimized using Binary Cross-Entropy (BCE) Loss for multi-label classification.
* **MixCLIP (Incorporating Compositional Prompts):**
This variant introduces **compositional prompts** and **cross-modal contrastive learning**. Multiple labels of a single sample are combined into a structured textual description (e.g., `"A [piano] [jazz piece] with a [calm mood]."`). Alongside BCE Loss, **InfoNCE Loss** is introduced to optimize image-text alignment, significantly accelerating model convergence.

---

##  Dataset & Implementation Details

* **Dataset:** MagnaTagATune (Filtered for top 50 tags, with synonymous terms deduplicated and cleaned into 38 final distinct tags spanning Genre, Instrument, Mood, etc.).
* **Evaluation Metrics:** ROC-AUC, PR-AUC, and F1-Score.
* **Latent Space Verification:** t-SNE visualization of the Text Embeddings confirms that the model successfully learns semantic clustering (e.g., grouping quiet/soft or electro/techno together) rather than merely memorizing the training data.

---

##  Limitations & Future Work

1. **Trade-off Between Audio Duration and Physical Meaning:** While 3s and 10s segments are utilized to align with standard music tagging practices, SSM/CSMs typically require longer contexts (e.g., >10s) to capture physically meaningful musical structures. However, longer contexts yield massive image dimensions that challenge hardware limits, presenting a critical trade-off.
2. **Direct Spectrogram-to-VLM Baseline:** Rigorous control experiments are needed to directly compare "feeding cropped spectrograms as images into a VLM" against "using SSM/CSM representations" to isolate the precise feature-level advantages.
3. **Quantifying Computational Savings and Image Operations:** Future steps include designing and evaluating image-level operations (e.g., rotations, flips, geometric splicing) and quantifying their impact on reducing computational overhead and enhancing data diversity.
4. **Investigating the Literature Gap:** Further theoretical exploration is required to understand why utilizing SSM/CSM as geometric textures in VLMs remains largely unexplored in past MIR literature, helping to uncover hidden technical bottlenecks or constraints.
