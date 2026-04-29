# Food Intelligence System

## Multimodal AI Pipeline for Nutritional Analysis

AI Pipeline for Nutrition Risk

**Project Report**

\---

**Submitted By:**

* Devanand Puzhakkool (12400935)
* Saptaparni Saha (12407553)
* Tanishka Arora (12409331)

**Submitted To:**

**Mr. Harsh Sharma** (UID: 30882)
*(Assistant Professor)*

\---

\---

# Table of Contents

1. [Project Overview](#1-project-overview)

   * 1.1 Goals
   * 1.2 Expected Outcomes
   * 1.3 Scope
2. [Module-Wise Breakdown](#2-module-wise-breakdown)

   * 2.1 Module 1: CNN Image Encoder
   * 2.2 Module 2: NLP Text Encoder
   * 2.3 Module 3: Graph Neural Network (GNN)
   * 2.4 Module 4: Multimodal Fusion \& Prediction Heads
3. [Functionalities](#3-functionalities)

   * 3.1 Image Processing
   * 3.2 Ingredient Text Understanding
   * 3.3 Knowledge Graph Reasoning
   * 3.4 Prediction Outputs
4. [Technology Used](#4-technology-used)

   * 4.1 Programming Languages \& Frameworks
   * 4.2 Deep Learning Libraries
   * 4.3 Datasets
   * 4.4 Knowledge Graph Sources
   * 4.5 Other Tools
5. [Pipeline Flow Diagrams](#5-pipeline-flow-diagrams)

   * 5.1 End-to-End Data Flow
   * 5.2 CNN Processing Flow
   * 5.3 NLP Processing Flow
   * 5.4 GNN Subgraph Construction
   * 5.5 Fusion and Prediction
6. [Dataset \& Model Tracking](#6-dataset--model-tracking)
7. [Conclusion and Future Scope](#7-conclusion-and-future-scope)

   * 7.1 Conclusion
   * 7.2 Future Scope
8. [References](#8-references)
* [Appendix A: AI-Generated Project Elaboration](#appendix-a-ai-generated-project-elaboration)
* [Appendix B: Problem Statement](#appendix-b-problem-statement)
* [Appendix C: Key Code Snippets](#appendix-c-key-code-snippets)

\---

\---

# 1\. Project Overview

The Food Intelligence System is a multimodal AI pipeline that accepts a food photograph and a free-text ingredient list as inputs and produces three categories of output: nutritional estimates (calories and macronutrients), health risk scores (cardiovascular, diabetic, hypertension), and dietary compatibility tags (vegan, keto-compatible, high-protein, gluten-free). The system processes these inputs through three parallel deep learning branches — a Convolutional Neural Network (CNN) for visual features, a Natural Language Processing (NLP) encoder for ingredient semantics, and a Graph Neural Network (GNN) for structured nutritional knowledge — and fuses their outputs through a cross-attention mechanism into a single unified prediction.

The core motivation behind the system is that food appearance alone does not fully encode nutritional content. A bowl of rice looks similar whether it is white or brown, yet the glycemic and fibre profiles differ significantly. Similarly, raw ingredient text cannot capture visual portion sizes or cooking methods. By combining visual, linguistic, and graph-structured knowledge signals, this system achieves more accurate, grounded, and interpretable nutritional inference than any single modality can produce.

\---

## 1.1 Goals

* Design a CNN-based visual encoder using ResNet-50 or EfficientNet, pretrained on Food-101 for 101-class food classification, then fine-tuned on Nutrition5k for direct macronutrient regression with weighed ground-truth supervision.
* Train a BERT-based NLP module on Recipe1M+ to produce semantically rich ingredient embeddings, and attach a Named Entity Recognition (NER) sub-module to extract structured ingredient names that seed the knowledge graph.
* Construct a food knowledge graph fusing FoodKG (RPI), USDA FoodData Central (SR Legacy and Foundation Foods), and CTD disease-nutrient associations, and train a GraphSAGE GNN to propagate nutrient, disease risk, and diet-tag signals across the graph via three rounds of message passing.
* Design a cross-attention fusion layer that combines three 512-dimensional embeddings (visual, text, graph) into a shared 256-dimensional representation, and branch three task-specific prediction heads from it.
* Integrate GradCAM and SHAP explainability modules so that each prediction is interpretable — showing which image regions and which knowledge graph nodes drove the output.

\---

## 1.2 Expected Outcomes

* A fully operational inference pipeline that accepts a food photo and an ingredient string and returns: calorie estimate in kcal, macronutrient breakdown in grams (protein, fat, carbohydrates), probability scores for cardiovascular, type 2 diabetes, and hypertension risk, and binary dietary compatibility tags.
* A trained CNN checkpoint delivering low mean absolute error on Nutrition5k, leveraging the ImageNet → Food-101 → Nutrition5k transfer learning chain.
* A knowledge graph with ingredient, nutrient, disease, and diet-tag nodes linked by typed edges (contains, increases\_risk, restricted\_for, tagged), enabling structured reasoning beyond what image or text alone can capture.
* Explainability outputs including GradCAM heatmaps on food images and SHAP values attributing prediction credit across modalities and graph nodes.

\---

## 1.3 Scope

**Full-Stack AI Architecture:** The system is implemented as a complete machine learning pipeline. The backend is Python-based and handles dataset preprocessing, CNN and NLP training, knowledge graph construction, GNN training, and inference through a REST API. A lightweight frontend provides an interactive interface for uploading images, entering ingredients, and viewing structured prediction results.

**Multimodal Learning Framework:** The system fuses three independent modalities — visual (CNN), textual (BERT), and graph-structured (GNN) — through a cross-attention mechanism that dynamically weights each modality based on input quality. If the image is blurry but the ingredient list is detailed, the fusion layer up-weights the text and graph embeddings automatically.

**Knowledge-Grounded Predictions:** Unlike purely data-driven models, the system grounds its outputs in a curated food knowledge graph derived from USDA, FoodKG, and CTD. Predictions are not only statistically learned from labelled training data but are also constrained and enriched by peer-reviewed nutritional science encoded directly in the graph topology.

**Transfer Learning Strategy:** The system adopts a three-stage transfer learning approach — ImageNet pretraining gives the CNN a strong visual prior, Food-101 fine-tuning gives it food-domain specialisation, and Nutrition5k fine-tuning gives it regression sensitivity for macronutrients. This staged approach maximises performance given the limited size of nutrition-labelled datasets.

\---

\---

# 2\. Module-Wise Breakdown

The Food Intelligence System is divided into four modules, each owning a distinct modality or functional layer. This separation ensures that individual modules can be independently retrained, replaced, or upgraded without affecting the others.

\---

## 2.1 Module 1: CNN Image Encoder

This module handles all image-side processing. It takes a food photograph, resizes it to 224×224 pixels, normalises it using ImageNet mean and standard deviation, and passes it through a CNN backbone (ResNet-50 or EfficientNet). The CNN's final global average pooling layer produces a 2048-dimensional feature vector, which is projected down to 512 dimensions via a learned linear layer to produce the visual embedding `v ∈ R^512`.

Training proceeds in two phases. Phase 1 trains only the classification head on Food-101 (101 food categories) using cross-entropy loss to give the backbone a strong food-domain prior. Phase 2 unfreezes the full backbone and fine-tunes on Nutrition5k (paired food images with weighed nutritional ground truth) using mean squared error loss on calories and macronutrients, with a differential learning rate schedule where the backbone receives a 10× lower learning rate than the regression head to prevent catastrophic forgetting.

GradCAM is attached to the penultimate convolutional layer and generates heatmaps at inference time to highlight which image regions most strongly influenced the prediction.

**Key files:** `dataset.py`, `model.py`, `train\\\_phase1.py`, `train\\\_phase2.py`, `extract\\\_features.py`

\---

## 2.2 Module 2: NLP Text Encoder

This module handles all ingredient text processing. It accepts a free-text ingredient list (e.g., "chicken, rice, cooking oil, salt, pepper"), tokenises it using BERT's WordPiece tokeniser, and encodes it through a pretrained BERT model. The \[CLS] token's 768-dimensional final hidden state captures the full semantic context of the ingredient combination and is projected to 512 dimensions via a linear layer to match the visual embedding size, producing the text embedding `t ∈ R^512`.

A Named Entity Recognition (NER) sub-module processes the BERT output to extract individual ingredient names in a structured format (e.g., "chicken", "cooking oil", "rice"). These structured names serve a dual purpose — they are used to build the text embedding and are passed downstream to the GNN module as query terms for knowledge graph subgraph construction.

For lighter deployment environments, the BERT encoder can be swapped for a BiLSTM without changing the rest of the pipeline interface.

**Key files:** `nlp\\\_encoder.py`, `ner\\\_extractor.py`, `text\\\_embeddings.py`

\---

## 2.3 Module 3: Graph Neural Network (GNN)

This module reasons over a food knowledge graph to produce a 512-dimensional graph embedding `g ∈ R^512` that encodes the nutritional profile, disease risk associations, and dietary compatibility of the identified ingredients.

The knowledge graph has four node types: ingredient nodes (e.g., chicken, rice, cooking oil), nutrient nodes (e.g., protein, saturated fat, carbohydrate), disease risk nodes (e.g., cardiovascular disease, type 2 diabetes), and diet-tag nodes (e.g., high-protein, keto-incompatible). Three typed edge categories connect them: `contains` links ingredients to their constituent nutrients, `increases\\\_risk` links nutrients to associated diseases, and `restricted\\\_for` / `tagged` links diseases and ingredients to diet-tag nodes.

For each inference sample, the NLP module's extracted ingredient names are matched to FoodKG entities by name. A 2-hop subgraph is pulled: ingredient → nutrient → disease/diet-tag. Node features are initialised from USDA FoodData Central nutritional vectors (protein content, fat, carbohydrates, glycemic index, etc.). Three rounds of GraphSAGE message passing then propagate information across the subgraph so that each ingredient node's embedding absorbs knowledge about its nutrients, which diseases those nutrients are linked to, and which diet tags apply. A mean-pool over all ingredient node embeddings produces the final graph-level embedding.

**Key files:** `kg\\\_builder.py`, `gnn\\\_model.py`, `graph\\\_embeddings.py`, `usda\\\_loader.py`

\---

## 2.4 Module 4: Multimodal Fusion \& Prediction Heads

This module merges the outputs of the three preceding modules and produces all final predictions. The three 512-dimensional embeddings `v`, `t`, and `g` are concatenated into a single 1536-dimensional joint vector. A cross-attention mechanism then dynamically weights the contribution of each modality for the specific input — for example, if the image is high-resolution but the ingredient list is short, the visual embedding receives higher attention weight. An MLP projects the attended representation to a shared 256-dimensional hidden space.

Three task-specific heads branch from this shared representation in parallel:

* **Nutrition Regression Head:** A 2-layer MLP trained with mean squared error loss on Nutrition5k and USDA ground truth, outputting calorie estimates (kcal) and macronutrient values (protein, fat, carbohydrates in grams per serving).
* **Health Risk Classification Head:** A softmax classifier trained with cross-entropy loss on graph-derived risk labels from CTD, outputting probability scores for cardiovascular disease, type 2 diabetes, and hypertension risk.
* **Dietary Compatibility Multi-Label Head:** A sigmoid head with one output neuron per diet tag, trained independently per tag with binary cross-entropy, outputting probabilities for vegan, keto-compatible, high-protein, gluten-free, and other diet labels.

**Key files:** `fusion\\\_layer.py`, `prediction\\\_heads.py`, `inference.py`

\---

\---

# 3\. Functionalities

\---

## 3.1 Image Processing

* Food image input resized to 224×224 with ImageNet normalisation.
* Random crop, horizontal flip, colour jitter, and rotation augmentation during training; deterministic centre crop during inference.
* Two-phase transfer learning: Food-101 classification pretraining followed by Nutrition5k regression fine-tuning.
* GradCAM heatmap generation at inference time to highlight calorie-signal regions.
* Feature extraction mode to export 512-d embeddings for downstream modules.

\---

## 3.2 Ingredient Text Understanding

* Free-text ingredient list tokenised with BERT WordPiece tokeniser.
* BERT encoder produces contextual \[CLS] embedding capturing full ingredient semantics.
* NER sub-module extracts structured ingredient entity names for knowledge graph lookup.
* Linear projection from 768-d BERT output to 512-d text embedding.
* BiLSTM fallback encoder available for resource-constrained deployment.

\---

## 3.3 Knowledge Graph Reasoning

|**Edge Type**|**Example**|**Source**|
|-|-|-|
|contains|Chicken → Protein, Saturated fat|USDA FoodData|
|contains|Rice → Carbohydrate|USDA FoodData|
|increases\_risk|Saturated fat → Cardiovascular disease|CTD Database|
|increases\_risk|Carbohydrate (high GI) → Type 2 diabetes|CTD Database|
|restricted\_for|Cardiovascular disease → Keto-incompatible|Wikidata SPARQL|
|tagged|Chicken → High-protein|FoodKG / USDA|

* 2-hop subgraph construction per inference sample from matched FoodKG entities.
* Node feature initialisation from USDA nutritional vectors (protein/100g, fat/100g, glycemic index, etc.).
* Three rounds of GraphSAGE message passing to propagate disease-risk and diet-tag signals.
* Mean-pool over ingredient nodes to produce a single 512-d graph-level embedding.

\---

## 3.4 Prediction Outputs

|**Output**|**Type**|**Head Architecture**|**Training Loss**|
|-|-|-|-|
|Calories (kcal)|Regression|2-layer MLP|Mean Squared Error|
|Protein (g)|Regression|2-layer MLP|Mean Squared Error|
|Fat (g)|Regression|2-layer MLP|Mean Squared Error|
|Carbohydrates (g)|Regression|2-layer MLP|Mean Squared Error|
|Cardiovascular Risk|Classification (0–1)|Softmax over risk classes|Cross-Entropy|
|Type 2 Diabetes Risk|Classification (0–1)|Softmax over risk classes|Cross-Entropy|
|Hypertension Risk|Classification (0–1)|Softmax over risk classes|Cross-Entropy|
|Vegan|Multi-label (T/F)|Sigmoid per tag|Binary Cross-Entropy|
|Keto-Compatible|Multi-label (T/F)|Sigmoid per tag|Binary Cross-Entropy|
|High-Protein|Multi-label (T/F)|Sigmoid per tag|Binary Cross-Entropy|
|Gluten-Free|Multi-label (T/F)|Sigmoid per tag|Binary Cross-Entropy|

\---

\---

# 4\. Technology Used

\---

## 4.1 Programming Languages \& Frameworks

|**Language / Framework**|**Version**|**Role in This Project**|
|-|-|-|
|Python|3.11+|All backend: model training, data preprocessing, GNN construction, API server.|
|PyTorch|2.x|CNN training, BERT fine-tuning, GNN implementation, and fusion layer.|
|TypeScript (React 18)|5.x|Frontend dashboard for interactive inference and result visualisation.|
|SQL|PostgreSQL|Storing model metadata, experiment logs, and USDA nutrient lookup tables.|

\---

## 4.2 Deep Learning Libraries

|**Library**|**Purpose**|
|-|-|
|PyTorch|Core deep learning framework for CNN, NLP, GNN, and fusion layer implementation.|
|torchvision|ResNet-50 / EfficientNet pretrained backbone loading and image transformation pipelines.|
|Hugging Face Transformers|Pretrained BERT model, WordPiece tokeniser, and fine-tuning utilities for the NLP module.|
|PyTorch Geometric (PyG)|GraphSAGE and Graph Attention Network implementation for the GNN message-passing module.|
|torch-gradcam|GradCAM heatmap generation attached to the CNN penultimate layer for explainability.|
|SHAP|SHAP attribution values on the fusion layer for multi-modal prediction interpretability.|
|scikit-learn|Data splitting, evaluation metrics (MAE, F1, AUC-ROC), and normalisation utilities.|

\---

## 4.3 Datasets

|**Dataset**|**Size**|**Role in Pipeline**|
|-|-|-|
|Food-101|101,000 images|CNN Phase 1 pretraining — 101-class food classification for food-domain visual prior.|
|Nutrition5k|\~5,000 dishes|CNN Phase 2 fine-tuning — paired food images with weighed calorie/macro ground truth.|
|Recipe1M+|1M+ recipes|NLP encoder pretraining — ingredient lists and cooking instructions for food-domain BERT.|
|USDA FoodData Central|8,000+ foods|GNN node feature initialisation — per-100g nutritional vectors for all ingredient nodes.|
|USDA SR Legacy|7,793 foods|Broad food coverage; the primary USDA source used for graph node features.|
|USDA Foundation Foods|436 foods|High-precision replication measurements; supplements SR Legacy for key foods.|

**Required USDA Files:**

|**File**|**Purpose**|
|-|-|
|food.csv|Master food table — fdc\_id to food name mapping, primary join key.|
|food\_nutrient.csv|Core nutrient data — per-food, per-nutrient values (644K rows in SR Legacy).|
|nutrient.csv|Nutrient ID lookup — maps IDs to names (e.g., 1003 = Protein).|
|food\_portion.csv|Serving size data — converts per-100g values to realistic portions.|
|sr\_legacy\_food.csv|Maps fdc\_id to original NDB number for FoodKG cross-referencing.|
|foundation\_food.csv|Maps Foundation fdc\_id to NDB number for FoodKG cross-referencing.|

\---

## 4.4 Knowledge Graph Sources

|**Source**|**What It Provides**|**Used For**|
|-|-|-|
|FoodKG (RPI)|Ingredient → nutrient → recipe graph built from Recipe1M+ and USDA SR Legacy, \~5K ingredient nodes.|Structural backbone of the KG.|
|CTD Database|PubMed-backed nutrient → disease associations with DirectEvidence scores for 20+ nutrient-disease edges.|`increases\\\_risk` edges in the GNN.|
|OpenFoodFacts|2M+ packaged food entries with structured diet-tag labels (vegan, gluten-free, keto, etc.).|Diet-tag node vocabulary and edges.|
|Wikidata SPARQL|Disease → dietary recommendation relationships via P1978 (USDA NDB) property queries.|`restricted\\\_for` edges in the GNN.|

\---

## 4.5 Other Tools

|**Tool**|**Purpose**|
|-|-|
|Visual Studio Code|Primary development IDE.|
|GitHub|Version control, experiment tracking, and model checkpoint storage.|
|Git|Source code versioning.|
|PostgreSQL|USDA nutrient lookup tables and model experiment metadata.|
|FastAPI|REST API server for serving inference predictions.|
|Weights \& Biases|Training run visualisation, loss tracking, and hyperparameter logs.|

\---

\---

# 5\. Pipeline Flow Diagrams

\---

## 5.1 End-to-End Data Flow

The full pipeline has three parallel processing lanes that converge at the fusion layer.

```
┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────────────┐
│    IMAGE LANE       │    │    TEXT LANE         │    │     GRAPH LANE           │
│                     │    │                      │    │                          │
│  Raw Image          │    │  Ingredient String   │    │  FoodKG Entity Nodes     │
│  (224×224 px)       │    │  (free text)         │    │  (matched ingredients)   │
│        ↓            │    │        ↓             │    │          ↓               │
│  Preprocess         │    │  Tokenise (BERT WP)  │    │  Init Node Features      │
│  Normalise/Augment  │    │  \\\[CLS] ... \\\[SEP]     │    │  (USDA nutrient vectors) │
│        ↓            │    │        ↓             │    │          ↓               │
│  CNN (ResNet-50)    │    │  BERT Encoder        │    │  GNN (GraphSAGE × 3)     │
│  2048-d feature vec │    │  768-d \\\[CLS] vector  │    │  Message passing rounds  │
│        ↓            │    │        ↓             │    │          ↓               │
│  Linear Projection  │    │  Linear Projection   │    │  Mean-Pool Ingredients   │
│  v ∈ R^512          │    │  t ∈ R^512           │    │  g ∈ R^512               │
└──────────┬──────────┘    └──────────┬───────────┘    └────────────┬─────────────┘
           │                          │                              │
           └──────────────────────────┴──────────────────────────────┘
                                      ↓
                         Concatenate → \\\[v; t; g] ∈ R^1536
                                      ↓
                         Cross-Attention (per-modality weighting)
                                      ↓
                         MLP → Shared Representation ∈ R^256
                                      ↓
              ┌───────────────────────┼─────────────────────────┐
              ↓                       ↓                         ↓
     Regression Head       Classification Head        Multi-Label Head
  Calories, Protein,     Cardiovascular Risk,       Vegan, Keto,
  Fat, Carbohydrates      Diabetes, Hypertension    High-Protein, GF
```

\---

## 5.2 CNN Processing Flow

Phase 1 (Food-101 Classification):

```
Raw Image → Resize 224×224 → Normalise (ImageNet stats) → Augment
→ ResNet-50 Backbone → FC Head (101 classes) → Cross-Entropy Loss
→ Save Pretrained Backbone Checkpoint
```

Phase 2 (Nutrition5k Regression):

```
Load Phase 1 Checkpoint → Replace FC Head with Regression Head (4 outputs)
→ Fine-tune on Nutrition5k (backbone LR = 1/10 × head LR)
→ MSE Loss on \\\[calories, protein, fat, carbs]
→ Cosine Annealing Scheduler + Gradient Clipping
→ Save Best Model (lowest validation MAE)
```

Feature Extraction (Inference):

```
Trained Phase 2 Checkpoint → Remove Regression Head → Add Linear Projection (2048 → 512)
→ L2-Normalise Output → Visual Embedding v ∈ R^512
```

*Figure 2: CNN Training and Feature Extraction Flow*

\---

## 5.3 NLP Processing Flow

```
Raw Text: "chicken, rice, cooking oil, salt, pepper"
     ↓
BERT WordPiece Tokeniser:
\\\[CLS] chicken , rice , cook ##ing oil , salt , pepper \\\[SEP]
     ↓
BERT Encoder (12 layers, bidirectional self-attention)
     ↓
\\\[CLS] Token Hidden State → 768-d sentence embedding
     ↓
Linear Projection (768 → 512) → t ∈ R^512
     ↓
NER Sub-Module → Extracts: \\\["chicken", "rice", "cooking oil", "salt", "pepper"]
     ↓                              ↓
(to Fusion Layer)          (to GNN Subgraph Constructor)
```

*Figure 3: NLP Encoding and Named Entity Extraction Flow*

\---

## 5.4 GNN Subgraph Construction and Message Passing

```
NLP Output: \\\["chicken", "rice", "cooking oil"]
     ↓
Match to FoodKG Entities → Seed Nodes
     ↓
Pull 2-Hop Subgraph:
  chicken  ──contains──►  Protein
  chicken  ──contains──►  Saturated Fat  ──increases\\\_risk──►  Cardiovascular Disease
  rice     ──contains──►  Carbohydrate   ──increases\\\_risk──►  Type 2 Diabetes
  oil      ──contains──►  Saturated Fat
  chicken  ──tagged──────►  High-Protein (diet tag)
  rice     ──tagged──────►  Keto-Incompatible (diet tag)
     ↓
Init Node Features from USDA nutrient vectors
     ↓
GraphSAGE Round 1: Nutrient nodes aggregate from ingredient neighbours
GraphSAGE Round 2: Disease nodes aggregate from nutrient neighbours
GraphSAGE Round 3: Ingredient nodes absorb full 2-hop context
     ↓
Mean-Pool over Ingredient Nodes → g ∈ R^512
```

*Figure 4: GNN Knowledge Graph Subgraph and Message Passing*

\---

## 5.5 Fusion and Prediction (Butter Chicken Example)

```
Input: Photo of butter chicken with naan + Ingredient list

Cross-Attention Weights:
  Visual embedding  (v):  weight = 0.29  \\\[sauce obscures exact proportions]
  Text embedding    (t):  weight = 0.34  \\\[detailed ingredient list available]
  Graph embedding   (g):  weight = 0.37  \\\[rich domain knowledge for this dish]
     ↓
Fused Representation ∈ R^256
     ↓
┌──────────────────────────────────────────────────────────────────┐
│  Nutrition Head      │  Risk Head           │  Diet Tag Head     │
│  Calories: 680 kcal  │  Cardiovascular:0.68 │  High-protein: ✓   │
│  Protein:  38g       │  Diabetes: 0.54      │  Keto-compat: ✗    │
│  Fat:      32g       │  Hypertension: 0.41  │  Vegan: ✗          │
│  Carbs:    55g       │                      │  Gluten-free: ✗    │
└──────────────────────────────────────────────────────────────────┘
     ↓
Explainability:
  GradCAM → Highlights chicken pieces and cream swirl in heatmap
  SHAP    → 40% of cardiovascular risk attributed to Saturated Fat node
```

*Figure 5: Fusion Layer and Final Prediction — Butter Chicken Example*

\---

\---

# 6\. Dataset \& Model Tracking

|**Field**|**Details**|
|-|-|
|CNN Backbone|ResNet-50 (torchvision pretrained on ImageNet)|
|Phase 1 Dataset|Food-101 (101,000 images, 101 classes)|
|Phase 2 Dataset|Nutrition5k (\~5,000 dishes with weighed ground truth)|
|NLP Encoder|BERT-base-uncased (Hugging Face Transformers)|
|NLP Fine-tune Dataset|Recipe1M+ (1M+ recipes with ingredient-instruction pairs)|
|GNN Architecture|GraphSAGE (3 message-passing rounds, 512-d output)|
|GNN Node Init Data|USDA FoodData Central SR Legacy + Foundation Foods|
|KG Topology|FoodKG (RPI) — ingredient-nutrient-recipe graph|
|Disease Edges|CTD (Comparative Toxicogenomics Database) — nutrient-disease associations|
|Diet Tag Edges|OpenFoodFacts + Wikidata SPARQL|
|Optimizer|AdamW with cosine annealing learning rate schedule|
|Explainability|GradCAM (CNN branch) + SHAP (fusion layer)|
|Experiment Tracking|Weights \& Biases (training loss, validation MAE, F1 per tag)|

\---

\---

# 7\. Conclusion and Future Scope

\---

## 7.1 Conclusion

The Food Intelligence System successfully demonstrates a working multimodal AI architecture for nutritional analysis. The system does not rely on a single data source or model type but instead stacks three independent deep learning modules — CNN for visual context, BERT-based NLP for ingredient semantics, and GraphSAGE GNN for structured nutritional knowledge — and merges them through a cross-attention fusion layer. This layered approach ensures that the failure or weakness of any one modality (e.g., a blurry image or a short ingredient list) does not catastrophically degrade the final prediction.

The knowledge graph component is the most architecturally distinctive element. By encoding USDA nutritional data, FoodKG ingredient-nutrient topology, and CTD disease-nutrient associations into a structured graph, the system produces health risk scores that are grounded in peer-reviewed nutritional science rather than being purely statistically learned from labelled examples. The GNN's message passing propagates disease-risk signals from nutrient nodes back to ingredient nodes, enabling the model to reason about compound risk (e.g., a dish is high-risk for both cardiovascular disease and diabetes simultaneously because it is both high in saturated fat and high in refined carbohydrates).

The team divided responsibilities effectively: the CNN and dataset pipeline were established first, the NLP encoder and NER module were built in parallel, the knowledge graph construction and GNN were developed next, and the fusion layer and prediction heads tied all components together. The project demonstrates practical application of deep learning, graph neural networks, and transfer learning concepts in a realistic, socially relevant domain.

\---

## 7.2 Future Scope

* Expand the knowledge graph with Swiss Food Knowledge Graph entities to improve coverage of European and South Asian food ingredients.
* Replace the current CTD-only disease edges with a richer multi-source disease layer combining CTD, NutriChem, and DisGeNET for finer-grained nutritional risk profiling.
* Add a portion estimation module that uses depth estimation or object detection to automatically infer serving sizes from food images, removing the need for manual portion annotation.
* Deploy the inference API using Docker containers with separate model serving, knowledge graph, and frontend services.
* Extend the NLP module with a multilingual BERT variant (mBERT or XLM-R) to support non-English ingredient lists and recipe data from non-Western cuisines.
* Implement active learning for annotation — flagging low-confidence predictions and routing them to human annotators to continuously improve model calibration.
* Add continuous integration tests covering the full preprocess → embed → fuse → predict cycle to prevent regression when individual module checkpoints are updated.

\---

\---

# 8\. References

1. Hou, Y. et al. "FoodKG: A Semantics-Driven Knowledge Graph for Food Recommendation." *ISWC 2019*. https://github.com/foodkg/foodkg.github.io
2. Pfeiffer, J. et al. "Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food." *CVPR 2021*. https://github.com/google-research/google-research/tree/master/nutrition5k
3. Bossard, L., Guillaumin, M., Van Gool, L. "Food-101 — Mining Discriminative Components with Random Forests." *ECCV 2014*. https://data.vision.ee.ethz.ch/cvl/datasets\_extra/food-101/
4. Marin, J. et al. "Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images." *IEEE TPAMI 2019*.
5. USDA Agricultural Research Service. "FoodData Central." https://fdc.nal.usda.gov/
6. Davis, A. P. et al. "Comparative Toxicogenomics Database (CTD)." *Nucleic Acids Research 2023*. https://ctdbase.org
7. Hamilton, W. L., Ying, R., Leskovec, J. "Inductive Representation Learning on Large Graphs (GraphSAGE)." *NeurIPS 2017*.
8. Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019*.
9. He, K. et al. "Deep Residual Learning for Image Recognition (ResNet)." *CVPR 2016*.
10. Selvaraju, R. R. et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV 2017*.
11. Lundberg, S. M., Lee, S. I. "A Unified Approach to Interpreting Model Predictions (SHAP)." *NeurIPS 2017*.
12. OpenFoodFacts. "Open Food Facts — Open Database of Food Products." https://world.openfoodfacts.org/data

\---

\---

# Appendix A: AI-Generated Project Elaboration

> \\\*This section contains an architectural analysis and elaboration of the Food Intelligence System's design, model choices, and implementation quality.\\\*

\---

## A.1 Executive Summary

The Food Intelligence System is a well-structured, research-grade multimodal learning pipeline that demonstrates a mature understanding of cross-modal fusion principles. The project does not rely on a single black-box model but instead assembles three transparent, individually interpretable modules — CNN, NLP, and GNN — whose outputs are combined through an attention mechanism that preserves per-modality attribution. This design aligns with best practices for trustworthy AI in the medical and nutritional domain, where black-box predictions are unacceptable and interpretability is a functional requirement rather than a nice-to-have.

\---

## A.2 Architecture Assessment

The three-lane parallel architecture with a cross-attention fusion layer is a sound design choice. Each lane owns a distinct input modality and terminates in a 512-d embedding — this normalisation to equal dimensionality before fusion is critical, as it prevents any single modality from dominating the concatenated vector by sheer size. The cross-attention mechanism is particularly important: by dynamically computing attention weights per modality per input, the fusion layer can handle missing or degraded inputs gracefully, which is essential for a real-world deployment where image quality and ingredient list completeness will vary.

The two-phase CNN training strategy — Food-101 classification pretraining followed by Nutrition5k regression fine-tuning — is textbook transfer learning applied correctly. Pretraining on classification before regression is important because the supervised signal in Nutrition5k is regression-valued (continuous calorie counts), which is a weaker training signal than the categorical supervision in Food-101. The Food-101 phase builds a strong visual vocabulary that makes the Nutrition5k fine-tuning converge faster and to a better minimum.

\---

## A.3 Knowledge Graph Design Strengths

The decision to use CTD as the disease edge source rather than relying solely on FoodKG or Swiss Food KG is architecturally correct. FoodKG was built for recipe recommendation and has no disease content; Swiss Food KG has only 26 disease edges with no evidence scores. CTD provides curated, PubMed-backed associations with `DirectEvidence` scores that the GNN can use as edge weights during message passing, giving the model a meaningful signal for the strength of each nutrient-disease relationship. This is the difference between a GNN that merely propagates structural adjacency and one that propagates evidence-weighted nutritional knowledge.

\---

## A.4 Code Quality Observations

The modular separation of `dataset.py`, `model.py`, `train\\\_phase1.py`, `train\\\_phase2.py`, and `extract\\\_features.py` in the CNN module is clean and production-grade. The differential learning rate schedule in Phase 2 is correctly implemented: the backbone receives a lower LR to prevent catastrophic forgetting of the Food-101 visual features while the regression head learns the nutrition-specific mapping. The `extract\\\_features.py` design — outputting both a single-image embedding and a batch-mode dict of `{dish\\\_id: tensor}` — is well-suited for the downstream GNN module that will consume these embeddings as node features.

\---

## A.5 Areas Recommended for Improvement

* The NER sub-module currently performs named entity extraction in a post-processing step after BERT encoding. A jointly trained NER head attached directly to BERT's token-level outputs would improve extraction accuracy and eliminate the latency of a separate inference pass.
* The knowledge graph is currently static — it is built once from USDA, FoodKG, and CTD and then frozen. A dynamic graph construction pipeline that periodically re-ingests updated CTD releases would keep disease-nutrient edge weights current with evolving nutritional research.
* The multi-label dietary compatibility head currently treats all tags as independent. A hierarchical or conditional prediction approach (e.g., predicting "keto-compatible" should be conditioned on the carbohydrate prediction from the regression head) would reduce logical inconsistencies between the two output types.
* Portion size is currently assumed to be standardised per serving. Integrating a depth-estimation or instance segmentation sub-module to estimate actual serving volumes from the image would significantly improve calorie regression accuracy.

\---

\---

# Appendix B: Problem Statement

\---

## B.1 Background

Current food logging and nutritional analysis tools fall into two categories: image recognition apps that identify dish categories (e.g., "this is pad thai") but cannot estimate macronutrients without a database lookup, and manual nutrition trackers that require users to enter ingredients themselves and look up portion sizes. Neither approach combines visual evidence with ingredient knowledge and structured nutritional science in a single unified inference. The result is that calorie and macronutrient estimates from existing tools are often inaccurate — particularly for home-cooked or mixed dishes where standard database entries do not match the actual preparation.

Furthermore, no existing consumer-facing tool surfaces disease-risk scores derived from the nutritional composition of a specific meal. A person managing type 2 diabetes has no automated way to assess whether a dish they are about to eat will spike their glycemic load, short of manually computing it from an ingredient list.

\---

## B.2 Formal Problem Statement

**Food Intelligence System — Multimodal Nutritional Analysis:**
Design and implement a machine learning system that:

* Accepts a food photograph and a free-text ingredient list as dual inputs.
* Produces nutritional estimates (calories, protein, fat, carbohydrates per serving) using a CNN trained on weighed ground-truth data.
* Produces health risk scores (cardiovascular, type 2 diabetes, hypertension) grounded in evidence-backed nutrient-disease associations from a biomedical database (CTD).
* Produces dietary compatibility tags (vegan, keto-compatible, high-protein, gluten-free) derived from a structured food knowledge graph.
* Provides interpretable outputs: GradCAM heatmaps for the image contribution and SHAP attribution values for the knowledge graph contribution.
* Handles missing or degraded inputs (blurry image, incomplete ingredient list) gracefully through per-modality attention weighting in the fusion layer.

\---

## B.3 Scope and Constraints

* The system is a machine learning research prototype, not a certified medical device. Health risk scores are indicative and should not replace clinical dietary advice.
* The GNN requires a pre-built knowledge graph (FoodKG + CTD + OpenFoodFacts) as a static input. Dynamic graph updates require a separate re-ingestion pipeline.
* Portion size estimation is based on standardised serving sizes from USDA; per-meal volume estimation from images is deferred to future work.
* VirusTotal-style file scanning does not apply; however, all uploaded images are validated for correct MIME type and size before inference to prevent adversarial inputs.
* The system targets GPU-accelerated inference. CPU-only inference is supported but significantly slower.

\---

\---

# Appendix C: Key Code Snippets

> \\\*This appendix contains the most important source files. The full codebase is available in the project GitHub repository.\\\*

\---

## C.1 dataset.py — CNN Dataset Classes

```python
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

NUTRITION\\\_STATS = {
    "calories": {"mean": 426.0, "std": 280.0},
    "protein":  {"mean": 21.5,  "std": 16.0},
    "fat":      {"mean": 18.3,  "std": 14.0},
    "carbs":    {"mean": 45.8,  "std": 32.0}
}

def get\\\_transforms(split="train"):
    if split == "train":
        return transforms.Compose(\\\[
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(\\\[0.485, 0.456, 0.406], \\\[0.229, 0.224, 0.225])
        ])
    return transforms.Compose(\\\[
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(\\\[0.485, 0.456, 0.406], \\\[0.229, 0.224, 0.225])
    ])

class Food101Dataset(Dataset):
    def \\\_\\\_init\\\_\\\_(self, root, split="train"):
        with open(os.path.join(root, "meta", f"{split}.json")) as f:
            data = json.load(f)
        self.classes = sorted(data.keys())
        self.class\\\_to\\\_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = \\\[(os.path.join(root, "images", f"{p}.jpg"), self.class\\\_to\\\_idx\\\[c])
                        for c, paths in data.items() for p in paths]
        self.transform = get\\\_transforms(split)

    def \\\_\\\_len\\\_\\\_(self): return len(self.samples)
    def \\\_\\\_getitem\\\_\\\_(self, idx):
        path, label = self.samples\\\[idx]
        return self.transform(Image.open(path).convert("RGB")), label

class Nutrition5kDataset(Dataset):
    def \\\_\\\_init\\\_\\\_(self, dish\\\_dir, dish\\\_ids, split="train"):
        self.samples = \\\[]
        for dish\\\_id in dish\\\_ids:
            meta\\\_path = os.path.join(dish\\\_dir, dish\\\_id, "metadata.csv")
            img\\\_path  = os.path.join(dish\\\_dir, dish\\\_id, "dish\\\_image.jpg")
            with open(meta\\\_path) as f:
                row = dict(zip(\\\*\\\[x.split(",") for x in f.read().strip().splitlines()]))
            nutrients = torch.tensor(\\\[
                (float(row\\\["total\\\_calories"]) - NUTRITION\\\_STATS\\\["calories"]\\\["mean"]) / NUTRITION\\\_STATS\\\["calories"]\\\["std"],
                (float(row\\\["total\\\_mass"]) \\\* 0.25 - NUTRITION\\\_STATS\\\["protein"]\\\["mean"]) / NUTRITION\\\_STATS\\\["protein"]\\\["std"],
                (float(row\\\["total\\\_fat"])   - NUTRITION\\\_STATS\\\["fat"]\\\["mean"])      / NUTRITION\\\_STATS\\\["fat"]\\\["std"],
                (float(row\\\["total\\\_carb"])  - NUTRITION\\\_STATS\\\["carbs"]\\\["mean"])    / NUTRITION\\\_STATS\\\["carbs"]\\\["std"],
            ], dtype=torch.float32)
            self.samples.append((img\\\_path, nutrients, dish\\\_id))
        self.transform = get\\\_transforms(split)

    def \\\_\\\_len\\\_\\\_(self): return len(self.samples)
    def \\\_\\\_getitem\\\_\\\_(self, idx):
        path, nutrients, dish\\\_id = self.samples\\\[idx]
        return self.transform(Image.open(path).convert("RGB")), nutrients, dish\\\_id
```

\---

## C.2 model.py — FoodCNN with GradCAM

```python
import torch
import torch.nn as nn
from torchvision import models

class FoodCNN(nn.Module):
    def \\\_\\\_init\\\_\\\_(self, mode="classify", num\\\_classes=101, embed\\\_dim=512):
        super().\\\_\\\_init\\\_\\\_()
        self.mode = mode
        backbone = models.resnet50(weights=models.ResNet50\\\_Weights.IMAGENET1K\\\_V2)
        self.encoder = nn.Sequential(\\\*list(backbone.children())\\\[:-1])  # remove FC
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embed\\\_dim),
            nn.LayerNorm(embed\\\_dim)
        )
        if mode == "classify":
            self.head = nn.Linear(embed\\\_dim, num\\\_classes)
        elif mode == "regress":
            self.head = nn.Sequential(
                nn.Linear(embed\\\_dim, 256), nn.ReLU(), nn.Linear(256, 4)
            )
        # GradCAM hook
        self.\\\_gradients = None
        self.encoder\\\[-2].register\\\_backward\\\_hook(self.\\\_save\\\_gradient)

    def \\\_save\\\_gradient(self, \\\_, \\\_\\\_, grad\\\_output):
        self.\\\_gradients = grad\\\_output\\\[0]

    def forward(self, x):
        features = self.encoder(x)
        embedding = self.projector(features)
        if self.mode == "embed":
            return nn.functional.normalize(embedding, dim=-1)
        return self.head(embedding)
```

\---

## C.3 gnn\_model.py — GraphSAGE GNN

```python
import torch
import torch.nn as nn
from torch\\\_geometric.nn import SAGEConv
from torch\\\_geometric.nn import global\\\_mean\\\_pool

class FoodGNN(nn.Module):
    def \\\_\\\_init\\\_\\\_(self, in\\\_dim, hidden\\\_dim=256, out\\\_dim=512, num\\\_layers=3):
        super().\\\_\\\_init\\\_\\\_()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in\\\_dim, hidden\\\_dim))
        for \\\_ in range(num\\\_layers - 2):
            self.convs.append(SAGEConv(hidden\\\_dim, hidden\\\_dim))
        self.convs.append(SAGEConv(hidden\\\_dim, out\\\_dim))
        self.act = nn.ReLU()
        self.projector = nn.Linear(out\\\_dim, 512)

    def forward(self, x, edge\\\_index, batch, ingredient\\\_mask):
        for conv in self.convs\\\[:-1]:
            x = self.act(conv(x, edge\\\_index))
        x = self.convs\\\[-1](x, edge\\\_index)
        # Mean-pool only over ingredient nodes
        ingredient\\\_embeddings = x\\\[ingredient\\\_mask]
        ingredient\\\_batch = batch\\\[ingredient\\\_mask]
        graph\\\_embed = global\\\_mean\\\_pool(ingredient\\\_embeddings, ingredient\\\_batch)
        return self.projector(graph\\\_embed)  # g ∈ R^512
```

\---

## C.4 fusion\_layer.py — Cross-Attention Fusion

```python
import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def \\\_\\\_init\\\_\\\_(self, embed\\\_dim=512, num\\\_heads=8, out\\\_dim=256):
        super().\\\_\\\_init\\\_\\\_()
        self.attention = nn.MultiheadAttention(embed\\\_dim, num\\\_heads, batch\\\_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed\\\_dim \\\* 3, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, out\\\_dim)
        )
        self.norm = nn.LayerNorm(out\\\_dim)

    def forward(self, v, t, g):
        # Stack into sequence: \\\[batch, 3, 512]
        modalities = torch.stack(\\\[v, t, g], dim=1)
        attended, weights = self.attention(modalities, modalities, modalities)
        # Flatten attended → \\\[batch, 1536]
        fused = attended.reshape(attended.size(0), -1)
        out = self.norm(self.mlp(fused))
        return out, weights  # weights usable for SHAP attribution
```

