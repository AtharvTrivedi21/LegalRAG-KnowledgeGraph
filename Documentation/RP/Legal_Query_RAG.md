# Legal Query RAG

**Authors:** Rahman S. M. Wahidur¹, Sumin Kim², Haeung Choi¹, David S. Bhatti¹, and Heung-No Lee¹ *(Senior Member, IEEE)*

¹ School of Electrical Engineering and Computer Science, Gwangju Institute of Science and Technology, Gwangju 61005, South Korea
² Artificial Intelligence Graduate School, Gwangju Institute of Science and Technology, Gwangju 61005, South Korea

**Corresponding author:** Heung-No Lee (heungno@gist.ac.kr)

**DOI:** 10.1109/ACCESS.2025.3542125 | Received: 14 January 2025 | Accepted: 31 January 2025 | Published: 14 February 2025

---

## Abstract

Recently, legal practice has seen a significant rise in the adoption of Artificial Intelligence (AI) for various core tasks. However, these technologies remain in their early stages and face challenges such as understanding complex legal reasoning, managing biased data, ensuring transparency, and avoiding misleading responses, commonly referred to as hallucinations.

To address these limitations, this paper introduces **Legal Query RAG (LQ-RAG)**, a novel Retrieval-Augmented Generation framework with a recursive feedback mechanism specifically designed to overcome the critical shortcomings of standard RAG implementations in legal applications. The proposed framework incorporates four key components:

- A custom evaluation agent
- A specialized response generation model
- A prompt engineering agent
- A fine-tuned legal embedding LLM

Together, these components effectively minimize hallucinations, improve domain-specific accuracy, and deliver precise, high-quality responses for complex queries.

**Key experimental results:**
- Fine-tuned embedding LLM achieves a **13% improvement in Hit Rate** and a **15% improvement in Mean Reciprocal Rank (MRR)**
- **24% performance gain** when using the Hybrid Fine-Tuned Generative LLM (HFM) over general LLMs
- LQ-RAG achieves a **23% improvement in relevance score** over naive configurations
- **14% improvement** over RAG with Fine-Tuned LLMs (FTM)

**Index Terms:** Retrieval-augmented generation, legal query, LLM agent, information retrieval

---

## I. Introduction

Recent advancements in AI and NLP have propelled the development of powerful LLMs leveraging advanced deep learning techniques, transformer architectures, and extensive data. Models like OpenAI GPT and Meta LLaMA demonstrate remarkable versatility across fields including law, medicine, agriculture, coding, and psychology.

However, while proprietary models like BloombergGPT (finance) and Med-PaLM (medicine) have capitalized on domain-specific data, the legal domain has a relatively limited number of reliable LLMs. This scarcity has hindered the digital transformation of the legal sector.

**Key challenges in the legal domain:**
- Legal professionals must navigate complex legal language and nuanced interpretations
- Legislation is ever-evolving, requiring up-to-date information
- LLMs primarily trained on general corpora have limited access to domain-specific resources
- LLMs struggle with expanding parametric memory, leading to hallucinated information
- General-purpose LLMs hallucinate when responding to legal queries at an average rate of **58%–82%**
- High-profile incidents have seen attorneys disciplined for filing documents referencing fabricated AI-generated case law

A promising approach to address these limitations is **Retrieval-Augmented Generation (RAG)**, introduced by Lewis et al., which integrates external data retrieval into the generative process. RAG helps reduce hallucinations and facilitates continuous knowledge updates. However, conventional RAG may introduce irrelevant passages and lacks domain-specific training, resulting in inconsistent responses.

To address these constraints, this paper introduces **LQ-RAG**, which employs a hybrid approach to fine-tune the two principal components of the RAG system: the embedding generation module and the response generation module. These are augmented with chunk references, document hybrid retrieval, multi-document agents, and a recursive feedback evaluation mechanism.

**Key Contributions:**
1. A pioneering RAG framework incorporating agent-driven recursive feedback to refine response quality and precision
2. A custom-built LLM-based evaluation agent that independently assesses response accuracy and triggers regeneration when needed
3. A fine-tuned embedding LLM and a hybrid fine-tuned generative LLM providing enhanced domain adaptation and instruction-following
4. Extensive evaluations demonstrating that LQ-RAG consistently outperforms baseline models in the legal domain

---

## II. Background

### A. Generative LLMs and Embedding LLMs

The advancement of LLMs has given rise to two primary categories:

**Generative LLMs** excel in generating text using causal language modeling (auto-regression / next-token prediction), making them highly effective for producing contextually coherent content.

**Embedding LLMs** transform text into high-dimensional vector spaces, useful for indexing and determining semantic relationships. These excel at identifying semantic similarities between sentences, making them suitable for search engines and recommendation systems.

### B. LLM Fine-Tuning

Fine-tuning adapts a pre-trained language model to enhance its performance in domain-specific applications. Two methods are employed for generative LLMs:

- **Supervised Fine-Tuning (SFT)**
- **Instruction Tuning (IT)**

Benefits of fine-tuning include leveraging pre-training knowledge, reducing the need for labeled data, and enhancing model generalization. Fine-tuning an embedding LLM enriches semantic representation, thereby enhancing retrieval performance and commonly leading to significant improvements in RAG retrieval evaluation metrics.

### C. Retrieval Augmented Generation (RAG)

RAG is an architectural approach to enhance LLM applications using customized data sources. It marginalizes retrieved documents to produce a distribution over generated text through two methods:

**RAG-Sequence** uses the same retrieved document to generate the entire response:

$$p_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-K}(p(\cdot|x))} p_\eta(z|x) \prod_{i=1}^{N} p_\theta(y_i|x,z,y_{1:i-1})$$

**RAG-Token** utilizes multiple retrieved documents to produce an answer:

$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-K}(p(\cdot|x))} p_\eta(z|x) p_\theta(y_i|x,z,y_{1:i-1})$$

Where:
- *x* = input sequence, *y* = target sequence, *z* = retrieved documents
- *N* = target sequence length
- *p_η(z|x)* = retriever with parameters η
- *p_θ(yᵢ|x,z,y₁:ᵢ₋₁)* = generator with parameters θ

**RAG system types:**
- **Naive RAG:** Retrieve-Read framework (indexing → retrieval → generation); suffers from low retrieval precision and hallucinations
- **Advanced RAG:** Addresses shortcomings via refined retrieval, enhanced granularity, and optimized embedding models
- **Modular RAG:** Integrates search modules for similarity retrieval; facilitates adaptable approaches for complex tasks

---

## III. Related Work

### Pre-trained and Fine-tuned Legal LLMs

| Model | Base | Key Features |
|---|---|---|
| HanFei | 700M parameters | Legal Q&A, multi-turn dialogue, article generation, search |
| LawGPT_zh | ChatGLM-6B LoRA | Chinese legal LLM; integrates legal Q&A datasets |
| LawGPT | Chinese-LLaMA-7B | Expanded legal terminology and semantic understanding |
| LexiLaw | ChatGLM-6B | Legal consultation for professionals and general users |
| Lawyer LLaMA | Chinese-LLaMA-13B | Legal counsel, article generation, legal advice |

Despite advances, these models still exhibit hallucinations and biases, and their knowledge cutoff limits their ability to provide current information.

### RAG-based Legal Systems

- **DISC-LawLLM:** Integrates LLMs with a retrieval module to augment access to external legal knowledge
- **CBR-RAG:** Uses Case-Based Reasoning (CBR) to enrich LLM queries with contextual relevance for legal Q&A
- **LexDrafter:** Leverages RAG for drafting definition articles in legislative documents
- **KAB (Knowledge Augmented BERT2BERT):** Combines retrieval-based and generative techniques for Islamic jurisprudential legal questions
- **Hoppe et al.:** Intelligent legal advisor for German documents; BM25 outperforms pre-trained BERT in recall and MAP; fine-tuned DPR excels on GermanQuAD

This research extends prior work by incorporating fine-tuned LLMs with an agent-based RAG solution equipped with a feedback loop.

---

## IV. Proposed System

The LQ-RAG system is organized into two primary parts:

1. **Fine-Tuning (FT) Layer** — fine-tunes both the embedding LLM and the generative LLM
2. **RAG Layer** — integrates advanced RAG modules, an evaluation agent, a prompt engineering agent, and a feedback mechanism

### Fine-Tuning Layer

#### Embedding LLM Fine-Tuning

- **Data source:** Unstructured legal domain corpora (*C_legal*) from Library Genesis
- **Synthetic data generation:** A subset (*C_sub-legal*) is used by OpenAI GPT-3.5-turbo to create query-context pair synthetic dataset (*D_synthetic*)
- **Base model:** GIST Large Embedding v0
- **Loss function:** Multiple Negatives Ranking Loss (MNRL), which minimizes the distance between similar embeddings and maximizes the distance between dissimilar ones:

$$\mathcal{E}(x, y, \theta) = \frac{1}{B} \sum_{i=1}^{B} \left[ S(x_i, y_i) - \log \sum_{j=1}^{B} e^{S(x_i, y_j)} \right]$$

Where *S(xᵢ, yᵢ)* is the positive similarity score and *S(xᵢ, yⱼ)* is the negative similarity score using dot-product scoring.

**Algorithm 1 — Embedding LLM Fine-Tuning:**
```
Input: C_sub-legal
Output: Trained LLM network parameters θ_global

1. Generate D_synthetic from C_sub-legal using LLM
2. Split into D_train and D_eval
3. Initialize baseline pre-trained embedding model M
4. For each epoch:
   For each batch (x, y) in D_train:
     - Forward pass: ŷ = f(x; θ)
     - Compute MNRL loss
     - Backward pass: compute gradients ∇θL
     - Update weights: θ_local ← θ - η·∇θL
   Update θ_global
5. Return θ_global
```

#### Generative LLM Fine-Tuning

- **Base model:** LLaMA-3-8B (general-purpose pre-trained autoregressive LLM)
- **Datasets:** Domain-specific Q&A dataset (*D_QA*) and general-purpose instruction dataset (*D_Instr*)
- **Fine-tuning technique:** Low-Rank Adaptation (LoRA), which replaces weight updates η₀→η' with a smaller parameter set Θ, reducing trainable parameters while preserving performance
- **Objective:** Maximize log-likelihood:

$$\mathcal{G}(\Theta) = \max_{\Theta} \sum_{(x,y)\in D} \sum_{t=1}^{T} \log \left[ p_{\eta'(\Theta)}(y^t | x, y^{1:t-1}) \right]$$

- **Model merging:** The two fine-tuned models (M_QA and M_Instr) are combined via linear merging to create the **Hybrid Fine-Tuned Generative LLM (HFM)**

**Algorithm 2 — Generative LLM Fine-Tuning & Merging:**
```
Input: Training datasets D_QA and D_Instr, Eval dataset D_eval
Output: Merged model M_merged

1. Initialize LoRA parameters (A_l, B_l) for each trainable layer
2. Scale LoRA layers by α
3. Fine-tune separately on D_QA and D_Instr
4. Merge: M_merged ← Linear Merging(M_QA, M_Instr)
```

### RAG Layer

**Data Ingestion:**
- Remaining legal corpora (*C_rem*) are converted to document objects using parallel workers
- Documents are segmented into text chunks and processed through the fine-tuned embedding LLM to generate *d*-dimensional vectors: *E_document ∈ ℝ^(N×d)*
- An index is built using **Facebook AI Similarity Search (FAISS)** and stored in a vector database (*DB_vector*)

**Query Processing:**
1. User query *q* is embedded: *E_query ∈ ℝ^(N×d)*
2. A **ReAct (Reasoning and Action)** agent selects an appropriate query engine tool
3. **Hybrid retrieval** combines BM25 (lexical) and Dense Passage Retrieval (DPR) (semantic) to retrieve top-K relevant passages *C**
4. Passages are re-ranked by a re-ranker to produce *C_re-ranked*
5. A prompt *p* containing system instructions, user query, and retrieved context is fed into the generative LLM to produce initial response *r*

**Evaluation and Feedback Loop:**
- An **evaluation agent** (*A_evaluation*), powered by GPT-4, assesses:
  - **Answer relevance** — does the response address the query?
  - **Context relevance** — is the retrieved context pertinent?
  - **Groundedness** — is the response factually grounded in the retrieved context?
- Uses **Chain-of-Thought (CoT)** reasoning for thorough assessment
- If response meets criteria → output as final response
- If not → query enters feedback loop where a **prompt engineering agent** modifies the query and the process repeats (up to *N* iterations)

**Algorithm 3 — Response Generation Process:**
```
Input: C_rem, User Query q
Output: Final Response r

1. Ingest C_rem → E_document → FAISS index → DB_vector
2. Embed query q → E_query
3. Hybrid Retrieval → C* → Re-Rank → C_re-ranked
4. Generate initial response r = M_g(q, C_re-ranked)
5. Evaluate: Evaluation_result ← A_evaluation(r)
6. If result meets criteria: Return r
7. Else: While not meeting criteria and n ≤ N:
     - Modify query: q_modified ← ModifyQuery(q)
     - Regenerate: r ← M_g(q_modified, C_re-ranked)
     - Re-evaluate
8. Return r
```

---

## V. Tasks, Baseline LLMs, and Evaluation Metrics

### A. Tasks Description

This paper focuses on **six NLP tasks:**

1. Text classification
2. Multiple-choice
3. Sentence completion
4. Complex task understanding
5. Information retrieval
6. Question answering (both open-domain and closed-domain)

### B. Baseline LLMs

**Embedding LLMs:**

| Model | Version | Architecture | Params (B) | Embed. Dim | Intermediate Size |
|---|---|---|---|---|---|
| ColBERT | ColBERTv2 | HF_ColBERT | 0.11 | 768 | 3,072 |
| LLM-Embedder | LLM-Embedder | Bert Model | 0.10 | 1,024 | 3,072 |
| BGE Embedding | Small-en-v1.5 | Bert Model | 0.03 | 384 | 1,536 |
| BGE Embedding | Base-en-v1.5 | Bert Model | 0.10 | 768 | 3,072 |
| BGE Embedding | Large-en-v1.5 | Bert Model | 0.33 | 1,024 | 4,096 |
| GISTEmbed | Small-Embedding-v0 | Bert Model | 0.03 | 384 | 1,536 |
| GISTEmbed | GIST-Embedding-v0 | Bert Model | 0.10 | 768 | 1,536 |
| GISTEmbed | Large-Embedding-v0 | Bert Model | 0.33 | 1,024 | 4,096 |

**LLaMA Models:**

| Model | Architecture | Heads | Layers | Embed. Dim | Params (B) | Pretrain Tokens (B) | Vocab Size |
|---|---|---|---|---|---|---|---|
| LLaMA-2-7B | Decoder only | 32 | 32 | 4,096 | 7 | 2,000 | 32,000 |
| LLaMA-2-13B | Decoder only | 40 | 40 | 5,120 | 13 | 2,000 | 32,000 |
| LLaMA-3-8B | Decoder only | 32 | 32 | 4,096 | 8 | 15,000 | 128,256 |

**FLAN-T5 Models:**

| Model | Architecture | Heads | Layers | Embed. Dim | Params (B) |
|---|---|---|---|---|---|
| FLAN-T5 small | Encoder-decoder | 6 | 8 | 512 | 0.08 |
| FLAN-T5 base | Encoder-decoder | 12 | 12 | 768 | 0.25 |
| FLAN-T5 large | Encoder-decoder | 16 | 24 | 1,024 | 0.78 |
| FLAN-T5 XL | Encoder-decoder | 32 | 24 | 2,048 | 2.85 |

### C. Evaluation Metrics

#### 1. Hit Rate (HR)
Quantifies the ratio of queries in which the correct answer is present among the top-k retrieved documents:

$$\text{HR} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\{d_i \in D_{\text{true}}(q_i)\}$$

#### 2. Mean Reciprocal Rank (MRR)
Evaluates system precision by identifying the highest-ranked relevant document and calculating the mean reciprocal rank across all queries:

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

#### 3. Cosine Similarity (S)
Measures the similarity between two vectors by calculating the cosine of the angle between them:

$$S(\mathbf{C}, \mathbf{A}) = \frac{\mathbf{C} \cdot \mathbf{A}}{\|\mathbf{C}\| \|\mathbf{A}\|}$$

#### 4. Answer Relevance (AR)
Measures the degree to which the generated answer accurately addresses the given query:

$$\text{AR}(Q, A) = \frac{1}{N} \sum_{i=1}^{N} f_{\text{score}}(Q_i, A_i), \quad f_{\text{score}} \in [0, 1]$$

#### 5. Context Relevance (CR)
Measures how well the retrieved context fits the given query:

$$\text{CR}(Q, C) = \frac{1}{N} \sum_{i=1}^{N} f_{\text{score}}(Q_i, C_i)$$

#### 6. Groundedness (G)
Assesses the model's ability to differentiate between factual and hallucinatory input:

$$G(A, C) = \frac{1}{N} \sum_{i=1}^{N} f_{\text{score}}(A_i, C_i)$$

#### 7. Accuracy (Acc)
Evaluates whether the answer contains accurate and verified information:

$$\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN} \times 100\%$$

#### 8. Exact Match (EM)
Checks if the predicted answer exactly matches the true answer:

$$\text{EM}(Q, A) = \begin{cases} 1 & \text{if } A_{\text{pred}} = A_{\text{true}} \\ 0 & \text{otherwise} \end{cases}$$

#### 9. BLEU Score
Evaluates machine-translated text quality using n-gram precision and a brevity penalty:

$$\text{BLEU Score} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

#### 10. ROUGE Score
Measures similarity between generated and reference summaries using overlapping n-grams:

$$\text{ROUGE-N}_R = \frac{\sum_{w \in gen} \min(\text{Count}_{m\text{-gen}}(w), \text{Count}_{\text{ref}}(w))}{\sum_{w \in \text{ref}} \text{Count}_{\text{ref}}(w)}$$

---

## VI. Experiment and Evaluation

### A. Embedding LLM

The GIST Large Embedding v0 model (from Hugging Face) was fine-tuned using the LlamaIndex model fitting API with sentence transformers. Batch sizes of 8 and 10, and epoch sizes from 3 to 15 were tested.

**Before vs. After Fine-Tuning:**
- Average Hit Rate improved by **13%**
- Average MRR improved by **15%**

**GIST-Law-Embed vs. Baseline Models (@ Top K=5):**

| Model | Avg. Hit Rate | Avg. MRR |
|---|---|---|
| ColBERTv2 | 0.3160 | 0.2116 |
| LLM-Embedder | 0.3424 | 0.2430 |
| BGE Embedding small | 0.4129 | 0.3059 |
| BGE Embedding base | 0.4426 | 0.3263 |
| BGE Embedding large | 0.4524 | 0.3433 |
| GISTEmbed small | 0.4207 | 0.3025 |
| GISTEmbed base | 0.4541 | 0.3338 |
| GISTEmbed large | 0.4534 | 0.3369 |
| **GIST-Law-Embed** | **0.5145** | **0.3987** |

GIST-Law-Embed achieved the highest average Hit Rate of **51%** and average MRR of **40%**, outperforming all baselines. Larger model sizes consistently improved retrieval capabilities, and domain-specific tuning provided further gains.

**Effect of Top-K on Retrieval:**
- At k=15, the system successfully retrieves the original passage more than 60% of the time without significantly augmenting the input prompt size
- Increasing retrieved snippets consistently enhances RAG's retrieval from original context

### B. Generative LLM

LLaMA-3-8B was used as the baseline model. Key optimizations included:
- **PEFT** (Parameter-Efficient Fine-Tuning) for memory efficiency
- **4-bit quantization** with BitsAndBytesConfig to reduce model size and improve inference speed
- Early stopping and weight decay for regularization
- SFTTrainer from the `trl` library for training management

**Performance by Group:**

*Group 1 (Reasoning, Commonsense, Language Understanding, Q&A):*

| Model | BBH (EM) | Hellaswag (Acc) | TruthfulQA (BLEU) | SQuAD_v2 (Acc) |
|---|---|---|---|---|
| Flan-T5 XL | 0.35±0.01 | 0.47±0.01 | 0.48±0.02 | 0.03±0.02 |
| LLaMA-3-8B | 0.62±0.01 | 0.60±0.01 | 0.44±0.02 | **0.50±0.02** |
| **HFM** | **0.65±0.01** | **0.62±0.01** | **0.52±0.02** | 0.45±0.02 |

*Group 2 (Legal Domain-Specific Tasks):*

| Model | MMLU Int. Law | MMLU Prof. Law | Abercrombie | LRC | CTCO | CQA |
|---|---|---|---|---|---|---|
| LLaMA-3-8B | 0.77±0.04 | 0.46±0.01 | 0.45±0.05 | 0.52±0.01 | 0.41±0.01 | 0.19±0.01 |
| Flan-T5 large | 0.56±0.05 | 0.34±0.01 | 0.36±0.04 | 0.61±0.01 | **0.68±0.01** | 0.77±0.01 |
| **HFM** | **0.81±0.03** | **0.47±0.01** | **0.54±0.04** | **0.75±0.01** | 0.66±0.01 | **0.56±0.01** |

- Group 1 performance improved by **9%**
- Group 2 performance improved by **38%** over the pre-fine-tuned baseline

### C. LQ-RAG System

**Open-Domain Question Answering:**

| RAG Configuration | Avg. Relevance Score | Answer Relevance | Context Relevance | Groundedness |
|---|---|---|---|---|
| Naive RAG | 65% | 87% | 38% | 70% |
| RAG + FTM | 70% | 92% | 42% | 76% |
| **LQ-RAG** | **80%** | **88%** | **70%** | **82%** |

- LQ-RAG achieves a **23% improvement** over Naive RAG in average relevance
- LQ-RAG achieves a **14% improvement** over RAG with FTM
- LQ-RAG scored 88% in answer relevance, 70% in context relevance, and 82% in groundedness

**Closed-Domain Question Answering:**

| Configuration | Answer Relevance | Context Relevance | Groundedness |
|---|---|---|---|
| Naive RAG | 0.88 | 0.31 | 0.26 |
| RAG + FTM | 0.88 | 0.24 | 0.19 |
| LQ-RAG | 0.72 | 0.48 | 0.35 |

In the closed-domain scenario, all systems struggled — context relevance and groundedness remained below 50% across all configurations, as the system lacked relevant context information. This highlights that evaluating RAG systems requires considering all three criteria together.

**Time Complexity:**

| Configuration | Avg. Response Time (5 questions) |
|---|---|
| Naive RAG | 7.2 seconds |
| RAG + FTM | 11.2 seconds |
| LQ-RAG | 14.6 seconds |

LQ-RAG is approximately twice as slow as Naive RAG, as incorporating advanced modules increases time complexity.

---

## VII. Conclusion

This paper addresses domain-specific challenges in the legal field where traditional RAG systems often fail in information extraction and response generation. The LQ-RAG framework integrates RAG with a recursive feedback mechanism, combining specialized LLMs and an agent-driven approach for response evaluation and query engineering.

**Summary of results:**
- Fine-tuning a general-purpose LLM with legal corpora resulted in a **15% improvement** over baseline models
- A hybrid fine-tuned generative LLM achieved **up to 24% better performance** across tasks compared to general domain LLMs
- LQ-RAG outperformed all baselines with a **23% improvement** in average relevance over naive configuration and **14% improvement** over RAG with fine-tuned LLMs

The adaptable design facilitates adoption across other specialized domains with minimal adjustments, enabling professionals to make high-quality, informed decisions.

---

## VIII. Limitations & Future Work

**Current limitations:**
- Reliance on GPT-4 as the evaluation agent (proprietary model dependency)
- High response generation time (approximately double that of Naive RAG)
- Absence of feedback from domain experts (legal practitioners)

**Future directions:**
- Optimizing time complexity
- Developing a specialized legal evaluation agent with domain-specific expertise
- Incorporating feedback from legal practitioners to ensure practical utility and alignment with legal reasoning
- Incorporating benchmark datasets specifically designed for the legal domain
- Applying state-of-the-art optimization techniques to enhance Hit Rate and MRR
- Conducting empirical experiments in real-world legal scenarios

---

## Acknowledgment

The authors gratefully acknowledge the high-performance GPU computing support provided by HPC-AI Open Infrastructure through GIST SCENT.

This work was supported by:
- IITP grant funded by the Korea Government (MSIT) (IITP-2025-RS-2021-II210118)
- IITP-ITRC grant funded by the Korea Government [Ministry of Science and ICT] (IITP-2025-RS-2021-II211835)

---

## References

1. Q. Lang et al., "Exploring the answering capability of large language models in addressing complex knowledge in entrepreneurship education," *IEEE Trans. Learn. Technol.*, vol. 17, pp. 2053–2062, 2024.
2. G. B. Mohan et al., "An analysis of large language models: Their impact and potential applications," *Knowl. Inf. Syst.*, vol. 66, no. 9, pp. 5047–5070, Sep. 2024.
3. B. Meskó and E. J. Topol, "The imperative for regulatory oversight of large language models (or generative AI) in healthcare," *npj Digit. Med.*, vol. 6, no. 1, p. 120, Jul. 2023.
4. J. Lai et al., "Large language models in law: A survey," *AI Open*, vol. 5, pp. 181–196, 2024.
5. D. M. Katz et al., "GPT-4 passes the bar exam," *Phil. Trans. Roy. Soc. A*, vol. 382, Mar. 2023.
6. Q. Huang et al., "Lawyer LLaMA technical report," 2023, arXiv:2305.15062.
7. V. Magesh et al., "Hallucination-free? Assessing the reliability of leading AI legal research tools," 2024, arXiv:2405.20362.
8. W. Benjamin, "Here's what happens when your lawyer uses ChatGPT," *New York Times*, 2023.
9. M. Dahl et al., "Large legal fictions: Profiling legal hallucinations in large language models," *J. Legal Anal.*, vol. 16, no. 1, pp. 64–93, Jan. 2024.
10. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. Adv. Neural Inf. Process. Syst.*, 2020, pp. 9459–9474.
11. J. Chen et al., "Benchmarking large language models in retrieval-augmented generation," in *Proc. AAAI Conf. Artif. Intell.*, Mar. 2024, vol. 38, no. 16, pp. 17754–17762.
12. A. Asai et al., "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," in *Proc. 12th Int. Conf. Learn. Represent.*, Jan. 2023, pp. 1–30.
13. Y. Xia et al., "Generation of asset administration shell with large language model agents," *IEEE Access*, vol. 12, pp. 84863–84877, 2024.
14. R. S. M. Wahidur et al., "Enhancing zero-shot crypto sentiment with fine-tuned language model and prompt engineering," *IEEE Access*, vol. 12, pp. 10146–10159, 2024.
15. J. Bednár et al., "Some like it small: Czech semantic embedding models for industry applications," in *Proc. AAAI Conf. Artif. Intell.*, vol. 38, Mar. 2024, pp. 22734–22742.
16. X. Ma et al., "Query rewriting in retrieval-augmented large language models," in *Proc. EMNLP*, 2023, pp. 5303–5315.
17. R. Sharma, "Exploring Advanced RAG Techniques for AI," 2024.
18. ILIN, "Advanced RAG Techniques: An Illustrated Overview," 2023.
19. Z. Shao et al., "Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy," in *Proc. Findings ACL, EMNLP*, 2023, pp. 9248–9274.
20. W. Yu et al., "Generate rather than retrieve: Large language models are strong context generators," in *Proc. 11th Int. Conf. Learn. Represent.*, 2022.
21. J. Wen and W. He, "HanFei-1.0," 2023.
22. H. Liu et al., "Chinese Law Large Language Model," 2023.
23. H.-T. Nguyen, "A brief report on LawGPT 1.0," 2023, arXiv:2302.05729.
24. H. Li, "LexiLaw," 2023.
25. D. Soong et al., "Improving accuracy of GPT-3/4 results on biomedical data using a retrieval-augmented language model," *PLOS Digit. Health*, vol. 3, no. 8, Aug. 2024.
26. C. Zakka et al., "Almanac—Retrieval—Augmented language models for clinical medicine," *NEJM AI*, vol. 1, no. 2, pp. 1–45, 2024.
27. S. Yue et al., "DISC-LawLLM: Fine-tuning large language models for intelligent legal services," 2023, arXiv:2309.11325.
28. N. Wiratunga et al., "CBR-RAG: Case-based reasoning for retrieval augmented generation in LLMs for legal question answering," in *Proc. ICCBR*, vol. 14775, 2024, pp. 445–460.
29. A. Chouhan and M. Gertz, "LexDrafter: Terminology drafting for legislative documents using retrieval augmented generation," in *Proc. LREC-COLING*, 2024, pp. 10448–10458.
30. S. S. Alotaibi et al., "KAB: Knowledge augmented BERT2BERT automated questions-answering system for jurisprudential legal opinions," *IJCSNS*, vol. 22, pp. 346–356, Jun. 2022.
31. C. Hoppe et al., "Towards intelligent legal advisors for document retrieval and question-answering in German legal documents," in *Proc. IEEE 4th Int. Conf. AIKE*, Dec. 2021, pp. 29–32.
32. S. Robertson et al., "Simple BM25 extension to multiple weighted fields," in *Proc. 13th ACM Int. Conf. CIKM*, Nov. 2004, pp. 42–49.
33. V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in *Proc. EMNLP*, 2020, pp. 6769–6781.
34. M. Henderson et al., "Efficient natural language response suggestion for smart reply," 2017, arXiv:1705.00652.
35. J. E. Hu et al., "LoRA: Low-rank adaptation of large language models," in *Proc. ICLR*, Jan. 2021.
36. M. Douze et al., "The Faiss Library," 2024.
37. S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," 2022, arXiv:2210.03629.
38. J. Lee et al., "Chain-of-thought prompting elicits reasoning in large language models," in *Proc. NeurIPS*, 2022, pp. 24824–24837.
39. I. Bunescu, "QA Legal Dataset Train," 2023.
40. T. Rohan and G. Ishaan, "Stanford Alpaca: An Instruction-following LLaMA Model," 2023.
41. P. Rajpurkar et al., "Know what you don't know: Unanswerable questions for SQuAD," in *Proc. 56th ACL*, 2018, pp. 784–789.
42. S. Lin et al., "TruthfulQA: Measuring how models mimic human falsehoods," in *Proc. 60th ACL*, 2022, pp. 3214–3252.
43. J. Li et al., "Parameter-efficient legal domain adaptation," in *Proc. NLLP Workshop*, 2022, pp. 119–129.
44. M. Suzgun et al., "Challenging BIG-bench tasks and whether chain-of-thought can solve them," in *Proc. Findings ACL*, 2023, pp. 13003–13051.
45. N. Guha et al., "Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models," in *Proc. NeurIPS*, 2023, pp. 44123–44279.
46. D. Hendrycks et al., "Measuring massive multitask language understanding," in *Proc. ICLR*, May 2021.
47. R. Zellers et al., "HellaSwag: Can a machine really finish your sentence?" in *Proc. 57th ACL*, Florence, 2019, pp. 4791–4800.
48. O. Khattab and M. Zaharia, "ColBERT," in *Proc. 43rd ACM SIGIR*, Jul. 2020, pp. 39–48.
49. P. Zhang et al., "Retrieve anything to augment large language models," 2023, arXiv:2310.07554.
50. S. Xiao et al., "C-pack: Packed resources for general Chinese embeddings," in *Proc. 47th ACM SIGIR*, Jul. 2024, pp. 641–649.
51. A. V. Solatorio, "GISTEmbed: Guided in-sample selection of training negatives for text embedding fine-tuning," 2024, arXiv:2402.16829.
52. H. Touvron et al., "Llama 2: Open foundation and fine-tuned chat models," 2023, arXiv:2307.09288.
53. AI@Meta, "Llama 3 Model Card," 2024.
54. H. W. Chung et al., "Scaling instruction-finetuned language models," *J. Mach. Learn. Res.*, vol. 25, pp. 1–53, 2024.
55. X. Zhang et al., "A novel method to improve hit rate for big data quick reading," in *Proc. AIAM*, Oct. 2019, pp. 39–43.
56. S. Roychowdhury et al., "Evaluation of RAG metrics for question answering in the telecom domain," 2024, arXiv:2407.12873.
57. P. Xia et al., "Learning similarity with cosine similarity ensemble," *Inf. Sci.*, vol. 307, pp. 39–52, Jun. 2015.
58. A. Stolfo, "Groundedness in retrieval-augmented long-form generation: An empirical study," in *Proc. Findings ACL, NAACL*, 2024, pp. 1537–1552.
59. K. Papineni et al., "BLEU," in *Proc. 40th ACL*, 2001, p. 311.
60. C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in *Proc. Text Summarization Branches Out*, Jul. 2004, pp. 74–81.
61. N. Houlsby et al., "Parameter-efficient transfer learning for NLP," in *Proc. ICML*, 2019, pp. 2790–2799.
62. T. Dettmers et al., "GPT3.Int8(): 8-bit matrix multiplication for transformers at scale," in *Proc. NeurIPS*, vol. 35, 2022, pp. 30318–30332.
63. W. Leandro et al., "TRL: Transformer Reinforcement Learning," 2020.

---

## About the Authors

**Rahman S. M. Wahidur** received the B.Sc. degree in electrical and electronics engineering from the Ahsanullah University of Science and Technology, Dhaka, Bangladesh, in 2009. He is currently pursuing the combined M.S. and Ph.D. degree at GIST, South Korea, and is a Research Assistant at the INFONET LAB. He previously worked as a Telecommunication Engineer at various multinational corporations (2010–2019). Research interests: NLP, deep learning, blockchain price modeling, and generative AI.

**Sumin Kim** received the B.S. degree in communications and convergence software from Kwangwoon University, Seoul, South Korea, in 2021. She is currently pursuing the Ph.D. degree at the Artificial Intelligence Graduate School, GIST. Research interests: continual learning, reinforcement learning, NLP, and financial price modeling.

**Haeung Choi** received the B.S. degree from Kyungpook National University in 2013, and the M.S. degree from GIST in 2015, where he is currently pursuing the Ph.D. degree. He is also a Researcher at LiberVance Company. Research interests: blockchain and cybersecurity.

**David S. Bhatti** received the Ph.D. degree in computer science from NUST, Islamabad, Pakistan, in 2020. He is currently a Postdoctoral Researcher at GIST. Research interests: network security, deep learning, and hyperspectral imaging.

**Heung-No Lee** (Senior Member, IEEE) received the B.S., M.S., and Ph.D. degrees in electrical engineering from UCLA in 1993, 1994, and 1999, respectively. He was a Research Staff Member at HRL Laboratories (1999–2002) and an Assistant Professor at the University of Pittsburgh (2002–2008). He joined GIST in 2009. Research interests: information theory, signal processing, blockchain, communications/networking, compressive sensing, future internet, and brain–computer interface. He has received several prestigious national awards including the Top 100 National R&D Award (2012), the Top 50 Achievements of Fundamental Research Award (2013), and the Science/Engineer of the Month (January 2014).

---

*© 2025 The Authors. This work is licensed under a Creative Commons Attribution 4.0 License.*
*For more information, see https://creativecommons.org/licenses/by/4.0/*
*IEEE Access, Volume 13, 2025*
