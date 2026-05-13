### 1. What is Generative AI, and how does it differ from traditional discriminative AI?
**Answer:**  
Generative AI creates new content (text, images, code, audio) by learning the underlying data distribution and sampling from it. Examples include GPT models, Stable Diffusion, and Codex.  
Discriminative AI focuses on classification or regression (e.g., spam detection, image classification) by learning decision boundaries.  
Key difference: Generative models model P(X) or P(X|Y), while discriminative models focus on P(Y|X). In practice, GenAI is used for creation, augmentation, and reasoning tasks.

### 2. Explain the Transformer architecture and why it is foundational for modern GenAI.
**Answer:**  
The Transformer (Vaswani et al., 2017) uses self-attention mechanisms instead of RNNs/LSTMs, enabling parallel processing and better long-range dependencies.  
Core components:  
- Multi-Head Self-Attention (captures relationships between tokens)  
- Feed-Forward Networks  
- Positional Encoding  
- Encoder-Decoder structure (or decoder-only in GPT-style models)  
It powers nearly all LLMs because it scales efficiently with data and compute, supports in-context learning, and handles variable-length sequences well.

### 3. What are the main techniques for optimizing LLM inference (latency & cost)?
**Answer:**  
Common techniques include:  
- Quantization (8-bit, 4-bit, GPTQ, AWQ)  
- KV-Cache optimization & PagedAttention (vLLM)  
- Speculative decoding  
- Model distillation / pruning  
- Continuous batching / dynamic batching  
- Hardware-specific optimizations (FlashAttention-2, TensorRT-LLM, ONNX)  
In production, we often combine quantization + vLLM or TGI for 2-5x throughput gains.

### 4. Explain Retrieval-Augmented Generation (RAG) and when you would choose it over fine-tuning.
**Answer:**  
RAG retrieves relevant documents from a vector database and injects them into the prompt, allowing the model to ground responses in external/up-to-date knowledge.  
Use RAG when:  
- You need current information or domain-specific data  
- Frequent updates are required (no retraining)  
- Cost of fine-tuning is high  
Fine-tuning is better for teaching new behavior, style, or specialized reasoning. Hybrid approaches (fine-tune + RAG) are increasingly common.

### 5. How do you evaluate Generative AI outputs? Name key metrics and methods.
**Answer:**  
- **Quantitative:** ROUGE, BLEU, BERTScore, METEOR (text similarity); G-Eval, LLM-as-a-Judge; Human Preference (Elo, Win Rate)  
- **Qualitative:** Factuality (using tools like FactScore), Toxicity (Perspective API), Relevance, Coherence  
- **Task-specific:** Code (HumanEval, MBPP), Reasoning (GSM8K, BigBench), Safety benchmarks  
Best practice: Combine automated metrics with human/AI judge evaluation and monitor in production with guardrails.

### 6. What is prompt engineering? Describe advanced techniques you have used.
**Answer:**  
Prompt engineering is the art and science of crafting inputs to elicit desired outputs from LLMs.  
Advanced techniques I use:  
- Chain-of-Thought (CoT) & Self-Consistency  
- Tree-of-Thoughts / Graph-of-Thoughts  
- ReAct / Toolformer (reason + act)  
- Few-shot with examples selection  
- Automatic prompt optimization (DSPy, Promptbreeder)  
- Role prompting + delimiters + output formatting (JSON mode / Pydantic)

### 7. How would you handle hallucinations in a production GenAI application?
**Answer:**  
Multi-layered approach:  
1. **Prevention:** RAG with good retrieval, citation prompting, self-ask verification  
2. **Detection:** Uncertainty estimation, self-consistency checks, fact-checking against knowledge base  
3. **Mitigation:** Guardrails (NVIDIA NeMo Guardrails, LlamaGuard), fallback to human/review, confidence scoring  
4. **Monitoring:** Log prompts/responses, track hallucination rate in production, continuous evaluation  
Transparency (showing sources) significantly reduces user impact.

### 8. Describe your experience with fine-tuning LLMs. Which methods do you prefer?
**Answer:**  
I have experience with:  
- Full fine-tuning (for small models)  
- Parameter-Efficient Fine-Tuning: LoRA/QLoRA (most common), DoRA, Prefix Tuning  
- Alignment: RLHF, DPO, ORPO, KTO  
Preferred stack: Hugging Face PEFT + TRL + Unsloth (for speed) + Axolotl or Llama-Factory for experiments.  
For most enterprise use cases, QLoRA on 1-4 GPUs is sufficient and cost-effective.

### 9. What are the main challenges when deploying GenAI at scale?
**Answer:**  
- Cost (inference tokens can get expensive at volume)  
- Latency (especially for real-time apps)  
- Safety & compliance (PII leakage, bias, copyright)  
- Observability & drift (model behavior changes)  
- Rate limits & vendor lock-in  
Mitigations: Self-hosted open models (Llama 3, Mixtral, DeepSeek), multi-model routing, caching, and robust MLOps (LangSmith, Phoenix, Helicone).

### 10. How do you stay current with the fast-moving GenAI field?
**Answer:**  
- Follow key papers on arXiv (daily), Hugging Face daily papers, and conferences (NeurIPS, ICML, ICLR)  
- Experiment hands-on with new models on Groq, Together.ai, Fireworks, or local setups  
- Contribute to or use open-source (LangChain/LlamaIndex, vLLM, Axolotl)  
- Participate in communities (Discord, Twitter/X lists, Reddit r/LocalLLaMA)  
- Run personal benchmarks and side projects (e.g., building agents or RAG systems)
