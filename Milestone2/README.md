# **Infosys TextMorph – Milestone 2**
## **Advanced Text Summarization and Evaluation**

---

### **Overview**

This project is part of the Infosys TextMorph Internship Program – **Milestone 2**, focusing on advanced text summarization using multiple AI models:

- **Abstractive** and **Extractive** summarization techniques
- Performance evaluation using key metrics
- Interactive UIs for visualization and comparison

---

### **Objectives**

- **Implement multiple summarization models:**
  - *Abstractive Models*: TinyLlama, Phi, Gemma, BART
  - *Extractive Model*: TextRank
- **Develop interactive UIs using ipywidgets:**
  - Summarize with all models
  - Summarize with selected models
- **Evaluate summarization quality:**
  - ROUGE metrics
  - Semantic Similarity
  - Readability Scores
- **Test with diverse sample texts** from 10+ domains
- **Visualize performance** using comparison plots

---

### **💡 Why Text Summarization Is Important**

- **Save time:** Quickly understand long documents
- **Enhance decision-making:** Extract key insights from large data
- **Improve accessibility:** Make complex content concise and clear
- **Support AI applications:** Power chatbots, Q&A, recommendation engines

By exploring both **abstractive (human-like)** and **extractive (sentence-based)** summarization, you deepen your NLP and LLM knowledge.

---

### ** Key Concepts Covered**

| **Concept**                | **Description**                                                                 |
|----------------------------|--------------------------------------------------------------------------------|
| **Abstractive Summarization** | Generates new sentences that convey the core meaning of the text, similar to human summaries |
| **Extractive Summarization**  | Selects and combines important sentences directly from the source text         |
| **Transformer Models**        | Neural architectures (TinyLlama, Phi, Gemma, BART) for text understanding and generation |
| **Evaluation Metrics**        | Measures like ROUGE, semantic similarity to compare model output with references |
| **Readability Analysis**      | Assesses how easy the generated summaries are to read and understand          |

---

### ** System Components**

#### **1. Model Implementations**
- **Abstractive Methods (LLMs):**
  - **TinyLlama:** Lightweight LLM for summarization.
  - **Phi:** Efficient small language model with good comprehension.
  - **Gemma:** Instruction-tuned model by Google for advanced NLP tasks.
  - **BART:** Pretrained sequence-to-sequence model for abstractive summarization.
- **Extractive Method:**
  - **TextRank:** Graph-based extractive summarization algorithm.

#### **2. Evaluation Metrics**
- **ROUGE-1, ROUGE-2, ROUGE-L**
- **Semantic Similarity:** Cosine similarity via Sentence Transformers
- **Readability Metrics:** Flesch Reading Ease, Gunning Fog Index, etc.

#### **3. Interactive UIs**
- Built with **ipywidgets** for experimentation:
  - Enter custom text
  - Choose models to summarize
  - View summaries, scores, and comparison plots

#### **4. Visualization**
- **Matplotlib**-based plots to compare model performance

---

### ** Workflow Summary**

1. **Import Libraries**
   - Install & import: `transformers`, `datasets`, `torch`, `sentence-transformers`, `rouge-score`, `nltk`, `textstat`, `ipywidgets`, `matplotlib`
2. **Load Models and Tokenizers**
   - Initialize each summarization model (`device_map="auto"`, `torch.bfloat16`)
3. **Preprocess Input Texts**
   - Clean and tokenize the data
4. **Generate Summaries**
   - **Abstractive:** TinyLlama, Phi, Gemma, BART
   - **Extractive:** TextRank
5. **Evaluate Summaries**
   - Compute ROUGE, semantic similarity, readability
6. **Interactive Testing**
   - Use Colab UI for multi-input testing, visualize results
7. **Performance Visualization**
   - Generate comparison charts for insights

---

### ** Sample Text Domains**

- **Finance:** Stock market reports
- **Sports:** Match summaries
- **Weather:** Daily forecasts
- **Technology:** AI & innovation articles
- **Health:** Medical research abstracts
- **Education:** Academic papers
- **Politics:** Government announcements
- **Science:** Discovery reports
- **Entertainment:** Movie reviews
- **Business:** Company press releases

---

### **🧰 Tech Stack**

| **Category** | **Tools Used**                                                                 |
|--------------|--------------------------------------------------------------------------------|
| Language     | Python                                                                         |
| Libraries    | transformers, torch, sentence-transformers, rouge-score, textstat, ipywidgets, matplotlib, nltk |
| Environment  | Google Colab                                                                   |
| Models       | TinyLlama, Phi, Gemma, BART, TextRank                                          |

---

### ** Expected Outcome**

- Understand abstractive vs. extractive summarization
- Evaluate model-generated summaries
- Interactively compare model performances
- Gain hands-on experience with transformer-based NLP

---

### ** Conclusion**

**Milestone 2** bridges theory and practical implementation in text summarization.  
It showcases how modern LLMs transform text comprehension and generation — enabling efficient, intelligent, and human-like summarization systems.

---
