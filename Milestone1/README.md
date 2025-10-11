
**Text Summarization and Paraphrasing using T5, BART, and PEGASUS**

**Project Overview**:
    This project demonstrates text summarization and paraphrasing using three state-of-the-art transformer-based models — T5, BART, and PEGASUS — on classic literature from Project Gutenberg

The aim is to compare model performance in terms of summary length, compression ratio, and paraphrase quality, while visualizing the results through simple data analytics and plots.

Why Use GPU Instead of CPU?
     Large transformer models such as T5, BART, and PEGASUS involve millions of parameters (ranging from 220M to over 400M). These models perform numerous matrix multiplications and tensor operations during encoding and decoding steps.
| Feature             | CPU                                            | GPU                                      |
| ------------------- | ---------------------------------------------- | ---------------------------------------- |
| **Architecture**    | Sequential processing                          | Parallel processing (thousands of cores) |
| **Performance**     | Slow for large tensors                         | Extremely fast for tensor operations     |
| **Best suited for** | Simple numerical operations, smaller ML models | Deep learning models with large data     |
| **Example**         | Preprocessing, data loading                    | Model training, text generation          |
| **Speed (approx.)** | 10x–50x slower                                 | 10x–50x faster for transformers          |

GPU Advantage:
   When running summarization or paraphrasing, the model must repeatedly compute attention weights across all tokens. GPUs accelerate this process drastically, reducing inference time from minutes to seconds.
Using torch.cuda.is_available() ensures the code runs on GPU (if available) via NVIDIA CUDA, providing real-time generation efficiency.

**Text files**:

 "https://www.gutenberg.org/files/1661/1661-0.txt",  # The Adventures of Sherlock Holmes
 
 "https://www.gutenberg.org/files/98/98-0.txt"       # A Tale of Two Cities
  
**Models Used**:

1. **T5 (Text-to-Text Transfer Transformer)**
   - **Model Name:** t5-base
   - **Approach:** Treats every NLP task as a text-to-text problem.
   - **Summarization Input Format:** "summarize: <text>"
   - **Strengths:**
     - Flexible for multiple tasks (translation, summarization, paraphrasing)
     - Generates grammatically rich and concise summaries
   - **Weaknesses:**
     - May shorten too aggressively
     - Can remove rare context details

2. **BART (Bidirectional and Auto-Regressive Transformers)**
   - **Model Name:** facebook/bart-base
   - **Architecture:** Combines BERT’s encoder and GPT’s decoder
   - **Strengths:**
     - Excellent for abstractive summarization
     - Produces natural, fluent text with strong contextual coherence
   - **Weaknesses:**
     - Slightly slower than T5 due to complex bidirectional encoding

3. **PEGASUS (Pre-training with Gap Sentences)**
   - **Model Name:** google/pegasus-xsum
   - **Approach:** Pre-trained using “gap sentence generation”, where key sentences are masked and predicted
   - **Strengths:**
     - Especially designed for summarization tasks
     - Captures key semantic points
     - Performs well on news and story datasets
   - **Weaknesses:**
     - May over-summarize or paraphrase more freely
     - Sometimes loses factual detail

**Project Workflow:**
- **Data Loading:** Downloads text files (e.g., *A Tale of Two Cities*, *The Adventures of Sherlock Holmes*) from Project Gutenberg.
- **Text Cleaning:** Removes headers/footers and extra metadata using regex.
- **Summarization:** Each model generates summaries for multiple texts. Metrics such as original length, summary length, and compression ratio are computed and visualized.
- **Paraphrasing:** Uses fine-tuned models to rephrase sample sentences.
- **Visualization:** Bar plots for compression ratio comparison, word count comparison, optional similarity analysis.
  

| Metric                  | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| **Summary Length**      | Word count of generated summary                         |
| **Compression Ratio**   | Summary length ÷ Original text length                   |
| **Semantic Similarity** | Cosine similarity between embeddings (for paraphrasing) |
| **Fluency**             | Subjective evaluation of readability                    |



**Libraries Used:**
- transformers (Hugging Face)
- torch
- nltk
- pandas
- numpy
- matplotlib
- seaborn
- sentence-transformers (for similarity scoring)
- requests
- re

**Visual Outputs**:

 Compression Ratio Comparison: Shows how much each model compresses text.
 Paraphrased Word Count Comparison: Displays variation in paraphrase verbosity.
 Similarity Scores: Indicates how close paraphrases are to original sentences.

**Conclusion**:

 GPU usage significantly accelerates transformer inference, making large models feasible for interactive applications.
 
T5, BART, and PEGASUS each exhibit unique summarization styles:

  T5: Balanced and task-flexible
  BART: Fluent and detailed
  PEGASUS: Concise and key-focused

Together, they provide valuable insight into text abstraction and semantic rewording, essential for NLP-driven tasks like report generation, chatbot responses, and automatic content summarization.
