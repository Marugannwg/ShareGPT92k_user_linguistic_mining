# ShareGPT92K Analysis Project

This project analyzes human-AI interactions from the ShareGPT92K dataset, focusing on linguistic patterns, emotions, and anthropomorphic characteristics in early user-LLM conversations.

## Features

Here are the list of tools:

### Traditional NLP Analysis
- see [full_nlp.py](./full_nlp.py) and [sentiment_analysis.py](./sentiment_analysis.py)
- Lexical feature extraction (word frequencies, POS patterns)
- Sentiment analysis using VADER
- Discourse markers identification
- Prepared for Multi-threaded batch processing

### OpenAI API Integration
- see [openai_api_encode/](./openai_api_encode/)
- Asynchronous API calls at scale
- Token and rate limit management
- Streaming capabilities
- Threading management for parallel processing
- Support for both Chat (e.g., GPT-4o and more) and Text Embedding models
- The example here is an automated multi-shot labeling for expressiveness and anthropomorphism
- Same toolkit also used for a Two-step Chain of Thought for summary generation
- Human coding interface for manual message labeling \human_labeling.py -- for accuracy and robustness check

**Raw Data**: ShareGPT92K dataset in JSON format
- Raw data is cleaned and processed into batch, and go through the NLP analysis and customized OpenAI API labeling process
**Sample Intermediate Storage**: Processed conversations stored in `sampled_enriched_data/` as individual JSONs
- Sample json outcome for each conversation
**Final Output**: Combined all processed conversations into `merged_df_with_clusters.7z` -- Use the pickle file inside for visualization
- This version include:
   - Token count
   - VADER sentiment analysis
   - User of pronouns (first person, second person)
   - Expressiveness and anthropomorphism labeling (GPT-4o multi-shot result)
   - Summary and domain topic keyword labeling (GPT-4o CoT)
   - Clustering results based on domain topic keywords -- Each human input is assigned to one or more clusters representing similar topics (k=50), detail see this [csv](./processed_data/clustering_result_references/domain_clusters_detailed.csv))

### Visualization

- see [visualization.ipynb](./visualization.ipynb); Jupyter notebook for: visualing clustering results, pattern analysis...

## Setup and Usage

### Prerequisites

- Python 3.10+
- OpenAI API key
- Other dependencies (listed in `requirements.txt`)

**OpenAI API Authentication**:
- Ensure your API key is correctly set in `.env`
- Check environment variable is loaded: `echo $OPENAI_API_KEY`

Either use conda or pip to install the dependencies.

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
OR

```bash
conda env create -f environment.yml
conda activate sharegpt-analysis
```


## License

This project is open-sourced under the MIT License - see the LICENSE file for details.

