# ShareGPT92K Analysis Project

This project analyzes human-AI interactions from the ShareGPT92K dataset, focusing on linguistic patterns, emotions, and anthropomorphic characteristics in early user-LLM conversations.

├── raw_data/
│ └── ShareGPT_V3_unfiltered_cleaned_split.json
├── openai_api_encode/
│ └── [OpenAI API interaction modules]
├── sampled_enriched_data/
│ └── [Sample JSON files with processed conversations]
├── full_nlp.py
├── human_labeling.py
├── sentiment_analysis.py
├── visualization.ipynb
└── merged_df_with_clusters.7z *(pickle file inside)

## Features

### Traditional NLP Analysis
- Lexical feature extraction (word frequencies, POS patterns)
- Sentiment analysis using VADER
- Discourse markers identification
- Syntactic complexity measurement
- Named Entity Recognition
- Language detection
- Multi-threaded conversation processing

### OpenAI API Integration
- Asynchronous API calls at scale
- Token and rate limit management
- Streaming capabilities
- Threading management for parallel processing
- Support for both Chat (e.g., GPT-4) and Embedding models

### Labeling System
- Human coding interface for manual message labeling
- GPT-based automated labeling for expressiveness and anthropomorphism
- Two-step Chain of Thought for summary generation
- Domain/task topic classification

## Data Processing Pipeline

1. **Raw Data**: ShareGPT92K dataset in JSON format
2. **Processing**: 
   - NLP analysis via `full_nlp.py`
   - Sentiment analysis
   - OpenAI API processing for advanced features
3. **Intermediate Storage**: 
   - Processed conversations stored in `sampled_enriched_data/` as individual JSONs
   - Enables batch processing and scalability
4. **Final Output**: 
   - Compiled into `merged_df_with_clusters.7z`
   - Ready for EDA and visualization

### full_nlp.py
- Comprehensive NLP processing module
- Multi-threaded conversation analysis
- Feature extraction and aggregation
- Language detection and processing

### human_labeling.py
Interactive interface for manual message labeling:
- Expressive/directive speech act labeling
- Progress saving functionality
- Navigation between messages
- Quality control for automated labeling

### visualization.ipynb
Jupyter notebook for:
- Data exploration
- Feature visualization
- Pattern analysis
- Result presentation

## Setup and Usage

### Prerequisites

- Python 3.10+
- OpenAI API key
- Other dependencies (listed in `requirements.txt`)

**OpenAI API Authentication**:
- Ensure your API key is correctly set in `.env`
- Check environment variable is loaded: `echo $OPENAI_API_KEY`

Either use conda or pip to install the dependencies.

bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt

OR

conda env create -f environment.yml
conda activate sharegpt-analysis


## License

This project is open-sourced under the MIT License - see the LICENSE file for details.

