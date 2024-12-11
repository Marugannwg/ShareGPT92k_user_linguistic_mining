"""
VADER Sentiment Analyzer on ShareGPT

This script performs VADER sentiment analysis on ShareGPT conversations. It processes both
individual messages and overall conversations, enriching the original dataset with sentiment scores.

Features:
- Parallel processing for large datasets
- Individual message and conversation-level sentiment analysis
- Progressive saving of results
- Detailed logging
- Support for both full dataset and sample processing

Output Format:
{
    "id": "conversation_id",
    "conversations": [
        {
            "from": "human/assistant",
            "value": "message content",
            "sentiment": {
                "compound": float,  # [-1.0, 1.0]
                "pos": float,       # [0.0, 1.0]
                "neu": float,       # [0.0, 1.0]
                "neg": float        # [0.0, 1.0]
            }
        },
        ...
    ],
    "overall_sentiment": {
        "compound": float,  # Average compound score
        "pos": float,      # Average positive score
        "neu": float,      # Average neutral score
        "neg": float       # Average negative score
    }
}
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import json
from tqdm import tqdm
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)

class ConversationSentimentAnalyzer:
    """
    A class to analyze sentiment in ShareGPT conversations using VADER.
    
    Attributes:
        output_dir (Path): Directory for saving processed conversations
        max_workers (int): Number of parallel processing workers
        analyzer (SentimentIntensityAnalyzer): VADER sentiment analyzer instance
    """
    def __init__(self, output_dir='processed_data', max_workers=8):
        self.analyzer = SentimentIntensityAnalyzer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        logging.info(f"Initialized analyzer with {max_workers} workers")
        
    def analyze_message(self, message):
        try:
            content = message['value'] if 'value' in message else message['content']
            sentiment = self.analyzer.polarity_scores(content)
            
            enriched_message = message.copy()
            enriched_message['sentiment'] = sentiment
            return enriched_message
        except Exception as e:
            logging.error(f"Error in analyze_message: {str(e)}")
            logging.error(f"Message content: {message}")
            return message
    
    def analyze_conversation(self, item):
        try:
            # Debug logging
            logging.debug(f"Processing item: {item}")
            
            # Check if item is a dictionary
            if not isinstance(item, dict):
                logging.error(f"Item is not a dictionary: {type(item)}")
                return None
            
            # Check if conversations exists in the item
            if 'conversations' in item:
                conversation = item['conversations']
            elif 'conversation' in item:
                conversation = item['conversation']
            else:
                logging.error(f"No conversation found in item keys: {item.keys()}")
                return None
            
            enriched_messages = [self.analyze_message(msg) for msg in conversation]
            
            valid_sentiments = [msg['sentiment'] for msg in enriched_messages 
                              if 'sentiment' in msg]
            
            if valid_sentiments:
                overall_sentiment = {
                    'compound': sum(s['compound'] for s in valid_sentiments) / len(valid_sentiments),
                    'pos': sum(s['pos'] for s in valid_sentiments) / len(valid_sentiments),
                    'neu': sum(s['neu'] for s in valid_sentiments) / len(valid_sentiments),
                    'neg': sum(s['neg'] for s in valid_sentiments) / len(valid_sentiments)
                }
            else:
                overall_sentiment = {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
            
            enriched_item = {
                'id': item.get('id', 'unknown'),
                'conversations': enriched_messages,  # Changed to match original structure
                'overall_sentiment': overall_sentiment
            }
            
            # Save individual conversation
            output_path = self.output_dir / f"{enriched_item['id']}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_item, f, ensure_ascii=False, indent=2)
                
            return enriched_item
            
        except Exception as e:
            logging.error(f"Error processing conversation {item.get('id', 'unknown')}: {str(e)}")
            logging.error(f"Item structure: {item}")
            return None

    def process_dataset(self, input_path, sample_size=None):
        """
        Process the ShareGPT dataset with sentiment analysis.
        
        Args:
            input_path (str): Path to the input JSON file
            sample_size (int, optional): Number of conversations to process.
                                       If None, processes entire dataset.
        
        Returns:
            list: Enriched conversations with sentiment scores
        """
        try:
            # Load data
            logging.info(f"Loading data from {input_path}")
            with open(input_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Debug logging
            logging.info(f"Data type: {type(data)}")
            if data:
                logging.info(f"First item structure: {data[0].keys() if isinstance(data[0], dict) else type(data[0])}")
            
            # Sample if requested
            if sample_size and len(data) > sample_size:
                import random
                random.seed(42)
                data = random.sample(data, sample_size)
            
            logging.info(f"Processing {len(data)} conversations...")
            
            # Process conversations in parallel
            enriched_data = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(tqdm(
                    executor.map(self.analyze_conversation, data),
                    total=len(data),
                    desc="Analyzing conversations"
                ))
                
                enriched_data = [r for r in results if r is not None]
            
            # Save complete dataset
            complete_output_path = self.output_dir / 'complete_enriched_dataset.json'
            with open(complete_output_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Processing complete. Saved {len(enriched_data)} conversations.")
            return enriched_data
            
        except Exception as e:
            logging.error(f"Error in process_dataset: {str(e)}")
            raise

def main():
    # Configuration
    INPUT_PATH = 'raw_data/ShareGPT_V3_unfiltered_cleaned_split.json'
    OUTPUT_DIR = 'sentiment_enriched_data'
    SAMPLE_SIZE = None  # Reduced for initial testing
    MAX_WORKERS = 8 # Reduced for initial testing
    
    try:
        # Initialize and run analyzer
        analyzer = ConversationSentimentAnalyzer(
            output_dir=OUTPUT_DIR,
            max_workers=MAX_WORKERS
        )
        
        enriched_dataset = analyzer.process_dataset(
            input_path=INPUT_PATH,
            sample_size=SAMPLE_SIZE
        )
        
        # Display example results
        if enriched_dataset:
            sample_conv = enriched_dataset[0]
            logging.info(f"\nExample enriched conversation (ID: {sample_conv['id']}):")
            logging.info(f"Overall Sentiment: {sample_conv['overall_sentiment']}")
            logging.info("\nMessage-level sentiments:")
            for msg in sample_conv['conversations']:  # Changed to match structure
                role = msg['from'] if 'from' in msg else msg['role']
                sentiment = msg['sentiment']['compound'] if 'sentiment' in msg else 'N/A'
                logging.info(f"{role}: {sentiment}")
                
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()