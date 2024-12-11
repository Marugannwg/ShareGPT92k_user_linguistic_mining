from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ProcessPoolExecutor
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
import re
import json
from tqdm import tqdm
from pathlib import Path
import logging
from langdetect import detect, DetectorFactory
import numpy as np

# Set seed for language detection
DetectorFactory.seed = 42

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class ConversationAnalyzer:
    def __init__(self, output_dir='processed_data', max_workers=8):
        self.analyzer = SentimentIntensityAnalyzer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.stopwords = set(stopwords.words('english'))
        
        # Initialize patterns
        self._init_patterns()
        
        logging.info(f"Initialized analyzer with {max_workers} workers")
    
    def _init_patterns(self):
        """Initialize all regex patterns used in analysis"""
        # Self-referential and second person patterns
        self.self_ref_pattern = re.compile(r'\b(i|i\'m|me|we|us|my|mine|myself)\b', re.IGNORECASE)
        self.second_person_pattern = re.compile(r'\b(you|your|you\'re|yours|we|us|our)\b', re.IGNORECASE)
        
        # Question patterns
        self.question_patterns = {
            'what': re.compile(r'\b(what|whats|what\'s)\b', re.IGNORECASE),
            'how': re.compile(r'\b(how)\b', re.IGNORECASE),
            'why': re.compile(r'\b(why)\b', re.IGNORECASE),
            'can_could': re.compile(r'\b(can|could)\b', re.IGNORECASE),
            'general': re.compile(r'\?')
        }
        
        # Discourse markers
        self.discourse_markers = {
            'elaboration': re.compile(r'\b(specifically|particularly|for example|such as)\b', re.IGNORECASE),
            'contrast': re.compile(r'\b(however|but|although|though|nonetheless)\b', re.IGNORECASE),
            'causality': re.compile(r'\b(because|therefore|thus|consequently)\b', re.IGNORECASE),
            'hedging': re.compile(r'\b(maybe|perhaps|probably|possibly|might|could)\b', re.IGNORECASE)
        }
        
        # Clause markers
        self.clause_markers = re.compile(r'\b(because|although|if|unless|while|when|after|before)\b', re.IGNORECASE)

    def detect_language(self, text):
        """Detect language of text"""
        try:
            return detect(text)
        except:
            return 'unknown'

    def get_lexical_features(self, text, tokens, pos_tags):
        """Extract lexical features from text"""
        # Basic counts
        token_count = len(tokens)
        
        # Adjectives and adverbs
        adj_adv_count = sum(1 for _, tag in pos_tags if tag.startswith(('JJ', 'RB')))
        
        # Pronouns
        self_ref_count = len(self.self_ref_pattern.findall(text))
        second_person_count = len(self.second_person_pattern.findall(text))
        
        return {
            'adj_adv_freq': adj_adv_count,
            'adj_adv_ratio': adj_adv_count / token_count if token_count > 0 else 0,
            'self_ref_freq': self_ref_count,
            'self_ref_ratio': self_ref_count / token_count if token_count > 0 else 0,
            'second_person_freq': second_person_count,
            'second_person_ratio': second_person_count / token_count if token_count > 0 else 0,
            'token_count': token_count
        }

    def get_advanced_features(self, text, tokens, pos_tags):
        """Extract advanced linguistic features"""
        # Sentence analysis
        sentences = sent_tokenize(text)
        avg_sentence_length = len(tokens) / len(sentences) if sentences else 0
        
        # Question analysis
        question_types = {
            qtype: len(pattern.findall(text))
            for qtype, pattern in self.question_patterns.items()
        }
        
        # Vocabulary richness
        content_words = [word.lower() for word, tag in pos_tags 
                        if word.isalnum() and word.lower() not in self.stopwords]
        vocab_richness = len(set(content_words)) / len(content_words) if content_words else 0
        
        # Verb tense distribution
        verb_tenses = defaultdict(int)
        for _, tag in pos_tags:
            if tag.startswith('VB'):
                verb_tenses[tag] += 1
        
        # Named Entity Recognition
        named_entities = nltk.ne_chunk(pos_tags)
        entity_counts = defaultdict(int)
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entity_counts[chunk.label()] += 1
        
        # Discourse markers
        discourse_features = {
            marker: len(pattern.findall(text))
            for marker, pattern in self.discourse_markers.items()
        }
        
        # Syntactic complexity
        clause_count = len(self.clause_markers.findall(text))
        
        return {
            'structural_features': {
                'avg_sentence_length': avg_sentence_length,
                'clause_complexity': clause_count / len(sentences) if sentences else 0,
                'vocabulary_richness': vocab_richness
            },
            'interaction_features': {
                'question_types': question_types,
                'discourse_markers': discourse_features
            },
            'linguistic_features': {
                'verb_tense_dist': dict(verb_tenses),
                'named_entities': dict(entity_counts)
            }
        }
    def analyze_message(self, message):
        """Analyze a single message with core features (sentiment, pronouns) and optional advanced features."""
        try:
            # Get content safely
            content = message.get('value', message.get('content', ''))
            if not content:  # If no content found, return original message
                return message
                
            # Core features that we want for all messages
            enriched_message = message.copy()
            
            # 1. Always do sentiment analysis
            sentiment = self.analyzer.polarity_scores(content)
            
            # 2. Always do pronoun counting (core lexical features)
            core_lexical = {
                'self_ref_freq': len(self.self_ref_pattern.findall(content)),
                'second_person_freq': len(self.second_person_pattern.findall(content)),
            }
            
            # Try to detect language, but don't let it fail the analysis
            try:
                lang = self.detect_language(content)
            except:
                lang = 'unknown'
                
            # Basic enrichment that should always work
            enriched_message.update({
                'language': lang,
                'sentiment': sentiment,
                'core_lexical': core_lexical
            })
            
            # Only try advanced features for English text
            if lang == 'en':
                try:
                    # Advanced features - allow these to fail gracefully
                    tokens = word_tokenize(content)
                    pos_tags = pos_tag(tokens)
                    
                    # Add token count to core lexical features
                    core_lexical['token_count'] = len(tokens)
                    
                    # Calculate ratios
                    core_lexical.update({
                        'self_ref_ratio': core_lexical['self_ref_freq'] / len(tokens) if tokens else 0,
                        'second_person_ratio': core_lexical['second_person_freq'] / len(tokens) if tokens else 0
                    })
                    
                    # Try to get additional features but don't fail if they don't work
                    try:
                        lexical_features = self.get_lexical_features(content, tokens, pos_tags)
                        advanced_features = self.get_advanced_features(content, tokens, pos_tags)
                        
                        enriched_message.update({
                            'lexical_features': lexical_features,
                            'advanced_features': advanced_features
                        })
                    except Exception as e:
                        logging.debug(f"Advanced feature extraction failed: {str(e)}")
                        
                except Exception as e:
                    logging.debug(f"Token processing failed: {str(e)}")
            
            return enriched_message
                
        except Exception as e:
            logging.error(f"Error in analyze_message core processing: {str(e)}")
            # Return original message with minimal sentiment analysis
            try:
                return {
                    **message,
                    'sentiment': self.analyzer.polarity_scores(str(message.get('value', message.get('content', ''))))
                }
            except:
                return message
    # def analyze_message(self, message):
    #     try:
    #         content = message['value'] if 'value' in message else message['content']
            
    #         # Detect language
    #         lang = self.detect_language(content)
            
    #         # Only perform detailed analysis for English text
    #         if lang == 'en':
    #             # Basic tokenization and POS tagging
    #             tokens = word_tokenize(content)
    #             pos_tags = pos_tag(tokens)
                
    #             # Get all features
    #             sentiment = self.analyzer.polarity_scores(content)
    #             lexical_features = self.get_lexical_features(content, tokens, pos_tags)
    #             advanced_features = self.get_advanced_features(content, tokens, pos_tags)
                
    #             enriched_message = message.copy()
    #             enriched_message.update({
    #                 'language': lang,
    #                 'sentiment': sentiment,
    #                 'lexical_features': lexical_features,
    #                 'advanced_features': advanced_features
    #             })
    #         else:
    #             # Basic enrichment for non-English text
    #             enriched_message = message.copy()
    #             enriched_message.update({
    #                 'language': lang,
    #                 'sentiment': {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0},
    #                 'lexical_features': None,
    #                 'advanced_features': None
    #             })
            
    #         return enriched_message
            
    #     except Exception as e:
    #         logging.error(f"Error in analyze_message: {str(e)}")
    #         return message

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
            
            # Analyze each message
            enriched_messages = [self.analyze_message(msg) for msg in conversation]
            
            # Aggregate features
            language_dist = defaultdict(int)
            valid_sentiments = []
            valid_lexical = []
            valid_advanced = []
            
            for msg in enriched_messages:
                # Count languages
                language_dist[msg.get('language', 'unknown')] += 1
                
                # Collect valid features
                if 'sentiment' in msg:
                    valid_sentiments.append(msg['sentiment'])
                if msg.get('lexical_features'):
                    valid_lexical.append(msg['lexical_features'])
                if msg.get('advanced_features'):
                    valid_advanced.append(msg['advanced_features'])
            
            # Calculate overall metrics
            overall_sentiment = self._aggregate_sentiments(valid_sentiments)
            overall_lexical = self._aggregate_lexical(valid_lexical)
            overall_advanced = self._aggregate_advanced(valid_advanced)
            
            # Create enriched conversation item
            enriched_item = {
                'id': item.get('id', 'unknown'),
                'conversations': enriched_messages,
                'overall_sentiment': overall_sentiment,
                'overall_lexical': overall_lexical,
                'overall_advanced': overall_advanced,
                'conversation_metadata': {
                    'language_distribution': dict(language_dist),
                    'turn_count': len(enriched_messages),
                    'total_tokens': sum(msg.get('lexical_features', {}).get('token_count', 0) 
                                     for msg in enriched_messages)
                }
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

    def _aggregate_sentiments(self, sentiments):
        """Aggregate sentiment scores"""
        if not sentiments:
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
        
        return {
            'compound': np.mean([s['compound'] for s in sentiments]),
            'pos': np.mean([s['pos'] for s in sentiments]),
            'neu': np.mean([s['neu'] for s in sentiments]),
            'neg': np.mean([s['neg'] for s in sentiments])
        }

    def _aggregate_lexical(self, features):
        """Aggregate lexical features"""
        if not features:
            return None
        
        total_tokens = sum(f['token_count'] for f in features)
        
        return {
            'adj_adv_freq': sum(f['adj_adv_freq'] for f in features),
            'adj_adv_ratio': sum(f['adj_adv_freq'] for f in features) / total_tokens if total_tokens > 0 else 0,
            'self_ref_freq': sum(f['self_ref_freq'] for f in features),
            'self_ref_ratio': sum(f['self_ref_freq'] for f in features) / total_tokens if total_tokens > 0 else 0,
            'second_person_freq': sum(f['second_person_freq'] for f in features),
            'second_person_ratio': sum(f['second_person_freq'] for f in features) / total_tokens if total_tokens > 0 else 0,
            'total_tokens': total_tokens
        }

    def _aggregate_advanced(self, features):
        """Aggregate advanced features"""
        if not features:
            return None
        
        # Aggregate structural features
        structural = {
            'avg_sentence_length': np.mean([f['structural_features']['avg_sentence_length'] for f in features]),
            'clause_complexity': np.mean([f['structural_features']['clause_complexity'] for f in features]),
            'vocabulary_richness': np.mean([f['structural_features']['vocabulary_richness'] for f in features])
        }
        
        # Aggregate interaction features
        question_types = defaultdict(int)
        discourse_markers = defaultdict(int)
        for f in features:
            for qtype, count in f['interaction_features']['question_types'].items():
                question_types[qtype] += count
            for marker, count in f['interaction_features']['discourse_markers'].items():
                discourse_markers[marker] += count
        
        # Aggregate linguistic features
        verb_tenses = defaultdict(int)
        entities = defaultdict(int)
        for f in features:
            for tense, count in f['linguistic_features']['verb_tense_dist'].items():
                verb_tenses[tense] += count
            for entity, count in f['linguistic_features']['named_entities'].items():
                entities[entity] += count
        
        return {
            'structural_features': structural,
            'interaction_features': {
                'question_types': dict(question_types),
                'discourse_markers': dict(discourse_markers)
            },
            'linguistic_features': {
                'verb_tense_dist': dict(verb_tenses),
                'named_entities': dict(entities)
            }
        }

    def process_dataset(self, input_path, sample_size=None):
        """Process the dataset with all analyses"""
        try:
            # Load data
            logging.info(f"Loading data from {input_path}")
            with open(input_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
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
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('conversation_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    # Configuration
    INPUT_PATH = 'raw_data/ShareGPT_V3_unfiltered_cleaned_split.json'
    OUTPUT_DIR = 'enriched_data'
    SAMPLE_SIZE = None  # Process entire dataset
    MAX_WORKERS = 16
    
    try:
        # Initialize and run analyzer
        analyzer = ConversationAnalyzer(
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
            logging.info(f"Language distribution: {sample_conv['conversation_metadata']['language_distribution']}")
            logging.info(f"Overall Sentiment: {sample_conv['overall_sentiment']}")
            logging.info(f"Turn count: {sample_conv['conversation_metadata']['turn_count']}")
                
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()