from dataclasses import dataclass
from typing import Dict, List
import json
from pathlib import Path
import logging
import asyncio
from openai_manager_ebd import GptCompletion, OpenAiManager

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

class TopicEmbeddingProcessor:
    def __init__(self, openai_manager: OpenAiManager):
        self.openai_manager = openai_manager
        
    async def process_json_file(self, file_path: Path, output_path: Path = None) -> bool:
        """Process a JSON file containing topics and add embeddings"""
        try:
            # Create output directory if it doesn't exist
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Collect all unique topics that don't have embeddings yet
            topics_to_process = []
            for topic, topic_data in data.items():
                if 'embedding' not in topic_data:
                    topics_to_process.append(topic)

            if not topics_to_process:
                logging.info(f"No topics to process in {file_path.name}")
                return False

            logging.info(f"Processing {len(topics_to_process)} topics from {file_path.name}")

            # Process topics in batches
            batch_size = 500  # Adjust based on API limits
            modified = False
            
            for i in range(0, len(topics_to_process), batch_size):
                batch_topics = topics_to_process[i:i + batch_size]
                logging.info(f"Processing batch {i//batch_size + 1} of {(len(topics_to_process)-1)//batch_size + 1}")
                
                # Create completion objects for the batch
                completions = [
                    GptCompletion(prompt=topic)
                    for topic in batch_topics
                ]

                # Get embeddings for the batch
                try:
                    await self.openai_manager.add_gpt_completions(completions)
                except Exception as e:
                    logging.error(f"Batch processing error: {str(e)}")
                    continue

                # Update original data with embeddings
                for completion, topic in zip(completions, batch_topics):
                    if completion.error:
                        logging.error(f"API Error for topic {topic}: {completion.error}")
                        continue
                    
                    if completion.response is None:
                        logging.error(f"No embedding received for topic {topic}")
                        continue
                    
                    try:
                        data[topic]['embedding'] = completion.response
                        modified = True
                        logging.info(f"Successfully added embedding for topic: {topic}")
                    except Exception as e:
                        logging.error(f"Error updating embedding for topic {topic}: {str(e)}")
                        continue

                # Save after each batch
                if modified:
                    output_path = output_path or file_path
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logging.info(f"Saved batch progress to {output_path.name}")

            return modified

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False

async def main():
    # Initialize OpenAI manager
    openai_manager = OpenAiManager()
    processor = TopicEmbeddingProcessor(openai_manager)
    
    # Process domain topics
    domain_topics_path = Path('topic_words/domain_topics.json')
    domain_topics_output = Path('topic_words/domain_topics_with_embeddings.json')
    await processor.process_json_file(domain_topics_path, domain_topics_output)
    
    # # Process task types
    # task_types_path = Path('topic_words/task_types.json')
    # task_types_output = Path('topic_words/task_types_with_embeddings.json')
    # await processor.process_json_file(task_types_path, task_types_output)

if __name__ == "__main__":
    asyncio.run(main()) 