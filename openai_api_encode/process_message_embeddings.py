from dataclasses import dataclass
from typing import Dict, List
import json
from pathlib import Path
import logging
import asyncio
import pandas as pd
from openai_manager_ebd import GptCompletion, OpenAiManager

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

class MessageEmbeddingProcessor:
    def __init__(self, openai_manager: OpenAiManager):
        self.openai_manager = openai_manager
        
    async def process_dataframe(self, df: pd.DataFrame, output_path: Path) -> bool:
        """Process DataFrame messages and create embeddings"""
        try:
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize or load existing data
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logging.info(f"Loaded existing data from {output_path.name}")
            else:
                data = {}
                
            # Collect messages that don't have embeddings yet
            messages_to_process = []
            message_ids = []
            
            for _, row in df.iterrows():
                message_id = row['message_id']
                if message_id not in data:
                    messages_to_process.append(row['text'])
                    message_ids.append(message_id)

            if not messages_to_process:
                logging.info("No messages to process")
                return False

            logging.info(f"Processing {len(messages_to_process)} messages")

            # Process messages in batches
            batch_size = 500  # Adjust based on API limits
            modified = False
            
            for i in range(0, len(messages_to_process), batch_size):
                batch_messages = messages_to_process[i:i + batch_size]
                batch_ids = message_ids[i:i + batch_size]
                logging.info(f"Processing batch {i//batch_size + 1} of {(len(messages_to_process)-1)//batch_size + 1}")
                
                # Create completion objects for the batch
                completions = [
                    GptCompletion(prompt=msg)
                    for msg in batch_messages
                ]

                # Get embeddings for the batch
                try:
                    await self.openai_manager.add_gpt_completions(completions)
                except Exception as e:
                    logging.error(f"Batch processing error: {str(e)}")
                    continue

                # Update data with embeddings
                for completion, msg_id, msg_text in zip(completions, batch_ids, batch_messages):
                    if completion.error:
                        logging.error(f"API Error for message {msg_id}: {completion.error}")
                        continue
                    
                    if completion.response is None:
                        logging.error(f"No embedding received for message {msg_id}")
                        continue
                    
                    try:
                        data[msg_id] = {
                            "message_id": msg_id,
                            "text": msg_text,
                            "embedding": completion.response
                        }
                        modified = True
                        logging.info(f"Successfully added embedding for message: {msg_id}")
                    except Exception as e:
                        logging.error(f"Error updating embedding for message {msg_id}: {str(e)}")
                        continue

                # Save after each batch
                if modified:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logging.info(f"Saved batch progress to {output_path.name}")

            return modified

        except Exception as e:
            logging.error(f"Error processing messages: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False

async def main():
    # Initialize OpenAI manager
    openai_manager = OpenAiManager()
    processor = MessageEmbeddingProcessor(openai_manager)
    
    # Load DataFrame
    df = pd.read_pickle("merged_df.pkl")
    
    # Process messages
    output_path = Path('message_embeddings/message_embeddings.json')
    await processor.process_dataframe(df, output_path)

if __name__ == "__main__":
    asyncio.run(main())