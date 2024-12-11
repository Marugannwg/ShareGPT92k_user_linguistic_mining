import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
from openai_manager import OpenAiManager
from encode_sampled_data import SampledDataEncoder
from tqdm.asyncio import tqdm
import re
import argparse
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SampledDataProcessor:
    def __init__(self, input_dir: str, max_workers: int = 5, relabel: bool = False):
        self.input_dir = Path(input_dir)
        self.progress_file = self.input_dir / 'encoding_progress.json'
        self.error_log = self.input_dir / 'encoding_error.txt'
        self.max_workers = max_workers
        self.relabel = relabel
        
    def _load_progress(self) -> set:
        """Load set of completed files"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    return set(json.load(f)['completed_files'])
        except Exception as e:
            logging.error(f"Failed to load progress: {e}")
        return set()

    def _save_progress(self, completed_files: set):
        """Save progress of completed files"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({'completed_files': list(completed_files)}, f)
        except Exception as e:
            logging.error(f"Failed to save progress: {e}")

    def _log_error(self, error_msg: str):
        """Log errors to file"""
        try:
            with open(self.error_log, 'a') as f:
                f.write(f"{datetime.now()}: {error_msg}\n")
        except Exception as e:
            logging.error(f"Failed to log error: {e}")

    async def process_file(self, file_path: Path, encoder: SampledDataEncoder) -> bool:
        """Process a single file"""
        if file_path.name in self._load_progress() and not self.relabel:
            logging.info(f"Skipping completed file: {file_path.name}")
            return True

        logging.info(f"Processing file: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check for messages to process based on relabel flag
            has_messages = any(
                msg.get('from') == 'human' and (
                    self.relabel or (
                        'expressive_label' not in msg and 
                        'anthropomorphic_label' not in msg
                    )
                )
                for msg in data['conversations']
            )
            
            if not has_messages:
                logging.info(f"No messages to process in {file_path.name}, skipping")
                return True
                
        except Exception as e:
            self._log_error(f"Error checking file {file_path.name}: {str(e)}")
            return False

        # Process file with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = await encoder.encode_file(file_path, relabel=self.relabel)
                if success:
                    completed_files = self._load_progress()
                    completed_files.add(file_path.name)
                    self._save_progress(completed_files)
                    logging.info(f"Successfully processed {file_path.name}")
                    return True
                elif attempt < max_retries - 1:
                    logging.warning(f"Failed to process {file_path.name}, attempt {attempt + 2}/{max_retries}")
                    await asyncio.sleep(30)
            except Exception as e:
                self._log_error(f"Attempt {attempt + 1} failed for {file_path.name}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(30)
        
        self._log_error(f"Failed to process {file_path.name} after {max_retries} attempts")
        return False

    async def process_all_files(self):
        """Process all valid JSON files in parallel"""
        try:
            json_files = list(self.input_dir.glob('*.json'))
            json_files = [f for f in json_files if re.match(r'^[^_]+_[0-9]+\.json$', f.name)]
            
            logging.info(f"Found {len(json_files)} JSON files to process")
            logging.info(f"Mode: {'Relabeling' if self.relabel else 'New labeling only'}")
            
            openai_manager = OpenAiManager(max_requests_per_minute=2000)
            encoder = SampledDataEncoder(openai_manager)
            
            tasks = []
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def process_with_semaphore(file_path):
                async with semaphore:
                    return await self.process_file(file_path, encoder)
            
            tasks = [process_with_semaphore(f) for f in json_files]
            
            results = []
            for task in tqdm(asyncio.as_completed(tasks), 
                           total=len(tasks), 
                           desc="Processing files"):
                results.append(await task)
            
            successful = sum(1 for r in results if r)
            logging.info(f"Processing completed. Successfully processed {successful}/{len(json_files)} files")

        except Exception as e:
            self._log_error(f"Critical error in process_all_files: {str(e)}")
            raise

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--relabel', action='store_true', help='Relabel existing files')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        processor = SampledDataProcessor(
            'enriched_data', ###
            max_workers=args.workers,
            relabel=args.relabel
        )
        await processor.process_all_files()
        logging.info("Processing completed")
        
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 

## python process_sampled_data.py --workers 5