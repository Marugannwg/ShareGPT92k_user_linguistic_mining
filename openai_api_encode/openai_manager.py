import os
import asyncio
import logging
from typing import List
from dataclasses import dataclass
from openai import AsyncOpenAI, OpenAIError

@dataclass
class GptCompletion:
    prompt: str
    response: str = ""
    error: str = None

class OpenAiManager:
    def __init__(self, max_requests_per_minute: int = 3000, max_attempts: int = 3) -> None:
        """Initialize the OpenAiManager with API key, request limits, and model settings."""
        self.api_key = self._get_api_key()
        self.max_requests_per_minute = max_requests_per_minute
        self.max_attempts = max_attempts
        self.gpt_model = "gpt-4o-mini"  # or your preferred model
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    @staticmethod
    def _get_api_key() -> str:
        """Safely retrieve the OpenAI API key from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return api_key
        
    async def add_gpt_completion(self, completion_object: GptCompletion, semaphore: asyncio.Semaphore) -> None:
        attempts = 0
        delay = 20

        while attempts < self.max_attempts:
            attempts += 1
            try:
                async with semaphore:
                    #logging.info(f"Sending request to OpenAI (attempt {attempts}/{self.max_attempts})")
                    response = await self.client.chat.completions.create(
                        model=self.gpt_model,
                        messages=[{"role": "user", "content": completion_object.prompt}],
                        temperature=0
                    )

                completion_object.response = response.choices[0].message.content
                return
            except OpenAIError as e:
                completion_object.error = str(e)
                logging.error(f"OpenAI API error: {str(e)}")
                if attempts >= self.max_attempts:
                    break
                logging.info(f"Retrying in {delay * attempts} seconds...")
                await asyncio.sleep(delay * attempts)

    async def add_gpt_completions(self, completion_objects: List[GptCompletion]) -> List[GptCompletion]:
        """Add GPT completions to a list of completion objects concurrently."""
        semaphore = asyncio.Semaphore(self.max_requests_per_minute)
        tasks = [self.add_gpt_completion(obj, semaphore) for obj in completion_objects]
        await asyncio.gather(*tasks)
        return completion_objects