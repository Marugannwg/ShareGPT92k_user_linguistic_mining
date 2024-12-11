from dataclasses import dataclass
from typing import List, Dict
import json
from pathlib import Path
import re
import logging
from tqdm.asyncio import tqdm
from openai_manager import GptCompletion, OpenAiManager

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

class SampledDataEncoder:
    def __init__(self, openai_manager):
        self.openai_manager = openai_manager
        self.encoding_instruction = """
        Analyze this message query (made by a human in online dialogue) and provide a binary classification JSON response with these fields:
        
        **expressive_label**: 
        1 if the message shows explicitly emotional expression, personal attitudes, or feelings 
        (e.g., express personal apologize, thank, or other feelings; thinking aloud, using phrases like "Hi!", "Awesome!", "I understand...", "Let me think about this", "That's a great question.") ; 

        0 if: 
        -- it's purely informative or directive;
        -- Just ask a question like an order, even it use a second person pronoun;
        -- use long instructions and background details that not usually communicated in normal conversations;
        -- use very short instructional verbs or phrases.
        -- talking about a background, a question, or context generally, without clear illocutionary force.
        
        **anthropomorphic_label**: 
        1 if the message explicitly treats the other party as a person, using second personal pronouns
        (e.g., explicit using of you, your, we, etc. "Can you...", "Let's..." ); 
        0 if it's not communicating as if it's not explicitly addressing a human/second-person pronouns.
        
        Examples:

        ""Who is the most Badass fictional character of all time?"
        -- Just a question without clear personal reference or pronoun, even though using a strong adjective, it's not expressive, not anthropomorphic (expressive_label=0, anthropomorphic_label=0)

        "Can you update that so that resetForm is passed in to the original function instead of using the MyForm function?"
        -- it's a directive speech act: a instruction without personal feelings; but anthropomorphic: mention "you" politely (expressive_label=0, anthropomorphic_label=1)

        "Please explain this code to me."
        -- not expressive: just an order, not anthropomorphic: treating the other party as a machine/tool (expressive_label=0, anthropomorphic_label=0)

        "Please show me the full @keyframes rule called particlesAnimation that will be used to animate our particles."
        -- similar as the previous example, only an order, self-referencing but neither personal narrative nor addressing another party using second person pronouns, not expressive, not anthropomorphic (expressive_label=0, anthropomorphic_label=0)
        
        "I run a very long running script with autoit that takes even 24 hours to complete. Sometimes I accidentally interrupt it by doing something else on my computer so that the autoit script loses the focus and cannot continue. Is it somehow avoidable?” 
        -- expressive, anthropomorphic: with personal narrative and implying feeling, seeking help in a normal conversation (expressive_label=1, anthropomorphic_label=1)

        "Great. Here is the logline to the feature film...(continues with a long logline)"
        -- An continuation of a conversation, with clear expressive use of "great" or "okay" at beginning, but just providing information, not anthropomorphic pronouns like "you". (expressive_label=1, anthropomorphic_label=0)

        "I need your help to create a program to help people who are struggling with sleeping."
        -- expressive (directly mention need for help), anthromorphic (using you) (expressive_label=1, anthropomorphic_label=1)

        "This is a descent start. Now I will provide you with the synopsis."
        -- It's likely an continuation of a conversation, with clear expressive of personal satisfaction and response with pronouns "you". (expressive_label=1, anthropomorphic_label=1)

        "Start flirting with Jean."
        -- Unclear sentence background, likely an instruction during a roleplay senario, a direct order, not expressive, not anthropomorphic (expressive_label=0, anthropomorphic_label=0)

        "ok lets take into consideration that my wake up times on mon till wed a 445am and than I take 90 min to prepare and I start study those 3 days from 6 am till 1030 in 3 blocks of 80 min study and 15 min breaks in between."
        -- Although with long, technical details, it is using "ok" and "let's" to response to the other party with illocutionary force, also expressively thinking aloud; it's expressive, anthropomorphic (expressive_label=1, anthropomorphic_label=1)

        
        Respond only with the JSON object containing these two binary values.
        """

    def _prepare_prompt(self, message_input: str) -> str:
        return f"{self.encoding_instruction}\n\n Message starts here: \n {message_input}"
    
    def _parse_response(self, response: str) -> Dict:
        try:
            # Clean markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace('```json', '', 1)
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response.rsplit('```', 1)[0]
            
            # Clean and parse response
            cleaned_response = cleaned_response.strip()
            cleaned_response = re.sub(r'""', '"', cleaned_response)
            
            parsed = json.loads(cleaned_response)
            
            # Ensure values are binary
            parsed['expressive_label'] = 1 if parsed.get('expressive_label') == 1 else 0
            parsed['anthropomorphic_label'] = 1 if parsed.get('anthropomorphic_label') == 1 else 0
            
            return parsed
        except json.JSONDecodeError as e:
            return {
                "error": "Failed to parse response",
                "raw_response": response,
                "exception_message": str(e)
            }
    
    async def encode_file(self, file_path: Path, relabel: bool = False) -> bool:
        """Encode a single JSON file with AI message analysis"""
        try:
            # Load and process file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modified = False
            human_messages = []
            message_indices = []

            # Collect human messages that need processing
            for idx, msg in enumerate(data['conversations']):
                if msg.get('from') == 'human' and (
                    relabel or (
                        'expressive_label' not in msg and 
                        'anthropomorphic_label' not in msg
                    )
                ):
                    human_messages.append(msg['value'])
                    message_indices.append(idx)

            if not human_messages:
                logging.info(f"No messages to process in {file_path.name}")
                return False

            logging.info(f"Processing {len(human_messages)} messages from {file_path.name}")

            # Process messages in batches
            batch_size = 10
            for i in range(0, len(human_messages), batch_size):
                batch_msgs = human_messages[i:i + batch_size]
                batch_indices = message_indices[i:i + batch_size]

                completions = [
                    GptCompletion(prompt=self._prepare_prompt(msg))
                    for msg in batch_msgs
                ]

                await self.openai_manager.add_gpt_completions(completions)

                # Update original data with new labels
                for completion, orig_idx in zip(completions, batch_indices):
                    if completion.error:
                        logging.error(f"API Error for message {orig_idx}: {completion.error}")
                        continue
                        
                    encoding = self._parse_response(completion.response)
                    if 'error' in encoding:
                        logging.error(f"Parse error for message {orig_idx}: {encoding['error']}")
                        logging.error(f"Raw response: {encoding['raw_response']}")
                        continue
                        
                    try:
                        data['conversations'][orig_idx]['expressive_label'] = encoding['expressive_label']
                        data['conversations'][orig_idx]['anthropomorphic_label'] = encoding['anthropomorphic_label']
                        modified = True
                        logging.debug(f"Successfully labeled message {orig_idx}")
                    except KeyError as e:
                        logging.error(f"Missing key in encoding for message {orig_idx}: {str(e)}")
                        logging.error(f"Encoding received: {encoding}")
                        continue

            # Save updated file if modified
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logging.info(f"Successfully updated {file_path.name}")
                return True

            return False

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False 