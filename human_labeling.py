import json
from pathlib import Path
import argparse
import sys

def label_conversations(target_dir, start_file=None):
    """Manually label user messages in conversations"""
    target_path = Path(target_dir)
    
    print("\nLabeling Instructions:")
    print("For each user message, enter:")
    print("1 - Expressive speech act")
    print("0 - Directive speech act")
    print("q - Quit labeling")
    print("s - Skip current file")
    print("b - Back (re-label previous message)\n")
    
    # Get list of JSON files
    json_files = list(target_path.glob('*.json'))
    json_files = [f for f in json_files if f.name != 'sampling_metadata.json']
    
    # Sort files to ensure consistent ordering
    json_files.sort()
    
    # Find starting point if specified
    if start_file:
        try:
            start_idx = [f.name for f in json_files].index(start_file)
            json_files = json_files[start_idx:]
        except ValueError:
            print(f"Starting file {start_file} not found. Starting from beginning.")
    
    for json_file in json_files:
        print(f"\nProcessing file: {json_file.name}")
        
        # Load the JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        messages_to_label = [
            msg for msg in data['conversations'] 
            if msg.get('from') == 'human' and 'expressive_speech_act' not in msg
        ]
        
        i = 0
        while i < len(messages_to_label):
            msg = messages_to_label[i]
            print("\n" + "="*50)
            print("Message:", msg.get('value', msg.get('content', '')))
            
            while True:
                label = input("Enter label (1/0/q/s/b): ").strip().lower()
                
                if label == 'q':
                    # Save progress and exit
                    if modified:
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                    print("\nLabeling progress saved. Exiting...")
                    return
                
                elif label == 's':
                    print("Skipping current file...")
                    break
                
                elif label == 'b' and i > 0:
                    i -= 1
                    messages_to_label[i].pop('expressive_speech_act', None)
                    modified = True
                    break
                
                elif label in ['0', '1']:
                    msg['expressive_speech_act'] = int(label)
                    modified = True
                    i += 1
                    break
                
                else:
                    print("Invalid input. Please enter 1, 0, q, s, or b.")
            
            if label == 's':
                break
        
        # Save the updated JSON if modified
        if modified:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"\nSaved labels for {json_file.name}")

def main():
    parser = argparse.ArgumentParser(description='Manually label conversations in JSON files')
    parser.add_argument('--dir', default='sampled_enriched_data', help='Directory containing sampled JSON files')
    parser.add_argument('--start', help='Filename to start from (optional)')
    
    args = parser.parse_args()
    
    if not Path(args.dir).exists():
        print(f"Error: Directory '{args.dir}' does not exist!")
        sys.exit(1)
    
    print(f"Starting labeling process for files in {args.dir}")
    label_conversations(args.dir, args.start)

if __name__ == "__main__":
    main() 