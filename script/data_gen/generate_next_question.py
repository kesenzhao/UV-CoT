import json
import argparse

def convert_json_to_jsonl(input_file, output_file, include_answers=False):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    merged_data = {}
    
    for item in data:
        question_id = item['question_id']
        if question_id not in merged_data:
            merged_data[question_id] = {
                'ds_question_id': item['ds_question_id'],
                'image_path': item['metainfos']['image_path'],
                'question': item['raw_question'],
                'metainfos': item['metainfos']['metainfos'],
                'answers': item['answer']
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for question_id, item in merged_data.items():
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON to JSONL format.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSONL file')

    args = parser.parse_args()

    convert_json_to_jsonl(args.input_file, args.output_file)