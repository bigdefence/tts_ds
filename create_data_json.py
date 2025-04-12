import os
import json
from pathlib import Path

def create_data_json():
    # 입력 파일과 디렉토리 경로 설정
    wav_dir = "wavs"
    dataset_path = "ko_dataset.jsonl"
    output_path = "data.json"
    
    # 결과를 저장할 리스트
    result_data = []
    
    # ko_dataset.jsonl에서 질문과 응답 쌍 추출
    instructions = []
    responses = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 빈 줄 무시
                    json_obj = json.loads(line)
                    if 'conversations' in json_obj:
                        human_msg = None
                        gpt_msg = None
                        for msg in json_obj['conversations']:
                            if msg['from'] == 'human':
                                human_msg = msg['value']
                            elif msg['from'] == 'gpt':
                                gpt_msg = msg['value']
                        if human_msg and gpt_msg:  # 둘 다 있는 경우만 추가
                            instructions.append(human_msg)
                            responses.append(gpt_msg)
    except Exception as e:
        print(f"ko_dataset.jsonl 읽기 오류: {str(e)}")
        return
    
    # wavs 디렉토리에서 .wav 파일 목록 가져오기
    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    
    # 각 wav 파일에 대해 데이터 항목 생성
    for idx, wav_file in enumerate(wav_files):
            
        data_entry = {
            "id": f"{wav_file.split('.')[0]}",
            "speech": f"wavs/{wav_file}",
            "conversations": [
                {
                    "from": "human",
                    "value": "<speech>\nPlease directly answer the questions in the user's speech."
                },
                {
                    "from": "assistant",
                    "value": outputs[idx]
                }
            ]
        }
        result_data.append(data_entry)
    
    # data.json 파일로 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        print(f"{output_path} 파일 생성 완료: {len(result_data)}개 항목")
    except Exception as e:
        print(f"data.json 쓰기 오류: {str(e)}")

if __name__ == "__main__":
    create_data_json()