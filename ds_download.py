from datasets import load_dataset
import json
import os

# 데이터셋 로드
dataset = load_dataset("maywell/koVast")

# 'train' 데이터 저장 (처음 1000개만)
train_data = dataset['train'].select(range(1000))

# JSONL 파일을 현재 디렉토리에 저장
output_path = "ko_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for example in train_data:
        json.dump(example, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ koVast train 데이터 1000개가 현재 디렉토리에 저장되었습니다: {output_path}")
