import sys
sys.path.append('third_party/Matcha-TTS')
sys.path.append('third_party/WeTextProcessing')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from tqdm import tqdm
import os
import json
import torch
import time
import argparse
import gc
from functools import partial

# 실시간 출력 보장
print = partial(print, flush=True)

# PyTorch 고정 성능 설정
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configuration
MODEL_PATH = 'pretrained_models/CosyVoice2-0.5B'
FILE_PATH = 'ko_dataset.jsonl'
OUTPUT_DIR = 'wavs'
PROMPT_PATH = './asset/zero_shot_prompt.wav'
SPEAKER_DESC = '친절하고 따뜻한 한국어 여성 목소리로 말해줘.'
MAX_SAMPLES = 8000
START_INDEX = 1
NUM_MESSAGES = 5000

# Argument parser
parser = argparse.ArgumentParser(description='Process CosyVoice2 messages with custom start index and count.')
parser.add_argument('--start-index', type=int, default=START_INDEX)
parser.add_argument('--num-messages', type=int, default=NUM_MESSAGES)
parser.add_argument('--memory-limit', type=float, default=0.7)
args = parser.parse_args()
START_INDEX = args.start_index
NUM_MESSAGES = args.num_messages
MEMORY_LIMIT = args.memory_limit

# Output 디렉터리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GPU 메모리 비우기 함수
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# 사용자 메시지 추출 함수
def extract_user_messages(file_path, max_messages=8000):
    messages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                json_obj = json.loads(line)
                if 'conversations' in json_obj:
                    for msg in json_obj['conversations']:
                        if msg['from'] == 'user':
                            messages.append(msg['value'])
                            if len(messages) >= max_messages:
                                return messages
        return messages
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return []

# 단일 문장 처리 함수
@torch.inference_mode()
def process_single_text(cosyvoice, text, output_path, prompt_speech_16k, retries=1):
    from torch.cuda.amp import autocast
    try:
        clear_memory()
        torch.cuda.empty_cache()

        with autocast(dtype=torch.float16):
            result = next(cosyvoice.inference_zero_shot(
                tts_text=text,
                prompt_text=SPEAKER_DESC,  # 스타일 적용에 필요
                prompt_speech_16k=prompt_speech_16k,
                stream=False,
                text_frontend=False
            ))

        torchaudio.save(output_path, result['tts_speech'], cosyvoice.sample_rate)
        return True

    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and retries > 0:
            print("⚠️ CUDA OOM - retrying after 5s...")
            clear_memory()
            torch.cuda.empty_cache()
            time.sleep(5)
            return process_single_text(cosyvoice, text, output_path, prompt_speech_16k, retries - 1)
        else:
            print(f"❌ RuntimeError: {e}")
    except Exception as e:
        print(f"❌ Other error: {e}")
    return False

# 실행
try:
    print(f"📥 Loading dataset from {FILE_PATH}")
    user_messages = extract_user_messages(FILE_PATH, MAX_SAMPLES)

    if not user_messages:
        print("🚫 No messages found.")
        sys.exit(1)

    start_idx = START_INDEX - 1
    if start_idx >= len(user_messages):
        print(f"🚫 Start index {START_INDEX} exceeds message count.")
        sys.exit(1)

    if NUM_MESSAGES < 1:
        print("🚫 NUM_MESSAGES must be at least 1.")
        sys.exit(1)

    max_available = len(user_messages) - start_idx
    if NUM_MESSAGES > max_available:
        print(f"⚠️ Adjusting NUM_MESSAGES to available: {max_available}")
        NUM_MESSAGES = max_available

    user_messages = user_messages[start_idx:start_idx + NUM_MESSAGES]
    print(f"🔄 Processing {len(user_messages)} messages from index {START_INDEX}...")

    # Prompt 음성 로딩 및 앞부분 노이즈 제거 (0.5초 이후부터 사용)
    raw_prompt = load_wav(PROMPT_PATH, 16000)
    prompt_speech_16k = raw_prompt[:, int(0.5 * 16000):]  # 0.5초 제거

    # 메모리 확보 및 모델 로딩
    clear_memory()
    if torch.cuda.is_available():
        print("🚀 CUDA available - reserving memory buffer")
        reserve_memory_mb = 2048
        try:
            buffer_tensor = torch.empty(reserve_memory_mb * 1024 * 1024, dtype=torch.uint8, device='cuda')
            print(f"🧠 Reserved {reserve_memory_mb}MB buffer on GPU")
        except Exception as e:
            print(f"⚠️ Warning: Memory reserve failed: {e}")

    print(f"🔧 Loading CosyVoice2 model from {MODEL_PATH}")
    load_start = time.time()
    cosyvoice = CosyVoice2(
        MODEL_PATH,
        load_jit=False,
        load_trt=False,
        fp16=True,
        use_flow_cache=False
    )
    cosyvoice = cosyvoice.half()  # 전체 모델 FP16 적용
    print(f"✅ Model loaded in {time.time() - load_start:.2f}s")

    # 메시지 반복 처리
    total_start = time.time()
    processed_count = 0
    success_count = 0

    for i, text in enumerate(tqdm(user_messages, desc="🔊 Synthesizing", ncols=80)):
        output_path = os.path.join(OUTPUT_DIR, f'sample_{start_idx + i}.wav')
        success = process_single_text(cosyvoice, text, output_path, prompt_speech_16k, retries=2)
        if success:
            print(f"✅ sample_{start_idx + i} saved")
            success_count += 1
        processed_count += 1

    # 처리 결과 출력
    total_time = time.time() - total_start
    print(f"\n🎉 Finished: {processed_count} processed / {success_count} succeeded")
    print(f"⏱️ Total time: {total_time:.2f}s, Avg per message: {total_time / (processed_count or 1):.2f}s")
    print(f"📁 Output saved in: {OUTPUT_DIR}")

except Exception as e:
    print(f"❗ Unhandled error: {e}")
finally:
    if 'buffer_tensor' in locals():
        del buffer_tensor
    if 'cosyvoice' in locals():
        del cosyvoice
    clear_memory()
