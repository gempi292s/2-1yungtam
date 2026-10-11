# 가려진 영어 단어 이미지 복원

영어 단어 이미지에서 일부 글자가 가려졌을 때, 가려진 부분을 복원하는 딥러닝 모델입니다.

- **2024 버전**: Keras Conv 오토인코더 (`deeplearning.py`, `autoencoder_model.h5`, `result.py`)
- **2026 버전**: PyTorch `WordInpainter` (`inpaint/`, `train.py`, `app.py`, `word_inpainter.pt`)

2024 버전 파일은 그대로 두었습니다.

## 2026 버전에서 바뀐 점

| | 2024 Autoencoder | 2026 WordInpainter |
|---|---|---|
| 프레임워크 | TensorFlow/Keras | PyTorch |
| 구조 | Conv 5층, 스킵 연결 없음 | ConvNeXt U-Net + Transformer 병목 + 고해상도 refine |
| 마스크 정보 | 없음 (어디가 가려졌는지 모름) | 마스크 채널 입력 → 가려진 곳만 복원, 나머지는 원본 유지 |
| 문맥 이용 | 3x3 conv 의 좁은 시야 | Transformer 가 단어 전체를 보고 가려진 글자 추론 |
| 글자 인식 | 없음 | CTC 보조 헤드: 학습 중 단어 읽기를 함께 배우고, 추론 시 읽은 단어를 출력 |
| 손실 | BCE | 마스크 가중 BCE + 마스크 L1 + CTC |
| 학습 | 5 epoch, Adam | AdamW + warmup/cosine, EMA 가중치, gradient clipping, drop-path |
| 데이터 | 디스크에 저장한 고정 이미지 1세트 | 매 스텝 새로 합성 (여러 폰트·크기·위치·노이즈·블러, 글자/임의 사각형 마스크) |
| 마스크 위치 | 글자 bbox 폭을 더해서 계산 → 실제 글자와 어긋남 | 실제 렌더링 좌표(`textlength`/`textbbox`)로 정확히 계산 |
| 평가 | 없음 | 단어 단위 train/val/test 분리, 처음 보는 단어로 평가 |
| 입력 크기 | 200x50 고정 (GUI 에서 선택 영역을 늘려서 넣음) | 아무 크기 → 비율 유지로 맞춘 뒤 원래 해상도로 되돌림 |

## 사용법

```bash
pip install -r requirements.txt

python app.py                                  # GUI: 이미지 열기 → 드래그로 영역 선택 → 복원
python restore.py 입력.png --box 40 5 90 45     # 명령줄 복원 (box 는 여러 개 가능)
python restore.py --demo hello                  # 데모 단어로 바로 테스트
python evaluate.py                              # 2024 vs 2026 비교 + comparison.png
python train.py --steps 20000 --batch 64        # 다시 학습 (GPU 있으면 자동 사용)
```

## 결과

학습: CPU 4코어, 배치 32, 4,800 스텝 (약 80분), 파라미터 1.67M.
평가: 학습에 쓰지 않은 test 단어 1,000개.

**2024 방식 데이터** (고정 폰트/위치, 글자 단위 가림)

| 모델 | 가려진 영역 L1 ↓ | PSNR ↑ | 가려진 영역 픽셀 정확도 ↑ |
|---|---|---|---|
| 2024 Autoencoder | 0.365 | 16.65 | 0.635 |
| 2024 Autoencoder (+마스크 합성) | 0.365 | 19.07 | 0.635 |
| **2026 WordInpainter** | **0.252** | **23.02** | **0.771** |

**증강 데이터** (다양한 폰트/위치/노이즈/임의 사각형)

| 모델 | 가려진 영역 L1 ↓ | PSNR ↑ | 가려진 영역 픽셀 정확도 ↑ |
|---|---|---|---|
| 2024 Autoencoder | 0.255 | 15.47 | 0.771 |
| 2024 Autoencoder (+마스크 합성) | 0.255 | 21.03 | 0.771 |
| **2026 WordInpainter** | **0.107** | **27.95** | **0.870** |

![비교](comparison.png) (왼쪽부터: 입력 | 2024 | 2026 | 정답)

### 한계와 다음 단계

- 이 환경에는 GPU 가 없어서 CPU 로 80분만 학습했습니다. 가려진 글자가 아직 흐릿하게 나오고,
  여러 글자가 연속으로 가려지면 다른 글자로 복원되기도 합니다.
- CTC 글자 인식 헤드는 이 학습량에서는 아직 단어를 읽는 단계에 도달하지 못했습니다 (단어 정확도 0%).
  따라서 GUI 의 "모델이 읽은 단어" 는 현재 체크포인트에서는 의미가 없습니다.
- GPU 에서 `python train.py --steps 30000 --batch 128` 처럼 더 오래 학습하면 개선될 여지가 큽니다.
- 2024 모델은 Arial 로 학습됐는데, 이 평가 환경에는 Arial 이 없어 DejaVu 등 다른 폰트로 평가했습니다
  (2024 모델에 다소 불리한 조건).
- 2024 데이터에서 가린 상자는 실제 글자 위치와 어긋나 있었습니다 (`데이터전처리.py` 의 x 좌표 계산).
