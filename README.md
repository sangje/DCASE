# DCASE 2023 Challenge Task 6b: Audio-Text Cross-Modal Retrieval

이 프로젝트는 오디오-텍스트 상호 검색(Audio-Text Cross-Modal Retrieval)을 위한 딥러닝 모델을 제공합니다. 사용자는 이 코드를 활용하여 오디오 클립과 텍스트 설명 간의 검색 작업을 수행하는 모델을 학습하고 평가할 수 있습니다.

## 주요 기능

*   **다양한 모델 아키텍처 지원**: `ASE` (Audio-Sentence Embedding) 모델을 기반으로 하며, 필요에 따라 오디오 및 텍스트 인코더를 유연하게 변경할 수 있습니다.
*   **다중 손실 함수**: `Triplet Loss`, `NT-Xent Loss`, `InfoNCE Loss` 등 다양한 메트릭 러닝 손실 함수를 지원하여 연구 목적에 맞게 선택할 수 있습니다.
*   **데이터셋 지원**: `AudioCaps`와 `Clotho` 데이터셋을 기본으로 지원하며, `data_handling` 모듈을 통해 다른 데이터셋으로 확장할 수 있습니다.
*   **상세한 학습 과정 로깅**: `TensorBoard`와 `loguru`를 사용하여 학습 과정 중의 손실(loss) 및 성능 지표(recall, mAP 등)를 시각화하고 기록합니다.

## 프로젝트 구조

```
DCASE/
├── data/                 # 데이터셋 (AudioCaps, Clotho)
│   ├── AudioCaps/
│   └── Clotho/
├── data_handling/        # 데이터 로딩 및 전처리
│   └── DataLoader.py
├── models/               # 모델 아키텍처 정의
│   ├── ASE_model.py
│   ├── AudioEncoder.py
│   ├── TextEncoder.py
│   └── ...
├── settings/             # 설정 파일
│   └── settings.yaml
├── tools/                # 유틸리티 및 손실 함수
│   ├── loss.py
│   ├── InfoNCE.py
│   └── ...
├── trainer/              # 모델 학습 및 평가 로직
│   └── trainer.py
├── train.py              # 학습 실행 스크립트
└── README.md             # 프로젝트 설명
```

## 시작하기

### 1. 환경 설정

*   저장소 복제:
    ```bash
    git clone https://github.com/sangje/DCASE.git
    ```
*   Conda 환경 생성 및 활성화:
    ```bash
    conda env create -f environment.yaml -n dcase
    conda activate dcase
    ```
    *참고: 이 환경은 RTX 3090, CUDA 11에서 테스트되었습니다.*

### 2. 데이터셋 준비

*   **AudioCaps**: [AudioCaps 홈페이지](https://github.com/XinhaoMei/ACT)에서 다운로드할 수 있습니다.
*   **Clotho**: [Zenodo](https://zenodo.org/record/4783391#.YkRHxTx5_kk)에서 다운로드할 수 있습니다.

다운로드한 오디오 파일(`*.wav`)을 다음 구조에 맞게 `data` 디렉토리 내에 배치합니다.

```
data/
├── AudioCaps/
│   ├── csv_files/
│   └── waveforms/
│       ├── train/
│       ├── val/
│       └── test/
└── Clotho/
    ├── csv_files/
    └── waveforms/
        ├── train/
        ├── val/
        └── test/
```

### 3. 사전 학습된 인코더

*   사전 학습된 오디오 인코더(`Cnn14.pth`, `ResNet38.pth`)는 [여기](https://github.com/qiuqiangkong/audioset_tagging_cnn)에서 다운로드할 수 있습니다.
*   다운로드한 모델을 `pretrained_models/audio_encoder` 디렉토리에 저장합니다.

## 모델 학습 및 평가

### 1. 설정 수정

`settings/settings.yaml` 파일에서 데이터셋 경로, 모델 하이퍼파라미터, 손실 함수 등을 설정할 수 있습니다.

### 2. 학습 실행

다음 명령어를 사용하여 모델 학습을 시작합니다. 커맨드라인 인자를 통해 `settings.yaml`의 설정을 덮어쓸 수 있습니다.

```bash
python train.py -n <experiment_name> [options]
```

**주요 옵션:**

*   `-n`, `--exp_name`: 실험 이름 (기본값: `exp_name`)
*   `-d`, `--dataset`: 사용할 데이터셋 (`Clotho` 또는 `AudioCaps`)
*   `-l`, `--lr`: 학습률 (기본값: `0.0001`)
*   `-c`, `--config`: 설정 파일 이름 (기본값: `settings`)
*   `-o`, `--loss`: 손실 함수 (`triplet`, `ntxent`, `infonce` 등)
*   `-b`, `--batch`: 배치 크기 (기본값: `24`)
*   `-e`, `--epochs`: 총 에포크 수 (기본값: `50`)

학습이 완료되면 `outputs/<experiment_name>/models` 디렉토리에 가장 성능이 좋은 모델(`best_model.pth`)이 저장됩니다.

## DCASE 2023 Challenge 결과

이 프로젝트는 **DCASE 2023 Challenge Task 6b: Language-Based Audio Retrieval**에 제출되었으며, 제출된 시스템 중 6위를 기록했습니다.

*   **챌린지 결과**: [DCASE 2023 Task 6b 결과 페이지](https://dcase.community/challenge2023/task-language-based-audio-retrieval-results)
*   **기술 보고서**: [DCASE2023_Park_80_t6b.pdf](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Park_80_t6b.pdf)

## Citation

이 코드는 [XinhaoMei/audio-text_retrieval](https://github.com/XinhaoMei/audio-text_retrieval) 저장소를 기반으로 수정되었습니다. 원본 연구 및 코드에 대한 인용은 다음을 참조해 주십시오.

```
@article{Mei2022metric,
  title = {On Metric Learning for Audio-Text Cross-Modal Retrieval},
  author = {Mei, Xinhao and Liu, Xubo and Sun, Jianyuan and Plumbley, Mark D. and Wang, Wenwu},
  journal={arXiv preprint arXiv:2203.15537},
  year={2022}
}
```

```
@inproceedings{Mei2021ACT,
    author = "Mei, Xinhao and Liu, Xubo and Huang, Qiushi and Plumbley, Mark D. and Wang, Wenwu",
    title = "Audio Captioning Transformer",
    booktitle = "Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)",
    address = "Barcelona, Spain",
    month = "November",
    year = "2021",
    pages = "211--215",
    isbn = "978-84-09-36072-7",
    doi. = "10.5281/zenodo.5770113"
}
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참고하십시오.
