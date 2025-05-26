# VAND2025 Challenge - Track 1 Submission Readme Template

## Overview


**Challenge Website:** 
[https://sites.google.com/view/vand30cvpr2025/challenge](https://sites.google.com/view/vand30cvpr2025/challenge)

**CRITICAL:**
The reproducibility of your results is paramount. Please specify a project link 
to a github repo when submitting to the MVTec Benchmark Server
([https://benchmark.mvtec.com/](https://benchmark.mvtec.com/)).
Submissions that cannot be reproduced by the judges using the instructions and 
code provided **will not be considered** in the final ranking and will be 
removed from the leaderboard. Please test your instructions thoroughly.

## Dependencies
Install dependencies using:
pip install -r requirements.txt

## Dataset
This project uses the MVTec AD 2 dataset, which is publicly available at:
https://www.mvtec.com/company/research/datasets/

## Model Architecture
Defined in model, this file includes the implementation of the anomaly detection model used. 

## Training Pipeline
The training pipeline is implemented in train.py. 

**Example Training Command**
    python train.py \
        --dataset_path /path/to/mvtec_ad_2 \
        --save_path /path/to/save/model

## Testing & Evaluation Pipeline
The testing pipeline is implemented in test.py. It loads the trained model and evaluates it on the test set, producing anomaly maps and binary predictions.

**Example Testing Command**
    python test.py \
        --model_path /path/to/saved/model \
        --output_path /path/to/save/results \
        --dataset_path /path/to/mvtec_ad_2
## Reproducing Final Results
To reproduce the final results submitted to the benchmark server, use the provided shell script:
./VAND2025_track1_MAD2_reproduce_final_result.sh

**Sample Content of the Script**
 DATASET_PATH: Root directory of the MVTec AD 2 dataset
 MODEL_DIR: Directory to save/load trained model weights
 OUTPUT_DIR: Output directory for inference results (e.g., anomaly maps, binary masks)

## License
This code is released under the license specified in licence.txt.