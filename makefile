.PHONY: install data clean test test-yolo test-review prepare-yolo train-simple-cnn train-efficientnet train-resnet train-vit train-swin train-yolo infer-yolo eval-yolo eval-review agentic-report

install:
	bash scripts/install.sh

data:
	.venv/bin/python scripts/setup_data.py

clean:
	find src scripts -type f -name '*.pyc' -delete
	find src scripts -type d -name '__pycache__' -empty -delete

test:
	python scripts/test_setup.py

test-yolo:
	python scripts/test_yolo_pipeline.py

test-review:
	python scripts/test_claude_review_pipeline.py

prepare-yolo:
	python scripts/prepare_yolo_dataset.py

train-simple-cnn:
	python scripts/train.py --model simple_cnn --epochs 100 --save-dir experiments/simple_cnn_100/checkpoints --log-dir experiments/simple_cnn_100/logs

train-efficientnet:
	python scripts/train.py --model efficientnet_b3 --epochs 10

train-resnet:
	python scripts/train.py --model resnet50 --epochs 10

train-vit:
	python scripts/train.py --model vit_base --epochs 10

train-swin:
	python scripts/train.py --model swin_base_patch4_window7_224 --epochs 10 --save-dir experiments/agent_swin/checkpoints --log-dir experiments/agent_swin/logs

train-yolo:
	python scripts/train_yolo.py

infer-yolo:
	python scripts/infer_yolo.py --weights experiments/yolo_v8/full_e10local/best.pt --image data/test/002a34c58c5b758217ed1f584ccbcfe9.png

eval-yolo:
	python scripts/evaluate_yolo.py --weights experiments/yolo_v8/full_e10local/best.pt

eval-review:
	python scripts/evaluate_claude_review.py --max-cases 10

agentic-report:
	python scripts/run_agentic_report.py --image data/test/002a34c58c5b758217ed1f584ccbcfe9.png
