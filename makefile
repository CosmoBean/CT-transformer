.PHONY: install data clean test test-yolo test-review prepare-yolo train-simple-cnn train-efficientnet train-resnet train-vit train-swin train-yolo infer-yolo eval-yolo eval-review agentic-report reproduce-compare presentation-reports

install:
	bash scripts/install.sh

data:
	.venv/bin/python scripts/ct_transformer.py setup-data

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
	python scripts/ct_transformer.py prepare-yolo

train-simple-cnn:
	python scripts/ct_transformer.py train-classifier --model simple_cnn --epochs 100 --save-dir experiments/simple_cnn_100/checkpoints --log-dir experiments/simple_cnn_100/logs

train-efficientnet:
	python scripts/ct_transformer.py train-classifier --model efficientnet_b3 --epochs 10

train-resnet:
	python scripts/ct_transformer.py train-classifier --model resnet50 --epochs 10

train-vit:
	python scripts/ct_transformer.py train-classifier --model vit_base --epochs 10

train-swin:
	python scripts/ct_transformer.py train-classifier --model swin_base_patch4_window7_224 --epochs 10 --save-dir experiments/agent_swin/checkpoints --log-dir experiments/agent_swin/logs

train-yolo:
	python scripts/ct_transformer.py train-yolo

infer-yolo:
	python scripts/ct_transformer.py infer-yolo --weights experiments/yolo_v8/full_e10local/best.pt --image data/test/002a34c58c5b758217ed1f584ccbcfe9.png

eval-yolo:
	python scripts/ct_transformer.py eval-yolo --weights experiments/yolo_v8/full_e10local/best.pt

eval-review:
	python scripts/ct_transformer.py eval-review --max-cases 10

agentic-report:
	python scripts/ct_transformer.py report --image data/test/002a34c58c5b758217ed1f584ccbcfe9.png

reproduce-compare:
	python scripts/ct_transformer.py compare --max-cases 300

presentation-reports:
	python scripts/generate_presentation_comparison_reports.py --help
