.PHONY: install data clean test test-models train-efficientnet train-resnet train-vit train-swin train-autoencoder train-vae train-all view-results

install:
	bash scripts/install.sh

data:
	.venv/bin/python scripts/setup_data.py

clean:
	find src scripts -type f -name '*.pyc' -delete
	find src scripts -type d -name '__pycache__' -empty -delete
	rm -rf experiments/test_checkpoints experiments/test_logs

test:
	python scripts/test_setup.py

test-models:
	python scripts/validate_setup.py

train-efficientnet:
	python scripts/train.py --model efficientnet_b3 --epochs 10

train-resnet:
	python scripts/train.py --model resnet50 --epochs 10

train-vit:
	python scripts/train.py --model vit_base --epochs 10

train-swin:
	python scripts/train.py --model swin_base_patch4_window7_224 --epochs 10

train-autoencoder:
	python scripts/train.py --model autoencoder --epochs 20

train-vae:
	python scripts/train.py --model vae --epochs 20

train-all:
	python scripts/train_all_models.py

view-results:
	python scripts/view_results.py
