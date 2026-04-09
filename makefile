.PHONY: install clean test test-models test-agent train-efficientnet train-resnet train-vit train-swin train-autoencoder train-vae train-all eval-agent demo-agent run-agent-pipeline

install:
	bash scripts/install.sh

data:
	.venv/bin/python scripts/setup_data.py

clean:
	rm -rf .venv uv.lock cache
	rm -rf experiments/test_checkpoints experiments/test_logs

# Testing commands
test:
	python scripts/test_setup.py

test-models:
	python scripts/validate_setup.py

test-agent:
	python scripts/test_agent_workflow.py

# Training commands for different models
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

# Quick test runs (1 epoch)
test-efficientnet:
	python scripts/train.py --model efficientnet_b3 --epochs 1

test-resnet:
	python scripts/train.py --model resnet50 --epochs 1

test-vit:
	python scripts/train.py --model vit_base --epochs 1

test-swin:
	python scripts/train.py --model swin_base_patch4_window7_224 --epochs 1

# Train all models and log results
train-all:
	python scripts/train_all_models.py

# Train all models in background (for overnight runs)
train-all-bg:
	nohup python scripts/train_all_models.py > experiments/training_nohup.log 2>&1 &
	echo "Training started in background. PID: $$!"
	echo "Monitor with: tail -f experiments/training_nohup.log"
	echo "View results with: make view-results"

# View training results
view-results:
	python scripts/view_results.py

eval-agent:
	python scripts/evaluate_agent.py --checkpoint-path experiments/agent_swin/checkpoints/best_model.pth --output-dir reports/agent

demo-agent:
	python scripts/demo_case.py --checkpoint-path experiments/agent_swin/checkpoints/best_model.pth

run-agent-pipeline:
	bash scripts/run_agent_pipeline.sh
