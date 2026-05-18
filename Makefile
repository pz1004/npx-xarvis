XARVIS_HOME=$(CURDIR)
include ${XARVIS_HOME}/rvx_config.mh

PYTHON3_CMD ?= python3

include ./rvx_init.mh

config:
ifeq ($(OS),Windows_NT)
	setx XARVIS_HOME $(CURDIR)
	setx XARVIS_NPX_TRANIER_HOME %XARVIS_HOME%/npx_trainer
else
	${PYTHON3_CMD} ${RVX_UTIL_HOME}/configure_template.py -i ./rvx_setup.sh.template -o ./rvx_setup.sh -c CURDIR=$(CURDIR)
endif

path_name = Path('\"%\"RVX_MINI_HOME%') / output_dir.name

preinstall:
	@echo Checking OS type...
ifneq ($(OS),Windows_NT)
	@os_id=$$(. /etc/os-release && echo $$ID); \
	if [ "$$os_id" = "centos" ] || [ "$$os_id" = "rhel" ]; then \
		echo "Detected $$os_id -> using yum"; \
		sudo yum install -y sqlite-devel xz-devel; \
	elif [ "$$os_id" = "ubuntu" ] || [ "$$os_id" = "debian" ]; then \
		echo "Detected $$os_id -> using apt"; \
		sudo apt update && sudo apt install -y libsqlite3-dev liblzma-dev; \
	else \
		echo "Unknown OS: $$os_id"; \
		exit 1; \
	fi
endif
	@echo Preinstall is complete

#${PYTHON3_CMD} -m pip install pysqlite
#${PYTHON3_CMD} -m pip install sqlite3
#${PYTHON3_CMD} -m pip install numpy==1.26.4 # numpy must be < 2.0

install: install_verified

install_original:
	${PYTHON3_CMD} -m pip install tqdm tonic aedat
	${PYTHON3_CMD} -m pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
	${PYTHON3_CMD} -m pip install snntorch

install_verified:
	${PYTHON3_CMD} -m pip install tqdm==4.67.1 tonic==1.6.0 aedat==2.2.0
	${PYTHON3_CMD} -m pip install torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cpu
	${PYTHON3_CMD} -m pip install snntorch==0.9.1

install_recent:
	${PYTHON3_CMD} -m pip install --upgrade tqdm tonic aedat
	${PYTHON3_CMD} -m pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
	${PYTHON3_CMD} -m pip install --upgrade snntorch

install-paper:
	${PYTHON3_CMD} -m pip install snntorch tonic aedat pandas scikit-learn matplotlib seaborn psutil

train-paper:
	${PYTHON3_CMD} -m src.experiments.train_eval --dataset ${DATASET} --method ${METHOD} --seed ${SEED}

tune-paper:
	${PYTHON3_CMD} -m src.experiments.tune_filters --datasets ${DATASETS} --methods ${METHODS}

aggregate-paper:
	${PYTHON3_CMD} -m src.experiments.aggregate_results --regen-roc --roc-dataset ${DATASET}

smoke-paper:
	${PYTHON3_CMD} -m pytest tests
	${PYTHON3_CMD} -m src.experiments.train_eval --dataset nmnist --method proposed_conf --seed 0 --epochs-override 1 --max-train-samples 64 --max-val-samples 32 --max-test-samples 32 --force-cpu
	${PYTHON3_CMD} -m src.experiments.train_eval --dataset nmnist --method frame_snn --seed 0 --epochs-override 1 --max-train-samples 64 --max-val-samples 32 --max-test-samples 32 --force-cpu
