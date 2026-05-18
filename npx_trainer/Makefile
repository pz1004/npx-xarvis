ifdef XARVIS_HOME
	include ${XARVIS_HOME}/rvx_config.mh
endif

include ${RVX_UTIL_HOME}/template_makefile.mh
GIT_BASE = $(call absolute_path,${CURDIR}/..)

DATASET_DIR=${GIT_BASE}/dataset
VERIFY_DIR=${GIT_BASE}/verify
RESULT_DIR=${CURDIR}/result

ifndef GPU_ID
	GPU_ID=-1
endif

#NUM_EPOCH=10
#NUM_REPEAT=1
NUM_SAMPLE=1

#########
## RAW ##
#########

train:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd train quantize test -dataset ${DATASET_DIR} -o ${RESULT_DIR} -gpu ${GPU_ID}

test:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd quantize test -dataset ${DATASET_DIR} -o ${RESULT_DIR} -gpu ${GPU_ID}

reset:
	@${PYTHON3_CMD} ./npx_trainer.py -cfg $(CFG_FILE) -cmd reset -dataset ${DATASET_DIR} -o ${RESULT_DIR} -gpu ${GPU_ID}
	
riscv:
	@${PYTHON3_CMD} ./npx_converter.py -cfg $(CFG_FILE) -o ${RESULT_DIR}

tv:
	@${PYTHON3_CMD} ./npx_testvector.py -cfg $(CFG_FILE) -cmd testvector -dataset ${DATASET_DIR} -s ${NUM_SAMPLE} -o ${RESULT_DIR}

sample:
	@${PYTHON3_CMD} ./npx_testvector.py -cfg $(CFG_FILE) -cmd testvector -dataset ${DATASET_DIR} -s ${NUM_SAMPLE} -o ${RESULT_DIR} --sample_only

#########
## APP ##
#########

APP_CFG_DIR=${CURDIR}/app
ALL_CFG_LIST=$(wildcard ${APP_CFG_DIR}/*.cfg)
APP_CFG_LIST = $(filter-out %_preprocess.cfg, $(ALL_CFG_LIST))
APP_CFG_NAME_LIST=$(basename $(notdir ${APP_CFG_LIST}))

app_list:
	@echo ${APP_CFG_NAME_LIST}

${APP_CFG_NAME_LIST}:%: %.train %.test %.riscv

$(addsuffix .train, ${APP_CFG_NAME_LIST}):%.train:
	@make train CFG_FILE=${APP_CFG_DIR}/$(*).cfg

$(addsuffix .test, ${APP_CFG_NAME_LIST}):%.test:
	@make test CFG_FILE=${APP_CFG_DIR}/$(*).cfg

$(addsuffix .reset, ${APP_CFG_NAME_LIST}):%.reset:
	@make reset CFG_FILE=${APP_CFG_DIR}/$(*).cfg

$(addsuffix .riscv, ${APP_CFG_NAME_LIST}):%.riscv:
	@make riscv CFG_FILE=${APP_CFG_DIR}/$(*).cfg

$(addsuffix .tv, ${APP_CFG_NAME_LIST}):%.tv:
	@make tv CFG_FILE=${APP_CFG_DIR}/$(*).cfg

$(addsuffix .sample, ${APP_CFG_NAME_LIST}):%.sample:
	@make sample CFG_FILE=${APP_CFG_DIR}/$(*).cfg

##########
## TEST ##
##########

TEST_APP_CFG_FILE=${APP_CFG_DIR}/check_app.cfg

check0:
	@make reset CFG_FILE=${TEST_APP_CFG_FILE}

check1:
	@make train CFG_FILE=${TEST_APP_CFG_FILE}

check2:
	@make train riscv CFG_FILE=${TEST_APP_CFG_FILE}

check3:
	@make train riscv tv CFG_FILE=${TEST_APP_CFG_FILE} NUM_SAMPLE=${NUM_SAMPLE}

####################
## PREDEFINED APP ##
####################

PREDEFINED_APP_PREFIX_LIST=mnist_l2cf mnist_l2ff mnist_l3ccf mnist_l3fff mnist_l3cff mnist_l1f kmnist_l3fff fmnist_l2cf fmnist_l3cff fmnist_l3ccf cifar10_l5cccff gtsrb_l5cccff
NEURON_TYPE_LIST=q8ssf

gen_info:
	@echo ${PREDEFINED_APP_PREFIX_LIST}
	@echo ${NEURON_TYPE_LIST}

$(addsuffix .gen_cfg, ${PREDEFINED_APP_PREFIX_LIST}):%.gen_cfg:
	@${PYTHON3_CMD} ./npx_app_cfg_generator.py -app $(*) -neuron ${NEURON_TYPE_LIST} -dataset ${DATASET_DIR} -o ${APP_CFG_DIR}

$(addsuffix _q8ssf.reall, ${PREDEFINED_APP_PREFIX_LIST}):%_q8ssf.reall:
	@make $(*).gen_cfg NEURON_TYPE_LIST=q8ssf
	@make $(*)_q8ssf.reset
	@make $(*)_q8ssf.train

################
## FORWARDING ##
################

$(addsuffix .reall, ${PREDEFINED_APP_PREFIX_LIST}):%.reall: %_q8ssf.reall
