#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
VOLUME_MOUNTS="-v ${SCRIPTPATH}/data:/home/karasu/app/data -v ${SCRIPTPATH}/artifacts:/home/karasu/app/artifacts"
IMAGE_NAME="karasu-container:dev"

echo "### Creating the necessary folders (if required)... ###"
mkdir -p $SCRIPTPATH/artifacts
mkdir -p $SCRIPTPATH/data

echo "### Building the docker image (if required)... ###"
docker build -t $IMAGE_NAME .

# download data, if required
echo "### Downloading data (if required)... ###"
docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME bash /home/karasu/app/evaluation/data_download.sh

create_soo_data(){
  echo "### Create data for emulated repository for SOO using the baselines (this will take some time)... ###"
  PYTHON_CMD="python /home/karasu/app/evaluation/eval_soo.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

create_moo_data(){
  echo "### Create data for emulated repository for MOO using the baselines (this will take some time)... ###"
  PYTHON_CMD="python /home/karasu/app/evaluation/eval_moo.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

run_rq1_experiment(){
  echo "### Run full RQ1 experiments (this will take a long time)... ###"
  PYTHON_CMD="python /home/karasu/app/evaluation/eval_soo_rq1.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

run_rq2_experiment(){
  echo "### Run full RQ2 experiments (this will take a long time)... ###"
  PYTHON_CMD="python /home/karasu/app/evaluation/eval_soo_rq2.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

run_rq2_hetero_experiment(){
  echo "### Run full RQ2 (heterogeneous data) experiments (this will take a long time)... ###"
  PYTHON_CMD="python /home/karasu/app/evaluation/eval_soo_rq2_hetero.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

run_rq3_experiment(){
  echo "### Run full RQ3 experiments (this will take a long time)... ###"
  PYTHON_CMD="python /home/karasu/app/evaluation/eval_moo_rq3.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

analysis(){
  echo "### Create plots... ###"
  PYTHON_CMD="python /home/karasu/app/evaluation/analysis_data.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
  PYTHON_CMD="python /home/karasu/app/evaluation/analysis_results.py"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME $PYTHON_CMD
}

shell(){
  echo "### Open shell in container... ###"
  docker run -it --rm $VOLUME_MOUNTS -t $IMAGE_NAME bash
}

# Check if the function exists (bash specific)
if declare -f "$1" > /dev/null
then
  # call arguments verbatim
  "$@"
else
  # Show a helpful error
  echo "'$1' is not a known function name" >&2
  exit 1
fi