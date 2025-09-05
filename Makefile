# Names of the code directory and the Docker image, change them to match your project
DOCKER_IMAGE_NAME := livi
DOCKER_CONTAINER_NAME := livi
CODE_DIRECTORY := src

DOCKER_PARAMS=  -dit --name=$(DOCKER_CONTAINER_NAME) -v /data/nfs/analysis:/nfs 

# Specify GPU device(s) to use. Comment out this line if you don't have GPUs available
DOCKER_PARAMS+= --gpus '"device=2"' --shm-size=8g
DOCKER_RUN_MOUNT= docker run $(DOCKER_PARAMS) -v $(PWD):/workspace $(DOCKER_IMAGE_NAME)

usage:
	@echo "Available commands:\n-----------"
	@echo "	build		Build the Docker image"
	@echo "	run 		Run the Docker image in a container, after building it"
	@echo "	run-bash	Same as 'run', and launches an interactive bash session in the container while mounting the current directory"
	@echo "	stop		Stop the container if it is running"
	@echo "	logs		Stop the container if it is running"
	@echo "	poetry		Use poetry to modify 'pyproject.toml' and 'poetry.lock' files (e.g. 'make poetry add requests' to add the 'requests' package)"
	@echo "	check		Check coding conventions using multiple tools"
	@echo "	clean		Format your code using black and isort to fit coding conventions"


build:
	docker build --no-cache --progress=plain -t $(DOCKER_IMAGE_NAME) .

run: 
	docker run $(DOCKER_PARAMS) -v $(PWD):/workspace --env-file .env $(DOCKER_IMAGE_NAME)

run-bash:
	$(DOCKER_RUN_MOUNT) /bin/bash || true

stop:
	docker stop $(DOCKER_IMAGE_NAME) || true && docker rm $(DOCKER_IMAGE_NAME) || true

logs:
	docker logs -f $(DOCKER_CONTAINER_NAME)

poetry:
	$(DOCKER_RUN_MOUNT) poetry $(filter-out $@,$(MAKECMDGOALS))
%:	# Avoid printing anything after executing the 'poetry' target
	@:

check:
	$(DOCKER_RUN_MOUNT) poetry run mypy --show-error-codes $(CODE_DIRECTORY)
	$(DOCKER_RUN_MOUNT) poetry run ruff check --no-fix $(CODE_DIRECTORY)
	$(DOCKER_RUN_MOUNT) poetry run ruff format --check $(CODE_DIRECTORY)
	@echo "\nAll is good !\n"

clean:
	$(DOCKER_RUN_MOUNT) poetry run ruff check --fix $(CODE_DIRECTORY)
	$(DOCKER_RUN_MOUNT) poetry run ruff format $(CODE_DIRECTORY)
