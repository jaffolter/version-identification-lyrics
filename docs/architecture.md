
## Code Architecture

# LIVI Package

The `livi` package is the core of the project and regroups all the main logic, organized into several submodules for clarity and modularity.

### Key files and modules

- **`cli.py`** : Defines the command-line interface (CLI) using Typer. This file exposes commands to run main blocks of the package directly from the terminal, making it easy to launch scripts with arguments.

- **`main.py`** : The main entry point of the project, automatically called when running `docker run`. You can customize this file to call any function or workflow you want to execute at container startup. It typically loads configuration, sets up paths, and launches the main logic.

- **`utils/paths.py`** : Centralizes all important paths used throughout the package. The root data and audio directories are dynamically retrieved from `config.settings`, ensuring easy configuration via environment variables or the `.env` file.

### Additional structure

- **`config.py`** : Handles global configuration using Pydantic. All environment variables and project settings are defined here and loaded at runtime.
- **`utils/`** : Contains utility modules and helper functions used across the project (e.g., for file management, data processing, etc.).
- **`tests/`** : All unit and integration tests are grouped here, following best practices for test isolation and automation.

This modular architecture allows for easy extension and clear navigation, whether you are adding new features, running experiments, or deploying the project in production.
  