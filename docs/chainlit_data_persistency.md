# Data Persistency

A data layer can be activated using a git submodule:  
https://github.com/OpenLLM-France/chainlit-datalayer/tree/main (forked from https://github.com/Chainlit/chainlit-datalayer).  
With the fork, we industrialized the repo by dockerizing it.

- It deploys a PostgreSQL database for data persistency.
- The implementation by Chainlit is cloud-compatible, and the same applies for local data persistency.
- A local "fake" S3 bucket is deployed for local storage.

## Usage in Our App

### Cloning with Submodules

If you want to have everything at once, clone the project with submodule dependencies:

```bash
git clone --recurse-submodules <repository-url>
```

If you've already cloned this repo without submodules:

```bash
cd <project-name>
git submodule update --init --recursive
```

[!NOTE]  
The `--init --recursive` flags will:
- Initialize the submodules based on the existing `.gitmodules` file
- Download all the submodule content
- Handle nested submodules if any exist

### Environment Variables

After adding the submodule, add the following variables to your `.env` file:

```bash
# Chainlit data persistency
## Persistency services (localstack + Fake AWS S3 (Deployed Locally))
CHAINLIT_DATALAYER_COMPOSE=extern/chainlit-datalayer/compose.yaml # Path to docker compose file of the data layer service.

## Secret key for Chainlit authentication
CHAINLIT_AUTH_SECRET=... # Generate with: `uv run chainlit create-secret`

## The PostgreSQL instance for data storage.
POSTGRES_USER=root
POSTGRES_PASSWORD=root
POSTGRES_DB=postgres
POSTGRES_PORT=5432

DATABASE_URL=postgresql://${POSTGRES_USER:-root}:${POSTGRES_PASSWORD:-root}@postgres:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-postgres} # Used by the chainlit app: `ragondin/chainlit/app_front.py`

## S3 configuration.
BUCKET_NAME=my-bucket
APP_AWS_ACCESS_KEY=random-key
APP_AWS_SECRET_KEY=random-key
APP_AWS_REGION=eu-central-1

LOCALSTACK_PORT=4566
DEV_AWS_ENDPOINT=http://localstack:${LOCALSTACK_PORT:-4566}
```

>[!IMPORTANT]  
>If you want to deactivate the service, comment out these variables, especially `CHAINLIT_DATALAYER_COMPOSE`.