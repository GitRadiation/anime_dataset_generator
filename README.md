# 🌟📊 Anime Recomendator 🎬📚
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)  
[![NumPy 2.2.5](https://img.shields.io/badge/numpy-2.2.5-blue?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)  
[![Pandas 2.2.3](https://img.shields.io/badge/pandas-2.2.3-blue?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)  
[![SQLAlchemy 2.0.0](https://img.shields.io/badge/sqlalchemy-2.0.0-blue?style=for-the-badge&logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)  
[![Requests 2.32.3](https://img.shields.io/badge/requests-2.32.3-blue?style=for-the-badge&logo=python&logoColor=white)](https://docs.python-requests.org/)  
[![RAKE-NLTK 1.0.6](https://img.shields.io/badge/rake--nltk-1.0.6-blue?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/rake-nltk/)  
[![BeautifulSoup4 4.13.4](https://img.shields.io/badge/beautifulsoup4-4.13.4-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.crummy.com/software/BeautifulSoup/)  
[![python-dotenv 1.0.0](https://img.shields.io/badge/python--dotenv-1.0.0-blue?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/python-dotenv/)  
[![FastAPI 0.115.12](https://img.shields.io/badge/fastapi-0.115.12-blue?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)  
[![Uvicorn 0.34.2](https://img.shields.io/badge/uvicorn-0.34.2-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.uvicorn.org/)  
[![DiskCache 5.6.3](https://img.shields.io/badge/diskcache-5.6.3-blue?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/diskcache/)  
[![Jikan 4.0.0](https://img.shields.io/badge/jikan-4.0.0-blue?style=for-the-badge&logo=python&logoColor=white)](https://jikan.moe/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)


This project is designed to generate a comprehensive anime dataset using the 🚀 [Jikan API (4.0.0)](https://docs.api.jikan.moe/). It collects detailed information about anime titles, including scores, genres, synopses, producers, studios, and more. Additionally, it features functionality to generate lists of usernames from MyAnimeList, fetch user profiles, and retrieve their anime scores. Leveraging this data, the project employs an evolutionary algorithm to recommend anime tailored to each user’s watching history and preferences.

## Prerequisites

- 💻 Python 3.11 or higher
- 📦 Podman
- 📦 Poetry to run locally

## Deployment Modes

This project supports two main modes of operation to suit different environments and needs:

1. **Run Everything in Containers**  
   The recommended and easiest approach is to run all services (database, FastAPI backend, and Flet frontend) inside containers using `podman-compose`.
   - This mode ensures isolation and consistency across environments.  
   - Ideal for production or development setups where you want to avoid local dependency installation.

2. **Run Locally with Database in Containers**  
   You can also run all components except the database locally:  
   - The FastAPI backend and the Flet frontend can run directly on your machine (e.g., using Python).  
   - The database (`bbdd_maker`) is recommended to be run in containers to ensure proper setup and isolation, but it also supports local execution if preferred.  
   - The project supports mixed configurations, allowing the Flet web app to run in a container while the backend runs locally, or vice versa, depending on your development needs.

This flexibility enables iterative development and smooth deployment across various environments by allowing you to choose the configuration that best fits your needs. For local execution, the project uses Poetry to manage dependencies efficiently.

## Usage

1. Clone the repository:

   ```
   git clone https://github.com/your-username/anime-recommendator.git
   ```
2. Navigate to the project directory:
    ```
    cd anime-recommendator
    ```

3. Build and start the containers using Podman Compose:
    ```
    podman-compose up -d
    ```

4. Access the services:

- FastAPI backend: [http://localhost:8000](http://localhost:8000)  
- Flet web app: [http://localhost:8501](http://localhost:8501)  


## Notes

- Make sure you have the necessary Dockerfiles in the project root:  
- `Dockerfile.bbdd_maker`  
- `Dockerfile.uvicorn_server`  
- `Dockerfile.flet_runner`

- Make sure you have the necessary envs files in the project root:
- `.env`
- `.local.env`
- See the examples to do these files.
  
  
- If you face permission or image pull errors, check your Podman and container registry login credentials.

- To stop and remove containers, run:
    ```
    podman-compose down
    ```

## License

📝 This project is licensed under the [MIT License](LICENSE). You are free to modify and use the code in accordance with the terms and conditions of the license. Feel free to adapt the project to suit your needs and contribute to open-source development. 📜🔒
