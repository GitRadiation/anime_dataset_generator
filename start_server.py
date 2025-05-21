import subprocess


def main():
    cmd = [
        "poetry",
        "run",
        "uvicorn",
        "genetic_rule_miner.app.backend:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    # Lanza uvicorn en un proceso hijo sin esperar a que termine
    process = subprocess.Popen(cmd)

    try:
        # Esperar a que el proceso termine
        process.wait()
    except KeyboardInterrupt:
        print("Interrumpido, terminando servidor...")
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
