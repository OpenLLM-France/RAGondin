import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import git

def update_env_file(key: str, value: str):
    """Met à jour ou ajoute une variable dans le fichier .env."""
    env_path = Path('.env')
    
  
    if not env_path.exists():
        env_path.touch()
    

    with env_path.open('r') as f:
        lines = f.readlines()
    
    # Chercher si la variable existe déjà
    key_exists = False
    new_lines = []
    for line in lines:
        if line.strip() and not line.startswith('#'):
            current_key = line.split('=')[0].strip()
            if current_key == key:
                new_lines.append(f"{key}={value}\n")
                key_exists = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    if not key_exists:
        new_lines.append(f"{key}={value}\n")
    

    with env_path.open('w') as f:
        f.writelines(new_lines)
        
    # Recharger les variables d'environnement
    load_dotenv(override=True)

def get_current_branch():
    """Obtient le nom de la branche git actuelle et met à jour .env."""
    try:
        repo = git.Repo(os.getcwd())
        branch_name = repo.active_branch.name
        

        update_env_file('CURRENT_BRANCH', branch_name)
        
        return branch_name
    except Exception as e:
        print(f"Erreur lors de la récupération de la branche git: {e}")
        return "default"

def setup_environment():
    """Configure l'environnement et les répertoires nécessaires."""
    try:
        load_dotenv()
        
        container_name = os.getenv('CONTAINER_NAME')
        current_branch = os.getenv('CURRENT_BRANCH')
        venv_name = f"{container_name}_{current_branch}"
        
        update_env_file('VENV_NAME', venv_name)
        print(f"VENV_NAME: {venv_name}")

        venvs_path = Path(os.getenv('ENVS_PATH'))
        venvs_path.mkdir(exist_ok=True)
        
        venv_specific_path = venvs_path / venv_name
        venv_specific_path.mkdir(exist_ok=True)
        
        print(f"Configuration des répertoires terminée:")
        print(f"- Répertoire {os.getenv('ENVS_PATH')}: {venvs_path.absolute()}")
        print(f"- Environnement virtuel: {venv_specific_path.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors de la configuration de l'environnement: {e}")
        return False

def deploy_docker_compose():
    """Déploie l'application via Docker Compose."""
    try:
        load_dotenv()
        if not Path("docker-compose.yaml").exists():
            raise FileNotFoundError("docker-compose.yaml non trouvé")
        
        # Utiliser subprocess.run pour avoir les logs en direct
        process = subprocess.Popen(
            ["docker-compose", "up", "--build", "-d"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Afficher les logs en temps réel
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            print("Docker Compose déployé avec succès")
            
            logs_process = subprocess.Popen(
                ["docker", "logs", "-f", "RAGondin"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Afficher les logs en temps réel
            for line in logs_process.stdout:
                print(line, end='')
                
            logs_process.wait()
        else:
            print(f"Erreur lors du déploiement Docker Compose (code: {process.returncode})")
            
    except Exception as e:
        print(f"Erreur inattendue: {e}")

def main():
    """Fonction principale."""
    print("Démarrage du déploiement...")
    

    if not setup_environment():
        print("Échec de la configuration de l'environnement")
        return
    

    deploy_docker_compose()
main = main
if __name__ == "__main__":
    main()