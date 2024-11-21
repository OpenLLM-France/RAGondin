import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import git
import time


def get_qdrant_ip(image_name: str, port: int) -> str:
    """Récupère l'IP du conteneur Qdrant."""
    try:
        container_name = get_container_name(image_name, port)
        cmd = f"docker inspect -f '{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}}{{{{end}}}}' {container_name}"
        print(f"Commande: {cmd}")
        qdrant_ip = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        print(f"IP de Qdrant: {qdrant_ip}")
        return qdrant_ip
    except subprocess.CalledProcessError:
        print("Erreur: Impossible de trouver l'IP de Qdrant")
        return None
    
def get_container_name(image_name: str, port: int) -> str:
    """Récupère le nom du conteneur qui tourne sur l'image image_name et sur le port port."""
    try:
        cmd = f"docker ps -a --filter ancestor={image_name} --filter publish={port} --format '{{{{.Names}}}}'"
        print(f"Commande: {cmd}")
        container_name = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        print(f"Nom du conteneur {image_name} sur le port {port}: {container_name}")
        return container_name
    except subprocess.CalledProcessError:
        print("Erreur: Impossible de trouver le conteneur")
        return None




def find_port(start_port: int = 8001, max_attempts: int = 100) -> int:
    """
    Trouve un port disponible.
    Args:
        start_port: Port de départ
        max_attempts: Nombre maximum de tentatives
    Returns:
        int: Port disponible
    Raises:
        RuntimeError: Si aucun port n'est trouvé
    """
    current_port = start_port
    while current_port < (start_port + max_attempts):
        if not check_port(current_port):
            print(f"Port disponible trouvé: {current_port}")
            return current_port
        current_port += 1
    raise RuntimeError(f"Aucun port disponible trouvé entre {start_port} et {start_port + max_attempts}")

def modify_port(compose_path: Path, new_port: int = 8001):
    """Modifie temporairement le port dans le docker-compose."""
    import yaml
    
    # Lire le fichier
    with open(compose_path) as f:
        config = yaml.safe_load(f)
    
    # Modifier le port
    if 'services' in config:
        for service in config['services'].values():
            if 'ports' in service:
                for i, port in enumerate(service['ports']):
                    if ':8000' in port:
                        service['ports'][i] = f"{new_port}:8000"
    
    # Sauvegarder temporairement
    temp_path = compose_path.parent / 'docker-compose.temp.yaml'
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)
    
    return temp_path



def container_exists(container_name: str) -> bool:
    """Vérifie si un conteneur existe."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    return container_name in result.stdout.strip()

def check_port(port: int = 8000):
    """Vérifie en détail ce qui utilise un port."""
    
    try:
        print(f"\nDiagnostic du port {port}:")
        port_in_use = False
        print("\nProcessus utilisant le port (lsof):")
        result = subprocess.run(
            ["sudo", "lsof", "-i", f":{port}"],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            print(result.stdout)
            port_in_use = True
        
        print("\nConnexions sur le port (netstat):")
        subprocess.run(
            ["sudo", "netstat", "-tulpn", "|", "grep", str(port)],
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            print(result.stdout)
            port_in_use = True
        
        #  check docker sur le port
        print("\nConteneurs Docker utilisant des ports:")
        subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Ports}}"],
            check=False
        )
        if str(port) in result.stdout:
            print(result.stdout)
            port_in_use = True
            
        if not port_in_use:
            print(f"Le port {port} est libre")
        
        return port_in_use
        
    except Exception as e:
        print(f"Erreur lors du diagnostic: {e}")
        return False

def clean_container(compose_path: Path, container_name: str, remove_volumes: bool = False):
    """Nettoie l'environnement Docker avant déploiement."""
    try:

        if container_exists(container_name):
            print(f"Conteneur {container_name} trouvé, nettoyage en cours...")
            
            # down le container
            try :
                cmd = ["docker-compose", "-f", str(compose_path), "down"]
                if remove_volumes:
                    cmd.append("-v")
                    
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                print(f"Erreur lors de l'execution de docker-compose down: {e}")
                subprocess.run(
                    ["docker", "stop", container_name],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                

                time.sleep(2)
                
                # forcer suppression
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # vérifier les processus Docker résiduels
                subprocess.run(
                    ["docker", "system", "prune", "-f"],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                time.sleep(2)
                
                print("Nettoyage terminé avec succès")
            else:
                print(f"Aucun conteneur {container_name} trouvé, vérification des processus résiduels...")
                # Nettoyer quand même les processus résiduels
                subprocess.run(
                    ["docker", "system", "prune", "-f"],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )


                
            # dernière vérification et force remove si nécessaire
            if container_exists(container_name):
                print(f"Forçage de la suppression du conteneur {container_name}")
                subprocess.run(["docker", "stop", container_name], check=False)
                subprocess.run(["docker", "rm", "-f", container_name], check=False)
                
            print("Nettoyage terminé avec succès")
        else:
            print(f"Aucun conteneur {container_name} trouvé, pas besoin de nettoyage")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors du nettoyage: {e}")
        return False
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        return False

def update_env_file(key: str, value: str, env_path: Path = Path('.env')):
    """Met à jour ou ajoute une variable dans le fichier .env."""
    print(f"Mise à jour de {env_path}")
    print(os.getcwd())
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
    load_env_and_print(env_path)
    


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
def load_env_and_print(file_path: Path= Path('.env'), **kwargs):
    load_dotenv(dotenv_path=file_path, **kwargs)
    print(f"Variables d'environnement chargées avec succès: {file_path}")
    for key, value in os.environ.items():
        print(f"{key}: {value}")
def setup_environment():
    """Configure l'environnement et les répertoires nécessaires."""
    try:

        compose_path = Path(os.getenv('COMPOSE_PATH'))
        compose_env_path = compose_path/'.env.compose'
        load_env_and_print(compose_env_path)
        update_qdrant_host_env()
        container_name = os.getenv('CONTAINER_NAME')
        current_branch = get_current_branch().replace('-', '_')

        venv_name = f"{container_name}_{current_branch}"
        
        update_env_file('VENV_NAME', venv_name, env_path=compose_env_path)
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
        app_port = os.getenv('APP_PORT')
        check_port(app_port)
        
        compose_file = Path(os.getenv('COMPOSE_PATH'))/"docker-compose.yaml"

        container_name = os.getenv('CONTAINER_NAME')
        
        if not compose_file.exists():
            raise FileNotFoundError("docker-compose.yaml non trouvé")
        
        # nettoyer l'ancien container avant le rebuild et le déploiement
        if not clean_container(compose_file, container_name):
            raise Exception("Échec du nettoyage de l'environnement Docker")
        
        if check_port(app_port):
            print(f"Le port {app_port} est déjà utilisé")

            try:
                subprocess.run(
                    ["sudo", "fuser", "-k", f"{app_port}/tcp"],
                    check=False
                )
                time.sleep(2)
                if check_port(app_port):
                    print(f"Impossible de libérer le port {app_port}")
                try:
                    new_port = find_port()
                    print(f"Utilisation du port alternatif: {new_port}")
                    temp_compose = modify_port(compose_file, new_port)
                    compose_file = temp_compose
                except RuntimeError as e:
                    print(f"Erreur: {e}")
                    return

            except Exception as e:
                print(f"Erreur: {e}")
                return                
        
    
        print(os.getcwd())
        print(compose_file)
        print('Execution de docker-compose build')
        load_and_substitute_compose_file(compose_file)
        process = subprocess.Popen(
            ["docker-compose", "--progress=plain", "--env-file", ".env",  "-f", compose_file, "up", "--build", "-d" ],
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
                ["docker", "logs", "-f", container_name],
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
    finally:
        # Nettoyer le fichier temporaire si créé
        if 'temp_compose' in locals():
            temp_compose.unlink(missing_ok=True)

def update_qdrant_host_env():
    """Met à jour l'IP de Qdrant dans .env.compose"""
    env_compose_path = Path(os.getenv('COMPOSE_PATH'))/'.env.compose'
    
    try:

        qdrant_ip = get_qdrant_ip(os.getenv('qdrant_image_name'), os.getenv('port'))
        print(f"IP de Qdrant: {qdrant_ip}")
        # Lire le fichier .env.compose
        with open(env_compose_path, 'r') as f:
            lines = f.readlines()
        

        with open(env_compose_path, 'w') as f:
            for line in lines:
                if line.startswith('host='):
                    f.write(f'host={qdrant_ip}\n')
                else:
                    f.write(line)
                    
        print(f"HOST mis à jour avec l'IP de Qdrant: {qdrant_ip}")
        
    except subprocess.CalledProcessError:
        print("Erreur: Impossible de trouver l'IP de Qdrant")
    except Exception as e:
        print(f"Erreur lors de la mise à jour de .env.compose: {e}")

import click
import yaml

def load_and_substitute_compose_file(file_path):

    with open(file_path, 'r') as file:
        compose_content = yaml.safe_load(file)

    # Remplacer les variables d'environnement
    def substitute_variables(data):
        if isinstance(data, dict):
            return {key: substitute_variables(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [substitute_variables(item) for item in data]
        elif isinstance(data, str):
            return os.path.expandvars(data)  # Remplace les variables d'environnement
        return data

    substituted_content = substitute_variables(compose_content)


    print(yaml.dump(substituted_content, default_flow_style=False))

@click.command()
@click.option('--build', is_flag=True, help="Déployer la stack.")
@click.option('--up', is_flag=True, help="Déployer la stack.")
@click.option('--down', is_flag=True, help="Down la stack.")
@click.option('-d', is_flag=True, help="Dissocier.")
def main(build, up, down, d):
    """Fonction principale."""
    print("Démarrage du déploiement...")  

    load_dotenv()
    compose_path = Path(os.getenv('COMPOSE_PATH'))/'docker-compose.yaml'
    env_compose_path = Path(os.getenv('COMPOSE_PATH'))/'.env.compose'
    if not (up or down):
        build = True
    if down:
        down_stack(compose_path, env_compose_path, detach=d)
        
    else :
        if not setup_environment():
            print("Échec de la configuration de l'environnement")
            return   
        
        if build:
            deploy_docker_compose()
        

        
        
        if up:
            up_stack(compose_path, env_compose_path, detach=d)
    


def up_stack(compose_path, env_compose_path, detach=False):
    """Déployer la stack."""
    cmd = f"docker-compose -f {compose_path} --env-file {env_compose_path} up"
    if detach:
        cmd += " -d"
    subprocess.run(cmd, shell=True)

def down_stack(compose_path, env_compose_path, detach=False):
    """Down la stack."""
    cmd = f"docker-compose -f {compose_path} --env-file {env_compose_path} down"
    if detach:
        cmd += " -d"
    subprocess.run(cmd, shell=True)
main = main
if __name__ == "__main__":
    main()