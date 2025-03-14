#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour vérifier que l'API RAGondin est compatible avec l'API OpenAI.
Ce script envoie des requêtes à l'API comme si c'était l'API OpenAI.
"""

import requests
import json
import time
import argparse
import os
from typing import List, Dict, Any, Optional

class OpenAICompatTester:
    """Classe pour tester la compatibilité OpenAI de l'API RAGondin"""
    
    def __init__(self, base_url: str = "http://localhost:8083"):
        """
        Initialise le testeur avec l'URL de base de l'API.
        
        Args:
            base_url: URL de base de l'API RAGondin (sans le chemin /v1/chat/completions)
        """
        self.base_url = base_url
        self.completion_url = f"{base_url}/v1/chat/completions"
        
    def test_completion(self, messages: List[Dict[str, str]], model: str = "RAGondin", stream: bool = False) -> Dict[str, Any]:
        """
        Teste l'endpoint de completion chat en envoyant une requête au format OpenAI.
        
        Args:
            messages: Liste de messages au format OpenAI (role, content)
            model: Nom du modèle à utiliser
            stream: Activer le streaming de la réponse
            
        Returns:
            Réponse de l'API
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        print(f"\n\033[1;34m[INFO]\033[0m Envoi de la requête à {self.completion_url}")
        print(f"\033[1;34m[INFO]\033[0m Contenu: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(self.completion_url, headers=headers, json=data, stream=stream)
        
        if response.status_code != 200:
            print(f"\033[1;31m[ERREUR]\033[0m Statut HTTP: {response.status_code}")
            print(f"\033[1;31m[ERREUR]\033[0m Réponse: {response.text}")
            return {"error": response.text}
        
        print(f"\033[1;32m[SUCCÈS]\033[0m Statut HTTP: {response.status_code}")
        
        if stream:
            # Traiter le streaming
            result = {"streaming_response": []}
            print("\033[1;33m[RÉPONSE STREAMING]\033[0m")
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        json_str = line[6:]  # Enlever le "data: " au début
                        chunk = json.loads(json_str)
                        result["streaming_response"].append(chunk)
                        
                        # Afficher progressivement le contenu
                        if "delta" in chunk["choices"][0] and "content" in chunk["choices"][0]["delta"]:
                            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
            
            print("\n")
            return result
        else:
            # Réponse non-streaming
            result = response.json()
            print("\033[1;33m[RÉPONSE]\033[0m")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result

def main():
    """Fonction principale pour exécuter le test depuis la ligne de commande"""
    parser = argparse.ArgumentParser(description="Test de compatibilité OpenAI de l'API RAGondin")
    parser.add_argument("--url", default="http://localhost:8083", help="URL de base de l'API RAGondin")
    parser.add_argument("--stream", action="store_true", help="Activer le mode streaming")
    parser.add_argument("--question", default="Qu'est-ce que RAGondin?", help="Question à poser")
    args = parser.parse_args()
    
    tester = OpenAICompatTester(base_url=args.url)
    
    # Exemple de messages au format OpenAI
    messages = [
        {"role": "system", "content": "Tu es un assistant utile basé sur RAGondin."},
        {"role": "user", "content": args.question}
    ]
    
    # Exécution du test
    start_time = time.time()
    result = tester.test_completion(messages=messages, stream=args.stream)
    end_time = time.time()
    
    print(f"\n\033[1;34m[INFO]\033[0m Temps total: {end_time - start_time:.2f} secondes")
    
    # Vérification que la structure est correcte
    if not args.stream and "choices" in result:
        response_text = result["choices"][0]["message"]["content"]
        print(f"\n\033[1;32m[RÉSULTAT]\033[0m Réponse: {response_text}")
    
if __name__ == "__main__":
    main() 