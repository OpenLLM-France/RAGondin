#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test utilisant la bibliothèque officielle OpenAI pour vérifier 
la compatibilité de notre API RAGondin avec le standard OpenAI.
"""

import argparse
import time
import json
from openai import OpenAI
import os
import sys

# Ajouter le répertoire parent au chemin Python pour permettre les importations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_openai_client(base_url: str, question: str, stream: bool = False):
    """
    Teste l'API RAGondin en utilisant le client officiel OpenAI.
    
    Args:
        base_url: URL de base de l'API RAGondin (sans /v1)
        question: Question à poser à l'API
        stream: Activer le mode streaming
    """
    print("\n\033[1;34m[INFO]\033[0m Test avec le client OpenAI officiel")
    print(f"\033[1;34m[INFO]\033[0m URL de base: {base_url}")
    print(f"\033[1;34m[INFO]\033[0m Question: {question}")
    print(f"\033[1;34m[INFO]\033[0m Streaming: {stream}")
    
    # Initialiser le client OpenAI avec notre API personnalisée
    client = OpenAI(
        # Cette clé API est fictive, notre implémentation l'ignore
        api_key="sk-fake-key",
        # Point important : rediriger vers notre API
        base_url=base_url
    )
    
    # Créer les messages pour la requête
    messages = [
        {"role": "system", "content": "Tu es un assistant utile basé sur RAGondin."},
        {"role": "user", "content": question}
    ]
    
    start_time = time.time()
    
    try:
        # Appeler l'API avec le client OpenAI
        if stream:
            print("\n\033[1;33m[RÉPONSE STREAMING]\033[0m")
            
            # Mode streaming
            response = client.chat.completions.create(
                model="RAGondin",  # Valeur fixe, peu importe ce que l'on met
                messages=messages,
                stream=True
            )
            
            # Récupérer et afficher les chunks de la réponse en streaming
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
            
            print("\n")
            result = {"content": full_response}
            
        else:
            # Mode non-streaming
            response = client.chat.completions.create(
                model="RAGondin",  # Valeur fixe, peu importe ce que l'on met
                messages=messages,
                stream=False
            )
            
            # Afficher la réponse complète
            print("\n\033[1;33m[RÉPONSE]\033[0m")
            result = {
                "id": response.id,
                "model": response.model,
                "content": response.choices[0].message.content
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        end_time = time.time()
        print(f"\n\033[1;34m[INFO]\033[0m Temps total: {end_time - start_time:.2f} secondes")
        
        # Afficher le résultat final
        print(f"\n\033[1;32m[RÉSULTAT]\033[0m Test réussi! L'API RAGondin est compatible avec la bibliothèque OpenAI.")
        
    except Exception as e:
        print(f"\n\033[1;31m[ERREUR]\033[0m Une erreur s'est produite lors de l'appel à l'API: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de l'API RAGondin avec le client OpenAI")
    parser.add_argument("--url", type=str, default="http://localhost:8083", help="URL de base de l'API RAGondin")
    parser.add_argument("--question", type=str, default="Qu'est-ce que RAGondin?", help="Question à poser")
    parser.add_argument("--stream", action="store_true", help="Activer le mode streaming")
    
    args = parser.parse_args()
    
    test_openai_client(args.url, args.question, args.stream) 