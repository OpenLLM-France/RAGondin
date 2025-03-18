#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test utilisant requests pour vérifier l'API compatible OpenAI
sans dépendre de la bibliothèque openai.
"""

import argparse
import time
import json
import requests
import os
import sys
import sseclient

def test_openai_api_with_requests(base_url: str, question: str, stream: bool = False):
    """
    Teste l'API RAGondin compatible OpenAI en utilisant des requêtes HTTP directes.
    
    Args:
        base_url: URL de base de l'API RAGondin (sans /v1)
        question: Question à poser à l'API
        stream: Activer le mode streaming
    """
    print("\n\033[1;34m[INFO]\033[0m Test avec requests directement")
    print(f"\033[1;34m[INFO]\033[0m URL de base: {base_url}")
    print(f"\033[1;34m[INFO]\033[0m Question: {question}")
    print(f"\033[1;34m[INFO]\033[0m Streaming: {stream}")
    
    # Construire l'URL complète
    url = f"{base_url}/v1/chat/completions"
    
    # Préparer les headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-fake-key"  # Notre API ignore cette clé, mais on la met par convention
    }
    
    # Préparer les données de la requête
    data = {
        "model": "RAGondin",
        "messages": [
            {"role": "system", "content": "Tu es un assistant utile basé sur RAGondin."},
            {"role": "user", "content": question}
        ],
        "stream": stream
    }
    
    start_time = time.time()
    
    try:
        # Envoi de la requête
        if stream:
            print("\n\033[1;33m[RÉPONSE STREAMING]\033[0m")
            
            # Mode streaming - on utilise stream=True pour requests
            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            # Traiter la réponse SSE (Server-Sent Events)
            client = sseclient.SSEClient(response)
            full_response = ""
            
            for event in client.events():
                if event.data == "[DONE]":
                    break
                    
                try:
                    chunk = json.loads(event.data)
                    if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta and delta["content"]:
                            content = delta["content"]
                            full_response += content
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue
            
            print("\n")
            result = {"content": full_response}
            
        else:
            # Mode non-streaming
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            # Récupérer et afficher la réponse
            response_data = response.json()
            
            print("\n\033[1;33m[RÉPONSE]\033[0m")
            
            # Formater le résultat
            result = {
                "id": response_data["id"],
                "model": response_data["model"],
                "content": response_data["choices"][0]["message"]["content"]
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        end_time = time.time()
        print(f"\n\033[1;34m[INFO]\033[0m Temps total: {end_time - start_time:.2f} secondes")
        
        # Afficher les métadonnées des sources si disponibles
        if "X-Metadata-Sources" in response.headers:
            try:
                sources = json.loads(response.headers["X-Metadata-Sources"])
                print("\n\033[1;34m[SOURCES]\033[0m")
                for i, source in enumerate(sources):
                    print(f"{i+1}. {source.get('title', 'Document sans titre')} - URL: {source.get('url', 'N/A')}")
            except:
                pass
        
        # Afficher le résultat final
        print(f"\n\033[1;32m[RÉSULTAT]\033[0m Test réussi! L'API RAGondin est compatible avec le format OpenAI.")
        
    except Exception as e:
        print(f"\n\033[1;31m[ERREUR]\033[0m Une erreur s'est produite lors de l'appel à l'API: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de l'API RAGondin compatible OpenAI avec requests")
    parser.add_argument("--url", type=str, default="http://localhost:8083", help="URL de base de l'API RAGondin")
    parser.add_argument("--question", type=str, default="Qu'est-ce que RAGondin?", help="Question à poser")
    parser.add_argument("--stream", action="store_true", help="Activer le mode streaming")
    
    args = parser.parse_args()
    
    test_openai_api_with_requests(args.url, args.question, args.stream) 