"""
Arquivo de requisições do PokéAPI
"""

import requests

from config import BASE_URL, REQUEST_TIMEOUT

def get_pokemon_list(limit: int) -> list[dict]:
    url = f"{BASE_URL}/pokemon?limit={limit}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()['results']

def get_pokemon_detail(name_or_url: str) -> dict:
    
    url = name_or_url if name_or_url.startswith("http") else f"{BASE_URL}/pokemon/{name_or_url}"
 
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    raw = response.json() # Dados Gerais

    stats = {}

    for s in raw["stats"]:
        stats[s["stat"]["name"]] = s["base_stat"]

    types = []

    for t in raw["types"]:
        types.append(t["type"]["name"])  # Nome do tipo dentro de raw
    return {
        "id": raw["id"],
        "name": raw["name"],
        "hp": stats.get("hp"),
        "attack": stats.get("attack"),
        "defense": stats.get("defense"),
        "special-attack": stats.get("special-attack"),
        "special-defense": stats.get("special-defense"),
        "speed": stats.get("speed"),
        "type1": types[0] if len(types) > 0 else None,
        "type2": types[1] if len(types) > 1 else None,
    }
