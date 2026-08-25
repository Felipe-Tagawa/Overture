BASE_URL = "https://pokeapi.co/api/v2"

POKEMON_LIMIT = 151 # Evitar peso desnecessário

CACHE_PATH = "data/cache/pokemon_dataset.json"

REQUEST_TIMEOUT = 10 # em segundos

FEATURES = [
    "hp",
    "attack",
    "defense",
    "special_attack",
    "special_defense",
    "speed"
]

TARGET = "type1"