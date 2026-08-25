from api_client import get_pokemon_detail, get_pokemon_list

lista = get_pokemon_list(limit=5)
print(lista)

print("*"*5)

pikachu = get_pokemon_detail("pikachu")
print(pikachu)