FEATURES = [
    "H",
    "diameter",
    "albedo",
    "n",
    "per_y",
    "class"
]

TARGET = "moid"

# Colunas usadas só pra filtrar outliers (cometas quase-parabólicos),
# não entram no treino do modelo.
EXTRA_FILTER_COLUMNS = ["a", "e"]