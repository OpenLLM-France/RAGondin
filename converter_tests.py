import os

os.chdir("ragondin")

arguments = {
    "loader.image_captioning": ["true", "false"],
    "loader.file_loaders.pdf": ["MarkerLoader", "DoclingLoader"],
}


def generate_combinations(arguments):
    if len(arguments) == 0:
        return [{}]
    else:
        key, values = arguments.popitem()
        sub_combinations = generate_combinations(arguments)
        return [
            {key: value, **sub_combination}
            for value in values
            for sub_combination in sub_combinations
        ]


def add_arg(cmd, key, value):
    return f"{cmd} -{key} {value}"


combinations = generate_combinations(arguments)

cmd = "uv run python manage_collection.py -f ../data"
cmd = add_arg(cmd, "o", "vectordb.enable=false")

for combination in combinations:
    new_cmd = cmd
    for key, value in combination.items():
        val = key + "=" + value
        new_cmd = add_arg(new_cmd, "o", val)

    print("Running command:", new_cmd)
    os.system(new_cmd)
