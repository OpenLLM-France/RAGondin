#!/bin/bash

# Vérifier si un argument de chemin a été fourni

repo_path=$(pwd)


# Vérifier si un argument de taille minimale a été fourni
if [ -z "$1" ]; then
    min_size=500
else
    min_size=$2
fi

# Convertir la taille minimale en octets
min_size_bytes=$((min_size * 1024))

# Changer de répertoire
cd "$repo_path" || { echo "Le répertoire $repo_path n'existe pas."; exit 1; }

# Fichier de sortie
output_file="problematic_objects_report.csv"
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Initialiser le fichier de sortie
echo "Commit,Branch,Object ID,Object Path,Format,Author,Date,Row Size,Size" > $output_file

# Trouver les gros objets (excepté le .pack) et les trier par taille
problematic_objects=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | grep -v '\.pack' | awk -v min_size_bytes="$min_size_bytes" '$2 >= min_size_bytes' | sort -k2 -nr)

# Utiliser un tableau associatif pour éviter les doublons
declare -A seen_objects

# Parcourir chaque objet problématique
echo "$problematic_objects" | while read -r line; do
    object_id=$(echo $line | awk '{print $1}')
    object_size=$(echo $line | awk '{print $2}')
    object_path=$(echo $line | awk '{print $3}')
    object_format=$(echo $object_path | awk -F. '{print $NF}')

    # Trouver les commits responsables de l'objet
    commits=$(git log --all --find-object=$object_id --pretty=format:"%H")

    # Parcourir chaque commit et trouver les branches correspondantes
    for commit in $commits; do
        branches=$(git branch -r --contains $commit | sed 's/^[ \t]*//')
        for branch in $branches; do
            # Ignorer les entrées avec '->'
            if [[ "$branch" == *"->"* ]]; then
                continue
            fi
            # Obtenir l'auteur et la date du commit
            author=$(git log -1 --pretty=format:"%an" $commit)
            date=$(git log -1 --pretty=format:"%ad" --date=iso $commit)
            # Formater la taille en unités lisibles
            formatted_size=$(numfmt --to=iec-i --suffix=B $object_size)
            key="$commit,$branch,$object_id,$object_path,$object_format,$author,$date,$object_size,$formatted_size"
            if [[ -z "${seen_objects[$key]}" ]]; then
                echo "$key" >> $output_file
                seen_objects[$key]=1
            fi
        done
    done
done

# Afficher le nombre de fichiers trouvés
file_count=$(wc -l < "$output_file")
file_count=$((file_count - 1)) # Soustraire l'en-tête
echo "Number of problematic files found: $file_count"

echo "Report generated: $output_file"
