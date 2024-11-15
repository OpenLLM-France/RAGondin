#!/bin/bash

# Fichier de sortie
output_file="problematic_objects_report.txt"

# Initialiser le fichier de sortie
echo "Commit, Branch, Object ID, Object Path, Size" > $output_file

# Trouver les gros objets (excepté le .pack)
problematic_objects=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | grep -v '\.pack')

# Parcourir chaque objet problématique
echo "$problematic_objects" | while read -r line; do
    object_id=$(echo $line | awk '{print $1}')
    object_size=$(echo $line | awk '{print $2}')
    object_path=$(echo $line | awk '{print $3}')

    # Trouver les commits responsables de l'objet
    commits=$(git log --all --find-object=$object_id --pretty=format:"%H")

    # Parcourir chaque commit et trouver les branches correspondantes
    for commit in $commits; do
        branches=$(git branch -r --contains $commit | sed 's/^[ \t]*//')
        for branch in $branches; do
            # Formater la taille en unités lisibles
            formatted_size=$(numfmt --to=iec-i --suffix=B $object_size)
            echo "$commit, $branch, $object_id, $object_path, $formatted_size" >> $output_file
        done
    done
done

# Trier le fichier de sortie par taille (colonne 5)
sort -t, -k5 -hr $output_file -o $output_file

echo "Report generated: $output_file"