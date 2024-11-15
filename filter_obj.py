import pandas as pd
import click

@click.command()
@click.option('--ff', '--fformat', multiple=True, help='Liste des formats de fichier à inclure (ex: pdf, html)')
@click.option('--xf', '--xformat', multiple=True, help='Liste des formats de fichier à exclure (ex: pdf, html)')
@click.option('--ms', '--min-size', type=int, help='Taille minimum des fichiers à inclure (en bytes)')
@click.option('--h', '--head', type=int, help='Nombre de fichiers à récupérer, classés par ordre décroissant de taille')

def filter_files(ff, xf, ms, h):
    """
    Filtre le fichier CSV en fonction des critères spécifiés et génère un fichier files-to-remove.txt.
    """
    df = pd.read_csv("problematic_objects_report.csv")

    if h:
        df = df.head(h)
    if ms:
        df = df[df['Row Size'] >= ms]

    if ff:
        df = df[df['Format'].isin(ff)]


    if xf:
        df = df[~df['Format'].isin(xf)]

    print(df.columns.tolist())

    df = df.sort_values(by='Row Size', ascending=False)
    df = df.drop_duplicates(subset=['Object Path'])

    for index, row in df.iterrows():
        print(f"Path: {row['Object Path']}, Size: {row['Size']} ")

    # Écrire les chemins des fichiers à supprimer dans le fichier de sortie
    df['Object Path'].to_csv("gitfiles_to_clean.csv", index=False, header=False)

    click.echo(f"Report generated: gitfiles_to_clean.csv")
    total_size = df['Row Size'].sum()
    total_size_mb = total_size / (1024 * 1024)
    click.echo(f"Total size to be deleted: {total_size_mb:.2f} MB")

if __name__ == '__main__':
    filter_files()