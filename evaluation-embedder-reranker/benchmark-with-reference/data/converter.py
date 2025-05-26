import csv
import json
import unicodedata
import lzma
import gzip

def jsonl_gz_to_csv(input_path, output_path):
    with gzip.open(input_path, mode='rt', encoding='utf-8') as gz_file:
        with open(output_path, mode='w', encoding='utf-8', newline='') as csv_file:
            writer = None
            for line_number, line in enumerate(gz_file, start=1):
                try:
                    # Parse the JSON line
                    data = json.loads(line.strip())
                    
                    # Initialize the CSV writer with headers from the first JSON object
                    if writer is None:
                        writer = csv.DictWriter(csv_file, fieldnames=data.keys())
                        writer.writeheader()
                    
                    # Write the JSON object as a row in the CSV file
                    writer.writerow(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_number}: {e}")

def tsv_to_csv(tsv_path, csv_path):
    with open(tsv_path, mode='r', encoding='utf-8') as tsv_file:
        with open(csv_path, mode='w', encoding='utf-8', newline='') as csv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            csv_writer = csv.writer(csv_file, delimiter=',')
            
            for row in tsv_reader:
                csv_writer.writerow(row)

tsv_to_csv("./test.tsv", "qrels.csv")