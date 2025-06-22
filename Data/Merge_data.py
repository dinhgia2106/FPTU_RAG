import json
import csv

def merge_data():
    # Paths to files
    json_path = "e:/FPTU_RAG_V2/Data/flm_data_AI.json"
    csv_path = "e:/FPTU_RAG_V2/Data/Students_AI17D.csv"
    output_path = "e:/FPTU_RAG_V2/Data/combined_data.json"
    
    # Read the JSON data
    with open(json_path, 'r', encoding='utf-8') as json_file:
        curriculum_data = json.load(json_file)
    
    # Read the CSV data
    students = []
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        # Skip the first line if it contains the file path comment
        first_line = csv_file.readline().strip()
        if first_line.startswith('//'):
            csv_file.seek(0)
            next(csv_file)  # Skip the first line
        else:
            csv_file.seek(0)  # Reset to beginning if no comment
            
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            students.append(row)
    
    # Merge the data
    combined_data = curriculum_data.copy()
    combined_data['students'] = students
    
    # Write the combined data to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(combined_data, output_file, ensure_ascii=False, indent=4)
    
    print(f"Combined data has been saved to {output_path}")

if __name__ == "__main__":
    merge_data()