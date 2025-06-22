import json

with open('Data/combined_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Total syllabuses:", len(data["syllabuses"]))

# List first 10 subjects
for i, syllabus in enumerate(data["syllabuses"][:10]):
    metadata = syllabus.get("metadata", {})
    subject_code = metadata.get("subject_code_on_page", "UNKNOWN")
    title = metadata.get("title", "N/A")
    print(f"{i+1}. {subject_code}: {title}")

print("...")
print(f"Total: {len(data['syllabuses'])} môn học") 