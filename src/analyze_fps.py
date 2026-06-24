import openpyxl
import json
import glob
import os

def main():
    print('Loading Excel...')
    wb = openpyxl.load_workbook('data/Celiac Diagnosis by Manual Review.xlsx', data_only=True)
    sheet = wb.active

    gt_dict = {}
    gt_comments = {}
    gt_original = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        patient_id = str(row[0]).strip() if row[0] is not None else ''
        diagnosis = str(row[1]).strip() if row[1] is not None else ''
        comment = str(row[2]).strip() if row[2] is not None else ''
        if patient_id:
            gt_original[patient_id] = diagnosis
            gt_comments[patient_id] = comment
            if diagnosis in ['Yes', 'PMH']:
                gt_dict[patient_id] = 'Positive'
            elif diagnosis == 'No':
                gt_dict[patient_id] = 'Negative'
            else:
                gt_dict[patient_id] = 'Unknown'

    res_dir = 'results/celiac_agent'
    json_files = glob.glob(os.path.join(res_dir, '*.json'))

    fps = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            grid = str(data.get('grid', '')).strip()
            diag = data.get('diagnosis', 'Unknown')
            reasoning = data.get('reasoning', '')
            
            if grid in gt_dict:
                gt = gt_dict[grid]
                if gt == 'Negative' and diag == 'Positive':
                    fps.append({
                        'grid': grid,
                        'reasoning': reasoning,
                        'gt_comment': gt_comments[grid]
                    })

    # Group the false positives by common reasons or just list them.
    # We will format this directly into a markdown artifact.
    artifact_path = '.gemini/antigravity/brain/2844c27f-8836-4277-88ad-0b2f14784d1c/false_positives_analysis.md'
    
    report = "# False Positives Analysis\n\n"
    report += f"Total False Positives: {len(fps)}\n\n"
    
    for i, fp in enumerate(fps, 1):
        report += f"## {i}. Patient {fp['grid']}\n"
        report += f"**Manual Review Comment:**\n{fp['gt_comment']}\n\n"
        report += f"**Agent Reasoning:**\n{fp['reasoning']}\n\n"
        report += "---\n\n"
        
    with open(artifact_path, 'w') as f:
        f.write(report)
        
    print(f"Generated analysis at {artifact_path}")

if __name__ == "__main__":
    main()
