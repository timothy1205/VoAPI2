import os
import json
import argparse
import re
import time

DATASET_PATH = "../cvelistV5/cves"
OUTPUT_FILE = "endpoint_cve_dataset.json"

API_CWES = {
    "CWE-918", # SSRF
    "CWE-862", # Missing Authorization
    "CWE-639", # IDOR
    "CWE-306", # Missing Authentication
    "CWE-22",  # Path Traversal
    "CWE-502", # Deserialization
    "CWE-79"   # XSS
}

def get_nested_value(data, path, default=None):
    keys = path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list):
            if len(current) > 0 and isinstance(current[0], dict):
                if key in current[0]:
                    current = current[0][key]
                else:
                    return default
            else:
                return default
        else:
            return default
    return current

def extract_endpoints(text):
    # TODO: Reliably parse endpoints from description.
    return []

def parse_cve_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cve_data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
        return None

    cve_id = get_nested_value(cve_data, "cveMetadata.cveId")

    descriptions = get_nested_value(cve_data, "containers.cna.descriptions", [])
    description_text = ""
    for desc in descriptions:
        if desc.get("lang") == "en":
            description_text = desc.get("value", "")
            break

    if not description_text:
        return None

    problem_types = get_nested_value(cve_data, "containers.cna.problemTypes", [])
    found_cwes = []
    if problem_types:
        for pt in problem_types:
            for desc in pt.get("descriptions", []):
                cwe = desc.get("cweId")
                if cwe:
                    found_cwes.append(cwe)

    endpoints = extract_endpoints(description_text)

    if any(cwe in API_CWES for cwe in found_cwes):
        return {
            "cve_id": cve_id,
            "cwe_ids": found_cwes,
            "extracted_endpoints": endpoints,
            "description": description_text,
        }
    return None

def main():
    parser = argparse.ArgumentParser(description="Scan CVE dataset for REST API vulnerabilities")
    parser.add_argument('--cve', help="Process a single CVE file (e.g. CVE-2021-29490).")
    args = parser.parse_args()

    if args.cve:
        # Parse single file for testing
        cve_name = args.cve
        cve_file = f"{cve_name}.json"

        m = re.match(r'^CVE-(\d{4})-(\d+)(?:\.json)?$', cve_name, re.IGNORECASE)
        if not m:
            print("Invalid CVE name; expected format CVE-YYYY-NNNN")
            return

        year, idnum = m.group(1), m.group(2)
        prefix = idnum[:2] if len(idnum) >= 2 else idnum
        subdir = f"{prefix}xxx"
        year_dir = os.path.join(DATASET_PATH, year)
        file_path = None

        if os.path.isdir(year_dir):
            for d in os.listdir(year_dir):
                if not idnum.startswith(d.replace('x' ,'')):
                    continue

                candidate = os.path.join(year_dir, d, cve_file)
                if os.path.exists(candidate):
                    file_path = candidate
                    break


        if file_path:
            print(f"Loading single CVE file: {file_path}")
            result = parse_cve_file(file_path)
            if result:
                print(json.dumps(result, indent=2))
            else:
                print("No data was parsed for {cve_name}")
        else:
            print(f"No file named {cve_file} found under {DATASET_PATH}")

        return


    print(f"Scanning {DATASET_PATH} for REST API CVEs...")
    dataset = []

    start = time.perf_counter()

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                result = parse_cve_file(file_path)
                if result:
                    dataset.append(result)

    elapsed = time.perf_counter() - start

    print(f"Found {len(dataset)} relevant CVEs in {elapsed:.2f}s.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
