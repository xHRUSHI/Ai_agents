import requests
import json
import subprocess
from datetime import datetime
import os
import pandas as pd
from collections import defaultdict

# Configuration
SOURCES = {
    "github_malware_db": {
        "url": "https://github.com/aaryanrlondhe/Malware-Hash-Database.git",
        "types": ["MD5", "SHA-1", "SHA-256"],
        "path": "Malware-Hash-Database"
    },
    "malicious_hash_repo": {
        "url": "https://github.com/romainmarcoux/malicious-hash",
        "files": {
            "MD5": "https://raw.githubusercontent.com/romainmarcoux/malicious-hash/main/full-hash-md5-aa.txt",
            "SHA1": "https://raw.githubusercontent.com/romainmarcoux/malicious-hash/main/full-hash-sha1-aa.txt",
            "SHA256": "https://raw.githubusercontent.com/romainmarcoux/malicious-hash/main/full-hash-sha256-aa.txt"
        }
    },
    "malwarebazaar": {
        "recent": "https://bazaar.abuse.ch/export/csv/recent/",
        "full": "https://bazaar.abuse.ch/export/csv/full/"
    }
}

OUTPUT_FILE = f"malware_hashes_{datetime.now().strftime('%Y%m%d')}.json"
existing_hashes = set()

def update_git_repo(source):
    """Clone/update GitHub repositories with malware hashes:cite[1]:cite[5]"""
    if not os.path.exists(source['path']):
        subprocess.run(["git", "clone", source['url'], source['path']])
    else:
        subprocess.run(["git", "-C", source['path'], "pull"])

def process_github_data(source):
    """Process GitHub repository hash files"""
    hashes = []
    for root, _, files in os.walk(source['path']):
        for file in files:
            if any(file.endswith(ext) for ext in ['.md5', '.sha1', '.sha256']):
                with open(os.path.join(root, file)) as f:
                    for line in f:
                        hash_val = line.strip()
                        if hash_val not in existing_hashes:
                            hashes.append({
                                "hash": hash_val,
                                "type": file.split('.')[-1].upper(),
                                "source": "GitHub-Malware-Hash-DB",
                                "first_seen": datetime.now().isoformat()
                            })
                            existing_hashes.add(hash_val)
    return hashes

def fetch_malicious_hash_repo():
    """Fetch hourly updated hash lists:cite[5]"""
    hashes = []
    for hash_type, url in SOURCES['malicious_hash_repo']['files'].items():
        response = requests.get(url)
        for line in response.text.splitlines():
            if line.strip() and line not in existing_hashes:
                hashes.append({
                    "hash": line.strip(),
                    "type": hash_type,
                    "source": "Malicious-Hash-Repo",
                    "first_seen": datetime.now().isoformat()
                })
                existing_hashes.add(line.strip())
    return hashes

def fetch_malwarebazaar():
    """Get recent hashes from MalwareBazaar API:cite[7]:cite[9]"""
    hashes = []
    try:
        response = requests.get(SOURCES['malwarebazaar']['recent'])
        df = pd.read_csv(response.text.splitlines()[7:], sep=",", quotechar='"')
        
        for _, row in df.iterrows():
            if row['sha256_hash'] not in existing_hashes:
                hashes.append({
                    "hash": row['sha256_hash'],
                    "type": "SHA256",
                    "source": "MalwareBazaar",
                    "first_seen": row['first_seen'],
                    "file_type": row['file_type'],
                    "signature": row['signature']
                })
                existing_hashes.add(row['sha256_hash'])
    except Exception as e:
        print(f"Error fetching MalwareBazaar: {str(e)}")
    return hashes

def save_to_json(data):
    """Save collected hashes with metadata"""
    output = {
        "metadata": {
            "collection_time": datetime.now().isoformat(),
            "source_count": len(SOURCES),
            "total_hashes": len(data)
        },
        "hashes": data
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

def main():
    all_hashes = []
    
    # Process GitHub sources
    update_git_repo(SOURCES['github_malware_db'])
    all_hashes += process_github_data(SOURCES['github_malware_db'])
    
    # Fetch from malicious-hash repo
    all_hashes += fetch_malicious_hash_repo()
    
    # Get MalwareBazaar data
    all_hashes += fetch_malwarebazaar()
    # print(all_hashes)
    # Save results
    save_to_json(all_hashes)
    print(f"Collected {len(all_hashes)} new hashes. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()