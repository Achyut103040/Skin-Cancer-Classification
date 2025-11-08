"""
Google Drive File ID Extractor
Help you extract file IDs from Google Drive sharing links
"""

print("="*60)
print("  Google Drive File ID Extractor")
print("="*60)
print()
print("Paste your Google Drive sharing links to extract file IDs")
print("Example link format:")
print("https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing")
print()
print("Type 'done' when finished")
print("="*60)
print()

model_names = [
    'best_skin_cancer_model_balanced.pth (Binary)',
    'nv_model.pth',
    'bkl_model_cascade_fixed.pth',
    'bcc_model.pth',
    'akiec_model.pth',
    'vasc_model.pth'
]

file_ids = {}

for model_name in model_names:
    print(f"\n📦 {model_name}")
    link = input("   Paste Google Drive link: ").strip()
    
    if link.lower() == 'done':
        break
    
    # Extract file ID from various Google Drive link formats
    if '/file/d/' in link:
        # Format: https://drive.google.com/file/d/FILE_ID/view
        file_id = link.split('/file/d/')[1].split('/')[0]
    elif 'id=' in link:
        # Format: https://drive.google.com/open?id=FILE_ID
        file_id = link.split('id=')[1].split('&')[0]
    elif '/d/' in link:
        # Format: https://drive.google.com/d/FILE_ID
        file_id = link.split('/d/')[1].split('/')[0]
    else:
        print("   ⚠️  Could not extract ID. Please check the link format.")
        continue
    
    file_ids[model_name] = file_id
    print(f"   ✅ File ID: {file_id}")

print("\n" + "="*60)
print("  Configuration for streamlit_web_app.py")
print("="*60)
print("\nCopy this to your code (around line 40):\n")

print("GDRIVE_MODEL_URLS = {")
if 'best_skin_cancer_model_balanced.pth (Binary)' in file_ids:
    print(f"    'binary': 'https://drive.google.com/uc?id={file_ids['best_skin_cancer_model_balanced.pth (Binary)']}',")
if 'nv_model.pth' in file_ids:
    print(f"    'nv': 'https://drive.google.com/uc?id={file_ids['nv_model.pth']}',")
if 'bkl_model_cascade_fixed.pth' in file_ids:
    print(f"    'bkl': 'https://drive.google.com/uc?id={file_ids['bkl_model_cascade_fixed.pth']}',")
if 'bcc_model.pth' in file_ids:
    print(f"    'bcc': 'https://drive.google.com/uc?id={file_ids['bcc_model.pth']}',")
if 'akiec_model.pth' in file_ids:
    print(f"    'akiec': 'https://drive.google.com/uc?id={file_ids['akiec_model.pth']}',")
if 'vasc_model.pth' in file_ids:
    print(f"    'vasc': 'https://drive.google.com/uc?id={file_ids['vasc_model.pth']}',")
print("}")

print("\n" + "="*60)
print("✅ Done! Now:")
print("1. Copy the GDRIVE_MODEL_URLS dictionary above")
print("2. Paste it in streamlit_web_app.py (replace existing)")
print("3. Set USE_GDRIVE_MODELS = True")
print("4. Deploy!")
print("="*60)

input("\nPress Enter to exit...")
