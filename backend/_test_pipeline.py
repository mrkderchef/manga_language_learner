"""Quick test of manga-ocr pipeline on the benchmark image."""
import sys
sys.path.insert(0, ".")

from services.manga_ocr_service import extract_and_translate

img = r"c:\Users\kammmar\OneDrive - adidas\Desktop\manga_language_learner\panels\uploads\AisazuNihaIrarenai-003.jpg"
print("Running manga-ocr pipeline...")
result = extract_and_translate(img)

print(f"\nSuccess: {result['success']}")
print(f"Method: {result['method']}")
print(f"Annotations: {len(result['annotations'])}")
print()
for i, a in enumerate(result["annotations"]):
    print(f"  {i+1}. Text: {a['text']}")
    print(f"     Translation: {a['translated']}")
    print(f"     BBox: [{a['bbox'][0][0]},{a['bbox'][0][1]}] {a['bbox'][1][0]-a['bbox'][0][0]}x{a['bbox'][2][1]-a['bbox'][0][1]}")
    print()
