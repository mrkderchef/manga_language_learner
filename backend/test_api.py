#!/usr/bin/env python3
"""
Test script for Manga Language Learner API
"""

import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n📋 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_list_panels():
    """Test listing panels"""
    print("\n📋 Testing list panels endpoint...")
    response = requests.get(f"{BASE_URL}/api/panels/list")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('total', 0)} panels:")
    for panel in data.get('panels', []):
        print(f"  - {panel['filename']} ({panel['size']} bytes)")

def test_service_status():
    """Test service status"""
    print("\n📋 Testing service status...")
    response = requests.get(f"{BASE_URL}/api/panels/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_ocr(filename: str):
    """Test OCR on a panel"""
    print(f"\n📋 Testing OCR on {filename}...")
    response = requests.post(f"{BASE_URL}/api/panels/{filename}/ocr")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Extracted text: {data.get('text', 'N/A')}")
        print(f"Annotations found: {len(data.get('annotations', []))}")
    else:
        print(f"Error: {response.text}")

def test_extract_and_translate(filename: str):
    """Test OCR and translation"""
    print(f"\n📋 Testing extract and translate on {filename}...")
    response = requests.post(f"{BASE_URL}/api/panels/{filename}/extract-and-translate")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Original text: {data.get('original_text', 'N/A')}")
        print(f"Translated text: {data.get('translated_text', 'N/A')}")
        print(f"Annotations with translations: {len(data.get('annotations', []))}")
    else:
        print(f"Error: {response.text}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Manga Language Learner API - Test Suite")
    print("=" * 60)
    
    try:
        # Test basic endpoints
        test_health()
        test_service_status()
        test_list_panels()
        
        # Test OCR on first available panel
        response = requests.get(f"{BASE_URL}/api/panels/list")
        if response.status_code == 200:
            panels = response.json().get('panels', [])
            if panels:
                filename = panels[0]['filename']
                print(f"\nUsing first panel: {filename}")
                test_ocr(filename)
                test_extract_and_translate(filename)
        
        print("\n✅ All tests completed!")
    
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Make sure the server is running!")
        print(f"   Try running: python app.py")

if __name__ == "__main__":
    main()
