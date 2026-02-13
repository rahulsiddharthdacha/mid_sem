#!/usr/bin/env python3
"""
Example script demonstrating the complete table detection cycle:
1. Upload an Excel file
2. Detect tables using the trained ML model
3. Get structured JSON output

Prerequisites:
- Start the API server: uvicorn serving.app:app --host 0.0.0.0 --port 8000
- Have a trained model available (run: python model/train_model.py)
"""

import requests
import json
from pathlib import Path
import sys

def detect_tables_in_excel(file_path: str, api_url: str = "http://localhost:8000") -> dict:
    """
    Upload an Excel file and detect tables.
    
    Args:
        file_path: Path to the Excel file
        api_url: Base URL of the API server
        
    Returns:
        Dictionary containing detection results
    """
    endpoint = f"{api_url}/detect-tables"
    
    # Check if file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Upload and detect
    print(f"📤 Uploading file: {file_path}")
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        response = requests.post(endpoint, files=files, timeout=30)
    
    # Check response
    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")
    
    return response.json()


def display_results(result: dict):
    """Display detection results in a readable format."""
    print("\n" + "="*70)
    print("📊 TABLE DETECTION RESULTS")
    print("="*70)
    
    # Summary
    summary = result.get("summary", {})
    print(f"\n📋 File: {result.get('filename', 'N/A')}")
    print(f"   Status: {result.get('status', 'N/A')}")
    print(f"\n📈 Statistics:")
    print(f"   - Total Cells: {summary.get('total_cells', 0)}")
    print(f"   - Header Cells: {summary.get('header_cells', 0)}")
    print(f"   - Data Cells: {summary.get('data_cells', 0)}")
    dims = summary.get('dimensions', {})
    print(f"   - Dimensions: {dims.get('rows', 0)} rows × {dims.get('columns', 0)} columns")
    
    # Headers
    headers = result.get("detected_tables", {}).get("headers", [])
    print(f"\n📌 Detected Headers ({len(headers)}):")
    for i, header in enumerate(headers[:10], 1):  # Show first 10
        print(f"   {i}. [{header['row']}, {header['column']}]: {header['value']}")
    if len(headers) > 10:
        print(f"   ... and {len(headers) - 10} more headers")
    
    # Data
    data_rows = result.get("detected_tables", {}).get("data", [])
    print(f"\n📊 Data Rows ({len(data_rows)}):")
    for i, row in enumerate(data_rows[:5], 1):  # Show first 5 rows
        row_num = row['row_number']
        cells = row['cells']
        print(f"   Row {row_num}: {len(cells)} cells")
        # Show first 3 cells of each row
        for cell in cells[:3]:
            print(f"      - [{cell['row']}, {cell['column']}]: {cell['value']}")
        if len(cells) > 3:
            print(f"      ... and {len(cells) - 3} more cells")
    if len(data_rows) > 5:
        print(f"   ... and {len(data_rows) - 5} more rows")
    
    print("\n" + "="*70)


def main():
    """Main function."""
    # Default test file
    test_file = "data/sales_report.xlsx"
    
    # Allow user to specify a different file
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    try:
        # Check if API is running
        print("🔍 Checking API server...")
        try:
            health_response = requests.get("http://localhost:8000/health", timeout=5)
            if health_response.status_code == 200:
                health = health_response.json()
                print(f"✅ API server is running (model loaded: {health.get('model_loaded', False)})")
            else:
                print("⚠️  API server responded but returned an error")
        except requests.exceptions.ConnectionError:
            print("❌ API server is not running!")
            print("\nPlease start the server first:")
            print("   uvicorn serving.app:app --host 0.0.0.0 --port 8000")
            return 1
        
        # Detect tables
        result = detect_tables_in_excel(test_file)
        
        # Display results
        display_results(result)
        
        # Optionally save to file
        output_file = "table_detection_result.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full results saved to: {output_file}")
        
        print("\n✅ Table detection completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
