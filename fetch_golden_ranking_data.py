"""
Fetch golden_ranking comparison data from S3.

This script downloads comparison data from the quality-comparison-toolkit S3 bucket
specifically for golden_ranking patches.
"""

import boto3
import json
import os
import argparse
from typing import Dict, List
from botocore.config import Config as BotoConfig
from data_loader import get_s3_client


def list_golden_ranking_files(bucket_name: str, prefix: str) -> List[str]:
    """
    List all files in the golden_ranking S3 path.

    Args:
        bucket_name: S3 bucket name
        prefix: S3 prefix (e.g., 'quality-comparison-toolkit/data/golden_ranking_patches/')

    Returns:
        List of S3 keys
    """
    s3 = get_s3_client()

    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        files = []
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.json'):
                    files.append(key)

        print(f"Found {len(files)} JSON files in {prefix}")
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def download_s3_file(bucket_name: str, key: str, local_path: str):
    """
    Download a file from S3 to local path.

    Args:
        bucket_name: S3 bucket name
        key: S3 object key
        local_path: Local file path to save to
    """
    s3 = get_s3_client()

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading {key} to {local_path}")
        s3.download_file(bucket_name, key, local_path)
        print(f"Downloaded successfully")
    except Exception as e:
        print(f"Error downloading {key}: {e}")


def load_json_from_s3(bucket_name: str, key: str) -> Dict:
    """
    Load JSON data directly from S3.

    Args:
        bucket_name: S3 bucket name
        key: S3 object key

    Returns:
        Parsed JSON data
    """
    s3 = get_s3_client()

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        data = json.loads(obj['Body'].read().decode('utf-8'))
        return data
    except Exception as e:
        print(f"Error loading {key}: {e}")
        return {}


def fetch_golden_ranking_comparisons(
    bucket_name: str,
    prefix: str,
    output_dir: str
) -> List[Dict]:
    """
    Fetch all golden_ranking comparison data from S3.

    Args:
        bucket_name: S3 bucket name
        prefix: S3 prefix for golden_ranking data
        output_dir: Local directory to save downloaded data

    Returns:
        List of all comparison dicts
    """
    # List all files in the golden_ranking path
    files = list_golden_ranking_files(bucket_name, prefix)

    all_comparisons = []

    for key in files:
        print(f"\nProcessing {key}")

        # Load JSON data
        data = load_json_from_s3(bucket_name, key)

        # Handle different possible data structures
        if isinstance(data, list):
            # Direct list of comparisons
            comparisons = data
        elif isinstance(data, dict):
            # Dict with 'comparisons' key
            comparisons = data.get('comparisons', [])

            # Or check if it's a single comparison dict
            if not comparisons and 'img1' in data and 'img2' in data:
                comparisons = [data]
        else:
            print(f"Unknown data structure in {key}")
            continue

        print(f"Found {len(comparisons)} comparisons")

        # Add source file info to each comparison
        for comp in comparisons:
            comp['source_file'] = key

        all_comparisons.extend(comparisons)

        # Also save the raw file locally for inspection
        filename = os.path.basename(key)
        local_path = os.path.join(output_dir, 'raw', filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"\nTotal comparisons fetched: {len(all_comparisons)}")

    # Save aggregated comparisons
    aggregated_path = os.path.join(output_dir, 'all_golden_ranking_comparisons.json')
    with open(aggregated_path, 'w') as f:
        json.dump(all_comparisons, f, indent=2)

    print(f"Saved aggregated comparisons to {aggregated_path}")

    return all_comparisons


def main():
    parser = argparse.ArgumentParser(description='Fetch golden_ranking comparison data from S3')
    parser.add_argument('--bucket', type=str, default='surface-quality-dataset',
                        help='S3 bucket name')
    parser.add_argument('--prefix', type=str,
                        default='quack_v2_data_gold_ranking_logs/',
                        help='S3 prefix for golden_ranking data')
    parser.add_argument('--output-dir', type=str, default='training_data_golden_ranking2',
                        help='Local directory to save downloaded data')

    args = parser.parse_args()

    print("=" * 80)
    print("Fetching Golden Ranking Comparison Data from S3")
    print("=" * 80)
    print(f"Bucket: {args.bucket}")
    print(f"Prefix: {args.prefix}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch comparisons
    comparisons = fetch_golden_ranking_comparisons(
        args.bucket,
        args.prefix,
        args.output_dir
    )

    print("\n" + "=" * 80)
    print(f"Successfully fetched {len(comparisons)} golden_ranking comparisons")
    print("=" * 80)


if __name__ == '__main__':
    main()
