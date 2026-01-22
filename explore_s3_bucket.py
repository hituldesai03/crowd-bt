"""
Explore S3 bucket structure to find golden_ranking data.
"""

import boto3
from botocore.config import Config as BotoConfig


def get_s3_client():
    """Get S3 client with timeout config."""
    config = BotoConfig(
        connect_timeout=10,
        read_timeout=30,
        retries={'max_attempts': 3}
    )
    return boto3.client('s3', config=config)


def explore_s3_path(bucket_name: str, prefix: str = '', max_depth: int = 3):
    """
    Recursively explore S3 path structure.

    Args:
        bucket_name: S3 bucket name
        prefix: Starting prefix
        max_depth: Maximum depth to explore
    """
    s3 = get_s3_client()

    print(f"\nExploring: s3://{bucket_name}/{prefix}")
    print("=" * 80)

    try:
        # List "directories" (common prefixes)
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter='/'
        )

        directories = []
        files = []

        for page in pages:
            # Get subdirectories
            for common_prefix in page.get('CommonPrefixes', []):
                directories.append(common_prefix['Prefix'])

            # Get files
            for obj in page.get('Contents', []):
                key = obj['Key']
                size = obj['Size']
                if key != prefix:  # Skip the prefix itself
                    files.append((key, size))

        # Print directories
        if directories:
            print(f"\nDirectories ({len(directories)}):")
            for dir_path in directories[:20]:  # Limit to first 20
                print(f"  📁 {dir_path}")
            if len(directories) > 20:
                print(f"  ... and {len(directories) - 20} more")

        # Print files
        if files:
            print(f"\nFiles ({len(files)}):")
            for file_path, size in files[:20]:  # Limit to first 20
                size_mb = size / (1024 * 1024)
                print(f"  📄 {file_path} ({size_mb:.2f} MB)")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more")

        if not directories and not files:
            print("  (empty)")

        return directories, files

    except Exception as e:
        print(f"Error exploring {prefix}: {e}")
        return [], []


def main():
    bucket_name = 'surface-quality-dataset'

    print("=" * 80)
    print("Exploring S3 Bucket: surface-quality-dataset")
    print("=" * 80)

    # Explore root
    print("\n1. Exploring root level:")
    root_dirs, root_files = explore_s3_path(bucket_name, '')

    # Explore quality-comparison-toolkit
    print("\n2. Exploring quality-comparison-toolkit:")
    qct_dirs, qct_files = explore_s3_path(bucket_name, 'quality-comparison-toolkit/')

    # Explore quality-comparison-toolkit/data
    print("\n3. Exploring quality-comparison-toolkit/data:")
    data_dirs, data_files = explore_s3_path(bucket_name, 'quality-comparison-toolkit/data/')

    # Explore golden_ranking_patches
    print("\n4. Exploring quality-comparison-toolkit/data/golden_ranking_patches:")
    gr_dirs, gr_files = explore_s3_path(bucket_name, 'quality-comparison-toolkit/data/golden_ranking_patches/')

    # If there are subdirectories in golden_ranking_patches, explore them
    if gr_dirs:
        print("\n5. Exploring first subdirectory in golden_ranking_patches:")
        explore_s3_path(bucket_name, gr_dirs[0])


if __name__ == '__main__':
    main()
