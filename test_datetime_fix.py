#!/usr/bin/env python3
"""Test that the datetime.UTC bug is fixed."""

import sys
sys.path.insert(0, '/workspace/openpi/src')

def test_datetime_fix():
    """Test that download.py can be imported without datetime.UTC error."""
    try:
        from openpi.shared.download import _get_mtime
        # Test that _get_mtime works correctly
        mtime = _get_mtime(2025, 2, 17)
        assert mtime > 0, f"Expected positive mtime, got {mtime}"
        print(f"✓ datetime.UTC fix verified: _get_mtime(2025, 2, 17) = {mtime}")
        return 100.0
    except AttributeError as e:
        if "datetime.UTC" in str(e):
            print(f"✗ datetime.UTC bug still present: {e}")
            return 0.0
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 0.0

def test_libero_policy_import():
    """Test that libero_policy can be imported."""
    try:
        from openpi.policies import libero_policy
        print("✓ libero_policy import OK")
        return 100.0
    except Exception as e:
        print(f"✗ libero_policy import failed: {e}")
        return 0.0

def main():
    print("Running datetime.UTC fix tests...")
    print()
    
    score1 = test_datetime_fix()
    score2 = test_libero_policy_import()
    
    # Average score
    avg_score = (score1 + score2) / 2
    print()
    print(f"Final score: {avg_score}")
    return avg_score

if __name__ == "__main__":
    score = main()
    sys.exit(0 if score == 100 else 1)
