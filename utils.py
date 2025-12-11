# utils.py

"""
Utility functions for DeepJoin
"""

def build_column_text(col_name, values):
    """
    Create DeepJoin's text representation:
    'column: borough. sample values: Manhattan, Queens, Bronx'
    """
    sample_text = ", ".join([str(v) for v in values if v])
    return f"Column: {col_name}. Sample values: {sample_text}"
