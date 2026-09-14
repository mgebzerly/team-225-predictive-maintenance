#!/usr/bin/env python3
"""
Run all analytical queries against the SQLite database.
Verifies each query executes and prints row counts.
"""

import sqlite3
import re
import sys

DB_PATH = 'data/ai4i2020.db'

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    with open('sql/queries.sql', 'r') as f:
        content = f.read()
    
    # Split by numbered query headers
    pattern = r'(^--\s+--\d+\.\s+.*?$)'
    lines = content.split('\n')
    queries = []
    current_lines = []
    in_query = False
    
    for line in lines:
        m = re.match(r'^--\s+(\d+)\.\s+(.+)$', line)
        if m:
            if current_lines and in_query:
                queries.append('\n'.join(current_lines))
            current_lines = [line]
            in_query = True
        elif in_query:
            current_lines.append(line)
    
    if current_lines and in_query:
        queries.append('\n'.join(current_lines))
    
    print(f"Found {len(queries)} queries\n")
    
    passed = 0
    failed = 0
    
    for i, q in enumerate(queries, 1):
        if not q.strip():
            continue
        try:
            cursor.execute(q)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description] if cursor.description else []
            print(f"Query {i}: {len(rows)} rows, {len(cols)} cols")
            if rows:
                print(f"  Columns: {cols}")
                print(f"  First row: {rows[0]}")
            passed += 1
        except Exception as e:
            print(f"Query {i}: FAILED")
            print(f"  Error: {e}")
            print(f"  SQL (first 200 chars): {q[:200]}")
            failed += 1
            conn.rollback()
        print()
    
    conn.close()
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)