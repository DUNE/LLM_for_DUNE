import os
import sys

# Adjust this if needed
VENV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), 'env')
print(VENV_PATH)
PATCH_CODE = '''__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
'''

def patch_chromadb(venv_path):
    chromadb_path = None

    # Walk through the venv's site-packages to find chromadb
    for root, dirs, files in os.walk(venv_path):
        print(root, dirs)
        for d in dirs:
            if d == 'chromadb':
                chromadb_path = os.path.join(root, d)
                break
        if chromadb_path:
            break

    if not chromadb_path:
        print("❌ chromadb package not found in venv.")
        return

    init_file = os.path.join(chromadb_path, '__init__.py')
    if not os.path.isfile(init_file):
        # Just in case it's not in __init__, pick a common entrypoint file
        for filename in os.listdir(chromadb_path):
            if filename.endswith('.py'):
                init_file = os.path.join(chromadb_path, filename)
                break

    if not os.path.isfile(init_file):
        print("❌ No Python files found in chromadb package to patch.")
        return

    with open(init_file, 'r') as f:
        content = f.read()

    if PATCH_CODE in content:
        print("✅ chromadb already patched.")
        return

    # Inject at top
    with open(init_file, 'w') as f:
        f.write(PATCH_CODE + '\n' + content)

    print(f"✅ Patched chromadb in: {init_file}")

if __name__ == '__main__':
    patch_chromadb(VENV_PATH)

