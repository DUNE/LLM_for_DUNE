import importlib
import os
import sys

PATCH_CODE = """__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
"""


def patch_chromadb():
    try:
        import chromadb
    except ImportError:
        print("❌ chromadb is not installed in the active Python environment.")
        print("Activate your environment and run:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    chromadb_path = os.path.dirname(chromadb.__file__)
    init_file = os.path.join(chromadb_path, "__init__.py")

    if not os.path.isfile(init_file):
        print(f"❌ Could not find chromadb __init__.py at {init_file}")
        sys.exit(1)

    with open(init_file, "r") as f:
        content = f.read()

    if PATCH_CODE in content:
        print("✅ chromadb already patched.")
        return

    with open(init_file, "w") as f:
        f.write(PATCH_CODE + "\n" + content)

    print(f"✅ Patched chromadb at: {init_file}")


if __name__ == "__main__":
    patch_chromadb()

