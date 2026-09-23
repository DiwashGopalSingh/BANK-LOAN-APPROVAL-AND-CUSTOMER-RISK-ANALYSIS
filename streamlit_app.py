"""Streamlit Cloud Default Entrypoint.

Redirects directly to app.py so Streamlit Community Cloud's default
configuration ('streamlit_app.py') works out-of-the-box.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import main

if __name__ == "__main__":
    main()
