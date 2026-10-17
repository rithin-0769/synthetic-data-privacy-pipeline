import streamlit.web.bootstrap
from src.ui.app import *

# Streamlit entry point for root directory discovery
if __name__ == "__main__":
    import sys
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())
