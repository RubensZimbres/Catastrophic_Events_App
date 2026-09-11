import sys
import os
from unittest.mock import MagicMock

os.environ.setdefault("OPENWEATHER_API_KEY", "test_openweather_key")
os.environ.setdefault("GOOGLE_MAPS_API_KEY", "test_maps_key")
os.environ.setdefault("GOOGLE_AI_API_KEY", "test_ai_key")
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_PROJECT_NAME", "test-project")

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app'))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

sys.modules.setdefault('streamlit', MagicMock())
sys.modules.setdefault('streamlit.components', MagicMock())
sys.modules.setdefault('streamlit.components.v1', MagicMock())
sys.modules.setdefault('extra_streamlit_components', MagicMock())
sys.modules.setdefault('markdown', MagicMock())
sys.modules.setdefault('IPython', MagicMock())
sys.modules.setdefault('IPython.display', MagicMock())
sys.modules.setdefault('google.genai', MagicMock())
sys.modules.setdefault('google.genai.types', MagicMock())
sys.modules.setdefault('langchain_google_genai', MagicMock())

from langchain_core.tools import BaseTool
from langchain_core.agents import AgentAction, AgentFinish

_lc_tools = MagicMock()
_lc_tools.BaseTool = BaseTool
sys.modules['langchain.tools'] = _lc_tools

_lc_schema = MagicMock()
_lc_schema.AgentAction = AgentAction
_lc_schema.AgentFinish = AgentFinish
sys.modules['langchain.schema'] = _lc_schema

for _mod in [
    'langchain.agents',
    'langchain.agents.agent',
    'langchain.memory',
    'langchain.prompts',
    'langchain.chains',
]:
    sys.modules.setdefault(_mod, MagicMock())

try:
    import google.cloud as _gc
    _sm_mock = MagicMock()
    _sm_mock.SecretManagerServiceClient.return_value.access_secret_version.side_effect = Exception(
        "No GCP credentials in test environment"
    )
    _gc.secretmanager = _sm_mock
    sys.modules['google.cloud.secretmanager'] = _sm_mock
except Exception:
    _sm_mock = MagicMock()
    _sm_mock.SecretManagerServiceClient.return_value.access_secret_version.side_effect = Exception(
        "Not installed"
    )
    sys.modules['google.cloud.secretmanager'] = _sm_mock
