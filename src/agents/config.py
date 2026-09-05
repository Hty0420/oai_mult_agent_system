"""
Configuration file for the Multi-Agent Osteoarthritis Medical Assistant

This file contains all the configuration parameters for the project.
"""

from llm_config import get_llm_config

class RouterAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.1  # Deterministic for routing
        )

class SegmentationAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.1  # Deterministic for segmentation tasks
        )

class FeatureExtractionAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.1  # Deterministic for feature extraction
        )

class ClassificationAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.1  # Deterministic for classification
        )

class ConversationAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.7  # More creative for general conversation
        )

class OrchestratorAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.1  # Deterministic for orchestration
        )

class BiomarkerReporterAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.65
        )

class FeatureCombinationAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.1  # Deterministic for feature combination
        )

class WebSearchAgentConfig:
    def __init__(self):
        self.llm = get_llm_config().create_chat_model(
            temperature=0.1  # Deterministic for web search
        )

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.max_image_upload_size = 50  # max upload size in MB

class Config:
    def __init__(self):
        self.router_agent = RouterAgentConfig()
        self.segmentation_agent = SegmentationAgentConfig()
        self.feature_extraction_agent = FeatureExtractionAgentConfig()
        self.feature_combination_agent = FeatureCombinationAgentConfig()
        self.classification_agent = ClassificationAgentConfig()
        self.conversation_agent = ConversationAgentConfig()
        self.orchestrator_agent = OrchestratorAgentConfig()
        self.biomarker_reporter_agent = BiomarkerReporterAgentConfig()
        self.web_search_agent = WebSearchAgentConfig()
        self.api = APIConfig()
        self.max_conversation_history = 20

# Example usage
# config = Config()
