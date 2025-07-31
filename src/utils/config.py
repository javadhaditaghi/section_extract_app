# # # src/utils/config.py
# # from pathlib import Path
# #
# # # Base directories - going up 2 levels from config.py to reach project root
# # PROJECT_ROOT = Path(__file__).resolve().parents[2]
# # DIR_DATA = PROJECT_ROOT / "data"
# # DIR_RAW = DIR_DATA / "raw"
# # DIR_PROCESSED = DIR_DATA / "processed"
# # DIR_COMPLETE_TXT = DIR_PROCESSED / "complete_txt"
# # DIR_SECTIONED = DIR_DATA / "sectioned"
# # DIR_ABSTRACTS = DIR_SECTIONED / "abstracts"
# # DIR_INTRODUCTIONS = DIR_SECTIONED / "introductions"
# # DIR_DATASHEETS = DIR_SECTIONED / "datasheets"
# #
# #
# # # Create all necessary directories
# # def ensure_directories():
# #     """Create all necessary directories if they don't exist."""
# #     directories = [
# #         DIR_RAW,
# #         DIR_COMPLETE_TXT,
# #         DIR_ABSTRACTS,
# #         DIR_INTRODUCTIONS,
# #         DIR_DATASHEETS
# #     ]
# #
# #     for directory in directories:
# #         directory.mkdir(parents=True, exist_ok=True)
# #         print(f"✓ Directory ensured: {directory}")
# #
# #
# # # Ensure directories exist when config is imported
# # ensure_directories()
#
# #
# # # src/utils/config.py
# # from pathlib import Path
# #
# # # Base directories - going up 2 levels from config.py to reach project root
# # PROJECT_ROOT = Path(__file__).resolve().parents[2]
# # DIR_DATA = PROJECT_ROOT / "data"
# # DIR_SRC = PROJECT_ROOT / "src"
# #
# # # Data directories
# # DIR_RAW = DIR_DATA / "raw"
# # DIR_PROCESSED = DIR_DATA / "processed"
# # DIR_COMPLETE_TXT = DIR_PROCESSED / "complete_txt"
# # DIR_SECTIONED = DIR_DATA / "sectioned"
# # DIR_ABSTRACTS = DIR_SECTIONED / "abstracts"
# # DIR_INTRODUCTIONS = DIR_SECTIONED / "introductions"
# # DIR_DATASHEETS = DIR_SECTIONED / "datasheets"
# #
# # # Annotation directories
# # DIR_ANNOTATIONS = DIR_DATA / "annotations"
# # DIR_ANNOTATIONS_LLM = DIR_ANNOTATIONS / "LLM"
# # DIR_ANNOTATIONS_GPT = DIR_ANNOTATIONS_LLM / "GPT"
# # DIR_ANNOTATIONS_Gemini = DIR_ANNOTATIONS_LLM / "Gemini"
# # DIR_ANNOTATIONS_CLAUDE = DIR_ANNOTATIONS_LLM / "Claude"  # For future use
# # DIR_ANNOTATIONS_HUMAN = DIR_ANNOTATIONS / "human"  # For future use
# #
# #
# # # Create all necessary directories
# # def ensure_directories():
# #     """Create all necessary directories if they don't exist."""
# #     directories = [
# #         DIR_RAW,
# #         DIR_COMPLETE_TXT,
# #         DIR_ABSTRACTS,
# #         DIR_INTRODUCTIONS,
# #         DIR_DATASHEETS,
# #         DIR_ANNOTATIONS_GPT,
# #         DIR_ANNOTATIONS_Gemini,
# #         DIR_ANNOTATIONS_CLAUDE,
# #         DIR_ANNOTATIONS_HUMAN
# #     ]
# #
# #     for directory in directories:
# #         directory.mkdir(parents=True, exist_ok=True)
# #         print(f"✓ Directory ensured: {directory}")
# #
# #
# # # Ensure directories exist when config is imported
# # ensure_directories()
# #
# #
# # # Configuration settings
# # class AnnotationConfig:
# #     """Configuration settings for annotation tasks."""
# #
# #     # OpenAI API settings
# #     OPENAI_MODEL = "gpt-4"
# #     OPENAI_TEMPERATURE = 0.2
# #     OPENAI_MAX_RETRIES = 3
# #     OPENAI_RETRY_DELAY = 1  # seconds
# #
# #     # Processing settings
# #     DEFAULT_BATCH_SIZE = 10  # Process in batches to handle large datasets
# #     SAMPLE_SIZE = None  # None = process all, or set to number (e.g., 100)
# #
# #     # Output settings
# #     SAVE_INTERMEDIATE = True  # Save progress periodically
# #     OUTPUT_FORMAT = "csv"  # "csv"
#
# # src/utils/config.py
# from pathlib import Path
#
# # Base directories - going up 2 levels from config.py to reach project root
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# DIR_DATA = PROJECT_ROOT / "data"
# DIR_SRC = PROJECT_ROOT / "src"
#
# # Data directories
# DIR_RAW = DIR_DATA / "raw"
# DIR_PROCESSED = DIR_DATA / "processed"
# DIR_COMPLETE_TXT = DIR_PROCESSED / "complete_txt"
# DIR_SECTIONED = DIR_DATA / "sectioned"
# DIR_ABSTRACTS = DIR_SECTIONED / "abstracts"
# DIR_INTRODUCTIONS = DIR_SECTIONED / "introductions"
# DIR_DATASHEETS = DIR_SECTIONED / "datasheets"
#
# # Annotation directories
# DIR_ANNOTATIONS = DIR_DATA / "annotations"
# DIR_ANNOTATIONS_LLM = DIR_ANNOTATIONS / "LLM"
# DIR_ANNOTATIONS_GPT = DIR_ANNOTATIONS_LLM / "GPT"
# DIR_ANNOTATIONS_DEEPSEEK = DIR_ANNOTATIONS_LLM / "DeepSeek"
# DIR_ANNOTATIONS_Gemini = DIR_ANNOTATIONS_LLM / "Gemini"
# DIR_ANNOTATIONS_CLAUDE = DIR_ANNOTATIONS_LLM / "Claude"  # For future use
# DIR_ANNOTATIONS_HUMAN = DIR_ANNOTATIONS / "human"  # For future use
#
#
# # Create all necessary directories
# def ensure_directories():
#     """Create all necessary directories if they don't exist."""
#     directories = [
#         DIR_RAW,
#         DIR_COMPLETE_TXT,
#         DIR_ABSTRACTS,
#         DIR_INTRODUCTIONS,
#         DIR_DATASHEETS,
#         DIR_ANNOTATIONS_GPT,
#         DIR_ANNOTATIONS_DEEPSEEK,
#         DIR_ANNOTATIONS_Gemini,
#         DIR_ANNOTATIONS_CLAUDE,
#         DIR_ANNOTATIONS_HUMAN
#     ]
#
#     for directory in directories:
#         directory.mkdir(parents=True, exist_ok=True)
#         print(f"✓ Directory ensured: {directory}")
#
#
# # Ensure directories exist when config is imported
# ensure_directories()
#
#
# # Configuration settings
# class AnnotationConfig:
#     """Configuration settings for annotation tasks."""
#
#     # OpenAI API settings
#     OPENAI_MODEL = "gpt-4"
#     OPENAI_TEMPERATURE = 0.2
#     OPENAI_MAX_RETRIES = 3
#     OPENAI_RETRY_DELAY = 1  # seconds
#
#     # DeepSeek API settings
#     DEEPSEEK_MODEL = "deepseek-chat"
#     DEEPSEEK_TEMPERATURE = 0.2
#     DEEPSEEK_MAX_RETRIES = 3
#     DEEPSEEK_RETRY_DELAY = 1  # seconds
#
#     # Gemini API settings (for future use)
#     GEMINI_MODEL = "gemini-pro"
#     GEMINI_TEMPERATURE = 0.2
#     GEMINI_MAX_RETRIES = 3
#     GEMINI_RETRY_DELAY = 1  # seconds
#
#     # Claude API settings (for future use)
#     CLAUDE_MODEL = "claude-3-sonnet-20240229"
#     CLAUDE_TEMPERATURE = 0.2
#     CLAUDE_MAX_RETRIES = 3
#     CLAUDE_RETRY_DELAY = 1  # seconds
#
#     # Processing settings
#     DEFAULT_BATCH_SIZE = 10  # Process in batches to handle large datasets
#     SAMPLE_SIZE = None  # None = process all, or set to number (e.g., 100)
#
#     # Output settings
#     SAVE_INTERMEDIATE = True  # Save progress periodically
#     OUTPUT_FORMAT = "csv"  # "csv"
#
#     # Rate limiting settings (to avoid hitting API limits)
#     REQUEST_DELAY = 0.1  # Delay between requests in seconds
#     CONCURRENT_REQUESTS = 1  # Number of concurrent requests (keep at 1 for safety)
#
#     # Timeout settings
#     REQUEST_TIMEOUT = 60  # Timeout for API requests in seconds


# src/utils/config.py
from pathlib import Path

# Base directories - going up 2 levels from config.py to reach project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIR_DATA = PROJECT_ROOT / "data"
DIR_SRC = PROJECT_ROOT / "src"

# Data directories
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"
DIR_COMPLETE_TXT = DIR_PROCESSED / "complete_txt"
DIR_SECTIONED = DIR_DATA / "sectioned"
DIR_ABSTRACTS = DIR_SECTIONED / "abstracts"
DIR_INTRODUCTIONS = DIR_SECTIONED / "introductions"
DIR_DATASHEETS = DIR_SECTIONED / "datasheets"

# Annotation directories
DIR_ANNOTATIONS = DIR_DATA / "annotations"
DIR_ANNOTATIONS_LLM = DIR_ANNOTATIONS / "LLM"
DIR_ANNOTATIONS_GPT = DIR_ANNOTATIONS_LLM / "GPT"
DIR_ANNOTATIONS_DEEPSEEK = DIR_ANNOTATIONS_LLM / "DeepSeek"
DIR_ANNOTATIONS_Gemini = DIR_ANNOTATIONS_LLM / "Gemini"
DIR_ANNOTATIONS_CLAUDE = DIR_ANNOTATIONS_LLM / "Claude"  # For future use
DIR_ANNOTATIONS_HUMAN = DIR_ANNOTATIONS / "human"  # For future use


# Create all necessary directories
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DIR_RAW,
        DIR_COMPLETE_TXT,
        DIR_ABSTRACTS,
        DIR_INTRODUCTIONS,
        DIR_DATASHEETS,
        DIR_ANNOTATIONS_GPT,
        DIR_ANNOTATIONS_DEEPSEEK,
        DIR_ANNOTATIONS_Gemini,
        DIR_ANNOTATIONS_CLAUDE,
        DIR_ANNOTATIONS_HUMAN
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ensured: {directory}")


# Ensure directories exist when config is imported
ensure_directories()


# Configuration settings
class AnnotationConfig:
    """Configuration settings for annotation tasks."""

    # OpenAI API settings
    OPENAI_MODEL = "gpt-4"
    OPENAI_TEMPERATURE = 0.2
    OPENAI_MAX_RETRIES = 3
    OPENAI_RETRY_DELAY = 1  # seconds

    # DeepSeek API settings
    DEEPSEEK_MODEL = "deepseek-chat"
    DEEPSEEK_TEMPERATURE = 0.2
    DEEPSEEK_MAX_RETRIES = 3
    DEEPSEEK_RETRY_DELAY = 1  # seconds

    # Gemini API settings (for future use)
    GEMINI_MODEL = "gemini-pro"
    GEMINI_TEMPERATURE = 0.2
    GEMINI_MAX_RETRIES = 3
    GEMINI_RETRY_DELAY = 1  # seconds

    # Claude API settings
    CLAUDE_MODEL = "claude-opus-4-20250514"
    CLAUDE_TEMPERATURE = 0.2
    CLAUDE_MAX_RETRIES = 3
    CLAUDE_RETRY_DELAY = 1  # seconds
    CLAUDE_MAX_TOKENS = 4096  # Maximum tokens for Claude response

    # Processing settings
    DEFAULT_BATCH_SIZE = 10  # Process in batches to handle large datasets
    SAMPLE_SIZE = None  # None = process all, or set to number (e.g., 100)

    # Output settings
    SAVE_INTERMEDIATE = True  # Save progress periodically
    OUTPUT_FORMAT = "csv"  # "csv"

    # Rate limiting settings (to avoid hitting API limits)
    REQUEST_DELAY = 0.1  # Delay between requests in seconds
    CONCURRENT_REQUESTS = 1  # Number of concurrent requests (keep at 1 for safety)

    # Timeout settings
    REQUEST_TIMEOUT = 60  # Timeout for API requests in seconds