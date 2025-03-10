import nltk # type: ignore
import os

# Define the NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download required NLTK data if not already present
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'wordnet')):
    nltk.download('wordnet', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'taggers', 'punkt_tab')):
    nltk.download('punkt_tab', download_dir=nltk_data_path)