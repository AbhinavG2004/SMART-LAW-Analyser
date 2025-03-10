import nltk # type: ignore
import os

# Define the NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)

print("NLTK data downloaded successfully!")