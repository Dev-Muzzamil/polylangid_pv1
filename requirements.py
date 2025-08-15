# requirements.py
# This file lists the Python dependencies used in the project, extracted from the codebase.
# It is not a standard requirements file, but a Python list for programmatic use.

requirements = [
    # Core ML stack
    'torch==2.7.1+cu118',
    'transformers==4.54.0',
    'fasttext-predict==0.9.2.4',
    'numpy==2.1.3',
    'scikit-learn==1.6.1',

    # Tokenizers / language-specific helpers
    'jieba==0.42.1',
    'Janome==0.5.0',
    'pythainlp==5.1.2',
    'pyvi',
    'Sastrawi',
]
