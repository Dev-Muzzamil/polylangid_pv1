#!/usr/bin/env python3

"""
Enhanced installation and evaluation script with network retry logic.
This handles network connectivity issues for transformer model downloads.
"""

import os
import sys
import time
import urllib.request
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_network_connectivity(max_retries=5, delay=5):
    """
    Test network connectivity with retry logic.
    """
    urls = [
        'https://huggingface.co',
        'https://hf.co',
        'https://files.pythonhosted.org'  # Alternative check
    ]
    
    for attempt in range(max_retries):
        logger.info(f"Network connectivity check - Attempt {attempt + 1}/{max_retries}")
        
        for url in urls:
            try:
                logger.info(f"Testing connection to {url}...")
                response = urllib.request.urlopen(url, timeout=30)
                logger.info(f"✓ Successfully connected to {url} (status: {response.status})")
                return True
            except Exception as e:
                logger.warning(f"✗ Failed to connect to {url}: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Waiting {delay} seconds before next attempt...")
            time.sleep(delay)
    
    logger.error("⚠️ Network connectivity failed after all attempts")
    return False

def download_fasttext_model():
    """
    Ensure FastText model is available.
    """
    fasttext_path = Path("lid.176.ftz")
    if fasttext_path.exists():
        logger.info(f"✓ FastText model found: {fasttext_path}")
        return True
    
    logger.warning(f"✗ FastText model not found: {fasttext_path}")
    logger.info("Please ensure lid.176.ftz is available in the current directory")
    return False

def test_core_imports():
    """
    Test that all core dependencies can be imported.
    """
    logger.info("Testing core dependency imports...")
    
    imports_to_test = [
        ('numpy', 'import numpy as np'),
        ('torch', 'import torch'),
        ('transformers', 'from transformers import pipeline'),
        ('sklearn', 'from sklearn.metrics import precision_recall_fscore_support'),
        ('fasttext', 'import fasttext'),
        ('jieba', 'import jieba'),
        ('janome', 'from janome.tokenizer import Tokenizer'),
        ('pythainlp', 'from pythainlp.tokenize import word_tokenize'),
        ('pyvi', 'from pyvi import ViTokenizer'),
        ('sastrawi', 'from Sastrawi.Stemmer.StemmerFactory import StemmerFactory'),
    ]
    
    failed_imports = []
    
    for name, import_statement in imports_to_test:
        try:
            exec(import_statement)
            logger.info(f"✓ {name} imported successfully")
        except ImportError as e:
            logger.error(f"✗ Failed to import {name}: {e}")
            failed_imports.append(name)
        except Exception as e:
            logger.warning(f"⚠️ Import {name} had issues: {e}")
    
    if failed_imports:
        logger.error(f"Failed to import: {failed_imports}")
        return False
    
    logger.info("✓ All core dependencies imported successfully")
    return True

def test_transformer_loading(max_retries=3):
    """
    Attempt to load the transformer model with retry logic.
    """
    logger.info("Attempting to load XLM-R transformer model...")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading transformer - Attempt {attempt + 1}/{max_retries}")
            
            # Import here to avoid early failure
            from transformers import pipeline
            import torch
            
            # Try to load the model
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Using device: {'CUDA' if device == 0 else 'CPU'}")
            
            model = pipeline(
                "text-classification", 
                model="papluca/xlm-roberta-base-language-detection",
                device=device,
                return_all_scores=True
            )
            
            # Test the model with a simple text
            test_result = model("Hello world")
            logger.info(f"✓ Transformer model loaded and tested successfully")
            logger.info(f"Test result sample: {test_result[0][:2] if test_result else 'No result'}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Attempt {attempt + 1} failed to load transformer: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting 10 seconds before retry...")
                time.sleep(10)
    
    logger.warning("⚠️ Transformer model could not be loaded - proceeding with FastText only")
    return False

def run_evaluation():
    """
    Run the evaluation script.
    """
    logger.info("Starting evaluation...")
    
    try:
        # Import and run the evaluation
        import evalm
        logger.info("✓ Successfully imported evalm module")
        
        logger.info("Running evaluation main function...")
        evalm.main()
        logger.info("✓ Evaluation completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main installation and evaluation workflow.
    """
    logger.info("="*80)
    logger.info("POLYLANGID ENHANCED INSTALLATION AND EVALUATION")
    logger.info("="*80)
    
    # Step 1: Check network connectivity
    logger.info("\n1. Checking network connectivity...")
    network_ok = check_network_connectivity()
    
    # Step 2: Test core imports
    logger.info("\n2. Testing core dependency imports...")
    imports_ok = test_core_imports()
    
    if not imports_ok:
        logger.error("Core dependencies are missing. Installation may have failed.")
        return False
    
    # Step 3: Check FastText model
    logger.info("\n3. Checking FastText model availability...")
    fasttext_ok = download_fasttext_model()
    
    # Step 4: Attempt transformer loading (optional if network fails)
    logger.info("\n4. Attempting transformer model loading...")
    if network_ok:
        transformer_ok = test_transformer_loading()
    else:
        logger.warning("Skipping transformer loading due to network issues")
        transformer_ok = False
    
    # Step 5: Run evaluation
    logger.info("\n5. Running evaluation...")
    eval_ok = run_evaluation()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("INSTALLATION AND EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Network connectivity: {'✓' if network_ok else '✗'}")
    logger.info(f"Core dependencies: {'✓' if imports_ok else '✗'}")
    logger.info(f"FastText model: {'✓' if fasttext_ok else '✗'}")
    logger.info(f"Transformer model: {'✓' if transformer_ok else '✗'}")
    logger.info(f"Evaluation: {'✓' if eval_ok else '✗'}")
    
    if eval_ok:
        logger.info("\n✓ Setup and evaluation completed successfully!")
        if not transformer_ok:
            logger.warning("Note: Running with FastText only due to network/transformer issues")
        return True
    else:
        logger.error("\n✗ Setup or evaluation failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)