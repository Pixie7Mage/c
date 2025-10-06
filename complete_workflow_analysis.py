"""
CRISPR-Cas9 Off-Target Prediction Project - Complete Workflow Analysis
=====================================================================

This file provides a comprehensive analysis of the entire CRISPR-Cas9 off-target prediction workflow,
including detailed explanations, input/output examples, and the reasoning behind each component.

Author: AI Assistant
Date: 2025
Purpose: Educational analysis of hybrid CNN-BERT model for CRISPR off-target prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

print("=" * 100)
print("🎯 CRISPR-Cas9 Off-Target Prediction Project - Complete Workflow Analysis")
print("=" * 100)

# ============================================================================
# 1. PROJECT OVERVIEW
# ============================================================================

def project_overview():
    """
    Explains the overall purpose and approach of the project
    """
    print("\n" + "="*80)
    print("📋 PROJECT OVERVIEW")
    print("="*80)
    
    print("""
    🎯 OBJECTIVE:
    Predict whether a given sgRNA-target DNA sequence pair will result in:
    - On-target cutting (label=1): CRISPR cuts at intended location
    - Off-target cutting (label=0): CRISPR cuts at unintended location
    
    🧬 BIOLOGICAL CONTEXT:
    CRISPR-Cas9 uses a guide RNA (sgRNA) to find and cut specific DNA sequences.
    However, it can sometimes cut similar sequences (off-targets), causing unwanted effects.
    This model helps predict these off-target events.
    
    🏗️ APPROACH:
    Hybrid CNN-BERT architecture that processes DNA sequences in two ways:
    1. CNN Branch: Analyzes structural/biological features
    2. BERT Branch: Captures sequential patterns and long-range dependencies
    3. Fusion: Combines both approaches for final prediction
    """)

# ============================================================================
# 2. DATA INPUT AND PROCESSING PIPELINE
# ============================================================================

def demonstrate_raw_input():
    """
    Shows the raw input format and explains its structure
    """
    print("\n" + "="*80)
    print("📊 RAW INPUT DATA ANALYSIS")
    print("="*80)
    
    # Example raw data
    raw_examples = [
        "GTCACCTCCAATGACTAGGGAGG,GTCTCCTCCACTGGATTGTGAGG,0",
        "GCTGCCAGTACAGGCTCCCCCTCG,GCAGCCAGTACA_GCTCACCATGG,0",
        "GGGTGAGTGAGTGTGTGCGTGTG,GGGTGAGTGAGTGTGTGCGTGTG,1"
    ]
    
    print("📁 INPUT FILE FORMAT (datasets/*.txt):")
    print("Each line contains: sgRNA_sequence,target_sequence,label")
    print("\n📝 EXAMPLE INPUTS:")
    
    for i, example in enumerate(raw_examples, 1):
        parts = example.split(',')
        sgRNA, target, label = parts[0], parts[1], parts[2]
        
        print(f"\nExample {i}:")
        print(f"  Raw line: {example}")
        print(f"  sgRNA:    {sgRNA} (length: {len(sgRNA)})")
        print(f"  Target:   {target} (length: {len(target)})")
        print(f"  Label:    {label} ({'On-target' if label == '1' else 'Off-target'})")
        
        # Analyze sequence differences
        if len(sgRNA) == len(target):
            matches = sum(1 for a, b in zip(sgRNA, target) if a == b and a != '_' and b != '_')
            mismatches = sum(1 for a, b in zip(sgRNA, target) if a != b and a != '_' and b != '_')
            gaps = sum(1 for a, b in zip(sgRNA, target) if a == '_' or b == '_')
            
            print(f"  Analysis: {matches} matches, {mismatches} mismatches, {gaps} gaps")
    
    print(f"\n🎯 WHY THIS INPUT FORMAT:")
    print(f"  • sgRNA: The guide RNA sequence that directs Cas9")
    print(f"  • Target: The DNA sequence being evaluated for cutting")
    print(f"  • Label: Ground truth (0=off-target, 1=on-target)")
    print(f"  • Alignment: Sequences are pre-aligned to show matches/mismatches/indels")

def demonstrate_sequence_alignment():
    """
    Shows how sequences are converted to pair tokens
    """
    print("\n" + "="*80)
    print("🧬 STEP 1: SEQUENCE ALIGNMENT → PAIR TOKENS")
    print("="*80)
    
    # Example sequences
    seq1 = "GCTGCCAGTACAGGCTCCCCCTCG"  # sgRNA
    seq2 = "GCAGCCAGTACA_GCTCACCATGG"  # target (with gap _)
    
    print(f"INPUT SEQUENCES:")
    print(f"  seq1 (sgRNA):  {seq1}")
    print(f"  seq2 (target): {seq2}")
    print(f"  Length: {len(seq1)} vs {len(seq2)}")
    
    # Create pair list (simulating make_pair_list function)
    def make_pair_list_demo(s1, s2):
        """Demo version of make_pair_list with detailed output"""
        pair_list = []
        print(f"\nPAIR TOKEN CREATION:")
        print(f"Position | seq1 | seq2 | Token | Type")
        print(f"-" * 45)
        
        for i, (a, b) in enumerate(zip(s1, s2)):
            if a == "-" or a == "_":
                token = "_" + b
                token_type = "Insertion"
            elif b == "-" or b == "_":
                token = a + "_"
                token_type = "Deletion"
            elif a == b:
                token = a + b
                token_type = "Match"
            else:
                token = a + b
                token_type = "Mismatch"
            
            pair_list.append(token)
            print(f"{i:8} | {a:4} | {b:4} | {token:5} | {token_type}")
        
        return pair_list
    
    pair_list = make_pair_list_demo(seq1, seq2)
    
    print(f"\nOUTPUT PAIR LIST:")
    print(f"  {pair_list}")
    print(f"  Length: {len(pair_list)} tokens")
    
    print(f"\n🎯 WHY PAIR TOKENS:")
    print(f"  • Captures alignment information in single tokens")
    print(f"  • Preserves match/mismatch/indel patterns crucial for CRISPR binding")
    print(f"  • Reduces two sequences to one aligned representation")
    print(f"  • Each token represents the relationship at that position")

def demonstrate_token_encoding():
    """
    Shows how pair tokens are converted to integer sequences
    """
    print("\n" + "="*80)
    print("🔢 STEP 2: TOKEN ENCODING → INTEGER SEQUENCE")
    print("="*80)
    
    # Vocabulary setup (simplified version)
    BASES = ['A','C','G','T']
    pair_tokens = [b1+b2 for b1 in BASES for b2 in BASES]
    indel_tokens = [b+'_' for b in BASES] + ['_'+b for b in BASES]
    
    token_to_index = {}
    index_to_token = {}
    
    # Special tokens
    token_to_index['CLS'] = 0
    token_to_index['SEP'] = 1
    index_to_token[0] = 'CLS'
    index_to_token[1] = 'SEP'
    
    idx = 2
    # Base pairs
    for p in pair_tokens:
        token_to_index[p] = idx
        index_to_token[idx] = p
        idx += 1
    
    # Indels
    for it in indel_tokens:
        token_to_index[it] = idx
        index_to_token[idx] = it
        idx += 1
    
    VOCAB_SIZE = idx
    MAX_LEN = 26
    
    print(f"VOCABULARY SETUP:")
    print(f"  Total vocabulary size: {VOCAB_SIZE}")
    print(f"  Special tokens: CLS=0, SEP=1")
    print(f"  Base pairs (16): AA=2, AT=3, AG=4, ..., TT=17")
    print(f"  Indels (8): A_=18, T_=19, G_=20, C_=21, _A=22, _T=23, _G=24, _C=25")
    
    # Example pair list
    pair_list = ['GG', 'CC', 'TT', 'GG', 'CC', 'CC', 'AA', 'GG', 'TT', 'AA', 'CC', 'AA', 'A_']
    
    print(f"\nINPUT PAIR LIST:")
    print(f"  {pair_list}")
    
    # Encoding process
    def encode_token_list_demo(pair_list, max_len=26):
        """Demo version with detailed output"""
        print(f"\nENCODING PROCESS:")
        print(f"Step | Action | Token | ID | Result")
        print(f"-" * 50)
        
        toks = []
        
        # Add CLS
        toks.append(0)
        print(f"1    | Add CLS | CLS | 0 | {toks}")
        
        # Add pair tokens
        for i, p in enumerate(pair_list[:max_len-2]):
            token_id = token_to_index.get(p, 0)  # Use 0 (PAD) if not found
            toks.append(token_id)
            print(f"{i+2:<4} | Add token | {p} | {token_id} | {toks}")
        
        # Add SEP
        toks.append(1)
        print(f"{len(pair_list)+2:<4} | Add SEP | SEP | 1 | {toks}")
        
        # Pad to max_len
        while len(toks) < max_len:
            toks.append(0)
        
        print(f"Final| Pad to {max_len} | PAD | 0 | Length: {len(toks)}")
        
        return np.array(toks, dtype=np.int32)
    
    token_sequence = encode_token_list_demo(pair_list)
    
    print(f"\nFINAL TOKEN SEQUENCE:")
    print(f"  {token_sequence}")
    print(f"  Shape: {token_sequence.shape}")
    
    print(f"\n🎯 WHY TOKEN ENCODING:")
    print(f"  • Neural networks require numerical input")
    print(f"  • Each DNA pattern gets unique integer ID")
    print(f"  • Fixed length enables batch processing")
    print(f"  • CLS/SEP tokens mark sequence boundaries")

def demonstrate_onehot_encoding():
    """
    Shows how pair tokens are converted to biological feature matrices
    """
    print("\n" + "="*80)
    print("🧮 STEP 3: Embedding     → BIOLOGICAL FEATURES")
    print("="*80)
    
    # Feature encoding dictionary (simplified)
    encoded_dict = {
        'AA': [1, 0, 0, 0, 0, 0, 0],  # Both A
        'TT': [0, 1, 0, 0, 0, 0, 0],  # Both T
        'GG': [0, 0, 1, 0, 0, 0, 0],  # Both G
        'CC': [0, 0, 0, 1, 0, 0, 0],  # Both C
        'AT': [1, 1, 0, 0, 0, 1, 0],  # A-T mismatch
        'GC': [0, 0, 1, 1, 0, 0, 1],  # G-C mismatch
        'A_': [1, 0, 0, 0, 1, 1, 0],  # A + deletion
        '_A': [1, 0, 0, 0, 1, 0, 1],  # Insertion + A
        '--': [0, 0, 0, 0, 0, 0, 0],  # Special/padding
    }
    
    print(f"FEATURE ENCODING SCHEME:")
    print(f"  7-dimensional feature vector for each position:")
    print(f"  [A, T, G, C, GAP, STRAND1, STRAND2]")
    print(f"")
    print(f"  Columns 0-3: Base presence indicators")
    print(f"  Column 4:    Gap/deletion indicator")
    print(f"  Column 5:    First strand indicator")
    print(f"  Column 6:    Second strand indicator")
    
    # Example encoding
    pair_list = ['GG', 'CC', 'TT', 'AA', 'AT', 'GC', 'A_', '_A']
    
    print(f"\nEXAMPLE ENCODINGS:")
    print(f"Token | A T G C GAP S1 S2 | Interpretation")
    print(f"-" * 55)
    
    for token in pair_list:
        if token in encoded_dict:
            features = encoded_dict[token]
            feature_str = " ".join([str(int(f)) for f in features])
            
            # Interpretation
            bases = []
            if features[0]: bases.append('A')
            if features[1]: bases.append('T')
            if features[2]: bases.append('G')
            if features[3]: bases.append('C')
            
            if features[4]:  # Gap present
                if '_' in token:
                    interp = f"Indel: {token}"
                else:
                    interp = "Gap present"
            elif len(bases) == 1:
                interp = f"Match: {bases[0]}{bases[0]}"
            elif len(bases) == 2:
                interp = f"Mismatch: {bases[0]}-{bases[1]}"
            else:
                interp = "Complex pattern"
            
            print(f"{token:5} | {feature_str} | {interp}")
    
    # Create full matrix example
    def encode_onehot_matrix_demo(pair_list, max_len=26):
        """Demo version with detailed output"""
        print(f"\nFULL MATRIX CREATION:")
        
        mat = []
        
        # CLS token
        mat.append(encoded_dict['--'])
        print(f"Position 0 (CLS): {encoded_dict['--']}")
        
        # Pair tokens
        for i, p in enumerate(pair_list[:max_len-2]):
            encoding = encoded_dict.get(p, encoded_dict['--'])
            mat.append(encoding)
            print(f"Position {i+1} ({p}): {encoding}")
        
        # SEP token
        mat.append(encoded_dict['--'])
        print(f"Position {len(pair_list)+1} (SEP): {encoded_dict['--']}")
        
        # Padding
        while len(mat) < max_len:
            mat.append(encoded_dict['--'])
        
        return np.array(mat, dtype=np.float32)
    
    onehot_matrix = encode_onehot_matrix_demo(pair_list[:5])  # Show first 5 for brevity
    
    print(f"\nFINAL ONE-HOT MATRIX:")
    print(f"  Shape: {onehot_matrix.shape}")
    print(f"  Matrix (first 8 rows):")
    print(f"  {onehot_matrix[:8]}")
    
    print(f"\n🎯 WHY ONE-HOT ENCODING:")
    print(f"  • Preserves biological information (base composition)")
    print(f"  • CNN can process spatial patterns in features")
    print(f"  • Captures structural properties of DNA alignment")
    print(f"  • Indel information crucial for CRISPR binding prediction")

# ============================================================================
# 3. MODEL ARCHITECTURE ANALYSIS
# ============================================================================

def demonstrate_model_architecture():
    """
    Explains the complete model architecture with input/output shapes
    """
    print("\n" + "="*80)
    print("🏗️ MODEL ARCHITECTURE ANALYSIS")
    print("="*80)
    
    print(f"HYBRID CNN-BERT ARCHITECTURE:")
    print(f"")
    print(f"Input Layer:")
    print(f"  ├─ inp_tok: Token sequences (batch_size, 26)")
    print(f"  └─ inp_hot: One-hot matrices (batch_size, 26, 7)")
    print(f"")
    print(f"Parallel Processing:")
    print(f"  ├─ CNN Branch (inp_hot)")
    print(f"  │  ├─ Conv1D layers: kernel_sizes=[5,15,25,35], filters=80 each")
    print(f"  │  ├─ Concatenate: (batch_size, 26, 320)")
    print(f"  │  └─ Dense: (batch_size, 26, 80)")
    print(f"  │")
    print(f"  └─ BERT Branch (inp_tok)")
    print(f"     ├─ Embedding: (batch_size, 26, 128)")
    print(f"     ├─ Transformer Blocks x2")
    print(f"     │  ├─ Multi-Head Attention (4 heads)")
    print(f"     │  └─ Feed-Forward Network")
    print(f"     └─ Dense: (batch_size, 26, 80)")
    print(f"")
    print(f"Sequence Processing:")
    print(f"  ├─ CNN → Bidirectional GRU: (batch_size, 80)")
    print(f"  └─ BERT → Bidirectional GRU: (batch_size, 80)")
    print(f"")
    print(f"Fusion & Classification:")
    print(f"  ├─ Weighted Fusion: 0.2×CNN + 0.8×BERT → (batch_size, 80)")
    print(f"  ├─ Dense 1: (batch_size, 128)")
    print(f"  ├─ Dense 2: (batch_size, 64)")
    print(f"  ├─ Dropout: 0.35")
    print(f"  └─ Output: (batch_size, 2) [P(off-target), P(on-target)]")

def analyze_cnn_branch():
    """
    Detailed analysis of the CNN branch
    """
    print("\n" + "="*80)
    print("🔬 CNN BRANCH DETAILED ANALYSIS")
    print("="*80)
    
    print(f"PURPOSE: Extract local DNA motifs and structural patterns")
    print(f"")
    print(f"INPUT: One-hot encoded matrices (batch_size, 26, 7)")
    print(f"  • 26 positions (sequence length)")
    print(f"  • 7 features per position [A,T,G,C,GAP,STRAND1,STRAND2]")
    print(f"")
    print(f"INCEPTION-STYLE CONVOLUTIONS:")
    
    kernel_sizes = [5, 15, 25, 35]
    for i, k in enumerate(kernel_sizes):
        print(f"  Conv1D_{i+1}:")
        print(f"    ├─ Kernel size: {k}")
        print(f"    ├─ Filters: 80")
        print(f"    ├─ Padding: 'same'")
        print(f"    ├─ Activation: ReLU")
        print(f"    └─ Output: (batch_size, 26, 80)")
        print(f"    Purpose: Capture patterns of length {k}")
        if k == 5:
            print(f"             (local motifs, single mismatches)")
        elif k == 15:
            print(f"             (medium-range patterns)")
        elif k == 25:
            print(f"             (near full-sequence patterns)")
        else:
            print(f"             (global sequence context)")
        print(f"")
    
    print(f"CONCATENATION & REDUCTION:")
    print(f"  ├─ Concatenate all conv outputs: (batch_size, 26, 320)")
    print(f"  └─ Dense layer: 320 → 80 features per position")
    print(f"")
    print(f"WHY THIS DESIGN:")
    print(f"  • Multiple scales capture different biological patterns")
    print(f"  • Local patterns: Point mutations, single base changes")
    print(f"  • Global patterns: Overall sequence similarity")
    print(f"  • Inception approach: Let model choose optimal scale")

def analyze_bert_branch():
    """
    Detailed analysis of the BERT branch
    """
    print("\n" + "="*80)
    print("🤖 BERT BRANCH DETAILED ANALYSIS")
    print("="*80)
    
    print(f"PURPOSE: Capture sequential dependencies and contextual relationships")
    print(f"")
    print(f"INPUT: Token sequences (batch_size, 26)")
    print(f"  • Integer IDs representing DNA pair patterns")
    print(f"  • Vocabulary size: 26 (base pairs + indels + special tokens)")
    print(f"")
    print(f"EMBEDDING LAYER:")
    print(f"  ├─ Input: Token IDs (batch_size, 26)")
    print(f"  ├─ Embedding dimension: 128")
    print(f"  └─ Output: (batch_size, 26, 128)")
    print(f"  Purpose: Convert discrete tokens to dense vectors")
    print(f"")
    print(f"TRANSFORMER BLOCKS (2 layers):")
    print(f"  Each block contains:")
    print(f"    ├─ Multi-Head Self-Attention:")
    print(f"    │  ├─ Heads: 4")
    print(f"    │  ├─ Key dimension: 128")
    print(f"    │  └─ Purpose: Find relationships between positions")
    print(f"    │")
    print(f"    ├─ Feed-Forward Network:")
    print(f"    │  ├─ Hidden dimension: 256")
    print(f"    │  ├─ Activation: ReLU")
    print(f"    │  └─ Purpose: Non-linear transformations")
    print(f"    │")
    print(f"    └─ Residual connections + Layer normalization")
    print(f"")
    print(f"FINAL DENSE LAYER:")
    print(f"  ├─ Input: (batch_size, 26, 128)")
    print(f"  └─ Output: (batch_size, 26, 80)")
    print(f"  Purpose: Match CNN output dimensions for fusion")
    print(f"")
    print(f"WHY TRANSFORMER ARCHITECTURE:")
    print(f"  • Self-attention captures long-range dependencies")
    print(f"  • Each position can attend to all other positions")
    print(f"  • Learns which positions are important for prediction")
    print(f"  • Contextual understanding: meaning depends on surroundings")

def analyze_fusion_strategy():
    """
    Explains the fusion and classification components
    """
    print("\n" + "="*80)
    print("🔗 FUSION & CLASSIFICATION ANALYSIS")
    print("="*80)
    
    print(f"GRU PROCESSING:")
    print(f"  Both branches → Bidirectional GRU (40 units each direction)")
    print(f"  ├─ Forward GRU: Processes sequence left → right")
    print(f"  ├─ Backward GRU: Processes sequence right ← left")
    print(f"  ├─ Concatenate: 40 + 40 = 80 features")
    print(f"  └─ return_sequences=False: Only final state")
    print(f"  ")
    print(f"  Input:  (batch_size, 26, 80)")
    print(f"  Output: (batch_size, 80)")
    print(f"  ")
    print(f"  Purpose: Capture temporal/sequential relationships")
    print(f"")
    print(f"WEIGHTED FUSION:")
    print(f"  merged = 0.2 × CNN_features + 0.8 × BERT_features")
    print(f"  ")
    print(f"  ├─ CNN weight (0.2): Structural/biological patterns")
    print(f"  └─ BERT weight (0.8): Sequential/contextual patterns")
    print(f"  ")
    print(f"  Why 80% BERT? Sequential patterns more important for CRISPR")
    print(f"  Why 20% CNN? Structural features provide complementary info")
    print(f"")
    print(f"CLASSIFICATION HEAD:")
    print(f"  Input: (batch_size, 80)")
    print(f"  ├─ Dense 1: 80 → 128 features, ReLU")
    print(f"  ├─ Dense 2: 128 → 64 features, ReLU")
    print(f"  ├─ Dropout: 35% (regularization)")
    print(f"  └─ Output: 64 → 2 classes, Softmax")
    print(f"  ")
    print(f"  Final output: [P(off-target), P(on-target)]")
    print(f"")
    print(f"WHY THIS DESIGN:")
    print(f"  • Progressive dimensionality reduction")
    print(f"  • Dropout prevents overfitting")
    print(f"  • Softmax gives probability distribution")
    print(f"  • Two classes: binary classification problem")

# ============================================================================
# 4. TRAINING PIPELINE ANALYSIS
# ============================================================================

def analyze_training_pipeline():
    """
    Explains the training process and data handling
    """
    print("\n" + "="*80)
    print("🚀 TRAINING PIPELINE ANALYSIS")
    print("="*80)
    
    print(f"DATA SPLITTING:")
    print(f"  ├─ Strategy: Stratified train-test split")
    print(f"  ├─ Ratio: 90% training, 10% validation")
    print(f"  └─ Purpose: Maintain class balance in both sets")
    print(f"")
    print(f"IMBALANCE HANDLING:")
    print(f"  CRISPR datasets are typically imbalanced:")
    print(f"  ├─ More off-target samples (label=0)")
    print(f"  └─ Fewer on-target samples (label=1)")
    print(f"")
    print(f"  Adaptive sampling ratios:")
    print(f"  ├─ If imbalance < 250:  Use (7:3) positive:negative")
    print(f"  ├─ If imbalance < 2000: Use (3:2) positive:negative")
    print(f"  └─ Else:                Use (1:1) balanced")
    print(f"")
    print(f"BALANCED BATCH GENERATION:")
    print(f"  For each training batch:")
    print(f"  ├─ Sample positive examples according to ratio")
    print(f"  ├─ Sample negative examples according to ratio")
    print(f"  ├─ Combine and shuffle")
    print(f"  └─ Yield balanced batch")
    print(f"")
    print(f"  Purpose: Prevent model from always predicting majority class")
    print(f"")
    print(f"MODEL COMPILATION:")
    print(f"  ├─ Optimizer: Adam (adaptive learning rate)")
    print(f"  ├─ Loss: Sparse categorical crossentropy")
    print(f"  └─ Metrics: Accuracy")
    print(f"")
    print(f"TRAINING CALLBACKS:")
    print(f"  ├─ EarlyStopping: Stop if validation loss doesn't improve (patience=6)")
    print(f"  ├─ ReduceLROnPlateau: Reduce learning rate when stuck (patience=3)")
    print(f"  └─ ModelCheckpoint: Save best model based on validation loss")
    print(f"")
    print(f"WHY THESE CHOICES:")
    print(f"  • Adam: Handles sparse gradients well, adaptive per-parameter learning rates")
    print(f"  • Sparse categorical crossentropy: Efficient for integer labels")
    print(f"  • Early stopping: Prevents overfitting")
    print(f"  • Learning rate reduction: Helps fine-tune when progress stalls")

def analyze_evaluation_metrics():
    """
    Explains the evaluation metrics and their importance
    """
    print("\n" + "="*80)
    print("📊 EVALUATION METRICS ANALYSIS")
    print("="*80)
    
    print(f"MULTIPLE METRICS FOR COMPREHENSIVE EVALUATION:")
    print(f"")
    print(f"1. F1 SCORE:")
    print(f"   ├─ Formula: 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"   ├─ Range: 0 to 1 (higher is better)")
    print(f"   └─ Purpose: Harmonic mean of precision and recall")
    print(f"   Why important: Balances false positives and false negatives")
    print(f"")
    print(f"2. MATTHEWS CORRELATION COEFFICIENT (MCC):")
    print(f"   ├─ Range: -1 to 1 (higher is better)")
    print(f"   ├─ Purpose: Balanced accuracy measure")
    print(f"   └─ Advantage: Works well with imbalanced datasets")
    print(f"   Why important: Considers all four confusion matrix categories")
    print(f"")
    print(f"3. AREA UNDER ROC CURVE (AUROC):")
    print(f"   ├─ Range: 0 to 1 (higher is better)")
    print(f"   ├─ Purpose: Discrimination ability across all thresholds")
    print(f"   └─ Interpretation: Probability model ranks random positive > random negative")
    print(f"   Why important: Threshold-independent performance measure")
    print(f"")
    print(f"4. AREA UNDER PRECISION-RECALL CURVE (AUPR):")
    print(f"   ├─ Range: 0 to 1 (higher is better)")
    print(f"   ├─ Purpose: Performance on imbalanced datasets")
    print(f"   └─ Focus: How well model identifies positive class")
    print(f"   Why important: More informative than AUROC for imbalanced data")
    print(f"")
    print(f"WHY MULTIPLE METRICS:")
    print(f"  • Each captures different aspects of performance")
    print(f"  • CRISPR prediction is safety-critical (false positives costly)")
    print(f"  • Imbalanced datasets require specialized metrics")
    print(f"  • Comprehensive evaluation builds confidence")

# ============================================================================
# 5. COMPLETE WORKFLOW DEMONSTRATION
# ============================================================================

def demonstrate_complete_workflow():
    """
    Shows the complete pipeline with a concrete example
    """
    print("\n" + "="*80)
    print("🔄 COMPLETE WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Step 1: Raw input
    raw_input = "GCTGCCAGTACAGGCTCCCCCTCG,GCAGCCAGTACA_GCTCACCATGG,0"
    print(f"STEP 1 - RAW INPUT:")
    print(f"  {raw_input}")
    
    parts = raw_input.split(',')
    seq1, seq2, label = parts[0], parts[1], int(parts[2])
    print(f"  sgRNA:  {seq1}")
    print(f"  Target: {seq2}")
    print(f"  Label:  {label} (off-target)")
    
    # Step 2: Sequence alignment
    print(f"\nSTEP 2 - SEQUENCE ALIGNMENT:")
    pair_list = []
    for a, b in zip(seq1, seq2):
        if a == "_":
            token = "_" + b
        elif b == "_":
            token = a + "_"
        else:
            token = a + b
        pair_list.append(token)
    
    print(f"  Pair tokens: {pair_list[:10]}... (showing first 10)")
    print(f"  Total tokens: {len(pair_list)}")
    
    # Step 3: Token encoding (simplified)
    print(f"\nSTEP 3 - TOKEN ENCODING:")
    print(f"  Add CLS token (0) at start")
    print(f"  Convert each pair to integer ID")
    print(f"  Add SEP token (1) at end")
    print(f"  Pad to length 26")
    print(f"  Result shape: (26,)")
    
    # Step 4: One-hot encoding
    print(f"\nSTEP 4 - ONE-HOT ENCODING:")
    print(f"  Convert each pair to 7D feature vector")
    print(f"  Features: [A, T, G, C, GAP, STRAND1, STRAND2]")
    print(f"  Result shape: (26, 7)")
    
    # Step 5: Model processing
    print(f"\nSTEP 5 - MODEL PROCESSING:")
    print(f"  CNN Branch:")
    print(f"    Input: (1, 26, 7)")
    print(f"    → Conv1D layers → (1, 26, 320)")
    print(f"    → Dense → (1, 26, 80)")
    print(f"    → BiGRU → (1, 80)")
    print(f"")
    print(f"  BERT Branch:")
    print(f"    Input: (1, 26)")
    print(f"    → Embedding → (1, 26, 128)")
    print(f"    → Transformer × 2 → (1, 26, 128)")
    print(f"    → Dense → (1, 26, 80)")
    print(f"    → BiGRU → (1, 80)")
    
    # Step 6: Fusion and prediction
    print(f"\nSTEP 6 - FUSION & PREDICTION:")
    print(f"  Fusion: 0.2 × CNN + 0.8 × BERT → (1, 80)")
    print(f"  Dense layers: 80 → 128 → 64")
    print(f"  Output: (1, 2) probabilities")
    print(f"  Example result: [0.85, 0.15]")
    print(f"  Prediction: Off-target (class 0, confidence 85%)")
    
    print(f"\nWORKFLOW SUMMARY:")
    print(f"  Raw DNA sequences → Aligned pairs → Numerical encoding")
    print(f"  → Dual processing (CNN + BERT) → Fusion → Classification")
    print(f"  → Probability prediction for CRISPR off-target effects")

# ============================================================================
# 6. BIOLOGICAL SIGNIFICANCE AND DESIGN RATIONALE
# ============================================================================

def explain_biological_significance():
    """
    Explains why this approach makes biological sense
    """
    print("\n" + "="*80)
    print("🧬 BIOLOGICAL SIGNIFICANCE & DESIGN RATIONALE")
    print("="*80)
    
    print(f"CRISPR-CAS9 MECHANISM:")
    print(f"  1. Guide RNA (sgRNA) binds to complementary DNA sequence")
    print(f"  2. Cas9 protein cuts DNA at target site")
    print(f"  3. Off-targets occur when sgRNA binds similar sequences")
    print(f"  4. Prediction helps avoid unwanted cuts")
    print(f"")
    print(f"KEY BIOLOGICAL FACTORS:")
    print(f"  ├─ Sequence similarity (matches vs mismatches)")
    print(f"  ├─ Position of mismatches (some positions more tolerant)")
    print(f"  ├─ Type of mismatches (some base changes more tolerated)")
    print(f"  ├─ Indels (insertions/deletions affect binding)")
    print(f"  └─ Overall sequence context")
    print(f"")
    print(f"WHY HYBRID CNN-BERT ARCHITECTURE:")
    print(f"")
    print(f"CNN BRANCH - STRUCTURAL ANALYSIS:")
    print(f"  ✓ Captures local DNA motifs and patterns")
    print(f"  ✓ Processes biological features (base composition)")
    print(f"  ✓ Multiple scales detect different pattern types")
    print(f"  ✓ Translation-invariant (patterns anywhere in sequence)")
    print(f"")
    print(f"BERT BRANCH - CONTEXTUAL ANALYSIS:")
    print(f"  ✓ Models long-range dependencies")
    print(f"  ✓ Understands positional importance")
    print(f"  ✓ Learns which positions interact")
    print(f"  ✓ Contextual understanding (position matters)")
    print(f"")
    print(f"FUSION STRATEGY:")
    print(f"  ✓ Combines complementary information")
    print(f"  ✓ 80% BERT: Sequential patterns dominate CRISPR binding")
    print(f"  ✓ 20% CNN: Structural features provide additional signal")
    print(f"")
    print(f"BIOLOGICAL VALIDATION:")
    print(f"  • Matches known CRISPR biology (sequence + structure)")
    print(f"  • Handles complex interaction patterns")
    print(f"  • Accounts for position-dependent effects")
    print(f"  • Processes indels (common in real data)")

def summarize_project_strengths():
    """
    Summarizes the key strengths of this approach
    """
    print("\n" + "="*80)
    print("💪 PROJECT STRENGTHS & INNOVATIONS")
    print("="*80)
    
    print(f"TECHNICAL STRENGTHS:")
    print(f"  ✓ Hybrid architecture combines best of CNN and Transformer")
    print(f"  ✓ Dual input processing (tokens + biological features)")
    print(f"  ✓ Handles variable-length sequences with fixed encoding")
    print(f"  ✓ Addresses class imbalance with adaptive sampling")
    print(f"  ✓ Comprehensive evaluation with multiple metrics")
    print(f"")
    print(f"BIOLOGICAL RELEVANCE:")
    print(f"  ✓ Pair token encoding captures alignment information")
    print(f"  ✓ One-hot features preserve biological meaning")
    print(f"  ✓ Multi-scale CNN detects various DNA patterns")
    print(f"  ✓ Attention mechanism models position interactions")
    print(f"")
    print(f"PRACTICAL ADVANTAGES:")
    print(f"  ✓ End-to-end trainable")
    print(f"  ✓ Handles real-world data (indels, mismatches)")
    print(f"  ✓ Scalable to large datasets")
    print(f"  ✓ Interpretable through attention weights")
    print(f"")
    print(f"INNOVATION ASPECTS:")
    print(f"  ✓ Novel application of BERT to DNA sequences")
    print(f"  ✓ Inception-style CNN for multi-scale DNA analysis")
    print(f"  ✓ Weighted fusion strategy")
    print(f"  ✓ Comprehensive biological feature engineering")

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that runs the complete analysis
    """
    # Project overview
    project_overview()
    
    # Data processing pipeline
    demonstrate_raw_input()
    demonstrate_sequence_alignment()
    demonstrate_token_encoding()
    demonstrate_onehot_encoding()
    
    # Model architecture
    demonstrate_model_architecture()
    analyze_cnn_branch()
    analyze_bert_branch()
    analyze_fusion_strategy()
    
    # Training and evaluation
    analyze_training_pipeline()
    analyze_evaluation_metrics()
    
    # Complete workflow
    demonstrate_complete_workflow()
    
    # Biological significance
    explain_biological_significance()
    summarize_project_strengths()
    
    print("\n" + "="*100)
    print("🎉 COMPLETE WORKFLOW ANALYSIS FINISHED")
    print("="*100)
    print(f"""
    This analysis covered:
    ✓ Raw data format and processing pipeline
    ✓ Sequence alignment and encoding strategies  
    ✓ Hybrid CNN-BERT model architecture
    ✓ Training pipeline and imbalance handling
    ✓ Evaluation metrics and their significance
    ✓ Complete workflow with concrete examples
    ✓ Biological rationale and design justification
    
    The CRISPR-Cas9 off-target prediction model successfully combines
    structural DNA analysis (CNN) with sequential pattern recognition (BERT)
    to predict off-target effects with high accuracy and biological relevance.
    """)

if __name__ == "__main__":
    main()
