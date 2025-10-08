# demonstrate_model.py - Complete demonstration of model processing with one example
import numpy as np
import tensorflow as tf
from data_process import load_dataset, MAX_LEN, VOCAB_SIZE, DatasetEncoder, make_pair_list
from detailed_model import build_crispr_bert_model, predict_single_example

print("=" * 120)
print("CRISPR-Cas9 Off-Target Prediction Model - COMPLETE DEMONSTRATION")
print("=" * 120)

def demonstrate_data_processing():
    """
    Demonstrate the complete data processing pipeline
    """
    print(f"\n" + "="*80)
    print(f" [DATA PROCESSING DEMONSTRATION]")
    print(f"="*80)
    
    # Load a small sample from the dataset
    print(f"\n[Loading Dataset] Loading sample from c.txt...")
    X_tokens, X_onehot, y = load_dataset("datasets/sam.txt")
    print(f"    Dataset loaded: {len(y)} samples")
    print(f"    Token sequences shape: {X_tokens.shape}")
    print(f"    One-hot matrices shape: {X_onehot.shape}")
    print(f"    Labels shape: {y.shape}")
    
    # Take the first example
    sample_idx = 0
    token_seq = X_tokens[sample_idx]
    onehot_mat = X_onehot[sample_idx]
    label = y[sample_idx]
    
    print(f"\n [Sample Analysis] Analyzing sample {sample_idx}...")
    print(f"    Token sequence: {token_seq}")
    print(f"    One-hot matrix shape: {onehot_mat.shape}")
    print(f"    True label: {label} ({'Off-target' if label == 0 else 'On-target'})")
    
    # Show the original DNA sequences
    print(f"\n [DNA Sequence Analysis]")
    print(f"    This sample represents a DNA sequence pair for CRISPR off-target prediction")
    print(f"    The token sequence represents encoded DNA base pairs")
    print(f"    The one-hot matrix represents structural features of the DNA")
    
    # Decode the token sequence to show what it represents
    print(f"\n [Token Decoding] Decoding token sequence...")
    from data_process import index_to_token
    
    # Get complete token sequence
    print(f"\n Complete Token Sequence (length: {len(token_seq)}):")
    print("    " + " ".join([f"{i:2}" for i in range(len(token_seq))]))
    print("    " + "  ".join([f"{t:2}" for t in token_seq]))
    
    # Decode tokens to their string representation
    decoded_tokens = []
    token_meanings = []
    for i, token_id in enumerate(token_seq):
        # Handle special tokens first (0=CLS, 1=SEP, PAD tokens are typically 0 or 1)
        if token_id == 0:
            decoded_tokens.append("[CLS]")
            token_meanings.append("Classification token (start of sequence)")
        elif token_id == 1:
            decoded_tokens.append("[SEP]")
            token_meanings.append("Separator token (end of sequence)")
        elif token_id in index_to_token:
            token_str = index_to_token[token_id]
            decoded_tokens.append(token_str)
            # Add meaning for DNA base pairs and indels
            if '_' in token_str:
                if token_str.startswith('_'):
                    token_meanings.append(f"Insertion: {token_str[1]} added")
                elif token_str.endswith('_'):
                    token_meanings.append(f"Deletion: {token_str[0]} removed")
                else:
                    token_meanings.append(f"Base pair: {token_str}")
            else:
                token_meanings.append(f"DNA base pair: {token_str}")
        else:
            decoded_tokens.append(f"[PAD]")
            token_meanings.append("Padding token")
    
    # Print token meanings in a table format
    print("\n Token Details:")
    print(f"{'Index':<8} | {'Token ID':<8} | {'Token':<8} | Description")
    print("-" * 70)
    for i, (token_id, token, meaning) in enumerate(zip(token_seq, decoded_tokens, token_meanings)):
        print(f"{i:<8} | {token_id:<8} | {token:<8} | {meaning}")
    
    # Show special tokens summary
    print("\n Special Tokens Summary:")
    cls_positions = [i for i, token in enumerate(token_seq) if token == 0]
    sep_positions = [i for i, token in enumerate(token_seq) if token == 1]
    pad_positions = [i for i, token in enumerate(token_seq) if token not in index_to_token and token != 0 and token != 1]
    
    if cls_positions:
        print(f"    [CLS] tokens at positions: {cls_positions}")
    if sep_positions:
        print(f"    [SEP] tokens at positions: {sep_positions}")
    if pad_positions:
        print(f"    [PAD] tokens at positions: {pad_positions}")
    
    # Show actual sequence content
    actual_tokens = [token for token in token_seq if token in index_to_token]
    print(f"    Actual DNA sequence tokens: {len(actual_tokens)} base pairs")
    print(f"    Sequence: {' '.join([index_to_token[token] for token in actual_tokens])}")
    
    # Show one-hot matrix interpretation
    print(f"\n [One-Hot Matrix Analysis]")
    print(f"    Matrix dimensions: {onehot_mat.shape[0]} rows Ã— {onehot_mat.shape[1]} columns")
    print(f"    Why {onehot_mat.shape[0]} rows? MAX_LEN = {onehot_mat.shape[0]} (sequence length)")
    print(f"    Why {onehot_mat.shape[1]} columns? Each position has {onehot_mat.shape[1]} binary features:")
    print(f"        Column 0: A (Adenine) presence in base pair")
    print(f"        Column 1: T (Thymine) presence in base pair") 
    print(f"        Column 2: G (Guanine) presence in base pair")
    print(f"        Column 3: C (Cytosine) presence in base pair")
    print(f"        Column 4: Gap/Deletion indicator")
    print(f"        Column 5: First strand indicator")
    print(f"        Column 6: Second strand indicator")
    
    # Print complete one-hot matrix in a readable format
    print("\n Complete 26*7 Embedding Matrix:")
    # Print column headers
    print("Pos  | A T G C GAP FST SEC | Interpretation")
    print("-" * 60)
    
    # Print each position's encoding
    for i in range(onehot_mat.shape[0]):
        features = onehot_mat[i]
        base_encoding = " ".join(["1" if x > 0 else "." for x in features[:4]])
        meta_encoding = " ".join(["1" if x > 0 else "." for x in features[4:]])
        
        # Get the DNA base(s) at this position
        dna_bases = []
        if features[0] > 0: dna_bases.append('A')
        if features[1] > 0: dna_bases.append('T')
        if features[2] > 0: dna_bases.append('G')
        if features[3] > 0: dna_bases.append('C')
        
        # Interpretation - CORRECTED
        feature_sum = int(np.sum(features[:4]))  # Only sum the base indicators
        if feature_sum == 0:
            interpretation = "CLS/SEP token (padding)"
        elif feature_sum == 1:
            if features[4] > 0:  # If gap is present
                interpretation = f"Deletion: {dna_bases[0]}_"
            else:
                interpretation = f"Base pair: {dna_bases[0]}{dna_bases[0]}"
        elif feature_sum == 2:
            if features[4] > 0:  # If gap is present
                interpretation = f"Insertion/Deletion with gap"
            else:
                interpretation = f"Base pair: {''.join(dna_bases)}"
        else:
            interpretation = "Complex pattern"
        
        # Add position metadata
        if features[5] > 0 and features[6] > 0:
            interpretation += " (Both strands)"
        elif features[5] > 0:
            interpretation += " (First strand)"
        elif features[6] > 0:
            interpretation += " (Second strand)"
        
        print(f"{i:3d} | {base_encoding} | {meta_encoding} | {interpretation}")
    
    # Print summary of one-hot encoding
    print("\n One-Hot Matrix Summary:")
    print(f"    Total positions: {onehot_mat.shape[0]}")
    print(f"    Non-padding positions: {np.sum([np.any(row > 0) for row in onehot_mat])}")
    print(f"    Padding positions: {np.sum([not np.any(row > 0) for row in onehot_mat])}")
    
    # Show distribution of features
    print("\n Feature Distribution:")
    feature_names = ['A', 'T', 'G', 'C', 'GAP', 'FST', 'SEC']
    for i, name in enumerate(feature_names):
        count = np.sum(onehot_mat[:, i] > 0)
        print(f"    {name}: {count:3d} positions ({count/onehot_mat.shape[0]:.1%})")
    
    return token_seq.reshape(1, -1), onehot_mat.reshape(1, -1, 7), label

def demonstrate_segment_embedding():
    """
    Demonstrate segment embedding functionality for DNA sequences
    """
    print(f"\n" + "="*80)
    print(f"[SEGMENT EMBEDDING DEMONSTRATION]")
    print(f"="*80)
    
    # Load DNA sequence pair from sam.txt
    print(f"\n[Loading Sequences from sam.txt]")
    with open("datasets/sam.txt", "r") as f:
        line = f.readline().strip()
        seq1, seq2, label = line.split(",")
    
    print(f"\n[Input Sequences]")
    print(f"    sgRNA sequence: {seq1}")
    print(f"    Target sequence: {seq2}")
    print(f"    Label: {label}")
    print(f"    Sequence lengths: {len(seq1)} and {len(seq2)}")
    
    # Create pair list and tokenize
    from data_process import make_pair_list, DatasetEncoder, index_to_token
    pair_list = make_pair_list(seq1, seq2)
    enc = DatasetEncoder()
    token_sequence = enc.encode_token_list(pair_list)
    
    print(f"\n[Token Sequence]")
    print(f"    Token sequence: {token_sequence}")
    print(f"    Length: {len(token_sequence)}")
    
    # Create segment IDs - all zeros for length 26
    print(f"\n[Segment ID Creation]")
    segment_ids = [0] * 26  # All segment IDs are 0
    
    # Show segment mapping
    print(f"\n[Segment Mapping]")
    print("Position | Token ID | Token | Segment | Description")
    print("-" * 70)
    for i, (token_id, segment_id) in enumerate(zip(token_sequence, segment_ids)):
        if token_id == 0:
            token_str = "[CLS]"
            description = "Classification token"
        elif token_id == 1:
            token_str = "[SEP]"
            description = "Separator token"
        elif token_id in index_to_token:
            token_str = index_to_token[token_id]
            description = "DNA base pair"
        else:
            token_str = "[PAD]"
            description = "Padding token"
        
        print(f"{i:8} | {token_id:8} | {token_str:6} | {segment_id:7} | {description}")
    
    # Demonstrate embedding dimensions
    print(f"\n[Embedding Dimensions]")
    embed_dim = 128  # Small debug mode
    print(f"    Token embedding dimension: {embed_dim}")
    print(f"    Position embedding dimension: {embed_dim}")
    print(f"    Segment embedding dimension: {embed_dim}")
    print(f"    Combined embedding dimension: {embed_dim}")
    print(f"    Sequence length: 26")
    
    # Show how segment embeddings would be created
    print(f"\n[Segment Embedding Matrix]")
    print(f"    Segment embedding shape: (1, {embed_dim})")
    print(f"    Segment 0: Learnable vector of shape ({embed_dim},)")
    print(f"    All positions use segment 0 embedding")
    
    print(f"\n[How Segment Embeddings Work]")
    print(f"    1. All 26 positions use the same segment embedding (segment 0)")
    print(f"    2. Segment embedding vector is added to token embeddings")
    print(f"    3. Combined with position embeddings of length 26")
    print(f"    4. Final embedding = Token[26,{embed_dim}] + Position[26,{embed_dim}] + Segment[26,{embed_dim}]")
    
    print(f"\n[Segment Embedding Complete]")
    print(f"    Segment IDs: All zeros (length 26)")
    print(f"    This matches the token and position embedding dimensions")
    
    return token_sequence, segment_ids

def demonstrate_position_embedding():
    """
    Demonstrate position embedding functionality for DNA sequences
    """
    print(f"\n" + "="*80)
    print(f"[POSITION EMBEDDING DEMONSTRATION]")
    print(f"="*80)
    
    print(f"\n[Position Embedding Overview]")
    print(f"    Position embeddings encode the position of each token in the sequence")
    print(f"    Each position gets a unique learnable embedding vector")
    print(f"    Helps the model understand the order and relative positions of tokens")
    
    # Show position IDs for our example
    print(f"\n[Position ID Creation]")
    position_ids = list(range(26))
    print(f"    Position IDs: {position_ids}")
    print(f"    Position ID range: 0 to 25")
    print(f"    Sequence length: 26")
    
    # Show how position embeddings would work
    print(f"\n[Position Embedding Matrix]")
    embed_dim = 128  # Small debug mode
    print(f"    Position embedding shape: (26, {embed_dim})")
    print(f"    Each position gets a unique {embed_dim}-dimensional vector")
    
    # Show first few positions
    print(f"\n[Position Embedding Examples]")
    print("Position | Description | Embedding Shape")
    print("-" * 50)
    for i in range(min(10, 26)):
        if i == 0:
            description = "Position 0"
        elif i == 25:
            description = "Last position (25)"
        else:
            description = f"Position {i}"
        print(f"{i:8} | {description:20} | ({embed_dim},)")
    
    print(f"\n[How Position Embeddings Work]")
    print(f"    1. Each position in the sequence gets a unique embedding")
    print(f"    2. Position embeddings are learned during training")
    print(f"    3. Model learns position-specific patterns and relationships")
    print(f"    4. Helps distinguish between tokens at different positions")
    print(f"    5. Enables the model to understand sequence order")
    
    print(f"\n[Combined Embeddings]")
    print(f"    Final embedding = Token Embedding + Position Embedding + Segment Embedding")
    print(f"    All three embeddings have the same dimension: {embed_dim}")
    print(f"    All three embeddings have the same length: 26")
    print(f"    They are added together element-wise")
    
    print(f"\n[Position Embedding Complete]")
    print(f"    Position IDs: 0 to 25 (length 26)")
    print(f"    This demonstrates how position embeddings help the model")
    print(f"    understand the sequential nature of DNA sequences")
    
    return position_ids

def demonstrate_complete_embedding():
    """
    Demonstrate how token, position, and segment embeddings work together
    """
    print(f"\n" + "="*80)
    print(f"[COMPLETE EMBEDDING DEMONSTRATION]")
    print(f"="*80)
    
    # Load DNA sequence pair from sam.txt
    print(f"\n[Loading Sequences from sam.txt]")
    with open("datasets/sam.txt", "r") as f:
        line = f.readline().strip()
        seq1, seq2, label = line.split(",")
    
    print(f"\n[Input Sequences]")
    print(f"    sgRNA: {seq1}")
    print(f"    target: {seq2}")
    print(f"    Label: {label}")
    
    # Create all components
    from data_process import make_pair_list, DatasetEncoder, index_to_token
    pair_list = make_pair_list(seq1, seq2)
    enc = DatasetEncoder()
    token_sequence = enc.encode_token_list(pair_list)
    
    # Create segment IDs - all zeros for length 26
    segment_ids = [0] * 26
    
    # Position IDs
    position_ids = list(range(26))
    
    print(f"\n[Embedding Components]")
    embed_dim = 128
    print(f"    Embedding dimension: {embed_dim}")
    print(f"    Sequence length: 26")
    print(f"    Token embedding shape: (26, {embed_dim})")
    print(f"    Position embedding shape: (26, {embed_dim})")
    print(f"    Segment embedding shape: (26, {embed_dim})")
    print(f"    All segment IDs are 0")
    
    # Show detailed breakdown for first few positions
    print(f"\n[Detailed Embedding Breakdown]")
    print("Pos | Token | Token ID | Position | Segment | Combined Embedding")
    print("-" * 80)
    
    for i in range(min(10, 26)):
        token_id = token_sequence[i]
        position_id = position_ids[i]
        segment_id = segment_ids[i]
        
        # Get token string
        if token_id == 0:
            token_str = "[CLS]"
        elif token_id == 1:
            token_str = "[SEP]"
        elif token_id in index_to_token:
            token_str = index_to_token[token_id]
        else:
            token_str = "[PAD]"
        
        print(f"{i:3} | {token_str:5} | {token_id:8} | {position_id:8} | {segment_id:7} | Token + Pos + Seg = {embed_dim}D")
    
    print(f"\n[Embedding Combination Process]")
    print(f"    1. Token Embedding: Lookup token ID in embedding table")
    print(f"    2. Position Embedding: Lookup position ID in position table")
    print(f"    3. Segment Embedding: Lookup segment ID in segment table")
    print(f"    4. Element-wise Addition: E_final = E_token + E_position + E_segment")
    print(f"    5. Layer Normalization: Normalize the combined embedding")
    
    print(f"\n  [Why This Works]")
    print(f"    Token embeddings capture DNA base pair meanings")
    print(f"    Position embeddings capture sequence order and position")
    print(f"    Segment embeddings provide consistent context (all segment 0)")
    print(f"    Combined embeddings provide rich contextual information")
    print(f"    Model can learn complex DNA sequence relationships")
    
    print(f"\n[CRISPR-Specific Benefits]")
    print(f"    All positions use the same segment embedding (segment 0)")
    print(f"    Position embeddings help identify binding sites")
    print(f"    Token embeddings capture base pair interactions")
    print(f"    Combined information improves off-target prediction")
    
    print(f"\n [Complete Embedding Demonstration Complete]")
    
    return token_sequence, position_ids, segment_ids

def demonstrate_model_architecture():
    """
    Demonstrate the model architecture in detail
    """
    print(f"\n" + "="*80)
    print(f"[MODEL ARCHITECTURE DEMONSTRATION]")
    print(f"="*80)
    
    print(f"\n[Model Configuration]")
    print(f"    Vocabulary size: {VOCAB_SIZE}")
    print(f"    Maximum sequence length: {MAX_LEN}")
    print(f"    Model type: Hybrid CNN-BERT architecture")
    print(f"    Purpose: CRISPR-Cas9 off-target prediction")
    
    print(f"\n[CNN Branch Details]")
    print(f"    Input: One-hot encoded DNA sequences ({MAX_LEN} x 7)")
    print(f"    Convolution layers: 4 parallel Conv2D with kernels (1,1), (2,2), (3,3), (5,5)")
    print(f"    Filters per conv: 5, 15, 25, 35 (total 80 channels)")
    print(f"    Activation: ReLU")
    print(f"    Purpose: Capture local DNA sequence patterns")
    
    print(f"\[BERT Branch Details]")
    print(f"    Input: Tokenized DNA sequences ({MAX_LEN} tokens)")
    print(f"    Embedding dimension: 128 (small debug mode)")
    print(f"    Transformer layers: 2")
    print(f"    Attention heads: 4")
    print(f"    Feed-forward dimension: 256")
    print(f"    Purpose: Capture long-range dependencies")
    
    print(f"\n[Embedding Components]")
    print(f"    Token Embedding: Maps each DNA token to dense vector")
    print(f"    Position Embedding: Encodes position information in sequence")
    print(f"    Segment Embedding: Distinguishes between different sequence parts")
    print(f"    Combined Embedding: Sum of all three embeddings")
    
    print(f"\n[Segment Embedding Details]")
    print(f"    Segment 0: CLS token and first DNA sequence (sgRNA)")
    print(f"    Segment 1: SEP token and second DNA sequence (target)")
    print(f"    Segment 2: PAD tokens (padding)")
    print(f"    Purpose: Helps model understand sequence boundaries")
    
    print(f"\n[GRU Processing]")
    print(f"    Both branches processed through Custom Bidirectional GRU")
    print(f"    BiGRU units: 40 per direction (80 total)")
    print(f"    Purpose: Sequence modeling and temporal dependencies")
    
    print(f"\n[Fusion Strategy]")
    print(f"    CNN contribution: 20% (0.2x weight)")
    print(f"    BERT contribution: 80% (0.8x weight)")
    print(f"    Fusion method: Weighted addition")
    
    print(f"\n[Classification Head]")
    print(f"    Dense layer 1: 128 units, ReLU")
    print(f"    Dense layer 2: 64 units, ReLU")
    print(f"    Dropout: 0.35")
    print(f"    Output: 2 units, Softmax (On-target vs Off-target)")
    
    # Build the model
    print(f"\n[Building Model] Creating the model...")
    model = build_crispr_bert_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, small_debug=True)
    
    print(f"\nðŸ“Š [Model Summary]")
    model.summary()
    
    return model

def demonstrate_training_process():
    """
    Demonstrate a simplified training process
    """
    print(f"\n" + "="*80)
    print(f"[TRAINING PROCESS DEMONSTRATION]")
    print(f"="*80)
    
    print(f"\n[Data Preparation]")
    print(f"    Loading training data...")
    X_tokens, X_onehot, y = load_dataset("datasets/c.txt")
    
    # Use a small subset for demonstration
    subset_size = 1000
    X_tokens_sub = X_tokens[:subset_size]
    X_onehot_sub = X_onehot[:subset_size]
    y_sub = y[:subset_size]
    
    print(f"    Using subset of {subset_size} samples for demonstration")
    print(f"    Positive samples: {np.sum(y_sub==1)}")
    print(f"    Negative samples: {np.sum(y_sub==0)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    Xt, Xv, Ot, Ov, yt, yv = train_test_split(
        X_tokens_sub, X_onehot_sub, y_sub, 
        test_size=0.2, stratify=y_sub, random_state=42
    )
    
    print(f"    Training samples: {len(yt)}")
    print(f"    Validation samples: {len(yv)}")
    
    # Build and compile model
    print(f"\n[Model Setup]")
    model = build_crispr_bert_model(vocab_size=VOCAB_SIZE, max_len=MAX_LEN, small_debug=True)
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    print(f"    Model compiled with Adam optimizer")
    print(f"    Loss function: sparse_categorical_crossentropy")
    print(f"    Metrics: accuracy")
    
    # Quick training for demonstration
    print(f"\n [Quick Training] Training for 3 epochs...")
    print(f"    This is a demonstration - full training would use more epochs")
    
    history = model.fit(
        [Xt, Ot], yt,
        validation_data=([Xv, Ov], yv),
        epochs=3,
        batch_size=32,
        verbose=1
    )
    
    print(f"\n[Training Complete]")
    print(f"    Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"    Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model

def main_demonstration():
    """
    Main demonstration function
    """
    print(f"\n[MAIN DEMONSTRATION] Starting complete model demonstration...")
    
    # Step 1: Data Processing
    token_input, onehot_input, true_label = demonstrate_data_processing()
    
    # Step 1.5: Segment Embedding Demonstration
    token_seq, segment_ids = demonstrate_segment_embedding()
    
    # Step 1.6: Position Embedding Demonstration
    position_ids = demonstrate_position_embedding()
    
    # Step 1.7: Complete Embedding Demonstration
    tokens, positions, segments = demonstrate_complete_embedding()
    
    # Step 2: Model Architecture
    model = demonstrate_model_architecture()
    
    # Step 3: Training Process
    trained_model = demonstrate_training_process()
    
    # Step 4: Single Example Prediction
    print(f"\n" + "="*80)
    print(f"[SINGLE EXAMPLE PREDICTION]")
    print(f"="*80)
    
    print(f"\n[Making Prediction] Using trained model on sample...")
    prediction = predict_single_example(trained_model, token_input, onehot_input, true_label)
    
    print(f"\n[Prediction Summary]")
    print(f"    Input shape: Token {token_input.shape}, One-hot {onehot_input.shape}")
    print(f"    True label: {true_label}")
    print(f"    Predicted probabilities: {prediction[0]}")
    print(f"    Predicted class: {np.argmax(prediction[0])}")
    print(f"    Confidence: {np.max(prediction[0]):.4f}")
    
    # Step 5: Model Interpretation
    print(f"\n" + "="*80)
    print(f"[MODEL INTERPRETATION]")
    print(f"="*80)
    
    print(f"\n[What the Model Learned]")
    print(f"    The model combines two types of information:")
    print(f"    1. CNN branch: Local DNA sequence patterns and motifs")
    print(f"    2. BERT branch: Long-range dependencies and context")
    print(f"    The GRU layers capture temporal relationships in the sequence")
    print(f"    The fusion layer combines both perspectives for final prediction")
    
    print(f"\n[Prediction Confidence]")
    confidence = np.max(prediction[0])
    if confidence > 0.8:
        print(f"    High confidence prediction ({confidence:.4f})")
    elif confidence > 0.6:
        print(f"    Medium confidence prediction ({confidence:.4f})")
    else:
        print(f"    Low confidence prediction ({confidence:.4f})")
    
    print(f"\n[DEMONSTRATION COMPLETE]")
    print(f"    This demonstration showed:")
    print(f"    1. How DNA sequences are processed and encoded")
    print(f"    2. The hybrid CNN-BERT model architecture")
    print(f"    3. The training process with real data")
    print(f"    4. Step-by-step prediction on a single example")
    print(f"    5. Model interpretation and confidence analysis")
    
    print(f"\n" + "="*120)

if __name__ == "__main__":
    main_demonstration()
