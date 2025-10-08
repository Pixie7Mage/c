# detailed_model.py - Enhanced version with comprehensive print statements
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

print("=" * 80)
print("CRISPR-Cas9 Off-Target Prediction Model - Detailed Processing")
print("=" * 80)

# ---------- Transformer building blocks ----------
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        print(f"    [MultiHeadSelfAttention] Initialized with embed_dim={embed_dim}, num_heads={num_heads}")

    def call(self, x, training=False, mask=None):
        print(f"    [MultiHeadSelfAttention] Input shape: {x.shape}")
        print(f"    [MultiHeadSelfAttention] Computing self-attention...")
        result = self.att(x, x, x, attention_mask=mask, training=training)
        print(f"    [MultiHeadSelfAttention] Output shape: {result.shape}")
        return result

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        print(f"    [TransformerBlock] Initializing with embed_dim={embed_dim}, num_heads={num_heads}, ff_dim={ff_dim}")
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)
        print(f"    [TransformerBlock] Feed-forward network: {ff_dim} -> {embed_dim}")

    def call(self, x, training=False, mask=None):
        print(f"    [TransformerBlock] Input shape: {x.shape}")
        
        # Self-attention
        print(f"    [TransformerBlock] Step 1: Computing self-attention...")
        attn = self.att(x, training=training, mask=mask)
        print(f"    [TransformerBlock] Step 2: Applying dropout and residual connection...")
        out1 = self.norm1(x + self.drop1(attn, training=training))
        print(f"    [TransformerBlock] Step 3: Feed-forward network...")
        ffn = self.ffn(out1)
        print(f"    [TransformerBlock] Step 4: Final dropout and residual connection...")
        out2 = self.norm2(out1 + self.drop2(ffn, training=training))
        print(f"    [TransformerBlock] Output shape: {out2.shape}")
        return out2

# ---------- Model parts ----------
def inception_cnn_branch(inp, max_len=26):
    print(f"\n🔬 [CNN Branch] Starting Inception-style CNN processing...")
    print(f"    [CNN Branch] Input shape: {inp.shape}")
    print(f"    [CNN Branch] Max sequence length: {max_len}")
    
    # Match model.py: expand to 4D and apply parallel Conv2D with kernels (1,1),(2,2),(3,3),(5,5)
    x4d = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(inp)  # (batch, max_len, 7, 1)
    print(f"    [CNN Branch] Expanded to 4D for Conv2D: {x4d.shape}")

    print(f"    [CNN Branch] Applying parallel Conv2D filters: [(1,1)->5, (2,2)->15, (3,3)->25, (5,5)->35]")
    conv1 = layers.Conv2D(5, (1, 1), padding='same', activation='relu')(x4d)
    conv2 = layers.Conv2D(15, (2, 2), padding='same', activation='relu')(x4d)
    conv3 = layers.Conv2D(25, (3, 3), padding='same', activation='relu')(x4d)
    conv4 = layers.Conv2D(35, (5, 5), padding='same', activation='relu')(x4d)

    merged = layers.Concatenate(axis=-1)([conv1, conv2, conv3, conv4])  # (batch, max_len, 7, 80)
    print(f"    [CNN Branch] Concatenated feature maps shape: {merged.shape}")

    # Collapse width dimension (7 features) to (batch, max_len, 80), as in model.py
    collapsed = layers.Lambda(lambda t: tf.reduce_max(t, axis=2))(merged)  # (batch, max_len, 80)
    x = layers.Reshape((max_len, 80))(collapsed)
    print(f"    [CNN Branch] Collapsed output shape: {x.shape}")
    return x

def bert_branch(inp, vocab_size, max_len, small_debug=False):
    print(f"\n🤖 [BERT Branch] Starting Transformer-based BERT processing...")
    print(f"    [BERT Branch] Input shape: {inp.shape}")
    print(f"    [BERT Branch] Vocabulary size: {vocab_size}")
    print(f"    [BERT Branch] Max sequence length: {max_len}")
    print(f"    [BERT Branch] Small debug mode: {small_debug}")
    
    embed_dim = 768 if not small_debug else 128
    num_heads = 12 if not small_debug else 4
    ff_dim = 3072 if not small_debug else 256
    num_layers = 12 if not small_debug else 2
    
    print(f"    [BERT Branch] Configuration: embed_dim={embed_dim}, num_heads={num_heads}, ff_dim={ff_dim}, num_layers={num_layers}")
    
    # Step 1: Token, Position, and Segment embeddings (as in model.py)
    print(f"    [BERT Branch] Step 1: Token embedding...")
    token_emb = layers.Embedding(vocab_size, embed_dim, name='token_embedding')(inp)
    print(f"    [BERT Branch] Token embedding shape: {token_emb.shape}")

    print(f"    [BERT Branch] Step 1b: Position embedding...")
    position_indices = tf.range(start=0, limit=max_len, delta=1)
    position_emb = layers.Embedding(max_len, embed_dim, name='position_embedding')(position_indices)
    position_emb = layers.Lambda(lambda x: tf.expand_dims(x, 0))(position_emb)
    # Tile to match batch size of token_emb: from (1, max_len, embed_dim) -> (batch, max_len, embed_dim)
    position_emb = layers.Lambda(lambda args: tf.tile(args[0], [tf.shape(args[1])[0], 1, 1]))([position_emb, token_emb])
    print(f"    [BERT Branch] Position embedding shape (broadcasted): {position_emb.shape}")

    print(f"    [BERT Branch] Step 1c: Add Token + Position embeddings...")
    x = layers.Add()([token_emb, position_emb])
    print(f"    [BERT Branch] Combined embedding shape: {x.shape}")
    
    # Step 2: Transformer blocks
    print(f"    [BERT Branch] Step 2: Processing through {num_layers} Transformer blocks...")
    for i in range(num_layers):
        print(f"    [BERT Branch] Transformer Block {i+1}/{num_layers}:")
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
    # Step 3: Final projection to match CNN branch
    print(f"    [BERT Branch] Step 3: Final dense layer...")
    x = layers.Dense(80, activation="relu")(x)
    print(f"    [BERT Branch] Final BERT output shape: {x.shape}")
    return x

def build_crispr_bert_model(vocab_size, max_len, small_debug=False):
    print(f"\n🏗️  [Model Builder] Building CRISPR-BERT model...")
    print(f"    [Model Builder] Vocabulary size: {vocab_size}")
    print(f"    [Model Builder] Max sequence length: {max_len}")
    print(f"    [Model Builder] Small debug mode: {small_debug}")
    
    print(f"\n📥 [Input Layer] Creating input layers...")
    inp_tok = layers.Input(shape=(max_len,), dtype=tf.int32, name="token_input")
    inp_hot = layers.Input(shape=(max_len,7), dtype=tf.float32, name="onehot_input")
    print(f"    [Input Layer] Token input shape: {inp_tok.shape}")
    print(f"    [Input Layer] One-hot input shape: {inp_hot.shape}")

    print(f"\n🔄 [Branch Processing] Processing both branches...")
    x_cnn = inception_cnn_branch(inp_hot, max_len=max_len)
    x_bert = bert_branch(inp_tok, vocab_size, max_len, small_debug=small_debug)

    print(f"\n🔄 [GRU Processing] Custom Bidirectional GRU layers...")

    # Custom GRU cell and Bidirectional implementation (mirrors model.py)
    class CustomGRUCell(layers.Layer):
        """Custom GRU cell with update/reset gates and candidate hidden state"""
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.state_size = units
        def build(self, input_shape):
            input_dim = input_shape[-1]
            self.W_z = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_z')
            self.U_z = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_z')
            self.b_z = self.add_weight(shape=(self.units,), initializer='zeros', name='b_z')
            self.W_r = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_r')
            self.U_r = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_r')
            self.b_r = self.add_weight(shape=(self.units,), initializer='zeros', name='b_r')
            self.W_h = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_h')
            self.U_h = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_h')
            self.b_h = self.add_weight(shape=(self.units,), initializer='zeros', name='b_h')
            super().build(input_shape)
        def call(self, inputs, states):
            h_prev = states[0]
            z = tf.sigmoid(tf.matmul(inputs, self.W_z) + tf.matmul(h_prev, self.U_z) + self.b_z)
            r = tf.sigmoid(tf.matmul(inputs, self.W_r) + tf.matmul(h_prev, self.U_r) + self.b_r)
            h_tilde = tf.tanh(tf.matmul(inputs, self.W_h) + tf.matmul(r * h_prev, self.U_h) + self.b_h)
            h = (1 - z) * h_prev + z * h_tilde
            return h, [h]
        def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
            # Gracefully handle dynamic batch size and dtype provided by Keras RNN
            if inputs is not None:
                inferred_batch = tf.shape(inputs)[0]
                inferred_dtype = inputs.dtype
            else:
                inferred_batch = 0 if batch_size is None else batch_size
                inferred_dtype = tf.float32 if dtype is None else dtype
            bsz = batch_size if batch_size is not None else inferred_batch
            dt = dtype if dtype is not None else inferred_dtype
            return [tf.zeros((bsz, self.units), dtype=dt)]

    class CustomBiGRU(layers.Layer):
        """Bidirectional GRU using the custom GRU cell"""
        def __init__(self, units, return_sequences=False, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.return_sequences = return_sequences
        def build(self, input_shape):
            self.forward_gru = layers.RNN(
                CustomGRUCell(self.units),
                return_sequences=self.return_sequences,
                return_state=False,
                go_backwards=False,
                name='forward_gru'
            )
            self.backward_gru = layers.RNN(
                CustomGRUCell(self.units),
                return_sequences=self.return_sequences,
                return_state=False,
                go_backwards=True,
                name='backward_gru'
            )
            super().build(input_shape)
        def call(self, inputs, training=False, mask=None):
            forward_output = self.forward_gru(inputs, training=training, mask=mask)
            backward_output = self.backward_gru(inputs, training=training, mask=mask)
            output = tf.concat([forward_output, backward_output], axis=-1)
            return output

        def compute_output_shape(self, input_shape):
            # input_shape: (batch, timesteps, features)
            batch = input_shape[0]
            time = input_shape[1]
            feat = 2 * self.units
            if self.return_sequences:
                return (batch, time, feat)
            else:
                return (batch, feat)

    print(f"    [GRU] CNN branch GRU: 40 units per direction (custom BiGRU)...")
    x_cnn = CustomBiGRU(40, return_sequences=False)(x_cnn)
    print(f"    [GRU] CNN branch GRU output shape: {x_cnn.shape}")

    print(f"    [GRU] BERT branch GRU: 40 units per direction (custom BiGRU)...")
    x_bert = CustomBiGRU(40, return_sequences=False)(x_bert)
    print(f"    [GRU] BERT branch GRU output shape: {x_bert.shape}")

    print(f"\n🔗 [Fusion Layer] Combining branches with weighted fusion...")
    print(f"    [Fusion] CNN weight: 0.2, BERT weight: 0.8")
    merged = layers.Add()([layers.Lambda(lambda z: 0.2*z)(x_cnn),
                           layers.Lambda(lambda z: 0.8*z)(x_bert)])
    print(f"    [Fusion] Merged output shape: {merged.shape}")

    print(f"\n🎯 [Classification Head] Building classification layers...")
    print(f"    [Classification] Dense layer 1: 128 units, ReLU activation...")
    x = layers.Dense(128, activation="relu")(merged)
    print(f"    [Classification] Dense layer 1 output shape: {x.shape}")
    
    print(f"    [Classification] Dense layer 2: 64 units, ReLU activation...")
    x = layers.Dense(64, activation="relu")(x)
    print(f"    [Classification] Dense layer 2 output shape: {x.shape}")
    
    print(f"    [Classification] Dropout layer: 0.35 rate...")
    x = layers.Dropout(0.35)(x)
    print(f"    [Classification] Dropout output shape: {x.shape}")
    
    print(f"    [Classification] Final output layer: 2 units, softmax activation...")
    out = layers.Dense(2, activation="softmax")(x)
    print(f"    [Classification] Final output shape: {out.shape}")

    print(f"\n✅ [Model Complete] Model built successfully!")
    return Model([inp_tok, inp_hot], out)

def predict_single_example(model, token_input, onehot_input, label=None):
    """
    Predict on a single example with simplified layer-by-layer analysis
    """
    print(f"\n" + "="*80)
    print(f"🔍 LAYER-BY-LAYER ANALYSIS - SIMPLIFIED VIEW")
    print(f"="*80)
    
    print(f"\n📊 [INPUTS]")
    print(f"    Token input shape: {token_input.shape}")
    print(f"    Token sequence: {token_input[0]}")
    print(f"    One-hot input shape: {onehot_input.shape}")
    if label is not None:
        print(f"    True label: {label}")
    
    # Create intermediate models to extract outputs at key stages
    print(f"\n" + "="*80)
    print(f"STAGE-BY-STAGE PROCESSING")
    print(f"="*80)
    
    # Stage 1: Input & Tokenization
    print(f"\n📥 [STAGE 1: INPUT & TOKENIZATION]")
    print(f"    Token Input: {token_input.shape} → {token_input[0]}")
    print(f"    One-hot Input: {onehot_input.shape}")
    
    # Stage 2: Embeddings (BERT branch)
    print(f"\n🔤 [STAGE 2: EMBEDDINGS - BERT Branch]")
    try:
        token_emb_layer = model.get_layer('token_embedding')
        token_emb_model = Model(inputs=model.input[0], outputs=token_emb_layer.output)
        token_emb = token_emb_model.predict(token_input, verbose=0)
        print(f"    Token Embedding Output Shape: {token_emb.shape}")
        print(f"    Complete Token Embedding Output:")
        print(f"{token_emb[0]}")
        
        # Position embedding
        pos_emb_layer = model.get_layer('position_embedding')
        print(f"\n    Position Embedding: (26, 128) - learned positional encodings")
        
        # Combined embedding (after Add layer)
        add_layer = model.get_layer('add')
        add_model = Model(inputs=model.inputs, outputs=add_layer.output)
        combined_emb = add_model.predict([token_input, onehot_input], verbose=0)
        print(f"\n    Combined Embedding (Token + Position) Output Shape: {combined_emb.shape}")
        print(f"    Complete Combined Embedding Output:")
        print(f"{combined_emb[0]}")
    except Exception as e:
        print(f"    Embedding analysis error: {e}")
    
    # Stage 3: CNN Branch
    print(f"\n🔬 [STAGE 3: CNN BRANCH]")
    try:
        # Find the reshape layer after CNN processing
        for layer in model.layers:
            if 'reshape' in layer.name.lower() and hasattr(layer, 'output'):
                cnn_model = Model(inputs=model.inputs, outputs=layer.output)
                cnn_output = cnn_model.predict([token_input, onehot_input], verbose=0)
                print(f"    CNN Branch Output Shape: {cnn_output.shape}")
                print(f"    Complete CNN Branch Output:")
                print(f"{cnn_output[0]}")
                break
    except Exception as e:
        print(f"    CNN analysis error: {e}")
    
    # Stage 4: BERT Transformers
    print(f"\n🤖 [STAGE 4: BERT TRANSFORMERS]")
    try:
        # Find the last dense layer in BERT branch (before GRU)
        for layer in model.layers:
            if layer.name.startswith('dense') and 'relu' in str(layer.activation):
                # Check if this is the BERT final projection (80 units)
                if hasattr(layer, 'units') and layer.units == 80:
                    bert_model = Model(inputs=model.inputs, outputs=layer.output)
                    bert_output = bert_model.predict([token_input, onehot_input], verbose=0)
                    print(f"    BERT Branch Output Shape (after Transformers): {bert_output.shape}")
                    print(f"    Complete BERT Branch Output:")
                    print(f"{bert_output[0]}")
                    break
    except Exception as e:
        print(f"    BERT analysis error: {e}")
    
    # Stage 5: BiGRU Processing
    print(f"\n🔄 [STAGE 5: BiGRU PROCESSING]")
    try:
        # Find BiGRU layers
        bigru_layers = [l for l in model.layers if 'custom_bi_gru' in l.name.lower()]
        if len(bigru_layers) >= 2:
            # CNN BiGRU
            cnn_gru_model = Model(inputs=model.inputs, outputs=bigru_layers[0].output)
            cnn_gru_out = cnn_gru_model.predict([token_input, onehot_input], verbose=0)
            print(f"    CNN BiGRU Output Shape: {cnn_gru_out.shape} (40 units × 2 directions = 80)")
            print(f"    Complete CNN BiGRU Output:")
            print(f"{cnn_gru_out[0]}")
            
            # BERT BiGRU
            bert_gru_model = Model(inputs=model.inputs, outputs=bigru_layers[1].output)
            bert_gru_out = bert_gru_model.predict([token_input, onehot_input], verbose=0)
            print(f"\n    BERT BiGRU Output Shape: {bert_gru_out.shape} (40 units × 2 directions = 80)")
            print(f"    Complete BERT BiGRU Output:")
            print(f"{bert_gru_out[0]}")
    except Exception as e:
        print(f"    BiGRU analysis error: {e}")
    
    # Stage 6: Fusion
    print(f"\n🔗 [STAGE 6: FUSION (Weighted Combination)]")
    try:
        # Find the Add layer for fusion
        add_layers = [l for l in model.layers if l.name.startswith('add') and l != add_layer]
        if add_layers:
            fusion_model = Model(inputs=model.inputs, outputs=add_layers[0].output)
            fusion_out = fusion_model.predict([token_input, onehot_input], verbose=0)
            print(f"    Fusion Layer Output Shape (0.2×CNN + 0.8×BERT): {fusion_out.shape}")
            print(f"    Complete Fusion Layer Output:")
            print(f"{fusion_out[0]}")
    except Exception as e:
        print(f"    Fusion analysis error: {e}")
    
    # Stage 7: Dense Layers
    print(f"\n🎯 [STAGE 7: CLASSIFICATION HEAD]")
    try:
        # Find dense layers in classification head
        dense_layers = [l for l in model.layers if l.name.startswith('dense')]
        # Skip the BERT projection layer (80 units), get classification layers
        classification_dense = [l for l in dense_layers if hasattr(l, 'units') and l.units in [128, 64, 2]]
        
        if len(classification_dense) >= 1:
            dense1_model = Model(inputs=model.inputs, outputs=classification_dense[0].output)
            dense1_out = dense1_model.predict([token_input, onehot_input], verbose=0)
            print(f"    Dense Layer 1 Output Shape (128 units, ReLU): {dense1_out.shape}")
            print(f"    Complete Dense Layer 1 Output:")
            print(f"{dense1_out[0]}")
        
        if len(classification_dense) >= 2:
            dense2_model = Model(inputs=model.inputs, outputs=classification_dense[1].output)
            dense2_out = dense2_model.predict([token_input, onehot_input], verbose=0)
            print(f"\n    Dense Layer 2 Output Shape (64 units, ReLU): {dense2_out.shape}")
            print(f"    Complete Dense Layer 2 Output:")
            print(f"{dense2_out[0]}")
        
        # Dropout
        dropout_layers = [l for l in model.layers if 'dropout' in l.name.lower()]
        if dropout_layers:
            print(f"\n    Dropout (rate=0.35): Applied")
    except Exception as e:
        print(f"    Dense layers analysis error: {e}")
    
    # Stage 8: Final Output
    print(f"\n🎲 [STAGE 8: FINAL OUTPUT]")
    prediction = model.predict([token_input, onehot_input], verbose=0)
    print(f"    Final Output Layer Shape (2 units, Softmax): {prediction.shape}")
    print(f"    Complete Final Output (Probabilities):")
    print(f"{prediction[0]}")
    print(f"\n    Predicted class: {np.argmax(prediction[0])} ({'Off-target' if np.argmax(prediction[0]) == 0 else 'On-target'})")
    print(f"    Confidence: {np.max(prediction[0]):.4f}")
    
    if label is not None:
        correct = "✅ CORRECT" if np.argmax(prediction[0]) == label else "❌ INCORRECT"
        print(f"    Ground truth: {label} ({'Off-target' if label == 0 else 'On-target'})")
        print(f"    Result: {correct}")
    
    print(f"\n" + "="*80)
    return prediction
