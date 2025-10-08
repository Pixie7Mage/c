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
    
    # Expand dims to 4D for Conv2D: (batch, time=max_len, channels=7, 1)
    x4d = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(inp)
    print(f"    [CNN Branch] Expanded to 4D for Conv2D: {x4d.shape}")

    convs = []
    kernel_sizes = [5, 15, 25, 35]
    print(f"    [CNN Branch] Using kernel sizes: {kernel_sizes}")
    
    for i, k in enumerate(kernel_sizes):
        # Compute asymmetric padding to preserve time length like padding='same' in Conv1D
        p_left = k // 2
        p_right = k - 1 - p_left
        print(f"    [CNN Branch] Conv2D {i+1}/4: kernel=(" + str(k) + ", 7), filters=80, time-padding=({p_left},{p_right}), width-padding=(0,0), activation='relu'")
        padded = layers.ZeroPadding2D(padding=((p_left, p_right), (0, 0)))(x4d)
        c2d = layers.Conv2D(filters=80, kernel_size=(k, 7), padding='valid', activation='relu')(padded)
        # c2d shape: (batch, time, 1, 80) -> squeeze width dim to get (batch, time, 80)
        c = layers.Lambda(lambda t: tf.squeeze(t, axis=2))(c2d)
        print(f"    [CNN Branch] Conv2D {i+1}/4 output shape (after squeeze): {c.shape}")
        convs.append(c)
    
    print(f"    [CNN Branch] Concatenating {len(convs)} convolution outputs...")
    x = layers.Concatenate()(convs)
    print(f"    [CNN Branch] Concatenated shape: {x.shape}")
    
    print(f"    [CNN Branch] Dense layer: 80 units with ReLU activation...")
    x = layers.Dense(80, activation="relu")(x)
    print(f"    [CNN Branch] Final CNN output shape: {x.shape}")
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
    
    print(f"    [BERT Branch] Step 1: Token embedding...")
    x = layers.Embedding(vocab_size, embed_dim)(inp)
    print(f"    [BERT Branch] Embedding output shape: {x.shape}")
    
    print(f"    [BERT Branch] Step 2: Processing through {num_layers} Transformer blocks...")
    for i in range(num_layers):
        print(f"    [BERT Branch] Transformer Block {i+1}/{num_layers}:")
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
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

    print(f"\n🔄 [GRU Processing] Bidirectional GRU layers...")
    print(f"    [GRU] CNN branch GRU: 40 units, bidirectional...")
    x_cnn = layers.Bidirectional(layers.GRU(40, return_sequences=False))(x_cnn)
    print(f"    [GRU] CNN branch GRU output shape: {x_cnn.shape}")
    
    print(f"    [GRU] BERT branch GRU: 40 units, bidirectional...")
    x_bert = layers.Bidirectional(layers.GRU(40, return_sequences=False))(x_bert)
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
    Predict on a single example with detailed step-by-step output
    """
    print(f"\n" + "="*80)
    print(f"🔍 SINGLE EXAMPLE PREDICTION - DETAILED ANALYSIS")
    print(f"="*80)
    
    print(f"\n📊 [Input Analysis]")
    print(f"    Token input shape: {token_input.shape}")
    print(f"    One-hot input shape: {onehot_input.shape}")
    print(f"    Token sequence: {token_input[0]}")
    print(f"    One-hot matrix shape: {onehot_input[0].shape}")
    if label is not None:
        print(f"    True label: {label}")
    
    print(f"\n🚀 [Prediction Process]")
    print(f"    Running forward pass through the model...")
    
    # Full per-layer input/output tracing
    print(f"\n🧭 [Full Layer-by-Layer Trace]")
    try:
        layer_inputs = []
        layer_outputs = []
        layer_names = []
        for layer in model.layers:
            try:
                # Some layers have multiple inputs/outputs
                inp_t = layer.input
                out_t = layer.output
                layer_inputs.append(inp_t)
                layer_outputs.append(out_t)
                layer_names.append(layer.name)
            except Exception:
                # Skip layers that do not expose tensors in functional graph
                continue

        # Build a single probe model that outputs all inputs and outputs of each layer
        probe_model = Model(
            inputs=model.inputs,
            outputs=list(layer_inputs) + list(layer_outputs)
        )

        probe_values = probe_model.predict([token_input, onehot_input], verbose=0)

        n = len(layer_names)
        print(f"    Tracing {n} layers (inputs and outputs)...")

        def normalize_array(x):
            # Handle Keras returning lists/tuples for some layers
            if isinstance(x, (list, tuple)) and len(x) > 0:
                x = x[0]
            return x

        def full_preview(arr):
            arr = normalize_array(arr)
            try:
                # If there is a batch dimension of 1, unwrap it for readability
                if hasattr(arr, 'ndim') and arr.ndim >= 1 and arr.shape[0] == 1:
                    arr_to_print = arr[0]
                else:
                    arr_to_print = arr

                # Decide whether to print entire array or truncate
                max_elems = 4096  # safety cap
                total = np.prod(arr_to_print.shape) if hasattr(arr_to_print, 'shape') else 0
                if total and total <= max_elems:
                    return np.array2string(arr_to_print, precision=6, separator=", ")
                else:
                    flat = np.asarray(arr_to_print).ravel()
                    head = min(flat.size, 256)
                    return f"{np.array2string(flat[:head], precision=6, separator=', ')} ... (truncated {flat.size-head} of {flat.size})"
            except Exception:
                return str(arr)

        for i, lname in enumerate(layer_names):
            # Fetch input and output arrays for this layer
            tin = probe_values[i]
            tout = probe_values[n + i]

            # Format shapes
            tin_shape = getattr(normalize_array(tin), 'shape', None)
            tout_shape = getattr(normalize_array(tout), 'shape', None)

            print(f"\n    ➤ Layer [{i+1:02d}/{n}] {lname} ({model.get_layer(lname).__class__.__name__})")
            print(f"       - Input shape:  {tin_shape}")
            print(f"       - Output shape: {tout_shape}")
            try:
                print(f"       - Input sample:  {full_preview(tin)}")
                print(f"       - Output sample: {full_preview(tout)}")
            except Exception:
                print(f"       - Samples: <unavailable>")
    except Exception as e:
        print(f"    Full layer-by-layer trace skipped due to error: {e}")

    # Get intermediate outputs for detailed analysis
    print(f"\n🔬 [CNN Branch Analysis]")
    try:
        cnn_branch = Model(model.inputs, model.get_layer('concatenate_1').output)
        cnn_output = cnn_branch([token_input, onehot_input])
        print(f"    CNN branch output shape: {cnn_output.shape}")
        print(f"    CNN branch sample values: {cnn_output[0][:5]}")
    except:
        print(f"    CNN branch analysis skipped (layer name issue)")
    
    print(f"\n🤖 [BERT Branch Analysis]")
    try:
        # We need to get the BERT output before GRU
        bert_embedding = model.get_layer('embedding_1').output
        bert_transformer = Model(model.inputs, bert_embedding)
        bert_emb = bert_transformer([token_input, onehot_input])
        print(f"    BERT embedding shape: {bert_emb.shape}")
        print(f"    BERT embedding sample values: {bert_emb[0][0][:5]}")
    except:
        print(f"    BERT branch analysis skipped (layer name issue)")
    
    print(f"\n🔄 [GRU Processing Analysis]")
    try:
        # Get outputs after GRU layers
        cnn_gru = Model(model.inputs, model.get_layer('bidirectional_2').output)
        cnn_gru_out = cnn_gru([token_input, onehot_input])
        print(f"    CNN GRU output shape: {cnn_gru_out.shape}")
        print(f"    CNN GRU sample values: {cnn_gru_out[0][:5]}")
        
        bert_gru = Model(model.inputs, model.get_layer('bidirectional_3').output)
        bert_gru_out = bert_gru([token_input, onehot_input])
        print(f"    BERT GRU output shape: {bert_gru_out.shape}")
        print(f"    BERT GRU sample values: {bert_gru_out[0][:5]}")
        
        print(f"\n🔗 [Fusion Analysis]")
        print(f"    CNN contribution (0.2x): {cnn_gru_out[0][:5] * 0.2}")
        print(f"    BERT contribution (0.8x): {bert_gru_out[0][:5] * 0.8}")
        fusion_layer = Model(model.inputs, model.get_layer('add_1').output)
        fusion_out = fusion_layer([token_input, onehot_input])
        print(f"    Fused output shape: {fusion_out.shape}")
        print(f"    Fused sample values: {fusion_out[0][:5]}")
    except:
        print(f"    GRU and fusion analysis skipped (layer name issue)")
    
    print(f"\n🎯 [Classification Analysis]")
    try:
        dense1_layer = Model(model.inputs, model.get_layer('dense_15').output)
        dense1_out = dense1_layer([token_input, onehot_input])
        print(f"    Dense layer 1 output shape: {dense1_out.shape}")
        print(f"    Dense layer 1 sample values: {dense1_out[0][:5]}")
        
        dense2_layer = Model(model.inputs, model.get_layer('dense_16').output)
        dense2_out = dense2_layer([token_input, onehot_input])
        print(f"    Dense layer 2 output shape: {dense2_out.shape}")
        print(f"    Dense layer 2 sample values: {dense2_out[0][:5]}")
    except:
        print(f"    Classification analysis skipped (layer name issue)")
    
    print(f"\n🎲 [Final Prediction]")
    prediction = model.predict([token_input, onehot_input], verbose=0)
    print(f"    Raw prediction probabilities: {prediction[0]}")
    print(f"    Predicted class: {np.argmax(prediction[0])}")
    print(f"    Confidence: {np.max(prediction[0]):.4f}")
    
    if label is not None:
        correct = "✅ CORRECT" if np.argmax(prediction[0]) == label else "❌ INCORRECT"
        print(f"    Prediction result: {correct}")
    
    print(f"\n" + "="*80)
    return prediction
