# model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------- Transformer building blocks ----------
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, x, training=False, mask=None):
        return self.att(x, x, x, attention_mask=mask, training=training)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn = self.att(x, training=training, mask=mask)
        out1 = self.norm1(x + self.drop1(attn, training=training))
        ffn = self.ffn(out1)
        out2 = self.norm2(out1 + self.drop2(ffn, training=training))
        return out2

# ---------- Model parts ----------
def CNN_branch(inputs, max_len=26):
    """
    Inception-like CNN branch using Conv2D over the one-hot input (max_len x 7).
    Steps:
    - Expand channel dimension to make input 4D.
    - Apply parallel Conv2D with kernel sizes (1,1), (2,2), (3,3), (5,5) and filter counts 5, 15, 25, 35.
    - Concatenate along channels to get 80 feature maps.
    - Collapse the width dimension (7) via max-reduction to produce (max_len, 80) for BiGRU input.
    """
    # inputs: (batch, max_len, 7)
    x = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(inputs)  # (batch, max_len, 7, 1)

    conv1 = layers.Conv2D(5, (1, 1), padding='same', activation='relu')(x)
    conv2 = layers.Conv2D(15, (2, 2), padding='same', activation='relu')(x)
    conv3 = layers.Conv2D(25, (3, 3), padding='same', activation='relu')(x)
    conv4 = layers.Conv2D(35, (5, 5), padding='same', activation='relu')(x)

    merged = layers.Concatenate(axis=-1)([conv1, conv2, conv3, conv4])  # (batch, max_len, 7, 80)

    # Collapse width dimension (the 7 features) to match desired shape (max_len, 80)
    collapsed = layers.Lambda(lambda t: tf.reduce_max(t, axis=2))(merged)  # (batch, max_len, 80)

    # Optional explicit reshape for clarity (no-op if shapes already align)
    cnn_out = layers.Reshape((max_len, 80))(collapsed)
    return cnn_out

def bert_branch(inp, vocab_size, max_len, small_debug=False):
    embed_dim = 768 if not small_debug else 128
    num_heads = 12 if not small_debug else 4
    ff_dim = 3072 if not small_debug else 256
    num_layers = 12 if not small_debug else 2

    # Token embedding
    token_emb = layers.Embedding(vocab_size, embed_dim, name='token_embedding')(inp)
    
    # Position embedding (0 to max_len-1)
    position_emb = layers.Embedding(max_len, embed_dim, name='position_embedding')(
        tf.range(start=0, limit=max_len, delta=1)
    )
    # Expand position embedding to match batch size
    position_emb = layers.Lambda(lambda x: tf.expand_dims(x, 0))(position_emb)
    
    # Segment embedding: [CLS]=0, sequence pairs=1, [SEP] and padding=0
    # Create segment IDs: position 0 is CLS (segment 0), positions 1 to max_len-2 are sequence (segment 1),
    # position max_len-1 is SEP (segment 0)
    def create_segment_ids(max_len):
        # Segment pattern: [0, 1, 1, 1, ..., 1, 0] where middle positions are segment 1
        segment_ids = tf.concat([
            tf.zeros([1], dtype=tf.int32),  # CLS token
            tf.ones([max_len - 2], dtype=tf.int32),  # Sequence pairs
            tf.zeros([1], dtype=tf.int32)  # SEP token
        ], axis=0)
        return segment_ids
    
    segment_input = layers.Lambda(lambda x: create_segment_ids(max_len))(inp)
    segment_emb = layers.Embedding(2, embed_dim, name='segment_embedding')(segment_input)
    
    # Add all three embeddings (Token + Position + Segment)
    x = layers.Add()([token_emb, position_emb, segment_emb])
    
    # Pass through transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.Dense(80, activation="relu")(x)
    return x

def build_crispr_bert_model(vocab_size, max_len, small_debug=False):
    inp_tok = layers.Input(shape=(max_len,), dtype=tf.int32)
    inp_hot = layers.Input(shape=(max_len,7), dtype=tf.float32)

    x_cnn = CNN_branch(inp_hot, max_len=max_len)
    x_bert = bert_branch(inp_tok, vocab_size, max_len, small_debug=small_debug)

    # ========== BiGRU Layer (Keras built-in) - COMMENTED OUT ==========
    # x_cnn = layers.Bidirectional(layers.GRU(40, return_sequences=False))(x_cnn)
    # x_bert = layers.Bidirectional(layers.GRU(40, return_sequences=False))(x_bert)
    
    # ========== Custom BiGRU Implementation from Scratch ==========
    # Custom GRU Cell
    class CustomGRUCell(layers.Layer):
        """Custom GRU cell with update gate, reset gate, and candidate hidden state"""
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.state_size = units
        
        def build(self, input_shape):
            input_dim = input_shape[-1]
            
            # Update gate weights: z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)
            self.W_z = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_z')
            self.U_z = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_z')
            self.b_z = self.add_weight(shape=(self.units,), initializer='zeros', name='b_z')
            
            # Reset gate weights: r_t = sigmoid(W_r * x_t + U_r * h_{t-1} + b_r)
            self.W_r = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_r')
            self.U_r = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_r')
            self.b_r = self.add_weight(shape=(self.units,), initializer='zeros', name='b_r')
            
            # Candidate hidden state weights: h_tilde = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)
            self.W_h = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_h')
            self.U_h = self.add_weight(shape=(self.units, self.units), initializer='orthogonal', name='U_h')
            self.b_h = self.add_weight(shape=(self.units,), initializer='zeros', name='b_h')
            
            super().build(input_shape)
        
        def call(self, inputs, states):
            h_prev = states[0]
            
            # Update gate
            z = tf.sigmoid(tf.matmul(inputs, self.W_z) + tf.matmul(h_prev, self.U_z) + self.b_z)
            
            # Reset gate
            r = tf.sigmoid(tf.matmul(inputs, self.W_r) + tf.matmul(h_prev, self.U_r) + self.b_r)
            
            # Candidate hidden state
            h_tilde = tf.tanh(tf.matmul(inputs, self.W_h) + tf.matmul(r * h_prev, self.U_h) + self.b_h)
            
            # Final hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde
            h = (1 - z) * h_prev + z * h_tilde
            
            return h, [h]
        
        def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
            return [tf.zeros((batch_size, self.units), dtype=dtype)]
    
    # Custom Bidirectional GRU
    class CustomBiGRU(layers.Layer):
        """Bidirectional GRU: processes sequence forward and backward, then concatenates"""
        def __init__(self, units, return_sequences=False, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.return_sequences = return_sequences
        
        def build(self, input_shape):
            # Forward GRU (left to right)
            self.forward_gru = layers.RNN(
                CustomGRUCell(self.units),
                return_sequences=self.return_sequences,
                return_state=False,
                go_backwards=False,
                name='forward_gru'
            )
            
            # Backward GRU (right to left)
            self.backward_gru = layers.RNN(
                CustomGRUCell(self.units),
                return_sequences=self.return_sequences,
                return_state=False,
                go_backwards=True,
                name='backward_gru'
            )
            
            super().build(input_shape)
        
        def call(self, inputs, training=False, mask=None):
            # Forward pass
            forward_output = self.forward_gru(inputs, training=training, mask=mask)
            
            # Backward pass
            backward_output = self.backward_gru(inputs, training=training, mask=mask)
            
            # Concatenate: output dimension = 2 * units
            output = tf.concat([forward_output, backward_output], axis=-1)
            
            return output
    
    # Apply custom BiGRU to both branches
    x_cnn = CustomBiGRU(40, return_sequences=False)(x_cnn)
    x_bert = CustomBiGRU(40, return_sequences=False)(x_bert)

    merged = layers.Add()([layers.Lambda(lambda z: 0.2*z)(x_cnn),
                           layers.Lambda(lambda z: 0.8*z)(x_bert)])

    x = layers.Dense(128, activation="relu")(merged)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(2, activation="softmax")(x)

    return Model([inp_tok, inp_hot], out)

