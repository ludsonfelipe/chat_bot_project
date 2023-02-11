import tensorflow as tf

class Seq2SeqModel:
    def __init__(self, vocab_size, embedding_size, hidden_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
    def create_model(self):
        # Define the encoder
        encoder_inputs = tf.keras.layers.Input(shape=(None,))
        encoder_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size)(encoder_inputs)
        encoder_lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Define the decoder
        decoder_inputs = tf.keras.layers.Input(shape=(None,))
        decoder_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size)(decoder_inputs)
        decoder_lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(units=self.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model
        model = tf.keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model