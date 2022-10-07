# -*- coding: utf-8 -*-
"""
Created on Sat May 28 07:34:16 2022

@author: Amma
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from time import perf_counter


# Defining hyperparameters

VOCAB_SIZE = 8192
MAX_SAMPLES = 50000
BUFFER_SIZE = 20000
MAX_LENGTH = 40
EMBED_DIM = 256
LATENT_DIM = 512
NUM_HEADS = 8
BATCH_SIZE = 64


def readTxtFile (strPath):
    with open(strPath, 'r', encoding="utf-8") as file:
        return file.read().replace("\n", " ")


def dsLoad(ins=498):
    
    refTexts, hSimTexts = [],[]    
    for i in range(1, ins):
        refTexts.append(readTxtFile("../resources/Simplification_Datasets/References/{}.txt".format(i)))
        hSimTexts.append(readTxtFile("../resources/Simplification_Datasets/hSimplification/{}.txt".format(i)))
          
    return refTexts, hSimTexts


refTexts, hSimTexts = dsLoad()
train_dataset = tf.data.Dataset.from_tensor_slices((refTexts[:350], hSimTexts[:350]))
val_dataset = tf.data.Dataset.from_tensor_slices((refTexts[350:], hSimTexts[350:]))


# Preprocessing and Tokenization

def preprocess_text(sentence):
    
    sentence = tf.strings.lower(sentence)
    
    # Adding a space between the punctuation and the last word to allow better tokenization
    #sentence = tf.strings.regex_replace(sentence, r"([?.!,])", r" \1 ")  
    
    # Replacing multiple continuous spaces with a single space
    #sentence = tf.strings.regex_replace(sentence, r"\s\s+", " ")
    sentence = tf.strings.regex_replace(sentence, '([^\n\u060C-\u064A\.:؟?])', " ")
    
    sentence = tf.strings.strip(sentence)
    sentence = tf.strings.join(["[start]", sentence, "[end]"], separator=" ")
    return sentence



vectorizer = layers.TextVectorization(
    VOCAB_SIZE,
    standardize=preprocess_text,
    output_mode="int",
    output_sequence_length=MAX_LENGTH,
)


# We will adapt the vectorizer to both the questions and answers
# This dataset is batched to parallelize and speed up the process
vectorizer.adapt(tf.data.Dataset.from_tensor_slices((refTexts + hSimTexts)).batch(5))



#Tokenizing and padding sentences using TextVectorization

def vectorize_text(inputs, outputs):
    inputs, outputs = vectorizer(inputs), vectorizer(outputs)
    # One extra padding token to the right to match the output shape
    outputs = tf.pad(outputs, [[0, 1]])
    return (
        {"Encoder_Input_Embedding": inputs, "Decoder_Input_Embedding": outputs[:-1]},
        {"outputs": outputs[1:]},
    )


train_dataset = train_dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = (
    train_dataset.cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = val_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


#  FNet Encoder

class FNetEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, **kwargs):
        super(FNetEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs):
        # Casting the inputs to complex64
        inp_complex = tf.cast(inputs, tf.complex64)
        # Projecting the inputs to the frequency domain using FFT2D and
        # extracting the real part of the output
        fft = tf.math.real(tf.signal.fft2d(inp_complex))
        proj_input = self.layernorm_1(inputs + fft)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


# Creating the Decoder

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class FNetDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(FNetDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)



# Creating model 

def create_model():
    encoder_inputs = keras.Input(shape=(None,), dtype="int32", name="Encoder_Input_Embedding")
    x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(encoder_inputs)
    encoder_outputs = FNetEncoder(EMBED_DIM, LATENT_DIM)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)
    decoder_inputs = keras.Input(shape=(None,), dtype="int32", name="Decoder_Input_Embedding")
    encoded_seq_inputs = keras.Input(
        shape=(None, EMBED_DIM), name="Decoder_state_inputs"
    )
    x = PositionalEmbedding(MAX_LENGTH, VOCAB_SIZE, EMBED_DIM)(decoder_inputs)
    x = FNetDecoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x, encoded_seq_inputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
    decoder = keras.Model(
        [decoder_inputs, encoded_seq_inputs], decoder_outputs, name="outputs"
    )
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    fnet = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="fnet")
    return fnet


#  Training the model

fnet = create_model()
fnet.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
fnet.fit(train_dataset, epochs=1, validation_data=val_dataset)

fnet.summary(expand_nested=True, show_trainable=True)
dot_img_file = 'model_1.png'
tf.keras.utils.plot_model(
    fnet, 
    to_file=dot_img_file,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    layer_range=None,
    show_layer_activations=True,
    dpi=150)


# Performing inference
VOCAB = vectorizer.get_vocabulary()


def decode_sentence(input_sentence):
    # Mapping the input sentence to tokens and adding start and end tokens
    tokenized_input_sentence = vectorizer(
        tf.constant("[start] " + preprocess_text(input_sentence) + " [end]")
    )
    # Initializing the initial sentence consisting of only the start token.
    tokenized_target_sentence = tf.expand_dims(VOCAB.index("[start]"), 0)
    decoded_sentence = ""

    for i in range(MAX_LENGTH):
        # Get the predictions
        predictions = fnet.predict(
            {
                "encoder_inputs": tf.expand_dims(tokenized_input_sentence, 0),
                "decoder_inputs": tf.expand_dims(
                    tf.pad(
                        tokenized_target_sentence,
                        [[0, MAX_LENGTH - tf.shape(tokenized_target_sentence)[0]]],
                    ),
                    0,
                ),
            }
        )
        # Calculating the token with maximum probability and getting the corresponding word
        sampled_token_index = tf.argmax(predictions[0, i, :])
        sampled_token = VOCAB[sampled_token_index.numpy()]
        # If sampled token is the end token then stop generating and return the sentence
        if tf.equal(sampled_token_index, VOCAB.index("[end]")):
            break
        decoded_sentence += sampled_token + " "
        tokenized_target_sentence = tf.concat(
            [tokenized_target_sentence, [sampled_token_index]], 0
        )

    return decoded_sentence


#print ("تعتبر جده البوابه الرئيسيه لمكه المكرمه، اقدس مدينه في الاسلام", "\n", decode_sentence("تعتبر جده البوابه الرئيسيه لمكه المكرمه، اقدس مدينه في الاسلام"))
#print ("كذلك جعل الله مكان الخوف والرعب هو القلب", "\n", decode_sentence("كذلك جعل الله مكان الخوف والرعب هو القلب"))
#print ("يجب الا يشغلنا جمال منظر الكسوف عن خطر الاشعه","\n",  decode_sentence("يجب الا يشغلنا جمال منظر الكسوف عن خطر الاشعه"))
#print ("كذلك يسمي الدروز كبار رجال الدين", "\n", decode_sentence("كذلك يسمي الدروز كبار رجال الدين"))

'''
icount=0
t1_start = perf_counter()
for i in range(len(refTexts)):    
    #
    resultFile = open("../evaluation/Rephrase/Fourier/{}.txt".format(i+1), "w", encoding="utf-8")
    resultFile.write("{}".format(decode_sentence(refTexts[i])))
    resultFile.close()  
    #
    icount+=1
    print("\r", "Progress {:2.1%}".format(icount / 498), end="") 


t1_stop = perf_counter()
print ("\t Fourier\t e-time: ", (t1_stop-t1_start), )

'''


print ("Done")






