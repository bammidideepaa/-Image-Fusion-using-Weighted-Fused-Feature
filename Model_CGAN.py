from keras import layers, Model
from Evaluation_Imagefusion import evaluation


def transformer_block(x, num_heads=4, ff_dim=256):
    """Transformer Encoder Block"""
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn_output = layers.Add()([x, attn_output])
    attn_output = layers.LayerNormalization()(attn_output)

    ffn_output = layers.Dense(ff_dim, activation='relu')(attn_output)
    ffn_output = layers.Dense(x.shape[-1])(ffn_output)
    ffn_output = layers.Add()([attn_output, ffn_output])
    return layers.LayerNormalization()(ffn_output)


def generator(input_shape):
    """Generator model using Transformer"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = transformer_block(x)
    x = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
    return Model(inputs, x, name='Generator')


def discriminator(input_shape):
    """Discriminator model"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, x, name='Discriminator')


def tcgan_fusion(ct_image, pet_image):
    """Fusion using TCGAN"""
    input_shape = ct_image.shape[1:]
    gener = generator(input_shape)
    discriminators = discriminator(input_shape)

    fused_image = gener(ct_image)
    discriminator.trainable = False
    validity = discriminators(fused_image)

    fusion_model = Model(inputs=[ct_image, pet_image], outputs=[fused_image, validity], name='TCGAN')
    return fusion_model, fused_image


def Model_CGAN(Image1, Image2):
    model, fused_Image = tcgan_fusion(Image1, Image2)
    Eval = evaluation(Image1, Image2, fused_Image)

    return Eval, fused_Image
