import numpy as np
from keras import backend as k
from Evaluation_Imagefusion import evaluation
from tensorflow.keras import layers, Model


def dilated_res_block(x, filters, dilation_rate):
    res = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    res = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=dilation_rate)(res)
    res = layers.Add()([x, res])
    res = layers.ReLU()(res)
    return res


def encoder(Image1):
    input_shape = Image1.shape[1], Image1.shape[2], Image1.shape[3]
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = dilated_res_block(x, 64, 2)
    x = dilated_res_block(x, 64, 4)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    model = Model(inputs, x, name='Encoder')
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [k.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 4
    Feats = []
    for i in range(Image1.shape[0]):
        test = Image1[:][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)

    return Feats


def decoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)
    model = Model(inputs, x, name='Decoder')
    return model


def image_fusion(ct_image, pet_image, sol):
    input_shape = ct_image.shape[1:]
    CTFeat = encoder(ct_image)
    PETFeat = encoder(pet_image)
    dec = decoder((input_shape[0], input_shape[1], 128))

    Feat = (CTFeat + CTFeat * (sol[0] / 100)) + (PETFeat + PETFeat * (sol[1] / 100))
    fused_image = dec(Feat)

    fusion_model = Model(inputs=[ct_image, pet_image], outputs=fused_image, name='FusionModel')
    return fusion_model, fused_image


def Model_RANet(CT_Image, PET_Image, sol=None):
    if sol is None:
        sol = [-56, 10]
    model, fused_Image = image_fusion(CT_Image, PET_Image, sol)
    Eval = evaluation(CT_Image, PET_Image, fused_Image)

    return Eval, fused_Image
