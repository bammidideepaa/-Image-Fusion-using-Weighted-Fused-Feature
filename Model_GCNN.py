import numpy as np
import cv2
from keras import layers, Model
from Evaluation_Imagefusion import evaluation


def gabor_filter_bank(ksize=31, sigma=4.0, lambd=10.0, gamma=0.5, num_filters=8):
    """Create a bank of Gabor filters with different orientations."""
    filters = []
    for theta in np.linspace(0, np.pi, num_filters, endpoint=False):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filters.append(kernel)
    return np.array(filters)


def apply_gabor_filters(image, filters):
    """Apply Gabor filters to an image and stack responses."""
    responses = [cv2.filter2D(image, cv2.CV_32F, kernel) for kernel in filters]
    return np.stack(responses, axis=-1)


def multi_cnn(input_shape):
    """Multi-scale CNN feature extractor."""
    inputs = layers.Input(shape=input_shape)
    x1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs)
    x3 = layers.Conv2D(128, (7, 7), padding='same', activation='relu')(inputs)
    x = layers.concatenate([x1, x2, x3])
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    return Model(inputs, x, name='Multi-CNN')


def gcnn_fusion(ct_image, pet_image):
    """Fusion using Gabor-CNN representation."""
    input_shape = ct_image.shape[1:]
    gabor_filters = gabor_filter_bank()

    ct_gabor = apply_gabor_filters(ct_image.numpy().squeeze(), gabor_filters)
    pet_gabor = apply_gabor_filters(pet_image.numpy().squeeze(), gabor_filters)

    ct_gabor = np.expand_dims(ct_gabor, axis=0)  # Add batch dimension
    pet_gabor = np.expand_dims(pet_gabor, axis=0)

    feature_extractor = multi_cnn(input_shape)
    ct_features = feature_extractor(ct_gabor)
    pet_features = feature_extractor(pet_gabor)

    fused_features = layers.Add()([ct_features, pet_features])
    fused_image = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(fused_features)

    fusion_model = Model(inputs=[ct_image, pet_image], outputs=fused_image, name='G-CNN_Fusion')
    return fusion_model, fused_image


def Model_GCNN(Image1, Image2):
    model, fused_Image = gcnn_fusion(Image1, Image2)
    Eval = evaluation(Image1, Image2, fused_Image)

    return Eval, fused_Image

