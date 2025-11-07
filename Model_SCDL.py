import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from Evaluation_Imagefusion import evaluation


def extract_patches(image, patch_size=8, stride=4):
    """Extract overlapping patches from an image."""
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, stride):
        for j in range(0, image.shape[1] - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size].flatten()
            patches.append(patch)
    return np.array(patches)


def reconstruct_image(patches, image_shape, patch_size=8, stride=4):
    """Reconstruct image from patches using averaging."""
    reconstructed = np.zeros(image_shape)
    count = np.zeros(image_shape)
    idx = 0
    for i in range(0, image_shape[0] - patch_size + 1, stride):
        for j in range(0, image_shape[1] - patch_size + 1, stride):
            reconstructed[i:i + patch_size, j:j + patch_size] += patches[idx].reshape((patch_size, patch_size))
            count[i:i + patch_size, j:j + patch_size] += 1
            idx += 1
    return reconstructed / np.maximum(count, 1)


def Model_SCDL(ct_image, pet_image, dict_size=256, alpha=0.1):
    """Perform Simultaneous Coupled Dictionary Learning for image fusion."""
    patch_size = 8
    stride = 4

    ct_patches = extract_patches(ct_image, patch_size, stride)
    pet_patches = extract_patches(pet_image, patch_size, stride)

    combined_patches = np.vstack((ct_patches, pet_patches))

    dict_learner = DictionaryLearning(n_components=dict_size, alpha=alpha, transform_algorithm='lasso')
    dictionary = dict_learner.fit(combined_patches).components_

    coder = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
    ct_coeffs = coder.fit(dictionary.T, ct_patches.T).coef_.T
    pet_coeffs = coder.fit(dictionary.T, pet_patches.T).coef_.T

    fused_coeffs = 0.5 * ct_coeffs + 0.5 * pet_coeffs
    fused_patches = np.dot(fused_coeffs, dictionary)

    fused_image = reconstruct_image(fused_patches, ct_image.shape, patch_size, stride)
    Eval = evaluation(ct_image, pet_image, fused_image)

    return Eval, fused_image
