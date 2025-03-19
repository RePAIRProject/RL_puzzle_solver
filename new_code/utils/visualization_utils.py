import numpy as np 

def crop_to_content(image:np.ndarray, padding:int=1, return_vals:bool=False, max_noise:int=0):

    if len(image.shape) > 2:
        x0 = np.clip(np.min(np.where(np.sum(image, axis=2) > max_noise)[1]) - padding, 0, image.shape[1])
        x1 = np.clip(np.max(np.where(np.sum(image, axis=2) > max_noise)[1]) + padding, 0, image.shape[1])
        y0 = np.clip(np.min(np.where(np.sum(image, axis=2) > max_noise)[0]) - padding, 0, image.shape[0])
        y1 = np.clip(np.max(np.where(np.sum(image, axis=2) > max_noise)[0]) + padding, 0, image.shape[0])
    else:
        x0 = np.min(np.where(image > max_noise)[1]) - padding
        x1 = np.max(np.where(image > max_noise)[1]) + padding
        y0 = np.min(np.where(image > max_noise)[0]) - padding
        y1 = np.max(np.where(image > max_noise)[0]) + padding

    if return_vals == True:
        return image[y0:y1, x0:x1, :], x0, x1, y0, y1
    return image[y0:y1, x0:x1, :]