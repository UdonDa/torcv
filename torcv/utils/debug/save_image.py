import torchvision

def save_image(tensor, file_name, denorm_func=None):
    if denorm_func is not None:
        tensor = denorm_func(tensor)
    torchvision.utils.save_image(tensor, file_name)
    

