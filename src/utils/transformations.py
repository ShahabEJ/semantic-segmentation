import albumentations as A
from albumentations.pytorch import ToTensorV2
from .config_loader import Config

def get_transformations(stage='train'):

    transform_list = []
    config = Config.get_instance().config
    augmentations = config['transformations'][stage]

    if 'resize' in augmentations:
        transform_list.append(A.Resize(*augmentations['resize']))
    
    if stage == 'train': 
        if augmentations.get('horizontal_flip', False):
            transform_list.append(A.HorizontalFlip())
        if augmentations.get('vertical_flip', False):
            transform_list.append(A.VerticalFlip())
        if 'rotation_degrees' in augmentations:
            transform_list.append(A.Rotate(limit=augmentations['rotation_degrees']))

    # This is based on natural images. For remote sensing images it may not work.    
    transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform_list.append(ToTensorV2())

    return A.Compose(transform_list)
