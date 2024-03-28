from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M 
from tensorflow.keras.applications import ResNet50
from functions.model.misc import summary

base_model = ResNet50 (
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="max",
)

summary(base_model, 1)