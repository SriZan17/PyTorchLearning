import torchvision
import torch
import pathlib
import matplotlib.pyplot as plt

custom_image_path = pathlib.Path("example.jpg")
# Load in custom image and convert the tensor values to float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1] if necessary
custom_image = custom_image / 255.0

# Print out image data
print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")

plt.imshow(custom_image.permute(1, 2, 0))
plt.show()
