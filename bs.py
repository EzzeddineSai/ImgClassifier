from PIL import Image

# Open the image file
im = Image.open('./test_data/frog.jpg')

# Resize the image to 32x32 pixels
img = im.resize((32, 32))

# Save the resized image
img.save('./test_data/resized_image.jpg', 'JPEG', quality=100)