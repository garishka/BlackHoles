from PIL import Image, ImageDraw, ImageFilter

width = 400  # Width of the image
height = 400  # Height of the image
square_size = 40  # Size of each square in pixels
radius = 10  # Adjust this value to control the level of blur

# Create a new blank image with a white background
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

for x in range(0, width, square_size):
    for y in range(0, height, square_size):
        if (x // square_size + y // square_size) % 2 == 0:
            color = (0, 0, 0)  # Black
        else:
            color = (255, 255, 255)  # White

        draw.rectangle([x, y, x + square_size, y + square_size], fill=color)

blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
blurred_image.save("blurred_image.png")

image = Image.open("blurred_image.png")
pixel_data = list(image.getdata())
print(pixel_data)

# Create a new blank image with a white background
image_circles = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image_circles)

# Define the properties of the circles (e.g., radius, spacing, and colors)
circle_radius = 5
circle_spacing = 30

# Define a list of different colors for the circles
circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Create a pattern of circles by drawing circles at regular intervals
color_index = 0
for x in range(0, width, circle_spacing):
    for y in range(0, height, circle_spacing):
        center = (x + circle_radius, y + circle_radius)
        color = circle_colors[color_index % len(circle_colors)]
        draw.ellipse((x, y, x + circle_radius * 2, y + circle_radius * 2), fill=color, outline=color)
        color_index += 1

image_circles = image_circles.filter(ImageFilter.GaussianBlur(radius/4))

# Save the patterned circle image to a file
image_circles.save("patterned_circles.png")
image.show()
