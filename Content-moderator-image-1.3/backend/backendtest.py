import base64
import ollama

MODEL_NAME = "cyber-moderator-G3:4b"

def moderate_image(image_path):
    try:
        # Read image and convert to base64
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Generate response using the model
        response = ollama.generate(
            model=MODEL_NAME,
            prompt="Analyze this image for harmful content.",
            images=[image_base64]  # Important: Send images here!
        )

        return response["response"]  # Extract the JSON response
    except Exception as e:
        return {"error": str(e)}

# Test with an image
image_path = "/home/harish/Project_works/Content-moderator-image/test_vision_model/t4.jpg"
output = moderate_image(image_path)
print(output)
