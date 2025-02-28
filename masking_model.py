import sys
import os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
# Uncomment the following line if you prefer the default font:
# FONT = ImageFont.load_default()

GRID_SIZE = 40
PIXELS_PER_WORD = 200

# Set the folder to save generated images
IMAGE_FOLDER = os.path.join("static", "generated-images")

def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    token_ids = inputs["input_ids"][0]
    for index, token in enumerate(token_ids):
        if token == mask_token_id:
            return index
    return None

def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value is in the range [0, 255].
    """
    score = float(attention_score.numpy())
    gray_value = score * 255
    gray_value = int(round(gray_value))
    return (gray_value, gray_value, gray_value)

def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each token,
    with cells shaded based on attention_weights. The diagram is saved
    with a filename that includes the layer and head numbers.
    """
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns (rotated text)
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw attention grid cells
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Ensure the static/generated-images folder exists
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    filename = os.path.join(IMAGE_FOLDER, f"Attention_Layer{layer_number}_Head{head_number}.png")
    img.save(filename)

def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.
    For each attention layer and each head within that layer, a diagram
    is generated using the generate_diagram function.
    """
    for i, layer in enumerate(attentions):
        # layer shape: [batch_size, num_heads, seq_len, seq_len]
        for j, head in enumerate(layer[0]):  # Using the first (and only) batch element
            generate_diagram(i + 1, j + 1, tokens, head)

def run_mask_prediction(input_text):
    """
    Process the input_text using the BERT masked language model.
    - Tokenizes the input text.
    - Finds the index of the [MASK] token.
    - Runs the model to predict the masked word.
    - Generates attention diagrams (saved in 'static/generated-images').
    
    Returns:
        predictions: A list of predicted texts (one for each of the top K predictions).
        image_files: A list of filenames for the generated attention diagrams.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(input_text, return_tensors="tf")
    
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        return None, []

    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions for the [MASK] token
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    predictions = []
    for token in top_tokens:
        prediction = input_text.replace(tokenizer.mask_token, tokenizer.decode([token]))
        predictions.append(prediction)

    # Convert input IDs to tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Generate and save attention diagrams
    visualize_attentions(tokens, result.attentions)

    # Get list of generated image filenames from the folder
    image_files = os.listdir(IMAGE_FOLDER)
    return predictions, image_files

def main():
    input_text = input("Text: ")
    predictions, image_files = run_mask_prediction(input_text)
    if predictions is None:
        sys.exit("Input must include mask token.")
    print("Predictions:")
    for pred in predictions:
        print(pred)
    # Not printing image filenames to terminal

if __name__ == "__main__":
    main()
