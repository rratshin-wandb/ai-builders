import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    logo = mo.image(
        src="http://wandb.cloud/static/images/Primary_gold_black_600x73.png",
        width=300,
        rounded=True,
    )
    logo
    return


@app.cell
def _():
    import marimo as mo
    import openai
    from google import genai
    from google.genai import types
    import os, io
    import requests
    from pathlib import Path

    import  weave
    from weave import Content
    from PIL import Image as PILImage
    from typing import Annotated

    from dotenv import load_dotenv
    load_dotenv()
    return PILImage, Path, genai, io, mo, os, requests, types, weave


@app.cell
def _(PILImage, Path, genai, io, os, requests, types, weave):
    class ImageGeneratorModel(weave.Model):

        model: str = "gemini-3.1-flash-image-preview"

        @weave.op
        def is_url(self, path: str) -> bool:
            """Returns True if the given string is an HTTP/HTTPS URL."""
            return path.startswith("http://") or path.startswith("https://")

        @weave.op
        def load_image(self, path: str) -> PILImage.Image:
            """
            Auto-detects whether the image is a local file or a URL and returns
            a PIL Image object either way.

            Args:
                path: Local file path or HTTP/HTTPS URL

            Returns:
                PIL Image object
            """
            if self.is_url(path):
                print(f"  Detected URL: {path}")
                response = requests.get(path)
                response.raise_for_status()
                return PILImage.open(io.BytesIO(response.content))
            else:
                print(f"  Detected local file: {path}")
                file_path = Path(path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Local image file not found: {path}")
                return PILImage.open(file_path)

        @weave.op
        def invoke(
            self,
            image_paths: list,
            prompt: str,
            output_file: str = "combined_image.png",
        ) -> dict:
            """
            Combine two or more images using Gemini Flash's native image input and output.
            Images can be local file paths or URLs — auto-detected automatically.

            Args:
                image_paths:   List of local paths or URLs to the input images  (e.g. a bedroom)
                prompt:        User instruction                       (e.g. "place the lamps in the bedroom")
                output_file:   Filename to save the generated image

            Returns:
                Dictionary containing the saved path and prompt used
            """
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

            # ------------------------------------------------------------------ #
            # Step 1 — load both images (local or URL) as PIL images              #
            # ------------------------------------------------------------------ #
            print("Log images...")
            images = []
            for image_path in image_paths:
                images.append(self.load_image(image_path))
            print("Images loaded successfully.\n")

            # ------------------------------------------------------------------ #
            # Step 2 — send both images + user prompt to Gemini in one call       #
            # Gemini Flash natively supports interleaved image+text input         #
            # and can return generated images directly in the response            #
            # ------------------------------------------------------------------ #
            print("Sending images to Gemini Flash for combination...")

            # Build parts list: start with text prompt, then add each image
            parts = [types.Part.from_text(text=prompt)]
            for image in images:
                # Convert PIL image to bytes for the new API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_bytes = img_byte_arr.getvalue()
                parts.append(
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                )

            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                )
            ]

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="MINIMAL",
                ),
                response_modalities=["IMAGE", "TEXT"],
            )

            # ------------------------------------------------------------------ #
            # Step 3 — extract the generated image from the streamed response     #
            # ------------------------------------------------------------------ #
            generated_image = None
            response_text = None

            for chunk in client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.parts is None:
                    continue
                for part in chunk.parts:
                    if part.inline_data and part.inline_data.data:
                        generated_image = part.inline_data.data  # raw bytes
                    elif part.text:
                        response_text = (response_text or "") + part.text

            if generated_image is None:
                raise ValueError("No image was returned by Gemini. Check your prompt or API access.")

            print("Image generated successfully!")
            if response_text:
                print(f"Gemini response: {response_text}")

            # ------------------------------------------------------------------ #
            # Step 4 — save the image                                             #
            # ------------------------------------------------------------------ #
            output_path = Path(output_file)
            pil_result = PILImage.open(io.BytesIO(generated_image))
            pil_result.save(output_path)
            saved_path = str(output_path.resolve())
            print(f"Image saved to: {saved_path}")

            return {
                "image":           pil_result,        # renders as an image in the Weave UI
                "saved_path":      saved_path,
                "user_prompt":     prompt,
                "image_paths":     image_paths,
                "response_text":   response_text,
            }

    return (ImageGeneratorModel,)


@app.cell
def _(ImageGeneratorModel, weave):
    
    WANDB_ENTITY = "ENTER_WANDB_ENTITY_HERE"
    WANDB_PROJECT = "InteriorDesignAgent"

    weave.init(WANDB_ENTITY + "/" + WANDB_PROJECT)

    # Mix and match freely — local paths and URLs both work
    image_paths = [
        "bedroom_1.png",
        "lamp_3.png",
        "cat.png",
    ]
    prompt = """
        You are an interior designer assisting customers to visualize how new bedside table lamps will look in their bedroom.
        Please generate a photorealistic image by combining the following photographs and images:
            - Image 1: A customer bedroom
            - Image 2: A new lamp
            - Image 3: A cat

        Instructions:
            - Please place a lamp [Image 2] on both the left and right bedside tables in the bedroom [Image 1], matching the lighting of the room and the position of the lamps currently on the bedside tables.
            - Please put the cat [Image 5] sleeping on the bench at the foot of the bed.
            - Please do not zoom or pan on the final bedroom image. Everything in the bedroom [Image 1] should remain exactly the same except for the changes requested in the instructions.
    """

    output_file = "combined_image.png"

    image_model = ImageGeneratorModel(
        model="gemini-3.1-flash-image-preview",
    )

    result = image_model.invoke(
        image_paths=image_paths,
        prompt=prompt,
        output_file=output_file,
    )

    print(f"\nDone!")
    print(f"  Saved to      : {result['saved_path']}")
    print(f"  Prompt        : {result['user_prompt']}")
    if result["response_text"]:
        print(f"  Gemini said   : {result['response_text']}")
    return image_paths, result


@app.cell
def _(image_paths, mo, result):
    tabs_content = {}
    for idx, image_path in enumerate(image_paths):
        tabs_content["Image " + str((idx + 1))] = mo.image(src=image_paths[idx], rounded=True),
    tabs_content["Final Image"] = mo.image(src=result['saved_path'], rounded=True),
    tabs = mo.ui.tabs(
        tabs_content,
        value="Final Image"
    )
    tabs
    return


if __name__ == "__main__":
    app.run()
