#!/usr/bin/env python3
"""
Combined Flux Meme Bot:

- Training: Users send `/train <model_key>` and then 5–10 images.
  The bot uploads images to AWS S3 and starts a training job on Replicate’s Flux LoRA trainer.
  The trained model is pushed to the destination model: "amirpraaven/meme-model-1308598820-myt".

- Generation: Users send `/meme <model_key> <prompt>`.
  The bot looks up the trigger token associated with the model key, fetches the latest version of the
  trained model from Replicate, runs prediction (with the prompt prefixed by the trigger), and sends
  the generated image back.

All credentials are loaded from a .env file.
"""

import os
import logging
import tempfile
import zipfile
import time
import requests
import openai
import replicate
import boto3

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    CallbackContext,
)

# --- Load Environment Variables ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "your_s3_bucket_name")

# Hard-coded destination model on Replicate – ensure this model exists on your Replicate dashboard.
DESTINATION_MODEL = "amirpraaven/meme-model-1308598820-myt"
# Flux trainer version must be in the format username/model_name:version_id.
FLUX_TRAINER_VERSION = os.getenv(
    "FLUX_TRAINER_VERSION",
    "ostris/flux-dev-lora-trainer:b6af14222e6bd9be257cbc1ea4afda3cd0503e1133083b9d1de0364d8568e6ef",
)

# --- Set API Tokens ---
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
openai.api_key = OPENAI_API_KEY

# --- Configure AWS S3 Client ---
s3_endpoint = f"https://s3.{AWS_DEFAULT_REGION}.amazonaws.com"
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
    endpoint_url=s3_endpoint,
)

# --- In-Memory Storage ---
# user_triggers: { user_id: { model_key: trigger } }
user_triggers = {}
# user_images: { user_id: [file_paths] }
user_images = {}

# Conversation state for training
WAITING_FOR_IMAGES = 1

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message with instructions."""
    await update.message.reply_text(
        "Welcome to the Flux Meme Bot!\n\n"
        "Commands:\n"
        "/train <model_key> - Train your custom meme model using your images\n"
        "/meme <model_key> <prompt> - Generate a meme using your trained model\n"
        "/list_models - List your trained models\n"
        "/cancel - Cancel the current operation\n\n"
        "All credentials are securely managed via the environment."
    )


async def list_models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all trained model keys and their trigger tokens."""
    user_id = update.effective_user.id
    if user_id not in user_triggers or not user_triggers[user_id]:
        await update.message.reply_text("You haven't trained any models yet.")
        return

    msg = "\n".join(
        f"- Model key: {key}, Trigger: {trigger}"
        for key, trigger in user_triggers[user_id].items()
    )
    await update.message.reply_text(f"Your trained models:\n{msg}")


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Start the training process.
    Usage: /train <model_key>
    After this command, send 5–10 images of your face or subject.
    (The model_key is a short identifier you choose; the destination is fixed.)
    """
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /train <model_key>")
        return ConversationHandler.END

    model_key = context.args[0]
    user_id = update.effective_user.id
    user_images[user_id] = []  # initialize empty list for this user
    context.user_data["train_model_key"] = model_key

    await update.message.reply_text(
        f"Training initiated for model key '{model_key}'.\n"
        "Please send 5–10 images of your face or subject. Use /cancel to abort."
    )
    return WAITING_FOR_IMAGES


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos: save them locally and add to user's image list."""
    user_id = update.effective_user.id
    photo = update.message.photo[-1]  # highest resolution
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"user_{user_id}_{update.message.message_id}.jpg")
    tg_file = await photo.get_file()
    await tg_file.download_to_drive(custom_path=file_path)
    user_images.setdefault(user_id, []).append(file_path)
    count = len(user_images[user_id])
    await update.message.reply_text(f"Received image {count}.")

    if count >= 5:
        await update.message.reply_text(
            "Sufficient images received. Initiating training (this may take 10-20 minutes)..."
        )
        await train_model_on_replicate(update, context)
        return ConversationHandler.END

    return WAITING_FOR_IMAGES


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the current training operation."""
    user_id = update.effective_user.id
    user_images.pop(user_id, None)
    context.user_data.pop("train_model_key", None)
    await update.message.reply_text("Operation cancelled.")
    return ConversationHandler.END


# --- Replicate Integration Functions ---

async def train_model_on_replicate(update: Update, context: CallbackContext):
    """
    Zip the user's images, upload them to S3, and start a training job on Replicate.
    The training job uses the Flux LoRA trainer and pushes a new version to the fixed destination.
    """
    user_id = update.effective_user.id
    model_key = context.user_data.get("train_model_key")
    images = user_images.get(user_id, [])

    if not model_key or not images:
        await update.message.reply_text("Training failed: No images or model key provided.")
        return

    try:
        # 1. Zip the images.
        zip_filename = os.path.join(tempfile.gettempdir(), f"train_{user_id}_{model_key}.zip")
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for img in images:
                zipf.write(img, os.path.basename(img))

        # 2. Upload the ZIP to S3.
        zip_s3_key = f"flux_training/{user_id}/{model_key}.zip"
        s3_client.upload_file(zip_filename, S3_BUCKET_NAME, zip_s3_key)

        # 3. Generate a presigned URL (valid for 1 hour).
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": zip_s3_key},
            ExpiresIn=3600,
        )

        await update.message.reply_text("Initiating training on Replicate...")

        # 4. Generate a unique trigger token.
        trigger = f"USER{user_id}TOKEN".upper()

        # 5. Use the fixed destination model.
        model_dest = DESTINATION_MODEL

        # 6. Initiate training via Replicate.
        training = replicate.trainings.create(
            version=FLUX_TRAINER_VERSION,  # Format: username/model_name:version_id.
            destination=model_dest,
            input={
                "steps": 1000,
                "lora_rank": 16,
                "optimizer": "adamw8bit",
                "batch_size": 1,
                "resolution": "512,768,1024",
                "autocaption": True,
                "input_images": presigned_url,
                "trigger_word": trigger,
                "learning_rate": 0.0004,
                "wandb_project": "flux_train_replicate",
                "wandb_save_interval": 100,
                "caption_dropout_rate": 0.05,
                "cache_latents_to_disk": False,
                "wandb_sample_interval": 100,
                "gradient_checkpointing": False,
            },
        )

        # Poll until training status is "succeeded" or "failed"
        while training.status not in ("succeeded", "failed"):
            time.sleep(10)
            training = replicate.trainings.get(training.id)

        if training.status == "failed":
            raise Exception("Training job failed.")

        await update.message.reply_text("Training complete! Your custom meme model is ready.")

        # Save the trigger for later use.
        user_triggers.setdefault(user_id, {})[model_key] = trigger

    except Exception as e:
        logger.exception("Training on Replicate failed")
        await update.message.reply_text(f"Training failed: {e}")

    finally:
        for img in images:
            try:
                os.remove(img)
            except Exception:
                pass
        try:
            os.remove(zip_filename)
        except Exception:
            pass
        user_images.pop(user_id, None)
        context.user_data.pop("train_model_key", None)


async def generate_meme_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Generate a meme image using the trained model.
    Usage: /meme <model_key> <prompt>
    The bot will prepend the trigger token to the prompt and fetch the latest version of the model.
    """
    user_id = update.effective_user.id
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /meme <model_key> <prompt>")
        return

    model_key = context.args[0]
    prompt_text = " ".join(context.args[1:])

    if user_id not in user_triggers or model_key not in user_triggers[user_id]:
        await update.message.reply_text(
            f"No trained model found for key '{model_key}'. Please train one using /train first."
        )
        return

    trigger = user_triggers[user_id][model_key]

    # Fetch the latest version of the destination model.
    try:
        model_obj = replicate.models.get(DESTINATION_MODEL)
        versions_list = model_obj.versions.list()
        if not versions_list:
            raise Exception("No versions available for the model. Ensure training completed successfully.")
        latest_version = versions_list[0].id  # Assume the first element is the latest version.
        model_ref = f"{DESTINATION_MODEL}:{latest_version}"
        logger.info(f"Using model reference: {model_ref}")
    except Exception as e:
        logger.exception("Failed to fetch model details")
        await update.message.reply_text(f"Failed to fetch model details: {e}")
        return

    full_prompt = f"{trigger} {prompt_text}"
    await update.message.reply_text("Generating your meme...")

    try:
        output = replicate.run(model_ref, input={"prompt": full_prompt})
        image_url = output[0] if isinstance(output, list) and len(output) > 0 else output
        logger.info(f"Generated image URL: {image_url}")
        image_data = requests.get(image_url).content
        await update.message.reply_photo(photo=image_data, caption="Here is your meme!")
    except Exception as e:
        logger.exception("Meme generation failed")
        await update.message.reply_text(f"Meme generation failed: {e}")


async def enhance_prompt(user_prompt: str) -> str:
    """
    Optionally enhance the prompt using OpenAI GPT-3.5.
    """
    system_msg = "You are a helpful AI that refines user prompts for better meme generation."
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Enhance this prompt: {user_prompt}"},
        ],
        max_tokens=50,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# --- Main Bot Setup ---

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in environment file.")

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("train", train_command)],
        states={
            WAITING_FOR_IMAGES: [
                MessageHandler(filters.PHOTO, photo_handler),
                CommandHandler("cancel", cancel_command),
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("list_models", list_models_command))
    application.add_handler(CommandHandler("meme", generate_meme_command))
    application.add_handler(conv_handler)

    logger.info("Bot starting. Polling for updates...")
    application.run_polling()


if __name__ == "__main__":
    main()
