import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import List

import discord
import openai
from pydantic import FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    discord_bot_token: str
    openai_api_key: str
    max_cache: int = 5
    cooldown_time: int = 2
    openai_model: str = "gpt-3.5-turbo-16k"
    allowed_channel_ids: List[int] = []
    system_message_file: FilePath = FilePath("./system_message.txt")

    @property
    def system_message(self):
        with open(self.system_message_file, "r", encoding="utf-8") as file:
            return file.read()

    model_config = SettingsConfigDict(
        # `.env.prod` takes priority over `.env`
        env_file=(".env", ".env.prod"),
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data structures for caching, cooldown, and last interaction
user_message_cache = defaultdict(deque)
user_cooldown = defaultdict(int)
user_last_channel = {}
user_last_interaction_time = {}

# Additional data structure for tracking the last interaction time and channel of a user with the bot
user_last_interaction = defaultdict(lambda: {"time": datetime.min, "channel": None})

# Bot setup
intents = discord.Intents.default()
intents.message_content = True  # This intent is required to read message content
intents.messages = True  # Required for message.reference
bot = discord.Client(intents=intents)  # Using the Client class


# Set up OpenAI API
openai.api_key = settings.openai_api_key


def split_message(content, limit=2000):
    """Split a message into chunks of a specified limit."""
    return [content[i : i + limit] for i in range(0, len(content), limit)]


async def send_large_message(channel, content):
    """Send content which may be longer than the discord character limit by splitting it into parts."""
    for chunk in split_message(content):
        await channel.send(chunk)


# Load the system message from file
with open("system_message_copy.txt", "r", encoding="utf-8") as file:
    SYSTEM_MESSAGE = file.read()


@bot.event
async def on_ready():
    logger.info(f"We have logged in as {bot.user.name}")


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    mentioned = bot.user in message.mentions
    replied_to_bot = (
        message.reference and message.reference.resolved.author.id == bot.user.id
    )
    last_interaction = user_last_interaction[message.author.id]
    within_timeframe = datetime.utcnow() - last_interaction["time"] <= timedelta(
        seconds=120
    )
    same_channel = last_interaction["channel"] == message.channel.id

    if not (mentioned or replied_to_bot or (within_timeframe and same_channel)):
        return

    logger.info(
        f"Initiating conversation with user: {message.author.name} (ID: {message.author.id})"
    )

    async with message.channel.typing():
        await chat_with_openai(message)
        user_last_interaction[message.author.id] = {
            "time": datetime.utcnow(),
            "channel": message.channel.id,
        }


async def chat_with_openai(message):
    if user_cooldown[message.author.id]:
        logger.info(
            f"Cooldown enforced for user: {message.author.name} (ID: {message.author.id})"
        )
        await message.channel.send("Please wait a bit before chatting again!")
        return

    async def keep_typing(channel):
        while True:
            await channel.trigger_typing()
            await asyncio.sleep(5)  # Trigger every 5 seconds

    typing_task = asyncio.create_task(keep_typing(message.channel))

    messages = [{"role": "system", "content": settings.system_message}]

    for role, msg_content in user_message_cache[message.author.id]:
        messages.append({"role": role, "content": msg_content})

    messages.append({"role": "user", "content": message.content})

    try:
        response = openai.ChatCompletion.create(
            model=settings.openai_model, messages=messages
        )

        typing_task.cancel()
        ai_message = response["choices"][0]["message"]["content"]
        await send_large_message(message.channel, ai_message)

        if len(user_message_cache[message.author.id]) >= settings.max_cache:
            removed_message = user_message_cache[message.author.id].popleft()
            logger.info(
                f"Removed oldest cached message for user: {message.author.name} (ID: {message.author.id}). Message: {removed_message[1]}"
            )

        user_message_cache[message.author.id].append(("user", message.content))
        logger.info(
            f"Added message to cache for user: {message.author.name} (ID: {message.author.id}). Message: {message.content}"
        )

        user_message_cache[message.author.id].append(("assistant", ai_message))
        logger.info(
            f"Added bot response to cache for user: {message.author.name} (ID: {message.author.id}). Response: {ai_message}"
        )

        user_cooldown[message.author.id] = settings.cooldown_time
        bot.loop.create_task(cooldown_user(message.author.id))
        logger.info(
            f"Set cooldown for user: {message.author.name} (ID: {message.author.id})"
        )

    except openai.error.OpenAIError as e:
        typing_task.cancel()
        await message.channel.send(f"An error occurred: {str(e)}")
        logger.error(
            f"OpenAI API error for user: {message.author.name} (ID: {message.author.id}). Error: {str(e)}"
        )
    except Exception as e:
        typing_task.cancel()
        logger.error(
            f"Unexpected error for user: {message.author.name} (ID: {message.author.id}). Error: {e}"
        )


async def cooldown_user(user_id):
    logger.info(f"Starting cooldown for user ID: {user_id}")
    await asyncio.sleep(settings.cooldown_time)
    user_cooldown[user_id] = 0
    logger.info(f"Cooldown ended for user ID: {user_id}")


bot.run(settings.discord_bot_token)
