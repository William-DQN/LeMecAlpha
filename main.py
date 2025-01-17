import json
import discord
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import os

# Charger les configurations
with open("config.json") as f:
    config = json.load(f)
with open("personnality.json") as f:
    personality_config = json.load(f)

token = config["bot_token"]
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="ma!", intents=intents)

# Charger le modèle NLP
model_name = config["model_name"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Charger le prompt de personnalité
prompt = personality_config["personality_prompt"]

# Fonction pour générer une réponse textuelle
def generate_response(user_input):
    input_text = prompt + f"\n\nQuestion : {user_input}\nRéponse :"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(inputs["input_ids"], max_length=200, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()  # Nettoyer la réponse
    return response

# Fonction pour convertir le texte en audio avec RVC
def text_to_speech_rvc(text, output_file="response.wav"):
    # Sauvegarder le texte dans un fichier temporaire
    text_file = "temp_text.txt"
    with open(text_file, "w") as f:
        f.write(text)

    # Commande pour appeler le modèle RVC
    rvc_model_path = "./models/rvc_model.pth"  # Chemin vers le modèle RVC
    command = [
        "python", "inference.py",  # Script d'inférence RVC
        "--input", text_file,
        "--model_path", rvc_model_path,
        "--output", output_file
    ]

    # Exécuter la commande
    subprocess.run(command)

# Événement déclenché lorsque le bot reçoit un message
@bot.event
async def on_message(message):
    # Éviter que le bot réponde à lui-même
    if message.author == bot.user:
        return

    # Vérifier si le message mentionne le bot
    if bot.user in message.mentions:
        # Générer une réponse textuelle
        user_input = message.content.replace(f"<@{bot.user.id}>", "").strip()
        response_text = generate_response(user_input)

        # Convertir la réponse en audio avec RVC
        audio_file = "response.wav"
        text_to_speech_rvc(response_text, audio_file)

        # Envoyer le fichier audio en tant que message vocal
        await message.channel.send(
            content=f"{message.author.mention} Voici votre réponse vocale :",
            file=discord.File(audio_file)
        )

# Démarrer le bot
bot.run(token)
