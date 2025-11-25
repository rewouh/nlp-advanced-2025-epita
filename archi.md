# Architecture Technique & Cognitive - Overhearing Agents

Ce document présente l'architecture globale du projet. Il détaille à la fois la "mécanique" technique (traitement du signal, modèles) et le "cerveau" du PNJ (mémoire, personnalité, contexte).


## 🛠️ Partie 1 : Stack Technologique (Le Moteur)

Cette section justifie les outils utilisés pour assurer la performance et le temps réel.

### 1. VAD (Voice Activity Detection)
*   **Outil :** `Silero-VAD`.
*   **Rôle :** Le "Portier".
*   **Pourquoi ?** : Le micro écoute en permanence. Sans VAD, le système enverrait du silence ou du bruit de fond au processeur. Cela gaspille des ressources (GPU) et fait "halluciner" le modèle (il invente des mots dans le bruit). Le VAD coupe le flux quand personne ne parle.

### 2. STT (Speech-to-Text)
*   **Outil :** `faster-whisper` (Implémentation optimisée du modèle OpenAI).
*   **Rôle :** L'"Oreille".
*   **Optimisation :** Utilisation de **Hot-Words** (injection de vocabulaire métier) pour reconnaître les noms propres fantaisie (ex: "Phandalin", "Tiamat") que le modèle standard écorcherait.
*   **Pourquoi ?** : C'est le meilleur compromis précision/vitesse actuel pour transformer la parole naturelle en texte exploitable par le LLM.

### 3. Local LLM (Le Modèle)
*   **Outil :** `Ollama` tournant `Qwen2.5:3b` ou `Mistral`.
*   **Rôle :** Le moteur d'intelligence brute.
*   **Pourquoi ?** :
    *   **Confidentialité :** Tout reste en local.
    *   **Latence :** Pas d'appel API réseau.
    *   **Taille :** Les modèles 3B-7B tournent fluidement sur des laptops gaming (RTX) avec une latence acceptable pour la conversation.

### 4. TTS (Text-to-Speech)
*   **Outil :** `Piper TTS` (Modèle `en_US-joe-medium`).
*   **Rôle :** La "Bouche".
*   **Pourquoi ?** : Contrairement aux solutions "lourdes" (Tortoise) ou cloud (ElevenLabs), Piper génère de l'audio en quelques millisecondes (< 200ms). C'est crucial pour éviter le "blanc" gênant entre la réponse du joueur et celle du NPC.

---

## 🧠 Partie 2 : Architecture Cognitive (Le Pilote)

Cette section explique comment **LangChain** orchestre le LLM pour créer un personnage crédible et cohérent.

### 1. Orchestration & Router
*   **Outil :** `LangChain Router Chain`.
*   **Concept :** L'agent ne répond pas à tout. Le Router analyse l'intention de la phrase transcrite :
    *   *Si "Je demande au tavernier..."* -> **Activation**.
    *   *Si "Passe-moi les chips..."* -> **Ignorer**.

### 2. Gestion de la Mémoire (Memory)
Un LLM brut est amnésique. Nous utilisons deux types de mémoires :

*   **A. Mémoire Court Terme (Short-Term):**
    *   *Composant :* `ConversationSummaryBufferMemory`.
    *   *Fonctionnement :* Garde les derniers échanges (ex: 5) en texte brut pour la fluidité, et **résume** automatiquement les échanges plus anciens pour économiser la fenêtre de contexte.
    *   *Usage :* Se souvenir du nom du joueur ou de sa commande récente.

*   **B. Mémoire Long Terme (RAG / Lore):**
    *   *Composant :* `ChromaDB` (Vector Store) + `Retriever`.
    *   *Fonctionnement :* Base de données contenant l'histoire du monde ("Lore"). Quand le joueur pose une question sur l'univers, le système retrouve le passage pertinent et l'injecte dans le prompt.
    *   *Usage :* Connaître l'histoire du Roi ou la géographie sans l'avoir apprise par cœur.

### 3. Contexte Dynamique & Persona
Pour que le NPC soit vivant, son "System Prompt" est reconstruit dynamiquement à chaque tour de parole :

```text
Prompt = [Persona Statique] + [État de la Scène] + [Mémoire] + [Lore RAG]
```

*   **Persona :** "Tu es Joe, un nain grincheux."
*   **État de la Scène (Injecté) :** "Il fait nuit, la taverne est vide, les joueurs sont armés."
*   **Règle d'or (Guardrails) :** Interdiction d'utiliser des emojis ou des actions entre astérisques (`*sourit*`), car le TTS les lirait à haute voix, brisant l'immersion.

---

## 📜 Exemple Concret : "L'Affaire du Dragon"

Voici la trace d'exécution interne du système lorsqu'un joueur pose une question complexe.

**Scénario :** Les joueurs sont dans la taverne de Joe. Le joueur "Léo" pose une question sur une rumeur locale.

**1. Input Joueur (Audio)**
> "Hé Joe, t'as entendu parler du Dragon Blanc dans les montagnes ?"

**2. Traitement STT (Processing)**
*   `VAD` : Détecte une voix (ignore le bruit des verres).
*   `Whisper` : Transcrit "Hey Joe, have you heard about the White Dragon in the mountains?".
*   `Router` : Détecte le mot-clé "Joe" + Intention de question -> **ACTIVATION**.

**3. Récupération de Connaissance (RAG)**
*   Le système cherche "White Dragon" et "Mountains" dans `ChromaDB`.
*   *Résultat trouvé (Lore.txt) :* "Une rumeur dit que Cryovain, un dragon blanc cruel, a été vu au sommet du Pic Icespire."

**4. Assemblage du Prompt (LangChain)**
Le LLM reçoit ceci (simplifié) :
> **System:** Tu es Joe, un tavernier nain peureux. Tu parles d'une voix grave.
> **Contexte:** Il fait nuit. Ambiance calme.
> **Lore:** Le dragon s'appelle Cryovain, il vit au Pic Icespire.
> **Mémoire:** Léo t'a salué il y a 2 minutes.
> **User:** T'as entendu parler du Dragon Blanc ?

**5. Génération (LLM Output)**
> "Par ma barbe... Tu parles de Cryovain ? Ce monstre gèle le sang des voyageurs ! Ne t'approche pas du Pic Icespire si tu tiens à la vie, petit !"

**6. Sortie Audio (TTS)**
*   `Piper` génère l'audio avec le modèle vocal "Joe".
*   Les haut-parleurs jouent la réponse.