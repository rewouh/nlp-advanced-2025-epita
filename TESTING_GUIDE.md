# Guide de Test - Overhearing Agents System

## Scénario de Test Complet

Ce guide teste toutes les fonctionnalités du système en une seule session.

---

## 🎯 Phase 1 : Déclenchement et Personnalité de Base

**Objectif** : Tester les triggers et la personnalité initiale de Joe

1. **Trigger Direct** (teste la détection directe)
   ```
   "Hey Joe, can I get a drink?"
   ```
   ✅ **Attendu** : Joe répond avec sa personnalité "grumpy" et "secretive"

2. **Personnalité - Réaction au ton** (teste la réactivité)
   ```
   "Joe, what's your problem? You seem grumpy."
   ```
   ✅ **Attendu** : Réaction selon sa personnalité (peut devenir plus distant)

3. **Personnalité - Approche amicale** (teste la construction de relation)
   ```
   "Joe, I heard you used to be a sailor. That must have been interesting."
   ```
   ✅ **Attendu** : Réaction plus ouverte, commence à construire la relation

---

## 🧠 Phase 2 : RAG - Retrieval de Lore

**Objectif** : Tester la récupération de connaissances depuis la base RAG

4. **Question sur l'histoire** (teste RAG - history)
   ```
   "What happened during the Great Blizzard?"
   ```
   ✅ **Attendu** : Joe mentionne des détails du lore (50 ans, Frost Giants, etc.)

5. **Question sur les factions** (teste RAG - factions)
   ```
   "Tell me about the Iron Vanguard."
   ```
   ✅ **Attendu** : Informations sur la milice, la guerre, etc.

6. **Question sur les lieux** (teste RAG - locations)
   ```
   "What do you know about the Frozen Harbor?"
   ```
   ✅ **Attendu** : Détails sur le port gelé, les navires, etc.

---

## 🔗 Phase 3 : Relations et Secrets

**Objectif** : Tester le système de relations et le partage de secrets

7. **Construire la confiance** (teste RelationshipTracker)
   ```
   "Joe, I'm looking for information. I can help you with something if you help me."
   ```
   ✅ **Attendu** : Relation s'améliore progressivement

8. **Demander un secret trop tôt** (teste should_share_secrets)
   ```
   "Do you know any secrets about this place?"
   ```
   ✅ **Attendu** : Refus ou réponse évasive (score < 40)

9. **Continuer à construire la relation** (teste l'accumulation)
   ```
   "I understand you're looking for information about a sea monster. Maybe I can help."
   ```
   ✅ **Attendu** : Relation continue de s'améliorer

10. **Demander le secret après confiance** (teste le partage conditionnel)
    ```
    "Now, about those secrets you mentioned..."
    ```
    ✅ **Attendu** : Si score >= 40, partage le secret (treasure map, smuggling tunnel)

---

## 🎭 Phase 4 : Multi-NPC et Contexte

**Objectif** : Tester le changement de contexte et les autres NPCs

11. **Mentionner un autre NPC** (teste la détection indirecte)
    ```
    "I heard Captain Hilda is looking for help."
    ```
    ✅ **Attendu** : Joe réagit selon sa connaissance de Hilda

12. **Changement de contexte - Location** (teste ContextManager)
    ```
    "I'm heading to the guard barracks next."
    ```
    ✅ **Attendu** : Le système détecte le changement de location

13. **Question sur un NPC absent** (teste la gestion des NPCs non-présents)
    ```
    "What do you know about Captain Hilda?"
    ```
    ✅ **Attendu** : Joe partage ce qu'il sait (vétéran, guerre, etc.)

---

## 🎤 Phase 5 : Détection de Triggers Variés

**Objectif** : Tester différents types de triggers

14. **Trigger Indirect - Mention du nom** (teste NPC_INDIRECT)
    ```
    "I wonder what Joe thinks about all this."
    ```
    ✅ **Attendu** : Détection indirecte, Joe peut réagir

15. **Conversation Player-to-Player** (teste PLAYER_TO_PLAYER)
    ```
    "This place is really interesting, isn't it?"
    ```
    ✅ **Attendu** : Détection comme conversation entre joueurs, pas d'activation NPC

16. **Trigger Direct avec nom complet** (teste la robustesse)
    ```
    "Hey Joe, one more thing..."
    ```
    ✅ **Attendu** : Détection directe fonctionne avec contexte précédent

---

## 🌍 Phase 6 : RAG Avancé - Items et Quêtes

**Objectif** : Tester la récupération d'informations sur items et quêtes

17. **Question sur un item** (teste RAG - items)
    ```
   "Have you heard of the Frost Reaver?"
    ```
    ✅ **Attendu** : Informations sur l'arme légendaire

18. **Question sur une quête active** (teste RAG - quests + contexte)
    ```
    "What do you know about the missing supplies?"
    ```
    ✅ **Attendu** : Joe mentionne la quête active (missing_supplies) et les goblins

19. **Question sur une quête non-active** (teste RAG - quests)
    ```
    "What about the frozen lighthouse?"
    ```
    ✅ **Attendu** : Informations sur la quête (lighthouse keeper, ghost, etc.)

---

## 🎨 Phase 7 : Émotions TTS et Disposition

**Objectif** : Tester les émotions TTS selon la disposition

20. **Test émotion - Neutre** (début de conversation)
    ```
    "Joe, how's business?"
    ```
    ✅ **Attendu** : TTS avec émotion NEUTRAL

21. **Test émotion - Amical** (après construction de relation)
    ```
    "Thanks for the help, Joe. You're a good friend."
    ```
    ✅ **Attendu** : TTS avec émotion HAPPY/FRIENDLY (si disposition friendly)

22. **Test émotion - Hostile** (si on est impoli)
    ```
    "You're being difficult, Joe. Just tell me what I need to know!"
    ```
    ✅ **Attendu** : TTS avec émotion ANGRY (si disposition hostile)

---

## 🔄 Phase 8 : Conversation Multi-Tours

**Objectif** : Tester la mémoire et la cohérence sur plusieurs tours

23. **Référence à une conversation précédente** (teste la mémoire)
    ```
    "Remember when you told me about the sea monster? I found something."
    ```
    ✅ **Attendu** : Joe se souvient de la conversation précédente

24. **Question de suivi** (teste la continuité)
    ```
    "Can you tell me more about that treasure map you mentioned?"
    ```
    ✅ **Attendu** : Référence au secret partagé précédemment

25. **Changement de sujet avec contexte** (teste la flexibilité)
    ```
    "Speaking of the docks, what's the situation with the smuggling?"
    ```
    ✅ **Attendu** : Joe fait le lien avec le contexte précédent

---

## ✅ Checklist de Validation

Après chaque phase, vérifier :

- [ ] **STT** : Transcription correcte de la voix
- [ ] **Trigger Detection** : Bon type de trigger détecté
- [ ] **RAG** : Informations pertinentes récupérées du lore
- [ ] **Personnalité** : Réponses cohérentes avec les traits du NPC
- [ ] **Relations** : Score de relation évolue correctement
- [ ] **Secrets** : Partage conditionnel fonctionne (score >= 40)
- [ ] **Contexte** : Détection des changements (location, mood, etc.)
- [ ] **TTS** : Émotions correctes selon la disposition
- [ ] **Mémoire** : Références aux conversations précédentes
- [ ] **Blocage d'écoute** : Le système attend que le NPC finisse de parler

---

## 🐛 Tests de Robustesse

**Test de récupération d'erreur** :
- Parler très vite → Vérifier que le système gère bien
- Parler très bas → Vérifier la détection VAD
- Parler pendant que le NPC parle → Vérifier le blocage

**Test de limites** :
- Questions très longues → Vérifier la gestion
- Questions ambiguës → Vérifier la détection de trigger
- Questions hors contexte → Vérifier la réaction du NPC

---

## 📊 Métriques à Observer

Pendant les tests, noter :
- **Latence STT** : Temps entre la fin de la phrase et la transcription
- **Latence NPC** : Temps entre la transcription et la réponse
- **Latence TTS** : Temps de synthèse vocale
- **Précision RAG** : Pertinence des informations récupérées
- **Cohérence** : Les réponses sont-elles cohérentes avec le contexte ?

---

## 🎯 Résultat Attendu Global

À la fin du test, tu devrais avoir :
- ✅ Testé tous les types de triggers
- ✅ Construit une relation avec Joe (score > 40)
- ✅ Obtenu des secrets partagés
- ✅ Testé le RAG sur tous les types de lore (history, factions, locations, items, quests)
- ✅ Vérifié les émotions TTS selon la disposition
- ✅ Validé la mémoire multi-tours
- ✅ Confirmé le blocage d'écoute pendant la parole du NPC

