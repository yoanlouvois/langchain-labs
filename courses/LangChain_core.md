bonnes pratiques ?

input schema et pydantic
Pourquoi c'est central dans LangChain
Sans schéma (texte libre)Avec Pydantic / input_schemaLe LLM écrit du texte, tu parses à la main (fragile)Le LLM remplit un schéma connu à l'avanceErreurs de format découvertes tard, difficiles à déboguerValidation immédiate, erreurs explicitesPas de contrat clair entre composantsChaque Runnable documente précisément ce qu'il attend/renvoieDifficile d'exposer en API propreSe convertit nativement en doc OpenAPI (FastAPI)
En résumé : Pydantic = la librairie qui définit et valide des schémas de données en Python. input_schema/output_schema = la façon dont LangChain utilise ces schémas Pydantic pour que chaque Runnable déclare précisément ce qu'il attend et renvoie — et c'est ce même mécanisme qui, poussé un cran plus loin, permet au tool calling et à la sortie structurée de fonctionner de façon fiable, comme dans ton agent.

Mémoire de conversation (checkpointer) : à chaque appel, tout l'historique de la conversation est renvoyé en input au modèle. Plus une conversation avec un thread_id donné s'allonge, plus tu consommes de tokens en input — jusqu'à potentiellement dépasser la fenêtre de contexte sur de très longues conversations (d'où l'intérêt du trimming de messages mentionné dans le cours agents, section middleware).