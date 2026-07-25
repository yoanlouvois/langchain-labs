# Cours complet — Les Subgraphs avec LangGraph

> Basé sur la documentation officielle : https://docs.langchain.com/oss/python/langgraph/use-subgraphs

---

## 1. Qu'est-ce qu'un subgraph ?

Un **subgraph** est un [graphe](https://docs.langchain.com/oss/python/langgraph/graph-api#graphs) utilisé comme **nœud** dans un autre graphe (le "parent graph"). Autrement dit : tu construis un graphe LangGraph complet et autonome, puis tu l'insères tel quel comme une simple étape d'un graphe plus grand.

Les subgraphs sont utiles pour trois cas d'usage principaux :

- **Construire des systèmes multi-agents** — chaque agent est un graphe indépendant, orchestré par un graphe parent.
- **Réutiliser un ensemble de nœuds dans plusieurs graphes** — factoriser une logique commune (ex: un sous-workflow de vérification) sans dupliquer le code.
- **Distribuer le développement entre équipes** : dès lors que l'interface du subgraph (ses schémas d'entrée et de sortie) est respectée, le graphe parent peut être construit sans connaître les détails internes du subgraph.

C'est un principe d'**encapsulation**, similaire à une fonction ou un module en programmation classique — sauf qu'ici l'unité encapsulée est un graphe entier, avec son propre state, ses propres nœuds, et potentiellement sa propre mémoire.

---

## 2. Les deux façons de connecter un subgraph au graphe parent

Le choix dépend d'une seule question : **le parent et le subgraph partagent-ils des clés d'état (state keys) ?**

| Pattern | Quand l'utiliser | Schémas d'état |
| --- | --- | --- |
| **Appeler le subgraph dans un nœud** | Le parent et le subgraph ont des **schémas d'état différents** (aucune clé partagée), ou tu dois transformer l'état entre les deux | Tu écris une fonction wrapper qui convertit l'état parent en entrée du subgraph, et la sortie du subgraph en état parent |
| **Ajouter le subgraph comme un nœud** | Le parent et le subgraph **partagent des clés d'état** — le subgraph lit et écrit dans les mêmes canaux que le parent | Tu passes directement le subgraph compilé à `add_node` — aucun wrapper nécessaire |

### 2.1 Appeler un subgraph dans un nœud (schémas différents)

Quand parent et subgraph ont des **schémas d'état différents** (aucune clé partagée), tu invoques le subgraph à l'intérieur d'une fonction de nœud. C'est courant quand tu veux garder un historique de messages **privé** pour chaque agent dans un système multi-agents. La fonction de nœud transforme l'état parent en état du subgraph avant de l'invoquer, puis transforme le résultat en état parent avant de le retourner.

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class SubgraphState(TypedDict):
    bar: str

# Subgraph
def subgraph_node_1(state: SubgraphState):
    return {"bar": "hi! " + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph
class State(TypedDict):
    foo: str

def call_subgraph(state: State):
    # Transforme l'état parent en état du subgraph
    subgraph_output = subgraph.invoke({"bar": state["foo"]})
    # Transforme la réponse en état parent
    return {"foo": subgraph_output["bar"]}

builder = StateGraph(State)
builder.add_node("node_1", call_subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

**Exemple complet avec plusieurs nœuds internes au subgraph :**

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# Subgraph
class SubgraphState(TypedDict):
    # aucune de ces clés n'est partagée avec l'état du parent
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

def node_2(state: ParentState):
    response = subgraph.invoke({"bar": state["foo"]})
    return {"foo": response["bar"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream({"foo": "foo"}, subgraphs=True, version="v2"):
    if chunk["type"] == "updates":
        print(chunk["ns"], chunk["data"])
```

Sortie :
```
() {'node_1': {'foo': 'hi! foo'}}
('node_2:577b710b-...',) {'subgraph_node_1': {'baz': 'baz'}}
('node_2:577b710b-...',) {'subgraph_node_2': {'bar': 'hi! foobaz'}}
() {'node_2': {'foo': 'hi! foobaz'}}
```

**Deux niveaux de subgraphs (parent → enfant → petit-enfant)** — le pattern s'imbrique sans limite :

```python
# Grandchild graph
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START, END

class GrandChildState(TypedDict):
    my_grandchild_key: str

def grandchild_1(state: GrandChildState) -> GrandChildState:
    # NOTE : les clés du child ou du parent ne sont pas accessibles ici
    return {"my_grandchild_key": state["my_grandchild_key"] + ", how are you"}

grandchild = StateGraph(GrandChildState)
grandchild.add_node("grandchild_1", grandchild_1)
grandchild.add_edge(START, "grandchild_1")
grandchild.add_edge("grandchild_1", END)
grandchild_graph = grandchild.compile()

# Child graph
class ChildState(TypedDict):
    my_child_key: str

def call_grandchild_graph(state: ChildState) -> ChildState:
    # NOTE : les clés du parent ou du grandchild ne sont pas accessibles ici
    grandchild_graph_input = {"my_grandchild_key": state["my_child_key"]}
    grandchild_graph_output = grandchild_graph.invoke(grandchild_graph_input)
    return {"my_child_key": grandchild_graph_output["my_grandchild_key"] + " today?"}

child = StateGraph(ChildState)
child.add_node("child_1", call_grandchild_graph)
child.add_edge(START, "child_1")
child.add_edge("child_1", END)
child_graph = child.compile()

# Parent graph
class ParentState(TypedDict):
    my_key: str

def parent_1(state: ParentState) -> ParentState:
    return {"my_key": "hi " + state["my_key"]}

def parent_2(state: ParentState) -> ParentState:
    return {"my_key": state["my_key"] + " bye!"}

def call_child_graph(state: ParentState) -> ParentState:
    child_graph_input = {"my_child_key": state["my_key"]}
    child_graph_output = child_graph.invoke(child_graph_input)
    return {"my_key": child_graph_output["my_child_key"]}

parent = StateGraph(ParentState)
parent.add_node("parent_1", parent_1)
parent.add_node("child", call_child_graph)
parent.add_node("parent_2", parent_2)
parent.add_edge(START, "parent_1")
parent.add_edge("parent_1", "child")
parent.add_edge("child", "parent_2")
parent.add_edge("parent_2", END)
parent_graph = parent.compile()

for chunk in parent_graph.stream({"my_key": "Bob"}, subgraphs=True, version="v2"):
    if chunk["type"] == "updates":
        print(chunk["ns"], chunk["data"])
```

**Point clé** : à chaque niveau, un nœud ne voit **que son propre état** — les clés du parent ou de l'enfant ne sont jamais accessibles directement, seule la fonction wrapper (`call_child_graph`, `call_grandchild_graph`) fait le pont entre les niveaux. C'est ce qui garantit l'isolation entre les couches.

### 2.2 Ajouter un subgraph directement comme un nœud (schémas partagés)

Quand le parent et le subgraph **partagent des clés d'état**, tu peux passer le subgraph compilé **directement** à `add_node` — aucune fonction wrapper n'est nécessaire. Le subgraph lit et écrit automatiquement dans les canaux d'état du parent. C'est le pattern typique des systèmes **multi-agents**, où les agents communiquent souvent via une clé `messages` partagée.

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class State(TypedDict):
    foo: str

# Subgraph
def subgraph_node_1(state: State):
    return {"foo": "hi! " + state["foo"]}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph
builder = StateGraph(State)
builder.add_node("node_1", subgraph)  # <-- le subgraph directement comme nœud
builder.add_edge(START, "node_1")
graph = builder.compile()
```

**Exemple avec des clés à la fois partagées et privées :**

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class SubgraphState(TypedDict):
    foo: str  # partagée avec le parent
    bar: str  # privée au subgraph

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    # utilise 'bar' (privée) et met à jour 'foo' (partagée)
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream({"foo": "foo"}, version="v2"):
    if chunk["type"] == "updates":
        print(chunk["data"])
```

Sortie :
```
{'node_1': {'foo': 'hi! foo'}}
{'node_2': {'foo': 'hi! foobar'}}
```

---

## 3. Persistance des subgraphs

Quand tu utilises un subgraph, une question se pose : **que doit-il se passer avec ses données internes entre plusieurs appels ?** Prends l'exemple d'un bot de support client qui délègue à des sous-agents spécialisés : est-ce que le sous-agent "expert facturation" doit se souvenir des questions précédentes du client, ou repartir de zéro à chaque appel ?

C'est le paramètre **`checkpointer`** passé à `.compile()` du subgraph qui contrôle ce comportement :

| Mode | `checkpointer=` | Comportement |
| --- | --- | --- |
| **Per-invocation** (défaut) | `None` | Chaque appel repart de zéro, mais hérite du checkpointer du parent pour supporter les interrupts et l'exécution durable **au sein d'un même appel**. |
| **Per-thread** | `True` | L'état s'accumule entre les appels sur le même thread. Chaque appel reprend là où le précédent s'est arrêté. |
| **Stateless** | `False` | Aucun checkpointing — s'exécute comme un simple appel de fonction. Pas d'interrupts, pas d'exécution durable. |

**Per-invocation est le bon choix pour la plupart des applications**, y compris les systèmes multi-agents où les sous-agents traitent des requêtes indépendantes. **Per-thread** est à utiliser quand un sous-agent a besoin d'une mémoire de conversation multi-tours (ex: un assistant de recherche qui construit du contexte sur plusieurs échanges).

⚠️ Le graphe parent doit être compilé avec un checkpointer pour que les fonctionnalités de persistance du subgraph (interrupts, inspection d'état, mémoire per-thread) fonctionnent.

Les exemples ci-dessous utilisent `create_agent` de LangChain (vu dans le premier cours de cette série). `create_agent` produit un graphe LangGraph en interne, donc tous les concepts de persistance de subgraphs s'appliquent directement à un agent créé de cette façon.

### 3.1 Per-invocation (par défaut)

Mode recommandé pour la plupart des applications, notamment les systèmes multi-agents où les sous-agents sont invoqués comme des **tools**. Supporte les interrupts, l'exécution durable, et les appels parallèles, tout en gardant chaque invocation isolée.

À utiliser quand chaque appel au subgraph est indépendant et que le sous-agent n'a pas besoin de se souvenir des appels précédents — le cas le plus courant, en particulier pour des requêtes ponctuelles comme "regarde la commande de ce client" ou "résume ce document".

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info."""
    return f"Info about {fruit_name}"

@tool
def veggie_info(veggie_name: str) -> str:
    """Look up veggie info."""
    return f"Info about {veggie_name}"

# Sous-agents - pas de checkpointer défini (hérite du parent)
fruit_agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[fruit_info],
    system_prompt="You are a fruit expert. Use the fruit_info tool. Respond in one sentence.",
)

veggie_agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[veggie_info],
    system_prompt="You are a veggie expert. Use the veggie_info tool. Respond in one sentence.",
)

# On enveloppe les sous-agents comme des tools pour l'agent externe
@tool
def ask_fruit_expert(question: str) -> str:
    """Ask the fruit expert. Use for ALL fruit questions."""
    response = fruit_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content

@tool
def ask_veggie_expert(question: str) -> str:
    """Ask the veggie expert. Use for ALL veggie questions."""
    response = veggie_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content

# Agent externe avec checkpointer
agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[ask_fruit_expert, ask_veggie_expert],
    system_prompt=(
        "You have two experts: ask_fruit_expert and ask_veggie_expert. "
        "ALWAYS delegate questions to the appropriate expert."
    ),
    checkpointer=MemorySaver(),
)
```

**Comportement — chaque invocation repart d'un état frais.** Le sous-agent ne se souvient pas des appels précédents :

```python
config = {"configurable": {"thread_id": "1"}}

# Premier appel
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples"}]},
    config=config,
)
# Nombre de messages du sous-agent : 4

# Deuxième appel - le sous-agent repart de zéro, aucune mémoire des pommes
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Now tell me about bananas"}]},
    config=config,
)
# Nombre de messages du sous-agent : 4 (toujours frais !)
```

Plusieurs appels au même subgraph fonctionnent sans conflit, car chaque invocation obtient son propre **namespace de checkpoint** :

```python
config = {"configurable": {"thread_id": "1"}}

# Le LLM appelle ask_fruit_expert pour les pommes ET les bananes
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples and bananas"}]},
    config=config,
)
# Nombre de messages du sous-agent : 4 (pommes - frais)
# Nombre de messages du sous-agent : 4 (bananes - frais)
```

**Support des interrupts** — chaque invocation peut utiliser `interrupt()` pour se mettre en pause et reprendre. Ajouter `interrupt()` dans une fonction de tool permet d'exiger une validation humaine avant de continuer :

```python
@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info."""
    interrupt("continue?")
    return f"Info about {fruit_name}"
```

```python
config = {"configurable": {"thread_id": "1"}}

# Invoque - le tool call du sous-agent déclenche interrupt()
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples"}]},
    config=config,
)
# response contient __interrupt__

# Reprend - on approuve l'interrupt
response = agent.invoke(Command(resume=True), config=config)
```

### 3.2 Per-thread

À utiliser quand un sous-agent doit **se souvenir des interactions précédentes** — par exemple un assistant de recherche qui construit du contexte sur plusieurs échanges, ou un assistant de code qui garde la trace des fichiers déjà édités. L'historique de conversation et les données du sous-agent s'accumulent entre les appels sur le même thread. On active ce comportement avec `checkpointer=True` à la compilation.

⚠️ **Les subgraphs per-thread ne supportent pas les appels d'outils en parallèle.** Quand un LLM a accès à un sous-agent per-thread comme tool, il peut essayer de l'appeler plusieurs fois en parallèle (ex: demander à l'expert fruits pour les pommes et les bananes simultanément). Cela cause des conflits de checkpoint, car les deux appels écrivent dans le même namespace. On utilise `ToolCallLimitMiddleware` de LangChain pour empêcher ça.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info."""
    return f"Info about {fruit_name}"

# Sous-agent avec checkpointer=True pour un état persistant
fruit_agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[fruit_info],
    system_prompt="You are a fruit expert. Use the fruit_info tool. Respond in one sentence.",
    checkpointer=True,
)

@tool
def ask_fruit_expert(question: str) -> str:
    """Ask the fruit expert. Use for ALL fruit questions."""
    response = fruit_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return response["messages"][-1].content

# Agent externe -- ToolCallLimitMiddleware empêche les appels parallèles
# vers un sous-agent per-thread, qui causeraient des conflits de checkpoint.
agent = create_agent(
    model="groq:llama-3.1-8b-instant",
    tools=[ask_fruit_expert],
    system_prompt="You have a fruit expert. ALWAYS delegate fruit questions to ask_fruit_expert.",
    middleware=[
        ToolCallLimitMiddleware(tool_name="ask_fruit_expert", run_limit=1),
    ],
    checkpointer=MemorySaver(),
)
```

**Comportement — l'état s'accumule entre les invocations**, le sous-agent se souvient des conversations passées :

```python
config = {"configurable": {"thread_id": "1"}}

# Premier appel
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples"}]},
    config=config,
)
# Nombre de messages du sous-agent : 4

# Deuxième appel - le sous-agent SE SOUVIENT de la conversation sur les pommes
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Now tell me about bananas"}]},
    config=config,
)
# Nombre de messages du sous-agent : 8 (accumulé !)
```

**Isolation de namespace entre plusieurs sous-agents per-thread différents** — si tu as plusieurs subgraphs per-thread **différents** (ex: un expert fruits et un expert légumes), chacun a besoin de son propre espace de stockage pour que leurs checkpoints ne s'écrasent pas mutuellement.

Si tu [appelles des subgraphs dans un nœud](#21-appeler-un-subgraph-dans-un-nœud-schémas-différents), LangGraph assigne les namespaces selon l'ordre d'appel (premier appel, deuxième appel...). Réordonner tes appels peut donc mélanger quel subgraph charge quel état. Pour éviter ça, on enveloppe chaque sous-agent dans son propre `StateGraph` avec un nom de nœud unique — ce qui donne à chaque subgraph un namespace stable :

```python
from langgraph.graph import MessagesState, StateGraph

def create_sub_agent(model, *, name, **kwargs):
    """Enveloppe un agent avec un nom de nœud unique pour l'isolation de namespace."""
    agent = create_agent(model=model, name=name, **kwargs)
    return (
        StateGraph(MessagesState)
        .add_node(name, agent)  # nom unique -> namespace stable
        .add_edge("__start__", name)
        .compile()
    )

fruit_agent = create_sub_agent(
    "groq:llama-3.1-8b-instant", name="fruit_agent",
    tools=[fruit_info], system_prompt="...", checkpointer=True,
)
veggie_agent = create_sub_agent(
    "groq:llama-3.1-8b-instant", name="veggie_agent",
    tools=[veggie_info], system_prompt="...", checkpointer=True,
)
```

Les subgraphs [ajoutés comme nœuds](#22-ajouter-un-subgraph-directement-comme-un-nœud-schémas-partagés) obtiennent déjà des namespaces basés sur leur nom automatiquement — ils n'ont pas besoin de ce wrapper.

### 3.3 Stateless

À utiliser quand tu veux exécuter un sous-agent comme un simple appel de fonction, **sans overhead de checkpointing**. Le subgraph ne peut pas être mis en pause/repris et ne bénéficie pas de l'exécution durable. On compile avec `checkpointer=False`.

⚠️ Sans checkpointing, le subgraph n'a aucune exécution durable. Si le processus crashe en plein milieu, le subgraph ne peut pas récupérer et doit être ré-exécuté depuis le début.

```python
subgraph_builder = StateGraph(...)
subgraph = subgraph_builder.compile(checkpointer=False)
```

### 3.4 Tableau récapitulatif

```python
subgraph = builder.compile(checkpointer=False)  # ou True / None
```

| Fonctionnalité | Per-invocation (défaut) | Per-thread | Stateless |
| --- | --- | --- | --- |
| `checkpointer=` | `None` | `True` | `False` |
| Interrupts (HITL) | ✅ | ✅ | ❌ |
| Mémoire multi-tours | ❌ | ✅ | ❌ |
| Appels multiples (subgraphs différents) | ✅ | ⚠️ | ✅ |
| Appels multiples (même subgraph) | ✅ | ❌ | ✅ |
| Inspection d'état | ⚠️ | ✅ | ❌ |

- **Interrupts (HITL)** : le subgraph peut utiliser `interrupt()` pour suspendre l'exécution et attendre une entrée utilisateur, puis reprendre où il s'était arrêté.
- **Mémoire multi-tours** : le subgraph conserve son état entre plusieurs invocations au sein du même thread, plutôt que de repartir de zéro.
- **Appels multiples (subgraphs différents)** : plusieurs instances différentes de subgraphs peuvent être invoquées dans un même nœud sans conflit de namespace.
- **Appels multiples (même subgraph)** : la même instance de subgraph peut être invoquée plusieurs fois dans un même nœud. En persistance stateful, ces appels écrivent dans le même namespace de checkpoint et entrent en conflit — utiliser la persistance per-invocation à la place.
- **Inspection d'état** : l'état du subgraph est disponible via `get_state(config, subgraphs=True)`, pour le debug et le monitoring.

---

## 4. Visualiser l'état d'un subgraph

Quand la persistance est activée, tu peux inspecter l'état d'un subgraph via l'option `subgraphs`. Avec le mode stateless (`checkpointer=False`), aucun checkpoint n'est sauvegardé, donc l'état du subgraph n'est pas disponible.

⚠️ Visualiser l'état d'un subgraph nécessite que LangGraph puisse le **découvrir statiquement** — c'est-à-dire qu'il soit [ajouté comme nœud](#22-ajouter-un-subgraph-directement-comme-un-nœud-schémas-partagés) ou [appelé dans un nœud](#21-appeler-un-subgraph-dans-un-nœud-schémas-différents). Ça ne fonctionne **pas** quand un subgraph est appelé à l'intérieur d'une fonction de **tool** (le pattern "sous-agents" utilisé plus haut). Les interrupts, eux, continuent de se propager jusqu'au graphe racine quel que soit le niveau d'imbrication.

**Per-invocation — état de l'invocation courante uniquement :**

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

class State(TypedDict):
    foo: str

def subgraph_node_1(state: State):
    value = interrupt("Provide value:")
    return {"foo": state["foo"] + value}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()  # hérite du checkpointer parent

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)

# Voir l'état du subgraph pour l'invocation courante
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state

graph.invoke(Command(resume="bar"), config)
```

**Per-thread — état accumulé sur toutes les invocations du thread :**

```python
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

subgraph_builder = StateGraph(MessagesState)
# ... nœuds et arêtes
subgraph = subgraph_builder.compile(checkpointer=True)

builder = StateGraph(MessagesState)
builder.add_node("agent", subgraph)
builder.add_edge(START, "agent")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "what did I say?"}]}, config)

# État accumulé du subgraph (inclut les messages des deux invocations)
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state
```

---

## 5. Streamer les sorties d'un subgraph

Pour inclure les sorties des subgraphs dans le flux streamé du graphe parent, on passe l'option `subgraphs=True` à `.stream()`. Ça permet de streamer à la fois les sorties du graphe parent **et** de n'importe quel subgraph.

Avec `version="v2"` (LangGraph >= 1.1), les événements de subgraph utilisent le même format `StreamPart`. Le champ **`ns`** (namespace) identifie le graphe source : `()` pour la racine, `("node_2:<task_id>",)` pour un subgraph.

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class SubgraphState(TypedDict):
    foo: str
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        print(chunk["ns"], chunk["data"])
```

Sortie :
```
() {'node_1': {'foo': 'hi! foo'}}
('node_2:e58e5673-...',) {'subgraph_node_1': {'bar': 'bar'}}
('node_2:e58e5673-...',) {'subgraph_node_2': {'foo': 'hi! foobar'}}
() {'node_2': {'foo': 'hi! foobar'}}
```

---

## 6. Points clés à retenir

- Un subgraph = un graphe complet utilisé comme un nœud dans un graphe parent — encapsulation, réutilisation, développement en équipes séparées.
- **Schémas différents** → appelle le subgraph dans une fonction de nœud avec transformation manuelle de l'état. **Schémas partagés** → passe le subgraph compilé directement à `add_node`.
- Le paramètre `checkpointer` sur `.compile()` du subgraph contrôle la persistance : `None` (per-invocation, le plus courant), `True` (per-thread, mémoire multi-tours), `False` (stateless, aucun overhead mais pas de reprise possible).
- Les sous-agents per-thread ne supportent pas les appels parallèles — utiliser `ToolCallLimitMiddleware` pour l'éviter.
- Avec plusieurs subgraphs per-thread différents, donner un **nom de nœud unique à chacun** pour garantir un namespace stable (isolation).
- L'inspection d'état (`get_state(..., subgraphs=True)`) ne fonctionne que si LangGraph peut découvrir le subgraph statiquement (ajouté comme nœud ou appelé dans un nœud) — pas s'il est cachdé derrière un tool.
- Pour streamer les sorties de subgraphs en plus du graphe parent, passer `subgraphs=True` à `.stream()`.

---

## 7. Pour aller plus loin

- Doc officielle Subgraphs : https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- Graph API : https://docs.langchain.com/oss/python/langgraph/graph-api
- Multi-agent systems : https://docs.langchain.com/oss/python/langchain/multi-agent
- Persistence : https://docs.langchain.com/oss/python/langgraph/persistence
- Interrupts (human-in-the-loop) : https://docs.langchain.com/oss/python/langgraph/interrupts
- Durable execution : https://docs.langchain.com/oss/python/langgraph/durable-execution