"""
Reasoning prompt helpers for Weird AI.


This module introduces the first lightweight reasoning-related utilities for
Weird AI. These functions do not train a reasoning model yet. Instead, they
help students explore how prompt structure can encourage a model to produce
intermediate reasoning steps before generating a final answer.


Lesson 8 connects the Weird AI parody assistant to the reasoning model portion
of the course by focusing on inference-time reasoning prompts.
"""


def build_reasoning_prompt(user_prompt):
    """
    Convert a normal user prompt into a general reasoning prompt.


    The returned prompt should encourage the model to think through the task
    before giving its final answer.


    Args:
        user_prompt: The original user request as a string.


    Returns:
        A formatted prompt string that includes the original task and asks for
        step-by-step reasoning.
    """

    prompt = (
        "You are Weird AI.\n"
        "Think step-by-step.\n"
        "\n"
        f"Task: {user_prompt}\n"
        "\n"
        "Before answering, walk through your reasoning one step at a time.\n"
        "\n"
        "Final Answer:\n"
    )
    return prompt


def build_parody_reasoning_prompt(topic):
    """
    Build a reasoning prompt specifically for parody lyric generation.


    This prompt should guide Weird AI through a simple creative planning process
    before asking it to write the final parody lyrics.


    Args:
        topic: The parody topic or user request as a string.


    Returns:
        A formatted prompt string with step-by-step parody planning sections.
    """

    prompt = (
        "You are Weird AI.\n"
        "Think step-by-step.\n"
        "\n"
        f"Topic: {topic}\n"
        "\n"
        "Follow these planning steps before writing your lyrics:\n"
        "1. Identify the topic: Restate the subject of the parody.\n"
        "2. Identify funny ideas or exaggerations: Brainlist at least three "
        "absurd or exaggerated angles on the topic.\n"
        "3. Identify possible rhymes: List a few words or phrases that could "
        "serve as rhyme candidates.\n"
        "4. Write the final parody lyrics: Compose the complete parody using "
        "the ideas and rhymes above.\n"
        "\n"
        "Final Parody:\n"
    )
    return prompt


def build_closed_world_prompt(premises, question):
    """
    Build a prompt for closed-world reasoning.


    Closed-world reasoning means the model should answer using only the facts
    explicitly provided in the prompt. It should not use outside knowledge.


    Args:
        premises: A list of premise strings.
        question: The question to answer from those premises.


    Returns:
        A formatted prompt string for closed-world reasoning.
    """

    premises_list = "\n".join(f"  - {p}" for p in premises)

    prompt = (
        "Answer the following question using only the provided premises below. "
        "Do NOT use any outside knowledge or assumptions beyond these facts.\n"
        "\n"
        "Provided Premises:\n"
        f"{premises_list}\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        "Reasoning:\n"
        "Walk through which premises apply and how they lead to your answer.\n"
        "\n"
        "Answer:\n"
    )
    return prompt


def build_open_world_prompt(premises, question):
    """
    Build a prompt for open-world reasoning.


    Open-world reasoning allows the model to consider both the provided premises
    and relevant background knowledge. This can reveal contradictions between
    the prompt and the model's background knowledge.


    Args:
        premises: A list of premise strings.
        question: The question to answer.


    Returns:
        A formatted prompt string for open-world reasoning.
    """

    premises_list = "\n".join(f"  - {p}" for p in premises)

    prompt = (
        "Answer the following question using the provided premises AND any "
        "relevant background knowledge you have. "
        "Check whether the premises conflict with known facts and note any "
        "contradictions explicitly.\n"
        "\n"
        "Provided Premises:\n"
        f"{premises_list}\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        "Reasoning:\n"
        "1. Consider how the provided premises relate to the question.\n"
        "2. Bring in relevant background knowledge where helpful.\n"
        "3. Flag any contradictions between the premises and established facts.\n"
        "\n"
        "Answer:\n"
    )
    return prompt