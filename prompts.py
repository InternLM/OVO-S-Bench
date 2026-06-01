"""
Pluggable prompt templates for OVO-S evaluation.

Usage:
    from prompts import build_prompt

    prompt = build_prompt(question, options)                  # default
    prompt = build_prompt(question, options, style="cot")     # chain-of-thought

Register new templates with the @register decorator:

    @register("my_style")
    def my_prompt(question, options):
        ...
        return prompt_string
"""

from typing import Dict


PROMPT_REGISTRY: Dict[str, callable] = {}


def register(name: str):
    """Decorator to register a prompt template function."""
    def decorator(fn):
        PROMPT_REGISTRY[name] = fn
        return fn
    return decorator


def build_prompt(question: str, options: Dict[str, str], style: str = "default") -> str:
    """Build a prompt using the named template.

    Falls back to 'default' if *style* is not found in the registry.
    """
    fn = PROMPT_REGISTRY.get(style, PROMPT_REGISTRY["default"])
    return fn(question, options)


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

@register("default")
def _default_prompt(question: str, options: Dict[str, str]) -> str:
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"""You are evaluating a video understanding task. Based on the video frames provided, answer the following multiple choice question.

Question: {question}

Options:
{options_text}

Instructions:
- Respond with ONLY the letter of your answer (e.g., "A" or "B").
- Do not include any explanation or additional text.

Your answer:"""


@register("verbose")
def _verbose_prompt(question: str, options: Dict[str, str]) -> str:
    """Original prompt with extra analysis instructions."""
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"""You are evaluating a video understanding task. Based on the video frames provided, answer the following multiple choice question.

Question: {question}

Options:
{options_text}

Instructions:
- Carefully analyze the video frames to understand the spatial relationships and scene content.
- Select the most appropriate answer from the given options.
- Respond with ONLY the letter of your answer (e.g., "A" or "B").
- Do not include any explanation or additional text.

Your answer:"""


@register("cot")
def _cot_prompt(question: str, options: Dict[str, str]) -> str:
    """Chain-of-thought: model reasons step-by-step then gives the answer."""
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    return f"""You are evaluating a video understanding task. Based on the video frames provided, answer the following multiple choice question.

Question: {question}

Options:
{options_text}

Instructions:
- Think step by step about the spatial relationships and scene content shown in the frames.
- After your reasoning, output your final answer on a new line in the format: Answer: X

Your reasoning and answer:"""
