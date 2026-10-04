"""Prompt templates used for binary and score-based VQA evaluation."""


def video_template_fn_v0(frame_num: int) -> str:
    return f"This is sampled image frame {frame_num} from the video."


# Binary VQA templates for YES/NO questions
system_template_binary_v0 = (
    """You are a helpful AI assistant that answers questions about videos. Answer with just YES or NO."""
)

begin_user_template_binary_v0 = (
    "I'll show you a video with several frames. Please look carefully at all frames to understand "
    "what's happening in the video, then answer the question about the video with either YES or NO."
)


def user_template_binary_v0(qa_pair: dict, is_reasoning: bool = False) -> str:
    """Format a binary YES/NO question without output format control.

    Args:
        qa_pair: A dictionary containing the question
        is_reasoning: Whether to include reasoning instructions

    Returns:
        Formatted question prompt
    """
    question_prompt = f"Question: {qa_pair['question']}"

    if is_reasoning:
        question_prompt += "\n\nPlease answer with YES or NO and explain your reasoning."
    else:
        question_prompt += "\n\nPlease answer with YES or NO."

    return question_prompt


binary_schema = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string", "enum": ["YES", "NO"]},
    },
    "required": [
        "reasoning",
        "answer",
    ],
    "additionalProperties": False,
}


def output_format_fn_binary_v0(api_service: str, output_string: bool) -> dict | None:
    if output_string:
        return None
    if api_service == "oai":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": binary_schema,
            },
        }
    elif api_service == "gemini":
        binary_schema_copy = binary_schema.copy()
        if "additionalProperties" in binary_schema_copy:
            binary_schema_copy.pop("additionalProperties")
        return binary_schema_copy
    elif api_service == "anthropic":
        binary_schema_copy = binary_schema.copy()
        if "additionalProperties" in binary_schema_copy:
            binary_schema_copy.pop("additionalProperties")
        return {
            "name": "get_response",
            "description": "Get the response from the model",
            "input_schema": binary_schema_copy,
        }
    else:
        raise ValueError(f"API service {api_service} not supported")


# Function to generate a combined schema for multiple binary questions
def generate_binary_batch_schema(question_ids: list) -> dict:
    properties = {}
    required = []

    for q_id in question_ids:
        question_key = f"question_{q_id}"
        answer_key = f"answer_{q_id}"
        properties[question_key] = {"type": "string"}
        properties[answer_key] = {"type": "string", "enum": ["YES", "NO"]}
        required.extend([question_key, answer_key])

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# Function to format output for batch binary questions
def output_format_fn_binary_batch(api_service: str, question_ids: list, output_string: bool = False) -> dict | None:
    if output_string:
        return None

    batch_schema = generate_binary_batch_schema(question_ids)

    if api_service == "oai":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": batch_schema,
            },
        }
    elif api_service == "gemini":
        if "additionalProperties" in batch_schema:
            batch_schema.pop("additionalProperties")
        return batch_schema
    elif api_service == "anthropic":
        if "additionalProperties" in batch_schema:
            batch_schema.pop("additionalProperties")
        return {
            "name": "get_response",
            "description": "Get the response from the model",
            "input_schema": batch_schema,
        }
    else:
        raise ValueError(f"API service {api_service} not supported")


templates = {
    # Binary VQA templates
    "system_template_v0_binary": system_template_binary_v0,
    "user_templates_v0_binary": [begin_user_template_binary_v0, user_template_binary_v0],
    "output_format_fn_v0_binary": output_format_fn_binary_v0,
    "video_template_fn_v0_binary": video_template_fn_v0,
}


# Score-based VQA templates for 0-5 rubric questions
system_template_score_v0 = (
    "You are a careful video evaluator. "
    "For each criterion, give a single integer score from 0 to 5, where "
    "0 means the criterion is not satisfied at all and 5 means it is fully satisfied."
)

begin_user_template_score_v0 = (
    "I'll show you a video with several frames. Please inspect the whole video carefully. "
    "Rate the criterion using a single integer from 0 to 5. "
    "Use this rubric: 0=no evidence or clearly violated, 1=very weak, "
    "2=partially satisfied, 3=mostly satisfied, 4=strongly satisfied, 5=fully satisfied."
)


def user_template_score_v0(qa_pair: dict, is_reasoning: bool = False) -> str:
    """Format a score-based rubric question."""
    min_score = qa_pair.get("min_score", 0)
    max_score = qa_pair.get("max_score", 5)
    rubric = qa_pair.get(
        "rubric",
        f"Use an integer from {min_score} to {max_score}. Higher means the criterion is better satisfied.",
    )
    question_prompt = f"Criterion: {qa_pair['question']}\nRubric: {rubric}"

    if is_reasoning:
        question_prompt += f"\n\nPlease answer with a score from {min_score} to {max_score} and explain briefly."
    else:
        question_prompt += f"\n\nPlease answer with a single integer score from {min_score} to {max_score}."

    return question_prompt


templates.update(
    {
        "system_template_v0_score": system_template_score_v0,
        "user_templates_v0_score": [begin_user_template_score_v0, user_template_score_v0],
        "video_template_fn_v0_score": video_template_fn_v0,
    }
)
