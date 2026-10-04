#!/usr/bin/env python3
"""
Generate custom per-video VQA files for PAI-bench evaluation.

This is useful when evaluating videos generated from a custom prompt suite rather
than the official benchmark metadata/VQA package.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# Allow standalone execution via `python path/to/generate_custom_vqa.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCORE_RUBRIC = (
    "Use an integer score from 0 to 5. "
    "Score 0 if the event is missing, severely off-prompt, or physically contradicted. "
    "Score 1 if there is only weak evidence and major artifacts such as teleporting, hovering, pre-contact reactions, or random motion. "
    "Score 2 if the intended event is partly visible but causality, continuity, or outcome is clearly flawed. "
    "Score 3 if the event is mostly correct but still has noticeable physical inconsistencies. "
    "Score 4 if the event is strongly supported with only minor issues. "
    "Score 5 if the event is clear, temporally coherent, causally correct, and physically plausible."
)

CATEGORY_CODE_MAP = {
    "FA": "falling",
    "PR": "projectile",
    "SW": "swinging",
    "CP": "compression",
    "SP": "spinning",
    "SL": "sliding",
    "BO": "bouncing",
    "FL": "fluid_flowing",
    "FT": "floating",
    "SH": "splashing",
}

ALIGNMENT_ONLY_QUESTIONS: List[str] = [
    # Prompt–video alignment
    "Score how well the overall video matches the main event described in the text prompt.",
    "Score how well the main object, scene, and setting in the video match the text prompt.",
    "Score how well the visible motion matches the action described in the text prompt.",
    "Score how consistently the video stays on-prompt without drifting to unrelated content.",
    "Score how clearly the intended interaction or outcome described in the text is depicted in the video.",
    # Physics plausibility (still judged in light of the text prompt)
    "Score how temporally continuous the main motion is, penalizing teleporting, freezing, or abrupt reversals that contradict the prompt sequence.",
    "Score how well cause and effect follow visible contact or interaction, penalizing reactions before contact or non-causal jumps relative to the prompt.",
    "Score how physically plausible support, weight, and gravity cues are, penalizing hovering or unsupported motion where the prompt implies contact or falling.",
    "Score how plausible changes in speed, damping, and settling are for the described action, penalizing endless identical cycles or impossible instantaneous stops.",
    "Score how consistent object identity, shape, and boundaries are across frames, penalizing flicker, swaps, or unrelated warping not implied by the prompt.",
    "Score how plausible materials and deformations look when the prompt implies soft, liquid, breaking, or splashing behavior versus stiff or disconnected motion.",
]


CATEGORY_QUESTIONS: Dict[str, List[str]] = {
    "falling": [
        "Score how clearly the prompted object undergoes a genuine fall rather than unrelated motion or prompt drift.",
        "Score the temporal continuity of the downward motion, penalizing jumps, teleporting, or abrupt reversals.",
        "Score whether the object remains unsupported while descending instead of hovering or pausing unnaturally in midair.",
        "Score whether visible surface contact happens before any stop, rebound, or settle behavior.",
        "Score how physically plausible the post-contact outcome looks, such as bounce, slide, or stable rest rather than a sudden freeze.",
        "Score whether the object ends meaningfully lower than it began without implausible upward recovery.",
    ],
    "projectile": [
        "Score how clearly the projectile, target, and impact event are all visible and consistent with the prompt.",
        "Score the coherence of the projectile trajectory toward the target, penalizing wandering, stalling, or unrealistic path changes.",
        "Score whether the target reacts only after visible contact, not before impact or without contact.",
        "Score the plausibility of the impact aftermath, such as deflection, breakage, scattering, or recoil.",
        "Score how strongly the observed change appears causally caused by the collision rather than unrelated motion.",
        "Score the temporal smoothness of the projectile motion before impact, penalizing teleporting, freezing, or sudden redirection.",
    ],
    "swinging": [
        "Score how clearly the object performs repeated back-and-forth swinging rather than random drifting.",
        "Score how well the path resembles a smooth arc instead of straight jumps or disconnected positions.",
        "Score whether the object stays attached to its support point, hinge, chain, or string throughout the motion.",
        "Score whether direction changes happen near the arc endpoints rather than abruptly in the middle of the swing.",
        "Score the physical plausibility of the amplitude evolution, including damping or stable repeated swings.",
        "Score the overall temporal coherence of the swing, penalizing jitter, flicker, or random motion.",
    ],
    "compression": [
        "Score how clearly the prompted compression, squashing, denting, or pressing event is visible.",
        "Score whether the shape change begins at the moment of visible force or contact rather than at random times.",
        "Score how localized the deformation is to the compressed object, penalizing unrelated scene warping.",
        "Score the strength and clarity of the peak deformation relative to the object's initial shape.",
        "Score the plausibility of recovery after force is reduced or removed, penalizing impossible frozen shapes.",
        "Score the causal timing between interaction, deformation, and recovery, especially whether deformation appears only after contact.",
    ],
    "spinning": [
        "Score how clearly the object spins or rotates rather than merely vibrating or jittering in place.",
        "Score whether the object rotates around a stable center or axis instead of changing orientation randomly.",
        "Score the continuity of the rotation across frames, penalizing flicker, teleporting, or identity changes.",
        "Score the physical plausibility of how spin speed evolves over time, including slowdown or wobble.",
        "Score whether any slowdown is gradual rather than an abrupt drop to rest.",
        "Score how well the visible motion matches a real spinning object rather than camera-only motion or jitter.",
    ],
    "sliding": [
        "Score how clearly the object moves along the visible surface in a way that matches a sliding or skidding event.",
        "Score whether the object stays in visible contact with the ground or support surface while moving.",
        "Score whether the motion progresses smoothly along the surface rather than jumping, teleporting, or lifting away.",
        "Score how consistent the direction of travel is with the visible slope, push, curb, or scene layout.",
        "Score how plausible the later motion looks, including slowing down, wobbling, tipping, or settling at the end.",
        "Score the overall realism of the sliding event shown in the video.",
    ],
    "bouncing": [
        "Score how clearly the object shows true bouncing behavior after impact with a surface.",
        "Score whether each rebound occurs only after visible contact with the surface, penalizing midair reversals.",
        "Score the clarity of upward rebound after impact rather than only downward motion or ambiguous movement.",
        "Score the physical plausibility of energy loss across successive bounces, including lower rebound heights.",
        "Score how well the clip avoids non-causal motion such as bouncing without contact or sudden direction changes.",
        "Score the plausibility of the ending state, such as settling, sliding, or stopping instead of endlessly identical rebounds.",
    ],
    "fluid_flowing": [
        "Score how clearly the clip shows a liquid or fluid moving rather than a rigid object sliding through the scene.",
        "Score whether the liquid appears to come from a visible source and move toward a visible surface, container, or destination.",
        "Score how continuously the liquid changes shape as it pours, streams, spreads, swirls, or sloshes.",
        "Score the temporal continuity of the flow, penalizing disconnected patches, freezing, or teleporting liquid.",
        "Score how consistent the overall direction of flow is over time, given the visible source, tilt, or gravity cue.",
        "Score the overall realism of the liquid-flow event, including plausible spreading, pooling, ripples, or trailing motion.",
    ],
    "floating": [
        "Score how clearly the object appears to float rather than fall, sink, or get thrown through the scene.",
        "Score whether the object remains supported by water, air, or another medium instead of dropping abruptly.",
        "Score how gently and continuously the object drifts, bobs, or rotates over time.",
        "Score the stability of the floating motion, penalizing jitter, teleporting, or violent direction changes.",
        "Score whether the object stays near the expected floating region instead of suddenly sinking or shooting upward.",
        "Score whether small motion changes look like natural buoyant drift instead of random unrelated motion.",
    ],
    "splashing": [
        "Score how clearly the splash event and its cause are both visible in the clip.",
        "Score whether the splash is triggered by a visible contact event such as impact, entry, stepping, or collision.",
        "Score how well the first burst of liquid matches the timing, location, and direction of the impact.",
        "Score the plausibility of secondary effects such as droplets, spray, or ripples after the initial splash.",
        "Score whether the disturbance expands and dissipates naturally instead of freezing into a static shape.",
        "Score how consistent the size and direction of the splash are with the strength and location of the cause.",
    ],
    "generic": [
        "Score how clearly the main object and physical event described in the prompt are visible.",
        "Score the temporal continuity of the main action, penalizing unrelated or inconsistent frames.",
        "Score the motion coherence of the objects, penalizing teleporting, freezing, or abrupt identity changes.",
        "Score whether interactions follow visible contact and plausible cause-and-effect.",
        "Score how consistently the main event remains understandable throughout the clip rather than disappearing into unrelated motion.",
        "Score the overall physical plausibility of the clip rather than visual inconsistency or prompt drift.",
    ],
}

CATEGORY_MOTION_FOCUS = {
    "falling": "the downward fall under gravity",
    "projectile": "the through-air motion toward impact",
    "swinging": "the back-and-forth arc motion",
    "compression": "the contact-driven deformation and recovery",
    "spinning": "the rotational motion around a stable axis",
    "sliding": "the surface-following sliding or skidding motion",
    "bouncing": "the contact-triggered rebound motion",
    "fluid_flowing": "the continuous liquid flow and shape change",
    "floating": "the supported drifting or bobbing motion",
    "splashing": "the impact-driven liquid burst and aftermath",
    "generic": "the main physical motion",
}

CATEGORY_CONTACT_FOCUS = {
    "falling": "surface contact before any stop, bounce, or settle behavior",
    "projectile": "target reaction only after visible collision",
    "swinging": "smooth direction changes near arc endpoints",
    "compression": "deformation appearing only after visible force or contact",
    "spinning": "rotation changes that come from the motion rather than random jitter",
    "sliding": "continuous contact with the support surface during travel",
    "bouncing": "rebound only after visible contact with the surface",
    "fluid_flowing": "flow changes caused by visible source, gravity, or surface interaction",
    "floating": "continued support by the surrounding medium rather than sudden falling",
    "splashing": "liquid burst and spray triggered by visible impact or entry",
    "generic": "visible cause-and-effect between interaction and motion",
}


def _load_prompt_manifest(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in prompt manifest: {path}")
    return data


def _infer_category_from_video_id(video_id: str) -> str | None:
    match = re.match(r"^([A-Za-z]{2})_\d+", video_id.strip())
    if not match:
        return None
    return CATEGORY_CODE_MAP.get(match.group(1).upper())


def _infer_category(prompt: str, video_id: str = "") -> str:
    category = _infer_category_from_video_id(video_id)
    if category is not None:
        return category

    text = prompt.lower()
    keyword_groups = [
        ("projectile", ["throw", "thrown", "toss", "tossed", "launch", "launched", "shoot", "shot", "projectile", "hit target", "impact target"]),
        ("bouncing", ["bounce", "bounces", "bouncing", "rebound", "rebounds", "trampoline"]),
        ("swinging", ["swing", "swings", "swinging", "pendulum", "back and forth", "arc"]),
        ("compression", ["compress", "compressed", "compression", "squash", "squashed", "dent", "dented", "flatten", "flattened", "press", "pressed"]),
        ("spinning", ["spin", "spins", "spinning", "rotate", "rotates", "rotating", "twirl", "twirls"]),
        ("sliding", ["slide", "slides", "sliding", "skid", "skids", "skidding", "glide", "glides", "gliding"]),
        ("fluid_flowing", ["pour", "pours", "pouring", "flow", "flows", "flowing", "stream", "streams", "spills", "spilling", "swirl"]),
        ("floating", ["float", "floats", "floating", "drift", "drifts", "drifting", "bob", "bobs", "buoyant"]),
        ("splashing", ["splash", "splashes", "splashing", "droplet", "droplets", "ripple", "ripples", "spray"]),
        ("falling", ["fall", "falls", "falling", "drop", "drops", "dropping", "slips from", "comes loose"]),
    ]
    for category_name, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return category_name
    return "generic"


def _split_prompt_into_parts(prompt: str) -> tuple[str, list[str]]:
    prompt_text = prompt.strip().rstrip(".")
    if "," in prompt_text:
        scene, rest = prompt_text.split(",", 1)
        scene = scene.strip()
        rest = rest.strip()
    else:
        scene, rest = "", prompt_text

    clauses = [
        part.strip()
        for part in re.split(r",\s+and\s+|,\s+then\s+|,\s*|\s+and then\s+|\s+and\s+", rest)
        if part.strip()
    ]
    if not clauses:
        clauses = [prompt_text]
    return scene, clauses


def _extract_object_phrase(first_clause: str, category: str) -> str:
    verb_patterns = {
        "falling": r"\b(slips?|falls?|drops?|comes loose|lands?)\b",
        "projectile": r"\b(is tossed|is thrown|is struck|is knocked|tips|thrown|tossed|slips|lands|sprays)\b",
        "swinging": r"\b(swings?|rocks?|moves?)\b",
        "compression": r"\b(is squeezed|is pressed|is crushed|is squashed|sags?|compresses?|pressed|squeezed|crushed)\b",
        "spinning": r"\b(spins?|rotates?|turns?|circles?|wobbles?)\b",
        "sliding": r"\b(slides?|skids?|glides?|moves?|travels?|bumps?)\b",
        "bouncing": r"\b(hits?|bounces?|rebounds?|lands?|drops?)\b",
        "fluid_flowing": r"\b(pours?|drips?|flows?|streams?|washes?|sloshes?|spills?)\b",
        "floating": r"\b(floats?|drifts?|rocks?|bobs?|turns?)\b",
        "splashing": r"\b(drops?|steps?|tips?|jumps?|is tossed|passes?|poured|splashes?)\b",
        "generic": r"\b(is|are|moves?|shows?)\b",
    }
    pattern = verb_patterns.get(category, verb_patterns["generic"])
    match = re.search(pattern, first_clause, flags=re.IGNORECASE)
    if match:
        candidate = first_clause[: match.start()].strip(" ,")
        if candidate:
            return candidate
    words = first_clause.split()
    return " ".join(words[: min(4, len(words))]).strip() or "the main object"


def _build_prompt_specific_questions(prompt: str, category: str) -> list[str]:
    prompt_text = prompt.strip().rstrip(".")
    scene, clauses = _split_prompt_into_parts(prompt_text)
    first_clause = clauses[0]
    middle_clause = clauses[min(1, len(clauses) - 1)]
    final_clause = clauses[-1]
    sequence_text = " -> ".join(clauses[:4])
    object_phrase = _extract_object_phrase(first_clause, category)
    motion_focus = CATEGORY_MOTION_FOCUS.get(category, CATEGORY_MOTION_FOCUS["generic"])
    contact_focus = CATEGORY_CONTACT_FOCUS.get(category, CATEGORY_CONTACT_FOCUS["generic"])

    scene_fragment = f' in the setting "{scene}"' if scene else ""
    return [
        f'Score how faithfully the video depicts "{first_clause}"{scene_fragment}, rather than drifting away from the prompt.',
        f'Score how physically plausible {motion_focus} is for {object_phrase}, especially during "{middle_clause}".',
        f'Score whether the sequence "{sequence_text}" unfolds in the correct temporal order without skips, teleporting, or abrupt reversals.',
        f'Score whether the video shows {contact_focus}, especially around "{middle_clause}" and "{final_clause}".',
        f'Score how plausible the final outcome "{final_clause}" looks given the earlier motion and interactions in the prompt.',
        f'Score the overall physical plausibility of the full prompt "{prompt_text}" as shown in the video.',
    ]


def _questions_for_prompt(prompt: str, category: str, question_set_name: str) -> tuple[list[str], str]:
    prompt_text = prompt.strip().rstrip(".")
    if question_set_name == "alignment_only":
        questions = [
            f'{question} Evaluate this specifically for the prompt: "{prompt_text}."'
            for question in ALIGNMENT_ONLY_QUESTIONS
        ]
        return questions, "alignment_only"

    questions = _build_prompt_specific_questions(prompt_text, category)
    return questions, category


def _make_qa(video_id: str, category: str, idx: int, question: str) -> dict:
    return {
        "uid": f"{video_id}_q{idx+1}",
        "question": question,
        "evaluation_mode": "score",
        "min_score": 0,
        "max_score": 5,
        "rubric": SCORE_RUBRIC,
        "task": category,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate custom VQA files for custom prompt evaluation")
    p.add_argument("--prompt-file", type=str, required=True, help="Prompt manifest JSON with video_id and prompt_en")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to write per-video VQA json files")
    p.add_argument(
        "--question-set",
        type=str,
        default="custom_physics",
        choices=["custom_physics", "alignment_only"],
        help="Which question set to generate.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _load_prompt_manifest(prompt_file)
    count = 0
    question_set_name = args.question_set
    for item in items:
        video_id = str(item["video_id"])
        prompt = str(item.get("prompt_en") or item.get("prompt") or "")
        category = _infer_category(prompt, video_id)
        questions, task_name = _questions_for_prompt(prompt, category, question_set_name)
        qa_data = [_make_qa(video_id, task_name, i, q) for i, q in enumerate(questions)]
        (output_dir / f"{video_id}.json").write_text(json.dumps(qa_data, indent=2))
        count += 1

    print(f"✅ Wrote {count} custom VQA files to: {output_dir}")
    if question_set_name == "alignment_only":
        print("Each file contains alignment-only 0-5 scoring criteria focused on text-video matching.")
    else:
        print("Each file contains category-specific 0-5 scoring criteria focused on physical plausibility.")


if __name__ == "__main__":
    main()
