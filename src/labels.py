TASKS = {
    "func": {
        "column": "type",
        "labels": ["FR", "NFR"],
    },
    "prio": {
        "column": "priority",
        "labels": ["Must", "Should", "Could"],
    },
    "hybrid": {
        "column": "hybrid_label",
        "labels": [
            "FR-Must",
            "FR-Should",
            "FR-Could",
            "NFR-Must",
            "NFR-Should",
            "NFR-Could",
        ],
    },
}


def label2id(task: str) -> dict:
    return {l: i for i, l in enumerate(TASKS[task]["labels"])}


def id2label(task: str) -> dict:
    return {i: l for i, l in enumerate(TASKS[task]["labels"])}
