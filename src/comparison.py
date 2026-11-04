"""Read-only joins of original labels, TypeSafe Jev and human reviews."""


FILTERS = ("", "complete", "any_disagreement", "original_jev", "original_human",
           "jev_human", "missing_jev", "missing_human", "no_original")


def original_verdict(item):
    if item["original_label"] == 1:
        return "malicious"
    if item["original_label"] == 0:
        return "non-malicious"
    return None


def comparison_state(item, jev, human):
    original = original_verdict(item)
    jev_verdict = jev["verdict"] if jev else None
    human_verdict = human["verdict"] if human else None
    # "uncertain" is a human abstention, not a binary disagreement.
    binary = ("malicious", "non-malicious")
    pairs = {
        "original_jev": original in binary and jev_verdict in binary and original != jev_verdict,
        "original_human": original in binary and human_verdict in binary and original != human_verdict,
        "jev_human": jev_verdict in binary and human_verdict in binary and jev_verdict != human_verdict,
    }
    complete = original in binary and jev_verdict in binary and human_verdict in binary
    return {"complete": complete, "agree": complete and len(set((original, jev_verdict, human_verdict))) == 1,
            "disagreements": pairs}


def matches(item, jev, human, status):
    state = comparison_state(item, jev, human)
    if status == "complete":
        return state["complete"]
    if status == "any_disagreement":
        return any(state["disagreements"].values())
    if status in ("original_jev", "original_human", "jev_human"):
        return state["disagreements"][status]
    if status == "missing_jev":
        return jev is None
    if status == "missing_human":
        return human is None
    if status == "no_original":
        return original_verdict(item) is None
    return True


def filter_prompts(prompts, jev_reviews, human_reviews, dataset="", status="", label="", search="", offset=0, limit=12):
    needle = search.casefold().strip()
    results = []
    count = 0
    for item in prompts:
        if dataset and item["dataset"] != dataset:
            continue
        if label and str(item["original_label"]) != label:
            continue
        if needle and needle not in item["prompt"].casefold() and needle not in str(item["source"] or "").casefold():
            continue
        jev, human = jev_reviews.get(item["id"]), human_reviews.get(item["id"])
        if not matches(item, jev, human, status):
            continue
        if offset <= count < offset + limit:
            results.append({**item, "jev": jev, "human": human,
                            "comparison": comparison_state(item, jev, human)})
        count += 1
    return {"total": count, "items": results}


def summary(prompts, jev_reviews, human_reviews):
    counts = {"total": len(prompts), "with_original": 0, "with_jev": 0,
              "with_human": 0, "complete": 0, "agree": 0, "disagree": 0}
    for item in prompts:
        jev, human = jev_reviews.get(item["id"]), human_reviews.get(item["id"])
        counts["with_original"] += original_verdict(item) is not None
        counts["with_jev"] += jev is not None
        counts["with_human"] += human is not None
        state = comparison_state(item, jev, human)
        if state["complete"]:
            counts["complete"] += 1
            counts["agree"] += state["agree"]
            counts["disagree"] += not state["agree"]
    return counts
