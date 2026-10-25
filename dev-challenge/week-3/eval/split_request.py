"""Split a lazy request into LIBERO-Object items, in the order named.

The rule splitter in front of the step-by-step robot: "grab the butter and
the ketchup" becomes ["butter", "ketchup"], and eval/chain_rollout.py then
asks for one item at a time. Items are found by name, or by a short name
that fits only one item ("soup", "pudding"); "sauce" alone fits two items
and is not matched.

Usage (from dev-challenge/week-3):
    python eval/split_request.py "grab the butter and the ketchup"
"""

import re
import sys

ITEMS = [
    "alphabet soup", "cream cheese", "salad dressing", "bbq sauce", "ketchup",
    "tomato sauce", "butter", "milk", "chocolate pudding", "orange juice",
]
SHORT_NAMES = {
    "soup": "alphabet soup",
    "cheese": "cream cheese",
    "dressing": "salad dressing",
    "bbq": "bbq sauce",
    "barbecue sauce": "bbq sauce",
    "pudding": "chocolate pudding",
    "juice": "orange juice",
}


def split_request(request: str) -> list[str]:
    text = request.lower()
    names = {item: item for item in ITEMS} | SHORT_NAMES
    found = []  # (start, end, item)
    # Longest names first, so "cream cheese" is taken before "cheese".
    for name in sorted(names, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(name)}\b", text):
            if not any(match.start() < end and start < match.end() for start, end, _ in found):
                found.append((match.start(), match.end(), names[name]))
    items = []
    for _, _, item in sorted(found):
        if item not in items:
            items.append(item)
    return items


if __name__ == "__main__":
    print(split_request(" ".join(sys.argv[1:])))
