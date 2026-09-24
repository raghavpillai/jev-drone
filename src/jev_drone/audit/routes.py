"""Check that active multi-leg bypass goals came from an accepted Jev choice."""


def audit_routes(calls, tasks=()):
    selected = {}
    problems = []
    for call in calls:
        if call["role"] != "control":
            continue
        state = call["state"]
        active = state.get("active_detour")
        if active and "planned_route" in active:
            route = selected.get((active["decision_time"], active["choice"]))
            leg = active["leg"]
            if (
                route != active["planned_route"]
                or not route
                or not 0 <= leg < len(route)
                or active["position"] != route[leg]
                or active["remaining_route"] != route[leg + 1 :]
            ):
                problems.append("Active detour leg lacks a matching accepted Jev route choice")
        if call.get("accepted"):
            effect = state["control_effects"][call["choice"]]
            if "planned_route" in effect:
                selected[(call["time"], call["choice"])] = effect["planned_route"]
    for task in tasks:
        if task["outcome"] != "replan_requested":
            continue
        matching = [
            call
            for call in calls
            if call["role"] == "control"
            and call.get("accepted")
            and call.get("choice") == "brake_and_replan"
            and call["state"]["task"]["name"] == task["name"]
            and task["time"] <= call["time"]
            and 0 <= task["ended"] - call["response_time"] <= 0.2
        ]
        if not matching:
            problems.append("Model-requested replanning lacks an accepted Jev handoff choice")
    return sorted(set(problems))
