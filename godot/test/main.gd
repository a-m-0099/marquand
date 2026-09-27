# headless check: godot --headless --path godot/test (with marq serve running)
extends Node

func _ready() -> void:
	var a := await Marq.decide({"hunger": 95, "energy": 80, "time": "noon", "nearby": ["tavern", "well"]}, {
		"next": {"type": "choice", "instructions": "What should this villager do next?",
				 "criteria": {"eat": "go eat at the tavern", "rest": "go to bed", "wander": "walk around"}},
		"starving": {"type": "noul", "instructions": "Is the villager very hungry?"}})
	print("MARQ_RESULT ", JSON.stringify(a))
	get_tree().quit(0 if a.has("next") and a.next.choice in ["eat", "rest", "wander"] and a.starving.has("noul") else 1)
