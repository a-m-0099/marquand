# Headless smoke test: godot --headless --path godot/test   (needs `jev serve` on :8765)
extends Node

func _ready() -> void:
	var a := await Jev.decide({"hunger": 95, "energy": 80, "time": "noon", "nearby": ["tavern", "well"]}, {
		"next": {"type": "choice", "instructions": "What should this villager do next?",
				 "criteria": {"eat": "go eat at the tavern", "rest": "go to bed", "wander": "walk around"}},
		"starving": {"type": "noul", "instructions": "Is the villager very hungry?"}})
	print("JEV_RESULT ", JSON.stringify(a))
	get_tree().quit(0 if a.has("next") and a.next.choice in ["eat", "rest", "wander"] and a.starving.has("noul") else 1)
