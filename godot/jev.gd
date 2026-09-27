# Local Jev client for Godot 4. Add as an autoload named "Jev" (Project Settings > Globals).
# Needs `jev serve --model fast` running (default http://127.0.0.1:8765).
#
#   var a := await Jev.decide({"hunger": 72, "energy": 30, "time": "evening"}, {
#       "next": {"type": "choice", "instructions": "What should this villager do next?",
#                "criteria": {"eat": "go eat", "rest": "go to bed", "wander": "walk around", "idle": "stay"}},
#       "tired": {"type": "noul", "instructions": "Is the villager exhausted?"}})
#   a.next.choice        -> "eat"
#   a.next.confidence    -> 0.83   (gate on this: low confidence = fall back to your rule-based choice)
#   a.tired.noul         -> 0.71
extends Node

@export var url := "http://127.0.0.1:8765/v1/systemone"
@export var model := "jev-fast"


## Returns the `answers` dictionary, or {} if the server is unreachable or rejects the request.
func decide(state: Variant, questions: Dictionary) -> Dictionary:
	var http := HTTPRequest.new()
	add_child(http)
	var body := JSON.stringify({"model": model, "state": state, "questions": questions})
	if http.request(url, ["Content-Type: application/json"], HTTPClient.METHOD_POST, body) != OK:
		http.queue_free()
		return {}
	var res: Array = await http.request_completed  # [result, code, headers, body]
	http.queue_free()
	if res[0] != HTTPRequest.RESULT_SUCCESS or res[1] != 200:
		push_warning("jev: request failed (result %d, HTTP %d)" % [res[0], res[1]])
		return {}
	var parsed: Variant = JSON.parse_string(res[3].get_string_from_utf8())
	return parsed.get("answers", {}) if parsed is Dictionary else {}
