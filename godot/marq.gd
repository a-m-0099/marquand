# Marquand client for Godot 4 - add it as an autoload named "Marq" and run `marq serve --model fast`
#
#   var a := await Marq.decide({"hunger": 72, "energy": 30}, {
#       "next": {"type": "choice", "instructions": "What should this villager do next?",
#                "criteria": {"eat": "go eat", "rest": "go to bed", "wander": "walk around"}}})
#   a.next.choice -> "eat", a.next.confidence -> 0.83 (if it's low, fall back to your own logic)
extends Node

@export var url := "http://127.0.0.1:8765/v1/systemone"
@export var model := "marq-fast"


## returns the answers, or {} if the server is down or says no
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
		push_warning("marq: request failed (result %d, HTTP %d)" % [res[0], res[1]])
		return {}
	var parsed: Variant = JSON.parse_string(res[3].get_string_from_utf8())
	return parsed.get("answers", {}) if parsed is Dictionary else {}
