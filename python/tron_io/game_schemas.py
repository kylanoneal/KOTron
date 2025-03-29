import jsonschema
from jsonschema import validate

PLAYER_SCHEMA = {
    "type": "object",
    "properties": {
        "player_num": {"type": "integer"},
        "direction": {"type": "integer"},
        "head_pos": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer"},
            },
        },
    },
    "required": ["game_height", "game_width", "positions", "winner"],
}


POSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "grid": {"type": "array"},
        "players": {"type": "array", "items": PLAYER_SCHEMA},
    },
    "required": ["grid", "players"],
}

GAME_SCHEMA = {
    "type": "object",
    "properties": {
        "game_width": {"type": "integer"},
        "game_height": {"type": "integer"},
        "positions": {"type": "array", "items": POSITION_SCHEMA},
        "winner": {"type": "integer"},
    },
    "required": ["game_height", "game_width", "positions", "winner"],
}


GAME_COLLECTION_SCHEMA = {
    "type": "array",
    "items": GAME_SCHEMA,
}


if __name__ == "__main__":
    example_json = [
        {
            "game_width": 10,
            "game_height": 10,
            "winner": 0,
            "positions": [{"grid": [], "players": []}],
        }
    ]

    print(validate(instance=example_json, schema=GAME_COLLECTION_SCHEMA))
