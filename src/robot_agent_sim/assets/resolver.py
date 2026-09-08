from .registry import AssetRecord, AssetRegistry


class AssetResolver:
    def __init__(self, registry: AssetRegistry | None = None):
        self.registry = registry or AssetRegistry()

    def resolve_entities(self, entities) -> dict[str, AssetRecord]:
        resolved = {}
        for entity in entities:
            resolved[entity.entity_id] = self.registry.resolve(entity.category, entity.semantic_name, entity.aliases)
        return resolved
