class Chunk:
    def __init__(self, content: str, source: str):
        self.content = content
        self.source = source

    def __repr__(self) -> str:
        return f"Chunk(content={self.content[:50]}..., source={self.source})"
